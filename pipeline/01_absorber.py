"""
pipeline/01_absorber.py
=======================
Step 1 — Absorb an existing codebase into the knowledge layer.

Runs once when taking over a legacy project, and on-demand when the codebase
changes significantly enough to warrant a refresh.

Phases:
  1. File tree scan     — apply absorber.ignored rules, build file inventory
  2. Content extraction — full / key-only / signature-only per file
  3. Semantic compression — single LLM call → codebase_map.md
  4. Config inventory   — aggregate key-only extractions → config_map.json
  5. Git crawl          — git log → git_history.json + blame_map.md

External integrations (optional, graceful fallback):
  - vfs CLI            — signature extraction (98% token reduction)
  - Serena MCP         — symbol-level call graph (future: via subprocess)

Change detection (CocoIndex-inspired):
  - absorber_cache.json tracks file hashes
  - Only re-extracts files that changed since last run
  - --force flag bypasses cache

Usage:
  python 01_absorber.py                         # interactive git scope prompt
  python 01_absorber.py --git-scope 6m          # last 6 months
  python 01_absorber.py --git-scope 500         # last 500 commits
  python 01_absorber.py --git-scope all         # full history
  python 01_absorber.py --force                 # ignore cache, re-extract all
  python 01_absorber.py --skip-git              # skip git crawl
  python 01_absorber.py --dry-run               # scan only, no writes
  python 01_absorber.py --target /path/to/repo  # explicit target (default: ROOT)

Writes (owner: 01_absorber):
  artifacts/knowledge/current/codebase_map.md
  artifacts/knowledge/current/config_map.json
  artifacts/knowledge/current/blame_map.md
  artifacts/knowledge/history/git_history.json
  artifacts/cache/absorber_cache.json

For taxonomy details see docs/artifacts.md
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# === WRITE AUTHORITY: 01_absorber ===
# OWNS  : artifacts/knowledge/current/codebase_map.md
#         artifacts/knowledge/current/config_map.json
#         artifacts/knowledge/current/blame_map.md
#         artifacts/knowledge/history/git_history.json
#         artifacts/cache/absorber_cache.json
# READS : project source files (target codebase)

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (
    ROOT, CACHE_DIR, CURRENT_DIR, HISTORY_DIR,
    ensure_dirs,
)
ensure_dirs()

# ── Output paths ─────────────────────────────────────────────────────────────
CODEBASE_MAP   = CURRENT_DIR / "codebase_map.md"        # owner: 01_absorber
CONFIG_MAP     = CURRENT_DIR / "config_map.json"         # owner: 01_absorber
BLAME_MAP      = CURRENT_DIR / "blame_map.md"            # owner: 01_absorber
GIT_HISTORY    = HISTORY_DIR / "git_history.json"        # owner: 01_absorber
ABSORBER_CACHE = CACHE_DIR   / "absorber_cache.json"     # owner: 01_absorber

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_TOKENS_MAP   = 16384
_MAX_FILE_BYTES   = 256 * 1024   # 256 KB — skip very large files
_IGNORED_FILE     = "absorber.ignored"

# Model config — use long-context model for codebase synthesis
_MODEL = os.environ.get("ABSORBER_MODEL", "gemini/gemini-2.5-flash")

# Built-in skip dirs (align with VFS + common sense)
_BUILTIN_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", "vendor", ".git", "testdata",
    "dist", "build", ".next", "__pycache__", ".venv",
    "venv", ".tox", ".terraform", "target", "coverage",
    ".nyc_output", "storybook-static", ".parcel-cache",
    ".turbo", ".cache", "tmp", "temp",
})

# Built-in test file patterns (skipped by default — same as VFS)
_BUILTIN_SKIP_PATTERNS: tuple[str, ...] = (
    "*_test.go", "*.test.ts", "*.test.tsx", "*.test.js",
    "*.spec.ts", "*.spec.tsx", "*.spec.js",
    "test_*.py", "*_test.py", "*Test.java", "*Tests.java",
)

# Extensions that support key-only extraction
_KEY_ONLY_EXTENSIONS: frozenset[str] = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".ini",
    ".env", ".properties", ".cfg", ".conf",
})

# Source code extensions for signature extraction
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".ts", ".tsx", ".js", ".jsx",
    ".py", ".go", ".java", ".rs",
    ".cs", ".cpp", ".c", ".h",
    ".rb", ".php", ".kt", ".swift",
})


# ─────────────────────────────────────────────────────────────────────────────
# absorber.ignored parser
# ─────────────────────────────────────────────────────────────────────────────

class AbsorberIgnoreRules:
    """
    Parses absorber.ignored, which extends .gitignore syntax with directives:

      # Standard — skip entirely (same as .gitignore)
      node_modules/
      *.lock

      # Key-only — extract keys, redact values
      [key-only]
      **/appsettings*.json
      **/.env*

      # Signature-only — extract exports/interfaces, no body
      [signature-only]
      src/generated/**
      migrations/**
    """

    def __init__(self, rules_path: Path) -> None:
        self.skip_patterns:      list[str] = []
        self.key_only_patterns:  list[str] = []
        self.sig_only_patterns:  list[str] = []
        self._parse(rules_path)

    def _parse(self, path: Path) -> None:
        if not path.exists():
            return
        mode = "skip"
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "[key-only]":
                mode = "key-only"
            elif line == "[signature-only]":
                mode = "signature-only"
            else:
                if mode == "skip":
                    self.skip_patterns.append(line)
                elif mode == "key-only":
                    self.key_only_patterns.append(line)
                else:
                    self.sig_only_patterns.append(line)

    @staticmethod
    def _matches(rel_path: str, pattern: str) -> bool:
        """
        Match rel_path against a gitignore-style glob pattern.

        Supports:
          **   — match zero or more path segments (cross-directory)
          *    — match any sequence of characters within one path segment
          ?    — match any single character within one path segment

        Simple patterns (no '/') are matched against both the full relative
        path and just the filename, so "*.lock" catches "yarn.lock" and
        "sub/yarn.lock".  Patterns containing '/' are path-anchored.
        """
        # Build a regex from the glob pattern (gitignore semantics)
        pat = pattern.rstrip("/")
        parts = re.split(r"(\*\*|\*|\?)", pat)
        rx = ""
        for part in parts:
            if part == "**":
                rx += ".*"         # matches across directory boundaries
            elif part == "*":
                rx += "[^/]*"      # matches within a single path segment
            elif part == "?":
                rx += "[^/]"
            else:
                rx += re.escape(part)

        # Simple patterns (no '/' in original, no '**') apply to the filename
        # segment anywhere in the tree, so prefix with an optional dir component.
        if "/" not in pat and "**" not in pat:
            rx = r"(?:.+/)?" + rx

        compiled = re.compile(r"^" + rx + r"$")
        return bool(compiled.match(rel_path))

    def mode_for(self, rel_path: str) -> str:
        """Return 'skip', 'key-only', 'signature-only', or 'full'."""
        # Check signature-only first (more restrictive)
        for pat in self.sig_only_patterns:
            if self._matches(rel_path, pat):
                return "signature-only"
        for pat in self.key_only_patterns:
            if self._matches(rel_path, pat):
                return "key-only"
        for pat in self.skip_patterns:
            if self._matches(rel_path, pat):
                return "skip"
        return "full"


# ─────────────────────────────────────────────────────────────────────────────
# Change detection cache (CocoIndex-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, Any]:
    if ABSORBER_CACHE.exists():
        try:
            return json.loads(ABSORBER_CACHE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    ABSORBER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ABSORBER_CACHE.write_text(json.dumps(cache, indent=2))


def _file_hash(path: Path) -> str:
    """SHA-256 of file content, truncated to 16 hex chars."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — File tree scan
# ─────────────────────────────────────────────────────────────────────────────

def _should_skip_dir(name: str) -> bool:
    return name in _BUILTIN_SKIP_DIRS or name.startswith(".")


def _should_skip_file(name: str) -> bool:
    for pat in _BUILTIN_SKIP_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _looks_like_config_file(rel_path: str, fname: str, ext: str) -> bool:
    """
    Heuristic to auto-promote likely config/manifest/infra files to key-only mode.
    Covers generic naming keywords, well-known filenames, and infra path patterns.
    """
    if ext not in _KEY_ONLY_EXTENSIONS:
        return False

    rel_lower  = rel_path.lower()
    name_lower = fname.lower()

    # .env files (any variant) are always secrets
    if ext == ".env" or name_lower.startswith(".env"):
        return True

    # Generic config-ish keywords in filename
    if any(kw in name_lower for kw in (
        "config", "settings", "secret", "appsetting",
        "credential", "password", "token",
        "manifest", "values", "override",
    )):
        return True

    # Well-known filenames that are always config/metadata
    if name_lower in {
        "package.json",
        "package-lock.json",
        "composer.json",
        "pom.xml",
        "launchsettings.json",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.spec.json",
        "tsconfig.base.json",
        "angular.json",
        "dynamic-env.json",
        "ecs-task-def.json",
        "db-migrator-task-def.json",
    }:
        return True

    # Infra / deployment descriptor paths
    if any(token in rel_lower for token in (
        "task-def",
        "cloudformation",
        "helm",
        "k8s",
        "kubernetes",
        "deploy",
        "deployment",
        "docker-compose",
    )):
        return True

    return False


def _should_include_in_config_inventory(rel_path: str) -> bool:
    """
    Filter out pure build/tooling files from the config inventory.
    These have no runtime config relevance (tsconfig, lock files, e2e scaffolding,
    package manifests whose service keyword hits are false positives).
    """
    rel_lower = rel_path.lower()
    name      = Path(rel_path).name.lower()

    # Pure build/tooling filenames — no runtime config value
    if name in {
        "package-lock.json",
        "package.json",         # npm deps contain keywords that false-positive service detect
        "angular.json",         # Angular CLI config, not runtime
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.spec.json",
        "tsconfig.base.json",
    }:
        return False

    # e2e directories contain test fixtures, not runtime config
    # Check without requiring a leading "/" so it matches "angular/e2e/..." too
    if "e2e/" in rel_lower or rel_lower.startswith("e2e/"):
        return False

    return True


def scan_files(
    target: Path,
    rules: AbsorberIgnoreRules,
) -> list[dict[str, Any]]:
    """
    Walk target directory and build file inventory.
    Returns list of {rel_path, abs_path, ext, size_bytes, mode, lang}.
    """
    inventory: list[dict[str, Any]] = []

    for root_dir, dirs, files in os.walk(target):
        root_path = Path(root_dir)

        # Prune skipped dirs in-place (affects os.walk)
        dirs[:] = [
            d for d in dirs
            if not _should_skip_dir(d)
        ]

        for fname in files:
            if _should_skip_file(fname):
                continue

            abs_path = root_path / fname
            rel_path = str(abs_path.relative_to(target))
            ext      = abs_path.suffix.lower()

            # Check absorber.ignored rules
            mode = rules.mode_for(rel_path)
            if mode == "skip":
                continue

            # Auto-promote to key-only for likely config/manifest files
            if mode == "full" and _looks_like_config_file(rel_path, fname, ext):
                mode = "key-only"

            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_BYTES:
                continue  # skip very large files
            if size == 0:
                continue

            lang = _detect_language(ext)
            if lang is None and mode == "full":
                continue  # skip binary/unknown unless explicitly mapped

            inventory.append({
                "rel_path": rel_path,
                "abs_path": str(abs_path),
                "ext":      ext,
                "size":     size,
                "mode":     mode,
                "lang":     lang,
            })

    inventory.sort(key=lambda x: x["rel_path"])
    return inventory


def _detect_language(ext: str) -> str | None:
    mapping = {
        ".ts": "TypeScript", ".tsx": "TypeScript",
        ".js": "JavaScript", ".jsx": "JavaScript",
        ".py": "Python",
        ".go": "Go",
        ".java": "Java",
        ".rs": "Rust",
        ".cs": "C#",
        ".cpp": "C++", ".c": "C", ".h": "C/C++",
        ".rb": "Ruby",
        ".php": "PHP",
        ".kt": "Kotlin",
        ".swift": "Swift",
        ".sql": "SQL",
        ".json": "JSON",
        ".yaml": "YAML", ".yml": "YAML",
        ".toml": "TOML",
        ".tf": "Terraform", ".hcl": "HCL",
        ".proto": "Protobuf",
        ".md": "Markdown",
        ".sh": "Shell", ".bash": "Shell",
        ".dockerfile": "Dockerfile",
        ".xml": "XML",
        ".env": "ENV",
        ".ini": "INI", ".cfg": "Config",
        ".properties": "Properties",
    }
    # Special case for files named "Dockerfile"
    return mapping.get(ext)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Content extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_content(
    entry: dict[str, Any],
    cache: dict[str, Any],
    force: bool,
) -> tuple[str, bool]:
    """
    Extract content from a file according to its mode.
    Returns (content, from_cache).
    Uses change-detection cache to skip unchanged files.
    """
    abs_path = Path(entry["abs_path"])
    rel_path = entry["rel_path"]
    mode     = entry["mode"]

    current_hash = _file_hash(abs_path)
    cached = cache.get(rel_path, {})

    # Cache hit — return cached extraction
    if (
        not force
        and cached.get("hash") == current_hash
        and cached.get("mode") == mode
        and "content" in cached
    ):
        return cached["content"], True

    # Extract fresh
    if mode == "full":
        content = _extract_full(abs_path)
    elif mode == "key-only":
        content = _extract_key_only(abs_path, entry["ext"])
    else:  # signature-only
        content = _extract_signature(abs_path, entry["ext"], entry["lang"])

    # Update cache
    cache[rel_path] = {
        "hash":    current_hash,
        "mode":    mode,
        "content": content,
        "lang":    entry["lang"],
        "size":    entry["size"],
    }
    return content, False


def _extract_full(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception as e:
        return f"[read error: {e}]"


def _extract_key_only(path: Path, ext: str) -> str:
    """
    Parse structured config files and return key structure only.
    Values are replaced with <redacted> to prevent secret leakage.
    """
    raw = _extract_full(path)

    if ext in (".json",):
        return _redact_json(raw)
    elif ext in (".yaml", ".yml"):
        return _redact_yaml(raw)
    elif ext in (".toml",):
        return _redact_toml(raw)
    elif ext in (".env",) or path.name.startswith(".env"):
        return _redact_env(raw)
    else:
        # Generic: mask values on lines with = or :
        return _redact_generic(raw)


def _redact_json(raw: str) -> str:
    """Recursively redact JSON values, keep keys."""
    def _walk(obj: Any, depth: int = 0) -> Any:
        indent = "  " * depth
        if isinstance(obj, dict):
            if not obj:
                return "{}"
            lines = ["{"]
            for k, v in obj.items():
                child = _walk(v, depth + 1)
                lines.append(f'{indent}  "{k}": {child}')
            lines.append(indent + "}")
            return "\n".join(lines)
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            return f"[... {len(obj)} item(s)]"
        elif isinstance(obj, (int, float, bool)):
            return str(obj)   # keep primitive types (not secrets)
        else:
            return '"<redacted>"'

    try:
        parsed = json.loads(raw)
        return _walk(parsed)
    except Exception:
        return _redact_generic(raw)


def _redact_yaml(raw: str) -> str:
    lines_out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            lines_out.append(line)
            continue
        # Match "key: value" — redact value part
        match = re.match(r'^(\s*[\w\-\.]+\s*:)\s*(.+)$', line)
        if match:
            lines_out.append(match.group(1) + " <redacted>")
        else:
            lines_out.append(line)
    return "\n".join(lines_out)


def _redact_toml(raw: str) -> str:
    lines_out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("[") or not stripped:
            lines_out.append(line)
            continue
        match = re.match(r'^(\s*[\w\-\.]+\s*=)\s*(.+)$', line)
        if match:
            lines_out.append(match.group(1) + " <redacted>")
        else:
            lines_out.append(line)
    return "\n".join(lines_out)


def _redact_env(raw: str) -> str:
    lines_out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            lines_out.append(line)
            continue
        match = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$', line, re.IGNORECASE)
        if match:
            lines_out.append(f"{match.group(1)}=<redacted>")
        else:
            lines_out.append(line)
    return "\n".join(lines_out)


def _redact_generic(raw: str) -> str:
    lines_out: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//") or not stripped:
            lines_out.append(line)
            continue
        for sep in ("=", ":"):
            match = re.match(rf'^(\s*[^{sep}\n]+{sep})\s*(.+)$', line)
            if match:
                lines_out.append(match.group(1) + " <redacted>")
                break
        else:
            lines_out.append(line)
    return "\n".join(lines_out)


def _extract_signature(path: Path, ext: str, lang: str | None) -> str:
    """
    Extract signatures via vfs CLI if available, else Python AST for .py,
    else regex-based fallback for other languages.
    """
    # Try vfs first (best token reduction, multi-language)
    if shutil.which("vfs"):
        try:
            result = subprocess.run(
                ["vfs", str(path)],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception:
            pass

    # Python AST parser (no external deps)
    if ext == ".py":
        return _extract_python_signatures(path)

    # TypeScript/JavaScript — regex-based
    if ext in (".ts", ".tsx", ".js", ".jsx"):
        return _extract_ts_signatures(path)

    # Fallback — return first N lines as preview
    try:
        lines = path.read_text(errors="replace").splitlines()
        preview = "\n".join(lines[:50])
        if len(lines) > 50:
            preview += f"\n... ({len(lines) - 50} more lines)"
        return preview
    except Exception:
        return "[signature extraction failed]"


def _extract_python_signatures(path: Path) -> str:
    """Use Python AST to extract class/function signatures and docstrings."""
    try:
        source = path.read_text(errors="replace")
        tree   = ast.parse(source)
    except SyntaxError:
        return _extract_ts_signatures(path)  # fallback to regex

    lines: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level and class methods
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args  = [a.arg for a in node.args.args]
            sig   = f"{prefix} {node.name}({', '.join(args)})"
            if node.returns:
                sig += f" -> {ast.unparse(node.returns)}"
            lines.append(sig)
            if (docstring := ast.get_docstring(node)):
                lines.append(f'    """{docstring[:120]}"""')

        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            sig   = f"class {node.name}"
            if bases:
                sig += f"({', '.join(bases)})"
            lines.append(sig + ":")
            if (docstring := ast.get_docstring(node)):
                lines.append(f'    """{docstring[:120]}"""')

    return "\n".join(lines) if lines else path.read_text(errors="replace")[:500]


def _extract_ts_signatures(path: Path) -> str:
    """Regex-based TypeScript/JS/other signature extraction."""
    try:
        source = path.read_text(errors="replace")
    except Exception:
        return "[read error]"

    lines: list[str] = []
    patterns = [
        # export function / export async function
        r'export\s+(?:async\s+)?function\s+\w+[^{]*',
        # export const fn = (...) =>
        r'export\s+const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=>{]+)?(?:=>)?',
        # export class / abstract class
        r'(?:export\s+)?(?:abstract\s+)?class\s+\w+[^{]*',
        # export interface
        r'export\s+interface\s+\w+[^{]*',
        # export type
        r'export\s+type\s+\w+\s*=\s*[^;]+',
        # export enum
        r'export\s+enum\s+\w+',
        # @Component / @Injectable decorators
        r'@\w+\([^)]*\)',
    ]
    combined = re.compile("|".join(f"(?:{p})" for p in patterns))

    for match in combined.finditer(source):
        sig = match.group().strip()
        sig = re.sub(r'\s+', ' ', sig)[:200]
        lines.append(sig)

    return "\n".join(lines) if lines else source[:500]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Semantic compression → codebase_map.md
# ─────────────────────────────────────────────────────────────────────────────

_MAP_SYSTEM = textwrap.dedent("""
    You are a senior software architect performing a codebase intake.
    You will receive the extracted signatures and structure of a codebase.
    Your job is to produce a structured codebase_map.md document.

    Output a SINGLE markdown document with these sections (no extra commentary):

    # Codebase Map
    _Generated: {date} | Absorber v1_

    ## Project Overview
    [3-4 paragraph summary: what the system does, primary tech stack,
     architectural style (monolith/micro/serverless), key patterns observed]

    ## Module Inventory
    [For each logical module/directory, a subsection:
     ### <module-name>
     - **Purpose**: one sentence
     - **Key files**: comma-separated
     - **Primary exports**: function/class names
     - **Depends on**: other modules
    ]

    ## Entry Points & Call Flows
    [Top 3-5 most important call chains, traced from entry to outcome]

    ## Data Flow
    [How data moves: sources → transformations → sinks]

    ## Config Dependencies
    [Services, env vars, feature flags detected from config files]

    ## Detected Tech Debt
    [High-churn single-author files, TODO/FIXME patterns, old migrations,
     deprecated patterns — be specific with file names]

    ## Absorber Notes
    [Any ambiguities, files that could not be parsed, recommended follow-ups]
""").strip()


def call_llm_for_map(
    context: str,
    target_name: str,
    model: str = _MODEL,
) -> str:
    """Single LLM call to synthesize codebase_map.md from extracted content."""
    api_key, base_url, model_id = _resolve_model(model)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system   = _MAP_SYSTEM.replace("{date}", date_str)
    user     = (
        f"Codebase: {target_name}\n\n"
        f"Extracted content:\n\n{context}"
    )

    tokens_est = len(context) // 4
    print(f"[01] LLM call: {model_id} | ~{tokens_est:,} input tokens")

    # Validate API key early — httpx raises a cryptic error if the key is empty
    if not api_key or not api_key.strip():
        env_var = (
            "GEMINI_API_KEY"     if model.startswith(("gemini/", "gemini-")) else
            "DEEPSEEK_API_KEY"   if model.startswith("deepseek")             else
            "OPENROUTER_API_KEY" if "/" in model                             else
            "OPENAI_API_KEY"
        )
        raise ValueError(
            f"API key not set. Export {env_var} and retry.\n"
            f"  e.g.  export {env_var}=<your-key>\n"
            f"  or pass --skip-llm to skip LLM synthesis."
        )

    try:
        with httpx.Client(timeout=300) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "max_tokens": _MAX_TOKENS_MAP,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data         = resp.json()
            choice       = data["choices"][0]
            content      = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "unknown")
            if finish_reason == "length":
                print(
                    f"[01][warn] LLM output was truncated (finish_reason=length). "
                    f"Consider increasing _MAX_TOKENS_MAP (currently {_MAX_TOKENS_MAP}) "
                    f"or reducing codebase context."
                )
            else:
                print(f"[01] LLM finish_reason: {finish_reason}")
            return content
    except Exception as e:
        print(f"[01][error] LLM call failed: {e}", file=sys.stderr)
        raise


def _resolve_model(model: str) -> tuple[str, str, str]:
    """Return (api_key, base_url, model_id) for the given model string."""
    if model.startswith("gemini/") or model.startswith("gemini-"):
        model_id = model.split("/")[-1]
        return (
            os.environ.get("GEMINI_API_KEY", ""),
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_id,
        )
    elif model.startswith("deepseek"):
        return (
            os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            "https://api.deepseek.com/v1",
            model.split("/")[-1],
        )
    elif "/" in model:
        # OpenRouter format: provider/model
        return (
            os.environ.get("OPENROUTER_API_KEY", ""),
            "https://openrouter.ai/api/v1",
            model,
        )
    else:
        return (
            os.environ.get("OPENAI_API_KEY", ""),
            "https://api.openai.com/v1",
            model,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Config inventory → config_map.json
# ─────────────────────────────────────────────────────────────────────────────

# Module-level so both build_config_map and tests can import directly
_SERVICE_PATTERNS: dict[str, re.Pattern] = {
    "database":   re.compile(r'(?:postgres|mysql|mongodb|redis|sqlite|db_|database|connectionstring)', re.I),
    "messaging":  re.compile(r'(?:kafka|rabbitmq|sqs|sns|pubsub|amqp)', re.I),
    "auth":       re.compile(r'(?:auth|oauth|jwt|saml|sso|oidc|keycloak|openiddict)', re.I),
    "storage":    re.compile(r'(?:s3|gcs|azure_blob|minio|storage|bucket)', re.I),
    "monitoring": re.compile(r'(?:datadog|newrelic|prometheus|grafana|sentry|cloudwatch)', re.I),
    "email":      re.compile(r'(?:smtp|sendgrid|ses|mailgun|email)', re.I),
    "cloud":      re.compile(r'(?:aws|gcp|azure|heroku|fly\.io|ecs|fargate|cloudformation)', re.I),
}


def _extract_env_vars_from_raw(path: Path, ext: str) -> set[str]:
    """
    Extract env var names from raw (un-redacted) config/template content.

    Covers:
      - ${VAR} and $VAR shell-style interpolation
      - process.env.VAR (Node.js)
      - KEY=value lines (.env files)
      - SECTION__KEY double-underscore notation (.NET / container env overrides)
    """
    try:
        raw = path.read_text(errors="replace")
    except Exception:
        return set()
    return _parse_env_vars_from_text(raw)


def _parse_env_vars_from_text(raw: str) -> set[str]:
    """Core env var extraction logic — operates on a raw string.

    Covers four sources:
      1. ${VAR} / $VAR — shell-style interpolation (docker-compose, CloudFormation Fn::Sub)
      2. process.env.VAR — Node.js env references
      3. KEY=value lines — .env file style
      4. SECTION__KEY — .NET double-underscore env override notation
      5. JSON top-level keys — appsettings.json PascalCase keys converted to SCREAMING_SNAKE
         (e.g. "ConnectionStrings" → CONNECTION_STRINGS, "App" → APP)
         This covers ABP/.NET configs that don't use ${VAR} syntax.
    """
    env_vars: set[str] = set()

    # 1. ${VAR} and $VAR
    for match in re.findall(r'\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)', raw):
        env_vars.update(g for g in match if g)

    # 2. process.env.VAR (case-insensitive key name)
    env_vars.update(re.findall(
        r'process\.env\.([A-Z_][A-Z0-9_]*)',
        raw,
        re.IGNORECASE,
    ))

    # 3. KEY=value lines — .env style
    env_vars.update(re.findall(
        r'^\s*([A-Z_][A-Z0-9_]*)\s*=',
        raw,
        re.MULTILINE,
    ))

    # 4. SECTION__KEY double-underscore (.NET environment variable overrides)
    env_vars.update(re.findall(
        r'([A-Z_][A-Z0-9_]*(?:__[A-Z0-9_]+)+)',
        raw,
    ))

    # 5. JSON top-level keys → SCREAMING_SNAKE
    # Covers appsettings.json / appsettings.*.json patterns where .NET apps
    # expose config keys as env vars using double-underscore notation.
    # Only lift keys that look like config sections (PascalCase or ALLCAPS, ≥3 chars).
    json_keys = re.findall(r'"([A-Za-z][A-Za-z0-9]{2,})"\s*:', raw)
    for key in json_keys:
        # Convert PascalCase/camelCase → SCREAMING_SNAKE
        # e.g. ConnectionStrings → CONNECTION_STRINGS, App → APP, authServer → AUTH_SERVER
        screaming = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', key).upper()
        # Only include keys that are plausibly env-var-shaped (all caps, underscores, ≥3 chars)
        if re.match(r'^[A-Z][A-Z0-9_]{2,}$', screaming):
            env_vars.add(screaming)

    return env_vars


def build_config_map(
    inventory: list[dict[str, Any]],
    cache: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate key-only config files into a structured config inventory.

    Env var detection reads from the RAW source file (before redaction) so that
    ${VAR}, process.env.VAR, and .env KEY= references are never lost to the
    redaction pass.  Service detection scans both raw and redacted content.
    """
    config_files: list[dict[str, Any]] = []
    all_env_vars: set[str] = set()
    all_services: set[str] = set()

    for entry in inventory:
        if entry["mode"] != "key-only":
            continue

        rel_path = entry["rel_path"]

        # Skip pure build/tooling files — they add noise, not signal
        if not _should_include_in_config_inventory(rel_path):
            continue

        abs_path = Path(entry.get("abs_path", ""))
        ext      = entry["ext"]

        # Read raw content for accurate env var + service detection.
        # Fall back to cached content when abs_path is unavailable (e.g. tests).
        raw = ""
        if abs_path.exists():
            try:
                raw = abs_path.read_text(errors="replace")
            except Exception:
                pass
        if not raw:
            raw = cache.get(rel_path, {}).get("content", "")

        # Redacted content (from cache) as fallback for service scan
        redacted = cache.get(rel_path, {}).get("content", "")

        # Env vars — always from raw to avoid losing context after redaction
        file_env_vars = _parse_env_vars_from_text(raw)
        all_env_vars.update(file_env_vars)

        # Services — prefer raw; fall back to redacted if raw unavailable
        service_scan_text = raw or redacted
        file_services: list[str] = []
        for svc_name, pattern in _SERVICE_PATTERNS.items():
            if pattern.search(service_scan_text):
                file_services.append(svc_name)
                all_services.add(svc_name)

        config_files.append({
            "path":     rel_path,
            "env_vars": sorted(file_env_vars),
            "services": sorted(file_services),
        })

    return {
        "generated":         datetime.now(timezone.utc).isoformat(),
        "total_configs":     len(config_files),
        "services_detected": sorted(all_services),
        "env_vars_detected": sorted(all_env_vars),
        "files":             config_files,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Git crawl → git_history.json + blame_map.md
# ─────────────────────────────────────────────────────────────────────────────

def _ask_git_scope() -> str:
    """Interactive prompt to choose git history scope."""
    print("\n[01] Git history scope:")
    print("  1. Last 3 months")
    print("  2. Last 6 months")
    print("  3. Last 1 year")
    print("  4. All history")
    print("  5. Custom (number of commits or date range e.g. '500' or '2024-01-01')")
    choice = input("→ Choose [1-5]: ").strip()

    mapping = {"1": "3m", "2": "6m", "3": "1y", "4": "all"}
    if choice in mapping:
        return mapping[choice]
    elif choice == "5":
        custom = input("  Enter commits count or start date (YYYY-MM-DD): ").strip()
        return custom
    else:
        print("[01] Invalid choice, defaulting to 6 months.")
        return "6m"


def _scope_to_git_args(scope: str) -> list[str]:
    """Convert scope string to git log arguments."""
    if scope == "all":
        return []
    elif scope.endswith("m"):
        months = int(scope[:-1])
        return [f"--since={months} months ago"]
    elif scope.endswith("y"):
        years = int(scope[:-1])
        return [f"--since={years} years ago"]
    elif scope.isdigit():
        return [f"-n", scope]
    elif re.match(r'\d{4}-\d{2}-\d{2}', scope):
        return [f"--since={scope}"]
    else:
        return [f"--since=6 months ago"]


def crawl_git(target: Path, scope: str) -> dict[str, Any] | None:
    """
    Crawl git history for the target repo.
    Returns structured git_history dict or None if not a git repo.
    """
    git_dir = target / ".git"
    if not git_dir.exists():
        print("[01] No .git directory found — skipping git crawl.")
        return None

    scope_args = _scope_to_git_args(scope)

    # Get commit log
    git_cmd = [
        "git", "-C", str(target), "log",
        "--format=%H|||%ai|||%ae|||%s",
        "--numstat",
    ] + scope_args

    try:
        result = subprocess.run(
            git_cmd, capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("[01][warn] git log timed out — skipping git crawl.")
        return None

    if result.returncode != 0:
        print(f"[01][warn] git log failed: {result.stderr[:200]}")
        return None

    commits = _parse_git_log(result.stdout)
    if not commits:
        print("[01] No commits found for scope.")
        return None

    # Build file churn stats
    churn: dict[str, dict] = {}
    for commit in commits:
        for f in commit.get("files_changed", []):
            if f not in churn:
                churn[f] = {"count": 0, "authors": set()}
            churn[f]["count"] += 1
            churn[f]["authors"].add(commit["author"])

    hotspots = sorted(
        [
            {
                "file":         fp,
                "change_count": data["count"],
                "authors":      sorted(data["authors"]),
            }
            for fp, data in churn.items()
        ],
        key=lambda x: x["change_count"],
        reverse=True,
    )[:50]  # Top 50 hotspots

    # Collect unique authors
    authors = sorted({c["author"] for c in commits})

    return {
        "scope":         scope,
        "generated":     datetime.now(timezone.utc).isoformat(),
        "total_commits": len(commits),
        "authors":       authors,
        "hotspots":      hotspots,
        "commits":       commits,
    }


def _parse_git_log(raw: str) -> list[dict]:
    """Parse git log --format=%H|||%ai|||%ae|||%s --numstat output."""
    commits: list[dict] = []
    current: dict | None = None

    for line in raw.splitlines():
        # Strip leading/trailing whitespace to handle indented test fixtures
        # and trailing spaces, but preserve the tab structure of numstat lines.
        stripped = line.strip()

        if "|||" in stripped:
            # New commit header
            if current is not None:
                commits.append(current)
            parts = stripped.split("|||", 3)
            if len(parts) < 4:
                continue
            current = {
                "hash":          parts[0][:7],   # 7-char short hash (git convention)
                "date":          parts[1][:10],
                "author":        parts[2],
                "message":       parts[3][:200],
                "files_changed": [],
                "insertions":    0,
                "deletions":     0,
            }
        elif current is not None and stripped:
            # numstat line: insertions \t deletions \t filename
            # Use the stripped version for splitting (tabs preserved after strip)
            parts = stripped.split("\t", 2)
            if len(parts) == 3:
                ins_str, del_str, fname = parts
                try:
                    current["insertions"] += int(ins_str) if ins_str != "-" else 0
                    current["deletions"]  += int(del_str) if del_str != "-" else 0
                except ValueError:
                    pass
                if fname:
                    current["files_changed"].append(fname)

    if current is not None:
        commits.append(current)

    return commits


def build_blame_map(git_data: dict[str, Any]) -> str:
    """Generate human-readable blame_map.md from git history data."""
    now    = git_data["generated"][:10]
    scope  = git_data["scope"]
    total  = git_data["total_commits"]
    authors = git_data["authors"]

    lines: list[str] = [
        "# Codebase Hotspot Map",
        f"_Generated: {now} | Scope: {scope} | Commits analyzed: {total}_",
        "",
        "## Team",
        f"Active contributors: {', '.join(authors[:10])}"
        + (f" (+{len(authors)-10} more)" if len(authors) > 10 else ""),
        "",
    ]

    hotspots = git_data.get("hotspots", [])
    high     = [h for h in hotspots if h["change_count"] >= 10]
    medium   = [h for h in hotspots if 5 <= h["change_count"] < 10]

    if high:
        lines += [
            "## 🔴 High Churn Files (≥10 changes)",
            "",
            "| File | Changes | Authors |",
            "|------|---------|---------|",
        ]
        for h in high[:20]:
            auth_str = ", ".join(h["authors"][:3])
            if len(h["authors"]) > 3:
                auth_str += f" (+{len(h['authors'])-3})"
            lines.append(f"| `{h['file']}` | {h['change_count']} | {auth_str} |")
        lines.append("")

    if medium:
        lines += [
            "## 🟡 Medium Churn Files (5–9 changes)",
            "",
            "| File | Changes |",
            "|------|---------|",
        ]
        for h in medium[:15]:
            lines.append(f"| `{h['file']}` | {h['change_count']} |")
        lines.append("")

    # Module-level activity summary
    module_activity: dict[str, int] = {}
    for h in hotspots:
        parts = h["file"].split("/")
        if len(parts) >= 2:
            module = parts[0] if parts[0] not in ("src",) else parts[1]
            module_activity[module] = module_activity.get(module, 0) + h["change_count"]

    if module_activity:
        total_changes = sum(module_activity.values()) or 1
        lines += ["## Module Activity", ""]
        for module, count in sorted(module_activity.items(), key=lambda x: -x[1])[:10]:
            pct = round(count / total_changes * 100)
            lines.append(f"- **{module}**: {pct}% of changes ({count} file-changes)")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _build_extraction_context(
    inventory: list[dict[str, Any]],
    cache: dict[str, Any],
) -> str:
    """
    Assemble the full extraction context string for the LLM call.
    Groups files by directory and truncates very long individual extractions.
    """
    _MAX_PER_FILE = 2000  # chars
    _MAX_TOTAL    = 800_000  # chars (~200k tokens for Gemini)

    sections: list[str] = []
    total_chars = 0

    # Group by top-level directory
    groups: dict[str, list[dict]] = {}
    for entry in inventory:
        parts = entry["rel_path"].split("/")
        top   = parts[0] if len(parts) > 1 else "(root)"
        groups.setdefault(top, []).append(entry)

    for group_name, entries in sorted(groups.items()):
        group_lines = [f"\n## {group_name}/\n"]
        for entry in entries:
            rel  = entry["rel_path"]
            lang = entry["lang"] or ""
            content = cache.get(rel, {}).get("content", "")
            if not content:
                continue
            if len(content) > _MAX_PER_FILE:
                content = content[:_MAX_PER_FILE] + f"\n... [truncated, {len(content)} chars total]"
            group_lines.append(f"### {rel} ({lang}, {entry['mode']})\n```\n{content}\n```\n")

        chunk = "\n".join(group_lines)
        if total_chars + len(chunk) > _MAX_TOTAL:
            sections.append(f"\n[...{len(inventory) - len(sections)} more files truncated due to context limit]")
            break
        sections.append(chunk)
        total_chars += len(chunk)

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Absorb a codebase into the pipeline knowledge layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 01_absorber.py                         # interactive
              python 01_absorber.py --git-scope 6m          # last 6 months
              python 01_absorber.py --git-scope 500         # last 500 commits
              python 01_absorber.py --git-scope all         # full history
              python 01_absorber.py --skip-git              # skip git crawl
              python 01_absorber.py --force                 # ignore cache
              python 01_absorber.py --target /path/to/repo  # explicit target
              python 01_absorber.py --dry-run               # scan only
        """),
    )
    parser.add_argument(
        "--target", type=Path, default=ROOT,
        help="Path to codebase root (default: project root)",
    )
    parser.add_argument(
        "--git-scope", metavar="SCOPE", default=None,
        help="Git history scope: 3m, 6m, 1y, all, N (commits), YYYY-MM-DD",
    )
    parser.add_argument(
        "--skip-git", action="store_true",
        help="Skip git crawl entirely",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore absorber_cache.json — re-extract all files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report only — no LLM call, no file writes",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM synthesis — write raw extraction only",
    )
    args = parser.parse_args()

    target: Path = args.target.resolve()
    if not target.exists():
        print(f"[01][error] Target path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'─'*50}")
    print(f"  Absorber — {target.name}")
    print(f"{'─'*50}\n")

    # ── Phase 1: File tree scan ───────────────────────────────────────────────
    print("[01] Phase 1 — Scanning file tree ...")
    rules_path = target / _IGNORED_FILE
    rules      = AbsorberIgnoreRules(rules_path)
    inventory  = scan_files(target, rules)

    lang_counts: dict[str, int] = {}
    for entry in inventory:
        lang_counts[entry["lang"] or "other"] = lang_counts.get(entry["lang"] or "other", 0) + 1

    print(f"[01] Found {len(inventory)} files to process")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"     {lang}: {count}")

    mode_counts = {"full": 0, "key-only": 0, "signature-only": 0}
    for entry in inventory:
        mode_counts[entry["mode"]] = mode_counts.get(entry["mode"], 0) + 1
    print(f"[01] Extraction modes: {mode_counts}")

    if args.dry_run:
        print("\n[01] --dry-run: stopping here. No files written.")
        return

    # ── Phase 2: Content extraction ───────────────────────────────────────────
    print("\n[01] Phase 2 — Extracting content ...")
    cache = _load_cache()
    cache_hits = 0

    for i, entry in enumerate(inventory, 1):
        content, from_cache = extract_content(entry, cache, args.force)
        if from_cache:
            cache_hits += 1
        if i % 50 == 0:
            print(f"     {i}/{len(inventory)} files processed ...")

    total_chars = sum(len(cache.get(e["rel_path"], {}).get("content", "")) for e in inventory)
    est_tokens  = total_chars // 4
    print(f"[01] Extracted {len(inventory)} files | cache hits: {cache_hits}")
    print(f"[01] Total content: {total_chars:,} chars (~{est_tokens:,} tokens)")

    # Save cache after extraction
    _save_cache(cache)
    print(f"[01] ✓ Cache saved → {ABSORBER_CACHE}")

    # ── Phase 3: Semantic compression → codebase_map.md ──────────────────────
    if not args.skip_llm:
        print("\n[01] Phase 3 — Semantic compression (LLM) ...")
        context = _build_extraction_context(inventory, cache)
        try:
            codebase_map = call_llm_for_map(context, target.name)
            CODEBASE_MAP.write_text(codebase_map)
            print(f"[01] ✓ Codebase map → {CODEBASE_MAP}")
        except Exception as e:
            print(f"[01][warn] LLM synthesis failed: {e} — skipping codebase_map.md")
    else:
        # Write raw extraction as codebase_map for human review
        raw_context = _build_extraction_context(inventory, cache)
        CODEBASE_MAP.write_text(
            f"# Codebase Map (raw extraction — no LLM synthesis)\n\n{raw_context}"
        )
        print(f"[01] ✓ Raw extraction → {CODEBASE_MAP} (--skip-llm)")

    # ── Phase 4: Config inventory ─────────────────────────────────────────────
    print("\n[01] Phase 4 — Config inventory ...")
    config_map = build_config_map(inventory, cache)
    CONFIG_MAP.write_text(json.dumps(config_map, indent=2))
    print(f"[01] ✓ Config map → {CONFIG_MAP}")
    print(f"     Services detected: {', '.join(config_map['services_detected']) or 'none'}")
    print(f"     Env vars detected: {len(config_map['env_vars_detected'])}")

    # ── Phase 5: Git crawl ────────────────────────────────────────────────────
    if not args.skip_git:
        print("\n[01] Phase 5 — Git crawl ...")
        scope = args.git_scope
        if scope is None:
            scope = _ask_git_scope()

        git_data = crawl_git(target, scope)
        if git_data:
            GIT_HISTORY.write_text(json.dumps(git_data, indent=2))
            print(f"[01] ✓ Git history → {GIT_HISTORY}")
            print(f"     Commits: {git_data['total_commits']} | Authors: {len(git_data['authors'])}")
            print(f"     Hotspots: {len(git_data['hotspots'])} files")

            blame_md = build_blame_map(git_data)
            BLAME_MAP.write_text(blame_md)
            print(f"[01] ✓ Blame map → {BLAME_MAP}")
        else:
            print("[01] Git crawl skipped or failed.")
    else:
        print("\n[01] Phase 5 — Git crawl skipped (--skip-git).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Done — {target.name} absorbed")
    print(f"{'─'*50}")
    print(f"  codebase_map.md  → {CODEBASE_MAP}")
    print(f"  config_map.json  → {CONFIG_MAP}")
    if not args.skip_git:
        print(f"  git_history.json → {GIT_HISTORY}")
        print(f"  blame_map.md     → {BLAME_MAP}")
    print(f"\n  Next: python harness.py --skip-absorb")
    print(f"        (or review codebase_map.md before running pipeline)\n")


if __name__ == "__main__":
    main()
