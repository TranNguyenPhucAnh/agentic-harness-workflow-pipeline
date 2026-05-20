"""
pipeline/01_absorber.py
=======================
Step 1 — Absorb an existing codebase into the knowledge layer.

Runs once when taking over a legacy project, and on-demand when the codebase
changes significantly enough to warrant a refresh.

Phases:
  1. File tree scan        — apply absorber.ignored rules, build file inventory
  2. Content extraction    — full / key-only / signature-only per file
  3. Semantic compression  — single LLM call → codebase_map.md (includes Config + Git/Blame)
  4. Git crawl             — git log → merged into codebase_map.md ## Git/Blame section
  5. Append codebase_log   — long-term audit trail

External integrations optional, graceful fallback:
  - vfs CLI     — signature extraction
  - Serena MCP  — symbol-level call graph, future via subprocess

Change detection:
  - absorber/cache/codebase_snapshot.json tracks file hashes
  - Only re-extracts files that changed since last run
  - --force flag bypasses cache

Usage:
  python 01_absorber.py
  python 01_absorber.py --project my-app
  PIPELINE_PROJECT=my-app python 01_absorber.py

  python 01_absorber.py --git-scope 6m
  python 01_absorber.py --git-scope 500
  python 01_absorber.py --git-scope all
  python 01_absorber.py --force
  python 01_absorber.py --skip-git
  python 01_absorber.py --dry-run
  python 01_absorber.py --target /path/to/repo

Writes, owner: absorber (01_absorber.py):
  artifacts_<slug>/absorber/codebase_map.md          (short-term, overwrite)
  artifacts_<slug>/absorber/codebase_log.json        (long-term, append)
  artifacts_<slug>/absorber/cache/codebase_snapshot.json  (internal cache)

Reads:
  project source files (target codebase)
  artifacts_<slug>/absorber/cache/codebase_snapshot.json if present

At the end of each run, prints:
  - artifacts/files read
  - artifacts/files created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
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

# === WRITE AUTHORITY: absorber ===
# OWNS  : artifacts_<slug>/absorber/codebase_map.md (short-term, overwrite)
#         artifacts_<slug>/absorber/codebase_log.json (long-term, append)
#         artifacts_<slug>/absorber/cache/codebase_snapshot.json (cache - internal, overwrite)

# READS : project source files (target codebase)
#         artifacts_<slug>/absorber/cache/codebase_snapshot.json (cache)

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_LOG,
    ABSORBER_CODEBASE_MAP,
    ABSORBER_CODEBASE_SNAPSHOT,
    artifact_root,
    ensure_dirs,
)
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402

# Local aliases
ABSORBER_CACHE = ABSORBER_CODEBASE_SNAPSHOT
CODEBASE_MAP   = ABSORBER_CODEBASE_MAP
CODEBASE_LOG   = ABSORBER_CODEBASE_LOG


# ─────────────────────────────────────────────────────────────────────────────
# ── Constants ─────────────────────────────────────────────────────────────────

ROLE             = "absorber"
_MAX_TOKENS_MAP  = 16384
_MAX_FILE_BYTES = 256 * 1024
_IGNORED_FILE = "absorber.ignored"

_BUILTIN_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules",
    "vendor",
    ".git",
    "testdata",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".terraform",
    "target",
    "coverage",
    ".nyc_output",
    "storybook-static",
    ".parcel-cache",
    ".turbo",
    ".cache",
    "tmp",
    "temp",
})

# Pipeline-owned artifact subdirectories.
_ARTIFACT_CONTROL_DIRS: frozenset[str] = frozenset({
    "absorber",
    "clarificator",
    "enricher",
    "spectracker",
    "scaffolder",
    "planner",
    "executor",
    "debugger",
    "reporter",
    "judge",
    "patcher",
    "archivist",
    "spec",
    "output",
})

_BUILTIN_SKIP_PATTERNS: tuple[str, ...] = (
    "*_test.go",
    "*.test.ts",
    "*.test.tsx",
    "*.test.js",
    "*.spec.ts",
    "*.spec.tsx",
    "*.spec.js",
    "test_*.py",
    "*_test.py",
    "*Test.java",
    "*Tests.java",
)

_KEY_ONLY_EXTENSIONS: frozenset[str] = frozenset({
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".env",
    ".properties",
    ".cfg",
    ".conf",
})


# ─────────────────────────────────────────────────────────────────────────────
# absorber.ignored parser
# ─────────────────────────────────────────────────────────────────────────────

class AbsorberIgnoreRules:
    """
    Parses absorber.ignored, which extends .gitignore syntax with directives:

      # Standard — skip entirely
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
        self.skip_patterns: list[str] = []
        self.key_only_patterns: list[str] = []
        self.sig_only_patterns: list[str] = []
        self._parse(rules_path)

    def _parse(self, path: Path) -> None:
        if not path.exists():
            return

        track_read(path)

        mode = "skip"
        for raw in path.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line == "[key-only]":
                mode = "key-only"
            elif line == "[signature-only]":
                mode = "signature-only"
            elif mode == "skip":
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
          **   — match zero or more path segments
          *    — match any sequence within one path segment
          ?    — match any single character within one path segment
        """
        pat = pattern.rstrip("/")
        parts = re.split(r"(\*\*|\*|\?)", pat)

        rx = ""
        for part in parts:
            if part == "**":
                rx += ".*"
            elif part == "*":
                rx += "[^/]*"
            elif part == "?":
                rx += "[^/]"
            else:
                rx += re.escape(part)

        if "/" not in pat and "**" not in pat:
            rx = r"(?:.+/)?" + rx

        return bool(re.compile(r"^" + rx + r"$").match(rel_path))

    def mode_for(self, rel_path: str) -> str:
        """Return skip, key-only, signature-only, or full."""
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
# Change detection cache
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, Any]:
    if ABSORBER_CACHE.exists():
        try:
            track_read(ABSORBER_CACHE)
            return json.loads(ABSORBER_CACHE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    ABSORBER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ABSORBER_CACHE.write_text(json.dumps(cache, indent=2))
    track_write(ABSORBER_CACHE)


def _file_hash(path: Path) -> str:
    try:
        track_read(path)
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — File tree scan
# ─────────────────────────────────────────────────────────────────────────────

def _should_skip_dir(name: str, *, skip_artifact_control_dirs: bool = False) -> bool:
    if name.startswith("."):
        return True

    if name in _BUILTIN_SKIP_DIRS:
        return True

    # Always skip pipeline artifact output dirs (artifacts_<slug>/)
    if name.startswith("artifacts_"):
        return True

    if skip_artifact_control_dirs and name in _ARTIFACT_CONTROL_DIRS:
        return True

    return False


def _should_skip_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in _BUILTIN_SKIP_PATTERNS)


def _looks_like_config_file(rel_path: str, fname: str, ext: str) -> bool:
    """
    Heuristic to auto-promote likely config/manifest/infra files to key-only mode.
    """
    if ext not in _KEY_ONLY_EXTENSIONS:
        return False

    rel_lower = rel_path.lower()
    name_lower = fname.lower()

    if ext == ".env" or name_lower.startswith(".env"):
        return True

    if any(kw in name_lower for kw in (
        "config",
        "settings",
        "secret",
        "appsetting",
        "credential",
        "password",
        "token",
        "manifest",
        "values",
        "override",
    )):
        return True

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
    Filter out pure build/tooling files from config inventory.
    """
    rel_lower = rel_path.lower()
    name = Path(rel_path).name.lower()

    if name in {
        "package-lock.json",
        "package.json",
        "angular.json",
        "tsconfig.json",
        "tsconfig.app.json",
        "tsconfig.spec.json",
        "tsconfig.base.json",
    }:
        return False

    if "e2e/" in rel_lower or rel_lower.startswith("e2e/"):
        return False

    return True


def scan_files(
    target: Path,
    rules: AbsorberIgnoreRules,
    *,
    skip_artifact_control_dirs: bool = False,
) -> list[dict[str, Any]]:
    """
    Walk target directory and build file inventory.

    Returns:
      [
        {
          "rel_path": str,
          "abs_path": str,
          "ext": str,
          "size": int,
          "mode": "full" | "key-only" | "signature-only",
          "lang": str | None,
        },
        ...
      ]
    """
    inventory: list[dict[str, Any]] = []

    for root_dir, dirs, files in os.walk(target):
        root_path = Path(root_dir)

        dirs[:] = [
            d for d in dirs
            if not _should_skip_dir(
                d,
                skip_artifact_control_dirs=skip_artifact_control_dirs,
            )
        ]

        for fname in files:
            if _should_skip_file(fname):
                continue

            abs_path = root_path / fname

            try:
                rel_path = str(abs_path.relative_to(target))
            except ValueError:
                continue

            ext = abs_path.suffix.lower()

            mode = rules.mode_for(rel_path)
            if mode == "skip":
                continue

            if mode == "full" and _looks_like_config_file(rel_path, fname, ext):
                mode = "key-only"

            try:
                size = abs_path.stat().st_size
            except OSError:
                continue

            if size <= 0 or size > _MAX_FILE_BYTES:
                continue

            lang = _detect_language(abs_path)
            if lang is None and mode == "full":
                continue

            inventory.append({
                "rel_path": rel_path,
                "abs_path": str(abs_path),
                "ext": ext,
                "size": size,
                "mode": mode,
                "lang": lang,
            })

    inventory.sort(key=lambda x: x["rel_path"])
    return inventory


def _detect_language(path: Path) -> str | None:
    ext = path.suffix.lower()
    name = path.name.lower()

    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "Dockerfile"

    mapping = {
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".py": "Python",
        ".go": "Go",
        ".java": "Java",
        ".rs": "Rust",
        ".cs": "C#",
        ".cpp": "C++",
        ".c": "C",
        ".h": "C/C++",
        ".rb": "Ruby",
        ".php": "PHP",
        ".kt": "Kotlin",
        ".swift": "Swift",
        ".sql": "SQL",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".tf": "Terraform",
        ".hcl": "HCL",
        ".proto": "Protobuf",
        ".md": "Markdown",
        ".sh": "Shell",
        ".bash": "Shell",
        ".xml": "XML",
        ".env": "ENV",
        ".ini": "INI",
        ".cfg": "Config",
        ".conf": "Config",
        ".properties": "Properties",
    }
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
    Extract content according to file mode.

    Returns:
      (content, from_cache)
    """
    abs_path = Path(entry["abs_path"])
    rel_path = entry["rel_path"]
    mode = entry["mode"]

    current_hash = _file_hash(abs_path)
    cached = cache.get(rel_path, {})

    if (
        not force
        and cached.get("hash") == current_hash
        and cached.get("mode") == mode
        and "content" in cached
    ):
        return cached["content"], True

    if mode == "full":
        content = _extract_full(abs_path)
    elif mode == "key-only":
        content = _extract_key_only(abs_path, entry["ext"])
    else:
        content = _extract_signature(abs_path, entry["ext"], entry["lang"])

    cache[rel_path] = {
        "hash": current_hash,
        "mode": mode,
        "content": content,
        "lang": entry["lang"],
        "size": entry["size"],
    }
    return content, False


def _extract_full(path: Path) -> str:
    try:
        track_read(path)
        return path.read_text(errors="replace")
    except Exception as e:
        return f"[read error: {e}]"


def _extract_key_only(path: Path, ext: str) -> str:
    raw = _extract_full(path)

    if ext == ".json":
        return _redact_json(raw)

    if ext in (".yaml", ".yml"):
        return _redact_yaml(raw)

    if ext == ".toml":
        return _redact_toml(raw)

    if ext == ".env" or path.name.startswith(".env"):
        return _redact_env(raw)

    return _redact_generic(raw)


def _redact_json(raw: str) -> str:
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

        if isinstance(obj, list):
            if not obj:
                return "[]"
            return f"[... {len(obj)} item(s)]"

        if isinstance(obj, (int, float, bool)):
            return str(obj)

        return '"<redacted>"'

    try:
        return _walk(json.loads(raw))
    except Exception:
        return _redact_generic(raw)


def _redact_yaml(raw: str) -> str:
    lines_out: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            lines_out.append(line)
            continue

        match = re.match(r"^(\s*[\w\-\.]+\s*:)\s*(.+)$", line)
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

        match = re.match(r"^(\s*[\w\-\.]+\s*=)\s*(.+)$", line)
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

        match = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$", line, re.IGNORECASE)
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
            match = re.match(rf"^(\s*[^{sep}\n]+{sep})\s*(.+)$", line)
            if match:
                lines_out.append(match.group(1) + " <redacted>")
                break
        else:
            lines_out.append(line)

    return "\n".join(lines_out)


def _extract_signature(path: Path, ext: str, lang: str | None) -> str:
    """
    Extract signatures via vfs CLI if available, else Python AST for .py,
    else regex fallback for TS/JS, else preview.
    """
    if shutil.which("vfs"):
        try:
            result = subprocess.run(
                ["vfs", str(path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception:
            pass

    if ext == ".py":
        return _extract_python_signatures(path)

    if ext in (".ts", ".tsx", ".js", ".jsx"):
        return _extract_ts_signatures(path)

    try:
        track_read(path)
        lines = path.read_text(errors="replace").splitlines()
        preview = "\n".join(lines[:50])
        if len(lines) > 50:
            preview += f"\n... ({len(lines) - 50} more lines)"
        return preview
    except Exception:
        return "[signature extraction failed]"


def _extract_python_signatures(path: Path) -> str:
    try:
        track_read(path)
        source = path.read_text(errors="replace")
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_ts_signatures(path)
    except Exception:
        return "[signature extraction failed]"

    lines: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args = [a.arg for a in node.args.args]
            sig = f"{prefix} {node.name}({', '.join(args)})"

            if node.returns:
                try:
                    sig += f" -> {ast.unparse(node.returns)}"
                except Exception:
                    pass

            lines.append(sig)

            docstring = ast.get_docstring(node)
            if docstring:
                lines.append(f'    """{docstring[:120]}"""')

        elif isinstance(node, ast.ClassDef):
            try:
                bases = [ast.unparse(b) for b in node.bases]
            except Exception:
                bases = []

            sig = f"class {node.name}"
            if bases:
                sig += f"({', '.join(bases)})"
            lines.append(sig + ":")

            docstring = ast.get_docstring(node)
            if docstring:
                lines.append(f'    """{docstring[:120]}"""')

    if lines:
        return "\n".join(lines)

    try:
        track_read(path)
        return path.read_text(errors="replace")[:500]
    except Exception:
        return "[signature extraction failed]"


def _extract_ts_signatures(path: Path) -> str:
    try:
        track_read(path)
        source = path.read_text(errors="replace")
    except Exception:
        return "[read error]"

    patterns = [
        r"export\s+(?:async\s+)?function\s+\w+[^{]*",
        r"export\s+const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=>{]+)?(?:=>)?",
        r"(?:export\s+)?(?:abstract\s+)?class\s+\w+[^{]*",
        r"export\s+interface\s+\w+[^{]*",
        r"export\s+type\s+\w+\s*=\s*[^;]+",
        r"export\s+enum\s+\w+",
        r"@\w+\([^)]*\)",
    ]

    combined = re.compile("|".join(f"(?:{p})" for p in patterns))
    lines: list[str] = []

    for match in combined.finditer(source):
        sig = match.group().strip()
        sig = re.sub(r"\s+", " ", sig)[:200]
        lines.append(sig)

    return "\n".join(lines) if lines else source[:500]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Semantic compression → codebase_map.md
# ─────────────────────────────────────────────────────────────────────────────

_MAP_SYSTEM = textwrap.dedent("""
    You are a senior software architect performing a codebase intake.
    You will receive extracted signatures and structure of a codebase.
    Your job is to produce a structured codebase_map.md document.

    Output a SINGLE markdown document with these sections, no extra commentary:

    # Codebase Map
    _Generated: {date} | Absorber v2_

    ## Project Overview
    [3-4 paragraph summary: what the system does, primary tech stack,
     architectural style, and key patterns observed]

    ## Module Inventory
    [For each logical module/directory:
     ### <module-name>
     - **Purpose**: one sentence
     - **Key files**: comma-separated
     - **Primary exports**: function/class names
     - **Depends on**: other modules]

    ## Entry Points & Call Flows
    [Top 3-5 most important call chains, traced from entry to outcome]

    ## Data Flow
    [How data moves: sources → transformations → sinks]

    ## Config
    [Services, env vars, feature flags detected from config files.
     Include: services detected, env vars count, key config files.]

    ## Git/Blame
    [High-churn files, module activity distribution, team contributors.
     Include hotspot table if data available.]

    ## Detected Tech Debt
    [High-churn single-author files, TODO/FIXME patterns, old migrations,
     deprecated patterns — be specific with file names]

    ## Absorber Notes
    [Ambiguities, files that could not be parsed, recommended follow-ups]
""").strip()


def call_llm_for_map(
    context: str,
    target_name: str,
    config_section: str = "",
    git_section: str = "",
) -> tuple[str, float]:
    """
    Call LLM to produce codebase_map.md with merged Config and Git/Blame sections.

    Returns:
      (content, call_cost)  — call_cost is 0.0 if usage unavailable or call failed.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system = _MAP_SYSTEM.replace("{date}", date_str)

    user_parts = [f"Codebase: {target_name}\n\nExtracted content:\n\n{context}"]

    if config_section:
        user_parts.append(f"\n\n--- CONFIG INVENTORY DATA ---\n{config_section}")

    if git_section:
        user_parts.append(f"\n\n--- GIT/BLAME DATA ---\n{git_section}")

    user = "".join(user_parts)

    tokens_est = len(user) // 4
    print(f"[absorber] LLM call: {ROLE} | ~{tokens_est:,} input tokens")

    try:
        resp = call_model(
            ROLE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=_MAX_TOKENS_MAP,
            temperature=0.2,
        )
        choice = resp.choices[0]
        usage  = getattr(resp, "usage", None)
        call_cost = 0.0
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=get_model(ROLE), provider=get_provider(ROLE))
            print_call(__file__, pt, ct, call_cost)
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", "unknown")

        if finish_reason == "length":
            print(
                f"[absorber][warn] LLM output was truncated. "
                f"Consider increasing _MAX_TOKENS_MAP={_MAX_TOKENS_MAP} "
                f"or reducing codebase context."
            )
        else:
            print(f"[absorber] LLM finish_reason: {finish_reason}")

        return content, call_cost

    except Exception as e:
        print(f"[absorber][error] LLM call failed: {e}", file=sys.stderr)
        return _fallback_map(context, target_name, config_section, git_section), 0.0


def _fallback_map(
    context: str,
    target_name: str,
    config_section: str = "",
    git_section: str = "",
) -> str:
    """
    Produce a minimal codebase_map.md without LLM when the call fails.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [
        f"# Codebase Map",
        f"_Generated: {date_str} | Absorber v2 (fallback — LLM unavailable)_\n",
        "## Project Overview",
        f"Target: {target_name}\n",
        "LLM call failed. Below is the raw extracted context for manual review.\n",
        "## Raw Extraction\n",
        context[:8000],
    ]

    if config_section:
        parts.append("\n## Config\n")
        parts.append(config_section[:4000])

    if git_section:
        parts.append("\n## Git/Blame\n")
        parts.append(git_section[:4000])

    parts.append("\n## Absorber Notes\n")
    parts.append("- This map was generated in fallback mode (no LLM). Re-run when model is available.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Git crawl
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_git_scope(scope: str) -> list[str]:
    """
    Convert --git-scope value to git log arguments.

    Accepts:
      "6m"   → --since=6.months.ago
      "3m"   → --since=3.months.ago
      "500"  → -n 500
      "all"  → no limit
    """
    if scope == "all":
        return []

    match = re.match(r"^(\d+)([mMwWdD])$", scope)
    if match:
        num, unit = match.group(1), match.group(2).lower()
        unit_map = {"m": "months", "w": "weeks", "d": "days"}
        return [f"--since={num}.{unit_map.get(unit, 'months')}.ago"]

    if scope.isdigit():
        return ["-n", scope]

    # Default: 6 months
    return ["--since=6.months.ago"]


def _git_log_stats(target: Path, scope: str) -> tuple[str, list[tuple[str, int]]]:
    """
    Run git log --stat and produce a summary for the Git/Blame section.

    Returns:
      (section_str, hotspots) — hotspots is [(file, change_count), ...] top-20,
      or ("", []) if not a git repo or git command fails.
    """
    if not (target / ".git").exists():
        return "", []

    scope_args = _resolve_git_scope(scope)

    # Get file churn (hotspots)
    try:
        result = subprocess.run(
            ["git", "log", "--format=", "--name-only", *scope_args],
            capture_output=True,
            text=True,
            cwd=target,
            timeout=30,
        )
        if result.returncode != 0:
            return "", []

        file_counts: dict[str, int] = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                file_counts[line] = file_counts.get(line, 0) + 1

        # Top 20 hotspots
        hotspots = sorted(file_counts.items(), key=lambda x: -x[1])[:20]
    except Exception:
        return "", []

    # Get contributor stats
    try:
        result = subprocess.run(
            ["git", "shortlog", "-sn", "--no-merges", *scope_args],
            capture_output=True,
            text=True,
            cwd=target,
            timeout=30,
        )
        contributors = result.stdout.strip().splitlines()[:15] if result.returncode == 0 else []
    except Exception:
        contributors = []

    # Get total commit count
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD", *scope_args],
            capture_output=True,
            text=True,
            cwd=target,
            timeout=10,
        )
        total_commits = result.stdout.strip() if result.returncode == 0 else "?"
    except Exception:
        total_commits = "?"

    # Build section
    lines: list[str] = []
    lines.append(f"Total commits (scope: {scope}): {total_commits}")
    lines.append("")

    if hotspots:
        lines.append("### Hotspot Files (most changed)")
        lines.append("| File | Changes |")
        lines.append("|------|---------|")
        for fname, count in hotspots:
            lines.append(f"| {fname} | {count} |")
        lines.append("")

    if contributors:
        lines.append("### Contributors")
        for c in contributors:
            lines.append(f"  {c.strip()}")
        lines.append("")

    # Module activity distribution
    if hotspots:
        module_counts: dict[str, int] = {}
        for fname, count in file_counts.items():
            parts = fname.split("/")
            module = parts[0] if len(parts) > 1 else "(root)"
            module_counts[module] = module_counts.get(module, 0) + count

        top_modules = sorted(module_counts.items(), key=lambda x: -x[1])[:10]
        lines.append("### Module Activity")
        lines.append("| Module | Total file changes |")
        lines.append("|--------|-------------------|")
        for mod, count in top_modules:
            lines.append(f"| {mod} | {count} |")

    return "\n".join(lines), hotspots


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Append codebase_log.json (long-term audit trail)
# ─────────────────────────────────────────────────────────────────────────────

def _append_codebase_log(
    target: Path,
    inventory: list[dict[str, Any]],
    cached_count: int,
    extracted_count: int,
    map_size: int,
    *,
    call_cost: float = 0.0,
    git_scope: str = "",
    hotspot_summary: list[tuple[str, int]] | None = None,
) -> None:
    """
    Append an entry to absorber/codebase_log.json for audit trail.

    Args:
      call_cost        — USD cost returned by record_usage(); 0.0 if not available.
      git_scope        — raw --git-scope flag value (e.g. "6m", "all"); empty if --skip-git.
      hotspot_summary  — top-N [(file, change_count)] from _git_log_stats(); None if skipped.
    """
    entry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": str(target),
        "total_files": len(inventory),
        "cached_files": cached_count,
        "extracted_files": extracted_count,
        "modes": {
            "full": sum(1 for f in inventory if f["mode"] == "full"),
            "key_only": sum(1 for f in inventory if f["mode"] == "key-only"),
            "signature_only": sum(1 for f in inventory if f["mode"] == "signature-only"),
        },
        "languages": _count_languages(inventory),
        "map_size_bytes": map_size,
        "cost": round(call_cost, 6),
        "git_scope": git_scope or None,
        "hotspot_summary": (
            [{"file": f, "changes": c} for f, c in hotspot_summary]
            if hotspot_summary
            else None
        ),
    }

    log_path = Path(str(CODEBASE_LOG))
    existing: list[dict[str, Any]] = []

    if log_path.exists():
        try:
            track_read(log_path)
            data = json.loads(log_path.read_text())
            if isinstance(data, dict) and "entries" in data:
                existing = data["entries"]
            elif isinstance(data, list):
                existing = data
        except Exception:
            pass

    existing.append(entry)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"entries": existing}, indent=2))
    track_write(log_path)

    print(f"[absorber] Appended codebase_log entry (total entries: {len(existing)})")


def _count_languages(inventory: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in inventory:
        lang = f.get("lang")
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Config inventory builder (merged into codebase_map.md ## Config)
# ─────────────────────────────────────────────────────────────────────────────

def _build_config_section(inventory: list[dict[str, Any]], cache: dict[str, Any]) -> str:
    """
    Build config inventory data from key-only files for the LLM prompt.
    """
    config_files = [
        f for f in inventory
        if f["mode"] == "key-only" and _should_include_in_config_inventory(f["rel_path"])
    ]

    if not config_files:
        return ""

    lines: list[str] = []
    lines.append(f"Config files detected: {len(config_files)}")
    lines.append("")

    for cf in config_files[:30]:  # Cap at 30 to avoid token explosion
        rel = cf["rel_path"]
        cached_entry = cache.get(rel, {})
        content = cached_entry.get("content", "")

        lines.append(f"### {rel}")
        # Truncate individual config content
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Context builder — assemble extraction results for LLM
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(
    inventory: list[dict[str, Any]],
    cache: dict[str, Any],
    force: bool,
) -> tuple[str, int, int]:
    """
    Extract content from all files and build the context string for LLM.

    Returns:
      (context_str, cached_count, extracted_count)
    """
    sections: list[str] = []
    cached_count = 0
    extracted_count = 0

    # Budget: ~100k chars for context to stay within token limits
    _MAX_CONTEXT_CHARS = 100_000
    total_chars = 0

    for entry in inventory:
        content, from_cache = extract_content(entry, cache, force)

        if from_cache:
            cached_count += 1
        else:
            extracted_count += 1

        rel = entry["rel_path"]
        mode = entry["mode"]
        lang = entry["lang"] or "text"

        section = f"--- {rel} [{mode}] ({lang}) ---\n{content}\n"

        if total_chars + len(section) > _MAX_CONTEXT_CHARS:
            sections.append(f"\n... (context budget reached, {len(inventory) - len(sections)} files omitted)")
            break

        sections.append(section)
        total_chars += len(section)

    context = "\n".join(sections)
    return context, cached_count, extracted_count


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_target(args: argparse.Namespace) -> Path:
    """Resolve the target codebase directory."""
    if args.target:
        target = Path(args.target).resolve()
    else:
        target = Path.cwd()

    if not target.is_dir():
        print(f"[absorber][error] Target is not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    return target


def run_absorber(args: argparse.Namespace) -> None:
    """Main entry point for the absorber step."""
    ensure_dirs()

    target = _resolve_target(args)
    target_name = target.name

    print(f"[absorber] Target: {target}")
    print(f"[absorber] Force: {args.force}")
    print(f"[absorber] Git scope: {args.git_scope}")
    print(f"[absorber] Skip git: {args.skip_git}")
    print(f"[absorber] Dry run: {args.dry_run}")
    print()

    # --- Phase 1: File tree scan ---
    rules_path = target / _IGNORED_FILE
    rules = AbsorberIgnoreRules(rules_path)

    # Determine if target is inside artifact root (skip artifact dirs)
    art_root = Path(str(artifact_root()))
    skip_artifact = target == art_root.parent or art_root.is_relative_to(target)

    inventory = scan_files(target, rules, skip_artifact_control_dirs=skip_artifact)

    print(f"[absorber] Phase 1 — Scanned: {len(inventory)} files")
    mode_counts = {}
    for f in inventory:
        mode_counts[f["mode"]] = mode_counts.get(f["mode"], 0) + 1
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")
    print()

    if not inventory:
        print("[absorber][warn] No files found. Check target directory and absorber.ignored rules.")
        return

    # --- Phase 2: Content extraction ---
    cache = _load_cache()
    context, cached_count, extracted_count = _build_context(inventory, cache, args.force)

    print(f"[absorber] Phase 2 — Extracted: {extracted_count} new, {cached_count} from cache")
    print()

    # Save cache
    _save_cache(cache)

    if args.dry_run:
        print("[absorber] Dry run — skipping LLM call and writes.")
        print(f"  Context size: {len(context):,} chars (~{len(context)//4:,} tokens)")
        print_artifact_summary()
        return

    # --- Phase 4: Git crawl (before LLM so we can include in prompt) ---
    git_section = ""
    hotspots: list[tuple[str, int]] = []
    if not args.skip_git:
        print(f"[absorber] Phase 4 — Git crawl (scope: {args.git_scope})")
        git_section, hotspots = _git_log_stats(target, args.git_scope)
        if git_section:
            print(f"  Git data: {len(git_section):,} chars")
        else:
            print("  No git data (not a git repo or no history in scope)")
        print()

    # --- Build config section ---
    config_section = _build_config_section(inventory, cache)

    # --- Phase 3: Semantic compression → codebase_map.md ---
    print("[absorber] Phase 3 — LLM semantic compression")
    map_content, call_cost = call_llm_for_map(context, target_name, config_section, git_section)

    # Write codebase_map.md
    map_path = Path(str(CODEBASE_MAP))
    map_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply markdown header (needs map_path to detect existing header for created_ts)
    map_content = apply_md_header(map_content, map_path, owner="absorber")

    map_path.write_text(map_content)
    track_write(map_path)
    print(f"[absorber] Wrote: {map_path} ({len(map_content):,} bytes)")

    # --- Phase 5: Append codebase_log ---
    _append_codebase_log(
        target,
        inventory,
        cached_count,
        extracted_count,
        len(map_content),
        call_cost=call_cost,
        git_scope="" if args.skip_git else args.git_scope,
        hotspot_summary=hotspots if hotspots else None,
    )

    # --- Summary ---
    print()
    print_summary()
    print()
    print_artifact_summary()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Absorber — scan and compress a codebase into knowledge artifacts.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PIPELINE_PROJECT", ""),
        help="Project slug (default: PIPELINE_PROJECT env or auto-detect)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Path to target codebase (default: current directory)",
    )
    parser.add_argument(
        "--git-scope",
        default="6m",
        help="Git history scope: 6m, 3m, 500, all (default: 6m)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass cache, re-extract all files",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip git log analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and extract but skip LLM call and artifact writes",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.project:
        os.environ["PIPELINE_PROJECT"] = args.project

    print("=" * 60)
    print("  STEP 1 — ABSORBER")
    print("=" * 60)
    print()

    try:
        run_absorber(args)
    except KeyboardInterrupt:
        print("\n[absorber] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[absorber][fatal] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    prompt_next_step("absorber")


if __name__ == "__main__":
    main()
