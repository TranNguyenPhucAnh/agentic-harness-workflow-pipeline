"""
pipeline/01_absorber.py
=======================
Step 1 — Absorb an existing codebase into the knowledge layer.

Runs once when taking over a legacy project, and on-demand when the codebase
changes significantly enough to warrant a refresh.

Phases:
  1. File tree scan        — apply absorber.ignored rules, build file inventory
  2. Content extraction    — full / key-only / signature-only per file
  3. Semantic compression  — single LLM call → codebase_map.md
  4. Config inventory      — aggregate key-only extractions → config_map.json
  5. Git crawl             — git log → absorber_overwrite_git_snapshot.json + absorber_blame_map.md

External integrations optional, graceful fallback:
  - vfs CLI     — signature extraction
  - Serena MCP  — symbol-level call graph, future via subprocess

Change detection:
  - absorber_overwrite_codebase_snapshot.json tracks file hashes
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
  artifacts_<slug>/knowledge/current/absorber_codebase_map.md
  artifacts_<slug>/knowledge/current/absorber_config_map.json
  artifacts_<slug>/knowledge/current/absorber_blame_map.md
  artifacts_<slug>/cache/absorber_overwrite_codebase_snapshot.json
  artifacts_<slug>/cache/absorber_overwrite_git_snapshot.json

Reads:
  project source files (target codebase)
  artifacts_<slug>/cache/absorber_overwrite_codebase_snapshot.json if present

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
# OWNS  : artifacts_<slug>/knowledge/current/absorber_codebase_map.md
#         artifacts_<slug>/knowledge/current/absorber_config_map.json
#         artifacts_<slug>/knowledge/current/absorber_blame_map.md
#         artifacts_<slug>/cache/absorber_overwrite_codebase_snapshot.json
#         artifacts_<slug>/cache/absorber_overwrite_git_snapshot.json
# READS : project source files (target codebase)
#         artifacts_<slug>/cache/absorber_overwrite_codebase_snapshot.json

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ABSORBER_BLAME_MAP,
    ABSORBER_CODEBASE_MAP,
    ABSORBER_CODEBASE_SNAPSHOT,
    ABSORBER_CONFIG_MAP,
    ABSORBER_GIT_SNAPSHOT,
    artifact_root,
    ensure_dirs,
)
from artifacts.models import call_model  # noqa: E402

# Local aliases — map canonical constants to the short names used internally
ABSORBER_CACHE = ABSORBER_CODEBASE_SNAPSHOT
BLAME_MAP      = ABSORBER_BLAME_MAP
CODEBASE_MAP   = ABSORBER_CODEBASE_MAP
CONFIG_MAP     = ABSORBER_CONFIG_MAP
GIT_HISTORY    = ABSORBER_GIT_SNAPSHOT


# ─────────────────────────────────────────────────────────────────────────────
# Artifact/source access tracking
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_READ: set[str] = set()
_ARTIFACTS_WRITTEN: set[str] = set()


def _track_read(path: Any) -> None:
    _ARTIFACTS_READ.add(str(path))


def _track_write(path: Any) -> None:
    _ARTIFACTS_WRITTEN.add(str(path))


def _print_artifact_access_summary() -> None:
    print("[01] Artifacts/files read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[01]   READ  {item}")
    else:
        print("[01]   READ  (none)")

    print("[01] Artifacts/files created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[01]   WRITE {item}")
    else:
        print("[01]   WRITE (none)")


# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_TOKENS_MAP = 16384
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
#
# When target defaults to artifact_root(), scanning these directories makes the
# knowledge layer self-referential, e.g. codebase_map.md ingesting previous
# codebase_map.md, test_report.json, cache files, etc.
_ARTIFACT_CONTROL_DIRS: frozenset[str] = frozenset({
    "state",
    "cache",
    "execution",
    "knowledge",
    "reports",
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

        _track_read(path)

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
            _track_read(ABSORBER_CACHE)
            return json.loads(ABSORBER_CACHE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    ABSORBER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ABSORBER_CACHE.write_text(json.dumps(cache, indent=2))
    _track_write(ABSORBER_CACHE)


def _file_hash(path: Path) -> str:
    try:
        _track_read(path)
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
    Filter out pure build/tooling files from config_map.json.
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
        _track_read(path)
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
        _track_read(path)
        lines = path.read_text(errors="replace").splitlines()
        preview = "\n".join(lines[:50])
        if len(lines) > 50:
            preview += f"\n... ({len(lines) - 50} more lines)"
        return preview
    except Exception:
        return "[signature extraction failed]"


def _extract_python_signatures(path: Path) -> str:
    try:
        _track_read(path)
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
        _track_read(path)
        return path.read_text(errors="replace")[:500]
    except Exception:
        return "[signature extraction failed]"


def _extract_ts_signatures(path: Path) -> str:
    try:
        _track_read(path)
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
    _Generated: {date} | Absorber v1_

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

    ## Config Dependencies
    [Services, env vars, feature flags detected from config files]

    ## Detected Tech Debt
    [High-churn single-author files, TODO/FIXME patterns, old migrations,
     deprecated patterns — be specific with file names]

    ## Absorber Notes
    [Ambiguities, files that could not be parsed, recommended follow-ups]
""").strip()


def call_llm_for_map(
    context: str,
    target_name: str,
) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system = _MAP_SYSTEM.replace("{date}", date_str)
    user = f"Codebase: {target_name}\n\nExtracted content:\n\n{context}"

    tokens_est = len(context) // 4
    print(f"[absorber] LLM call: absorber | ~{tokens_est:,} input tokens")

    try:
        resp = call_model(
            "absorber",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=_MAX_TOKENS_MAP,
            temperature=0.2,
        )
        choice = resp.choices[0]
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

        return content

    except Exception as e:
        print(f"[absorber][error] LLM call failed: {e}", file=sys.stderr)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Config inventory → config_map.json
# ─────────────────────────────────────────────────────────────────────────────

_SERVICE_PATTERNS: dict[str, re.Pattern[str]] = {
    "database": re.compile(
        r"(?:postgres|mysql|mongodb|redis|sqlite|db_|database|connectionstring)",
        re.I,
    ),
    "messaging": re.compile(
        r"(?:kafka|rabbitmq|sqs|sns|pubsub|amqp)",
        re.I,
    ),
    "auth": re.compile(
        r"(?:auth|oauth|jwt|saml|sso|oidc|keycloak|openiddict)",
        re.I,
    ),
    "storage": re.compile(
        r"(?:s3|gcs|azure_blob|minio|storage|bucket)",
        re.I,
    ),
    "monitoring": re.compile(
        r"(?:datadog|newrelic|prometheus|grafana|sentry|cloudwatch)",
        re.I,
    ),
    "email": re.compile(
        r"(?:smtp|sendgrid|ses|mailgun|email)",
        re.I,
    ),
    "cloud": re.compile(
        r"(?:aws|gcp|azure|heroku|fly\.io|ecs|fargate|cloudformation)",
        re.I,
    ),
}


def _extract_env_vars_from_raw(path: Path, ext: str) -> set[str]:
    """
    Extract env var names from raw config/template content.

    ext is kept for compatibility with tests/importers.
    """
    _ = ext
    try:
        _track_read(path)
        raw = path.read_text(errors="replace")
    except Exception:
        return set()
    return _parse_env_vars_from_text(raw)


def _parse_env_vars_from_text(raw: str) -> set[str]:
    """
    Extract env var-like names from raw text.

    Covers:
      1. ${VAR} and $VAR
      2. process.env.VAR
      3. KEY=value lines
      4. SECTION__KEY .NET/container overrides
      5. JSON top-level keys converted to SCREAMING_SNAKE
    """
    env_vars: set[str] = set()

    for match in re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}|\$([A-Z_][A-Z0-9_]*)", raw):
        env_vars.update(g for g in match if g)

    env_vars.update(re.findall(
        r"process\.env\.([A-Z_][A-Z0-9_]*)",
        raw,
        re.IGNORECASE,
    ))

    env_vars.update(re.findall(
        r"^\s*([A-Z_][A-Z0-9_]*)\s*=",
        raw,
        re.MULTILINE,
    ))

    env_vars.update(re.findall(
        r"([A-Z_][A-Z0-9_]*(?:__[A-Z0-9_]+)+)",
        raw,
    ))

    json_keys = re.findall(r'"([A-Za-z][A-Za-z0-9]{2,})"\s*:', raw)
    for key in json_keys:
        screaming = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).upper()
        if re.match(r"^[A-Z][A-Z0-9_]{2,}$", screaming):
            env_vars.add(screaming)

    return env_vars


def build_config_map(
    inventory: list[dict[str, Any]],
    cache: dict[str, Any],
) -> dict[str, Any]:
    """
    Aggregate key-only config files into config_map.json.

    Env var detection reads raw source file where possible, not redacted content,
    to avoid losing ${VAR}, process.env.VAR, and KEY=value references.
    """
    config_files: list[dict[str, Any]] = []
    all_env_vars: set[str] = set()
    all_services: set[str] = set()

    for entry in inventory:
        if entry["mode"] != "key-only":
            continue

        rel_path = entry["rel_path"]

        if not _should_include_in_config_inventory(rel_path):
            continue

        abs_path = Path(entry.get("abs_path", ""))

        raw = ""
        if abs_path.exists():
            try:
                _track_read(abs_path)
                raw = abs_path.read_text(errors="replace")
            except Exception:
                raw = ""

        if not raw:
            raw = cache.get(rel_path, {}).get("content", "")

        redacted = cache.get(rel_path, {}).get("content", "")

        file_env_vars = _parse_env_vars_from_text(raw)
        all_env_vars.update(file_env_vars)

        service_scan_text = raw or redacted
        file_services: list[str] = []

        for svc_name, pattern in _SERVICE_PATTERNS.items():
            if pattern.search(service_scan_text):
                file_services.append(svc_name)
                all_services.add(svc_name)

        config_files.append({
            "path": rel_path,
            "env_vars": sorted(file_env_vars),
            "services": sorted(file_services),
        })

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_configs": len(config_files),
        "services_detected": sorted(all_services),
        "env_vars_detected": sorted(all_env_vars),
        "files": config_files,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Git crawl → absorber_overwrite_git_snapshot.json + absorber_blame_map.md
# ─────────────────────────────────────────────────────────────────────────────

def _ask_git_scope() -> str:
    print("\n[absorber] Git history scope:")
    print("  1. Last 3 months")
    print("  2. Last 6 months")
    print("  3. Last 1 year")
    print("  4. All history")
    print("  5. Custom: number of commits or date, e.g. 500 or 2024-01-01")

    choice = input("→ Choose [1-5]: ").strip()
    mapping = {"1": "3m", "2": "6m", "3": "1y", "4": "all"}

    if choice in mapping:
        return mapping[choice]

    if choice == "5":
        custom = input("  Enter commits count or start date YYYY-MM-DD: ").strip()
        return custom or "6m"

    print("[absorber] Invalid choice, defaulting to 6 months.")
    return "6m"


def _scope_to_git_args(scope: str) -> list[str]:
    if scope == "all":
        return []

    if scope.endswith("m") and scope[:-1].isdigit():
        return [f"--since={int(scope[:-1])} months ago"]

    if scope.endswith("y") and scope[:-1].isdigit():
        return [f"--since={int(scope[:-1])} years ago"]

    if scope.isdigit():
        return ["-n", scope]

    if re.match(r"\d{4}-\d{2}-\d{2}", scope):
        return [f"--since={scope}"]

    return ["--since=6 months ago"]


def crawl_git(target: Path, scope: str) -> dict[str, Any] | None:
    git_dir = target / ".git"
    if not git_dir.exists():
        print("[absorber] No .git directory found — skipping git crawl.")
        return None

    git_cmd = [
        "git",
        "-C",
        str(target),
        "log",
        "--format=%H|||%ai|||%ae|||%s",
        "--numstat",
    ] + _scope_to_git_args(scope)

    try:
        result = subprocess.run(
            git_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("[absorber][warn] git log timed out — skipping git crawl.")
        return None

    if result.returncode != 0:
        print(f"[absorber][warn] git log failed: {result.stderr[:200]}")
        return None

    commits = _parse_git_log(result.stdout)
    if not commits:
        print("[absorber] No commits found for scope.")
        return None

    churn: dict[str, dict[str, Any]] = {}
    for commit in commits:
        for fpath in commit.get("files_changed", []):
            if fpath not in churn:
                churn[fpath] = {"count": 0, "authors": set()}
            churn[fpath]["count"] += 1
            churn[fpath]["authors"].add(commit["author"])

    hotspots = sorted(
        [
            {
                "file": fp,
                "change_count": data["count"],
                "authors": sorted(data["authors"]),
            }
            for fp, data in churn.items()
        ],
        key=lambda x: x["change_count"],
        reverse=True,
    )[:50]

    authors = sorted({c["author"] for c in commits})

    return {
        "scope": scope,
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_commits": len(commits),
        "authors": authors,
        "hotspots": hotspots,
        "commits": commits,
    }


def _parse_git_log(raw: str) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in raw.splitlines():
        stripped = line.strip()

        if "|||" in stripped:
            if current is not None:
                commits.append(current)

            parts = stripped.split("|||", 3)
            if len(parts) < 4:
                current = None
                continue

            current = {
                "hash": parts[0][:7],
                "date": parts[1][:10],
                "author": parts[2],
                "message": parts[3][:200],
                "files_changed": [],
                "insertions": 0,
                "deletions": 0,
            }

        elif current is not None and stripped:
            parts = stripped.split("\t", 2)
            if len(parts) != 3:
                continue

            ins_str, del_str, fname = parts
            try:
                current["insertions"] += int(ins_str) if ins_str != "-" else 0
                current["deletions"] += int(del_str) if del_str != "-" else 0
            except ValueError:
                pass

            if fname:
                current["files_changed"].append(fname)

    if current is not None:
        commits.append(current)

    return commits


def build_blame_map(git_data: dict[str, Any]) -> str:
    now = git_data["generated"][:10]
    scope = git_data["scope"]
    total = git_data["total_commits"]
    authors = git_data["authors"]

    lines: list[str] = [
        "# Codebase Hotspot Map",
        f"_Generated: {now} | Scope: {scope} | Commits analyzed: {total}_",
        "",
        "## Team",
        f"Active contributors: {', '.join(authors[:10])}"
        + (f" (+{len(authors) - 10} more)" if len(authors) > 10 else ""),
        "",
    ]

    hotspots = git_data.get("hotspots", [])
    high = [h for h in hotspots if h["change_count"] >= 10]
    medium = [h for h in hotspots if 5 <= h["change_count"] < 10]

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
                auth_str += f" (+{len(h['authors']) - 3})"
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

    module_activity: dict[str, int] = {}
    for h in hotspots:
        parts = h["file"].split("/")
        if len(parts) >= 2:
            module = parts[0] if parts[0] != "src" else parts[1]
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
# Main orchestrator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_extraction_context(
    inventory: list[dict[str, Any]],
    cache: dict[str, Any],
) -> str:
    _MAX_PER_FILE = 2000
    _MAX_TOTAL = 800_000

    sections: list[str] = []
    total_chars = 0

    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in inventory:
        parts = entry["rel_path"].split("/")
        top = parts[0] if len(parts) > 1 else "(root)"
        groups.setdefault(top, []).append(entry)

    for group_name, entries in sorted(groups.items()):
        group_lines = [f"\n## {group_name}/\n"]

        for entry in entries:
            rel = entry["rel_path"]
            lang = entry["lang"] or ""
            content = cache.get(rel, {}).get("content", "")

            if not content:
                continue

            if len(content) > _MAX_PER_FILE:
                content = (
                    content[:_MAX_PER_FILE]
                    + f"\n... [truncated, {len(content)} chars total]"
                )

            group_lines.append(
                f"### {rel} ({lang}, {entry['mode']})\n"
                f"```\n{content}\n```\n"
            )

        chunk = "\n".join(group_lines)
        if total_chars + len(chunk) > _MAX_TOTAL:
            sections.append(
                f"\n[...context truncated due to {_MAX_TOTAL:,} char limit]"
            )
            break

        sections.append(chunk)
        total_chars += len(chunk)

    return "\n".join(sections)


def _configure_project(
    project: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    """
    Configure project context for direct execution.

    Harness normally sets PIPELINE_PROJECT before invoking this script.
    Direct usage can pass --project.
    """
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 01_absorber.py directly."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Absorb a codebase into the pipeline knowledge layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 01_absorber.py --project my-app
              PIPELINE_PROJECT=my-app python 01_absorber.py

              python 01_absorber.py --project my-app --git-scope 6m
              python 01_absorber.py --project my-app --git-scope 500
              python 01_absorber.py --project my-app --git-scope all
              python 01_absorber.py --project my-app --skip-git
              python 01_absorber.py --project my-app --force
              python 01_absorber.py --project my-app --target /path/to/repo
              python 01_absorber.py --project my-app --dry-run
        """),
    )

    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Project name for direct execution. Sets PIPELINE_PROJECT before "
            "resolving artifact paths."
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Path to codebase root. Default: artifacts_<slug>/ for current project.",
    )
    parser.add_argument(
        "--git-scope",
        metavar="SCOPE",
        default=None,
        help="Git history scope: 3m, 6m, 1y, all, N commits, or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip git crawl entirely.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore absorber_overwrite_codebase_snapshot.json and re-extract all files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only. No LLM call and no file writes.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM synthesis and write raw extraction as codebase_map.md.",
    )

    return parser


def main() -> None:
    exit_code = 0

    try:
        parser = _build_parser()
        args = parser.parse_args()

        _configure_project(args.project, parser)

        # Important: do not call ensure_dirs() at import-time.
        # PIPELINE_PROJECT must be available before artifact paths are resolved.
        ensure_dirs()

        using_default_target = args.target is None
        target: Path = (args.target or artifact_root()).resolve()

        if not target.exists():
            print(f"[absorber][error] Target path does not exist: {target}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'─' * 50}")
        print(f"  Absorber — {target.name}")
        print(f"{'─' * 50}\n")

        # ── Phase 1: File tree scan ───────────────────────────────────────────────

        print("[absorber] Phase 1 — Scanning file tree ...")
        rules_path = target / _IGNORED_FILE
        rules = AbsorberIgnoreRules(rules_path)

        inventory = scan_files(
            target,
            rules,
            skip_artifact_control_dirs=using_default_target,
        )

        lang_counts: dict[str, int] = {}
        for entry in inventory:
            lang = entry["lang"] or "other"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print(f"[absorber] Found {len(inventory)} files to process")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"     {lang}: {count}")

        mode_counts = {"full": 0, "key-only": 0, "signature-only": 0}
        for entry in inventory:
            mode_counts[entry["mode"]] = mode_counts.get(entry["mode"], 0) + 1

        print(f"[absorber] Extraction modes: {mode_counts}")

        if using_default_target:
            skipped = ", ".join(sorted(_ARTIFACT_CONTROL_DIRS))
            print(f"[absorber] Default target detected; skipping artifact-control dirs: {skipped}")

        if args.dry_run:
            print("\n[absorber] --dry-run: stopping here. No files written.")
            return

        # ── Phase 2: Content extraction ───────────────────────────────────────────

        print("\n[absorber] Phase 2 — Extracting content ...")
        cache = _load_cache()
        cache_hits = 0

        for i, entry in enumerate(inventory, 1):
            _, from_cache = extract_content(entry, cache, args.force)

            if from_cache:
                cache_hits += 1

            if i % 50 == 0:
                print(f"     {i}/{len(inventory)} files processed ...")

        total_chars = sum(
            len(cache.get(e["rel_path"], {}).get("content", ""))
            for e in inventory
        )
        est_tokens = total_chars // 4

        print(f"[absorber] Extracted {len(inventory)} files | cache hits: {cache_hits}")
        print(f"[absorber] Total content: {total_chars:,} chars (~{est_tokens:,} tokens)")

        _save_cache(cache)
        print(f"[absorber] ✓ Cache saved → {ABSORBER_CACHE}")

        # ── Phase 3: Semantic compression → codebase_map.md ──────────────────────

        if not args.skip_llm:
            print("\n[absorber] Phase 3 — Semantic compression (LLM) ...")
            context = _build_extraction_context(inventory, cache)

            try:
                codebase_map = call_llm_for_map(context, target.name)
                CODEBASE_MAP.parent.mkdir(parents=True, exist_ok=True)
                CODEBASE_MAP.write_text(codebase_map)
                _track_write(CODEBASE_MAP)
                print(f"[absorber] ✓ Codebase map → {CODEBASE_MAP}")
            except Exception as e:
                print(f"[absorber][warn] LLM synthesis failed: {e} — skipping codebase_map.md")
        else:
            raw_context = _build_extraction_context(inventory, cache)
            CODEBASE_MAP.parent.mkdir(parents=True, exist_ok=True)
            CODEBASE_MAP.write_text(
                "# Codebase Map (raw extraction — no LLM synthesis)\n\n"
                + raw_context
            )
            _track_write(CODEBASE_MAP)
            print(f"[absorber] ✓ Raw extraction → {CODEBASE_MAP} (--skip-llm)")

        # ── Phase 4: Config inventory ─────────────────────────────────────────────

        print("\n[absorber] Phase 4 — Config inventory ...")
        config_map = build_config_map(inventory, cache)

        CONFIG_MAP.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_MAP.write_text(json.dumps(config_map, indent=2))
        _track_write(CONFIG_MAP)

        print(f"[absorber] ✓ Config map → {CONFIG_MAP}")
        print(f"     Services detected: {', '.join(config_map['services_detected']) or 'none'}")
        print(f"     Env vars detected: {len(config_map['env_vars_detected'])}")

        # ── Phase 5: Git crawl ────────────────────────────────────────────────────

        if not args.skip_git:
            print("\n[absorber] Phase 5 — Git crawl ...")

            scope = args.git_scope or _ask_git_scope()
            git_data = crawl_git(target, scope)

            if git_data:
                GIT_HISTORY.parent.mkdir(parents=True, exist_ok=True)
                GIT_HISTORY.write_text(json.dumps(git_data, indent=2))
                _track_write(GIT_HISTORY)

                print(f"[absorber] ✓ Git history → {GIT_HISTORY}")
                print(
                    f"     Commits: {git_data['total_commits']} | "
                    f"Authors: {len(git_data['authors'])}"
                )
                print(f"     Hotspots: {len(git_data['hotspots'])} files")

                blame_md = build_blame_map(git_data)
                BLAME_MAP.parent.mkdir(parents=True, exist_ok=True)
                BLAME_MAP.write_text(blame_md)
                _track_write(BLAME_MAP)

                print(f"[absorber] ✓ Blame map → {BLAME_MAP}")
            else:
                print("[absorber] Git crawl skipped or failed.")
        else:
            print("\n[absorber] Phase 5 — Git crawl skipped (--skip-git).")

        # ── Summary ───────────────────────────────────────────────────────────────

        print(f"\n{'─' * 50}")
        print(f"  Done — {target.name} absorbed")
        print(f"{'─' * 50}")
        print(f"  absorber_codebase_map.md              → {CODEBASE_MAP}")
        print(f"  absorber_config_map.json              → {CONFIG_MAP}")

        if not args.skip_git:
            print(f"  absorber_overwrite_git_snapshot.json  → {GIT_HISTORY}")
            print(f"  absorber_blame_map.md                 → {BLAME_MAP}")

    except SystemExit as exc:
        # Preserve explicit sys.exit(...) behavior but still print access summary.
        code = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[absorber][error] {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
