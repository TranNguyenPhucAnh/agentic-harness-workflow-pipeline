"""
pipeline/01_absorber.py
=======================
Step 1 — Absorb an existing codebase into the knowledge layer.

Runs once when taking over a legacy project, and on-demand when the codebase
changes significantly enough to warrant a refresh.

Phases:
  1. File tree scan        — apply absorber.ignored rules, build file inventory
  2. Content extraction    — full / key-only / signature-only per file
  3. Semantic compression  — single LLM call → codebase_map.md (narrative)
  4. Git crawl             — git log → structured into codebase_map.json["git"]
  5. Write artifacts       — codebase_map.md + codebase_map.json
  6. Append codebase_log   — long-term audit trail

Artifact split:
  codebase_map.md   — LLM-generated narrative (human/LLM readable)
                       Sections: Project Overview, Module Inventory, Entry Points,
                       Data Flow, Config, Git/Blame, Tech Debt, Absorber Notes
  codebase_map.json — Structured machine-readable data
                       Fields: meta, config (env vars, services), git (hotspots,
                       contributors, module activity)

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
  python 01_absorber.py --dry-run
  python 01_absorber.py --target /path/to/repo

Writes, owner: absorber (01_absorber.py):
  artifacts_<slug>/absorber/codebase_map.md          (short-term, overwrite — LLM narrative)
  artifacts_<slug>/absorber/codebase_map.json        (short-term, overwrite — structured data)
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
# OWNS  : artifacts_<slug>/absorber/codebase_map.md  (short-term, overwrite — LLM narrative)
#         artifacts_<slug>/absorber/codebase_map.json (short-term, overwrite — structured data)
#         artifacts_<slug>/absorber/codebase_log.json (long-term, append)
#         artifacts_<slug>/absorber/cache/codebase_snapshot.json (cache - internal, overwrite)

# READS : project source files (target codebase)
#         artifacts_<slug>/absorber/cache/codebase_snapshot.json (cache)

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_LOG,
    ABSORBER_CODEBASE_MAP,
    ABSORBER_CODEBASE_MD,
    ABSORBER_CODEBASE_SNAPSHOT,
    artifact_root,
    ensure_dirs,
)
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402

# Local aliases
ABSORBER_CACHE = ABSORBER_CODEBASE_SNAPSHOT
CODEBASE_MAP   = ABSORBER_CODEBASE_MAP
CODEBASE_MD    = ABSORBER_CODEBASE_MD
CODEBASE_LOG   = ABSORBER_CODEBASE_LOG


# ─────────────────────────────────────────────────────────────────────────────
# ── Constants ─────────────────────────────────────────────────────────────────

ROLE            = "absorber"
_MAX_TOKENS_MAP = 16384
_MAX_FILE_BYTES = 256 * 1024
_IGNORED_FILE   = "absorber.ignored"

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

# Pipeline-owned artifact subdirectories — skip when target == repo root.
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
        pat   = pattern.rstrip("/")
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
    if skip_artifact_control_dirs and name in _ARTIFACT_CONTROL_DIRS:
        return True
    return False


def _should_skip_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in _BUILTIN_SKIP_PATTERNS)


def _looks_like_config_file(rel_path: str, fname: str, ext: str) -> bool:
    """Heuristic to auto-promote likely config/manifest/infra files to key-only mode."""
    if ext not in _KEY_ONLY_EXTENSIONS:
        return False

    rel_lower  = rel_path.lower()
    name_lower = fname.lower()

    if ext == ".env" or name_lower.startswith(".env"):
        return True

    if any(kw in name_lower for kw in (
        "config", "settings", "secret", "appsetting",
        "credential", "password", "token", "manifest", "values", "override",
    )):
        return True

    if name_lower in {
        "package.json", "package-lock.json", "composer.json", "pom.xml",
        "launchsettings.json", "tsconfig.json", "tsconfig.app.json",
        "tsconfig.spec.json", "tsconfig.base.json", "angular.json",
        "dynamic-env.json", "ecs-task-def.json", "db-migrator-task-def.json",
    }:
        return True

    if any(token in rel_lower for token in (
        "task-def", "cloudformation", "helm", "k8s",
        "kubernetes", "deploy", "deployment", "docker-compose",
    )):
        return True

    return False


def _should_include_in_config_inventory(rel_path: str) -> bool:
    """Filter out pure build/tooling files from config inventory."""
    rel_lower = rel_path.lower()
    name      = Path(rel_path).name.lower()

    if name in {
        "package-lock.json", "package.json", "angular.json",
        "tsconfig.json", "tsconfig.app.json", "tsconfig.spec.json", "tsconfig.base.json",
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

    Returns list of dicts:
      { rel_path, abs_path, ext, size, mode, lang }
    """
    inventory: list[dict[str, Any]] = []

    for root_dir, dirs, files in os.walk(target):
        root_path = Path(root_dir)

        dirs[:] = [
            d for d in dirs
            if not _should_skip_dir(d, skip_artifact_control_dirs=skip_artifact_control_dirs)
        ]

        for fname in files:
            if _should_skip_file(fname):
                continue

            abs_path = root_path / fname

            try:
                rel_path = str(abs_path.relative_to(target))
            except ValueError:
                continue

            ext  = abs_path.suffix.lower()
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
                "ext":      ext,
                "size":     size,
                "mode":     mode,
                "lang":     lang,
            })

    inventory.sort(key=lambda x: x["rel_path"])
    return inventory


def _detect_language(path: Path) -> str | None:
    ext  = path.suffix.lower()
    name = path.name.lower()

    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "Dockerfile"

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
        ".xml": "XML",
        ".env": "ENV",
        ".ini": "INI",
        ".cfg": "Config", ".conf": "Config",
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
    mode     = entry["mode"]

    current_hash = _file_hash(abs_path)
    cached       = cache.get(rel_path, {})

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
        "hash":    current_hash,
        "mode":    mode,
        "content": content,
        "lang":    entry["lang"],
        "size":    entry["size"],
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
                capture_output=True, text=True, timeout=15,
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
        lines   = path.read_text(errors="replace").splitlines()
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
        tree   = ast.parse(source)
    except SyntaxError:
        return _extract_ts_signatures(path)
    except Exception:
        return "[signature extraction failed]"

    lines: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            args   = [a.arg for a in node.args.args]
            sig    = f"{prefix} {node.name}({', '.join(args)})"
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
# Config inventory helpers
# ─────────────────────────────────────────────────────────────────────────────

_SERVICE_PATTERNS: dict[str, re.Pattern] = {
    "database":   re.compile(r"(?i)(mysql|postgres|mongodb|redis|sqlite|sqlserver|mssql|mariadb|aurora|dynamodb|rds)"),
    "auth":       re.compile(r"(?i)(oauth|openid|jwt|keycloak|auth0|okta|cognito|openiddict|identity)"),
    "cloud":      re.compile(r"(?i)(aws|azure|gcp|s3|lambda|ecs|eks|cloudformation|cloudfront|route53|ecr|iam)"),
    "email":      re.compile(r"(?i)(smtp|sendgrid|mailgun|ses|email|mailchimp)"),
    "storage":    re.compile(r"(?i)(s3|blob|minio|gcs|cloudinary|uploadcare)"),
    "monitoring": re.compile(r"(?i)(datadog|newrelic|prometheus|grafana|sentry|cloudwatch|loki|opentelemetry)"),
    "messaging":  re.compile(r"(?i)(kafka|rabbitmq|sqs|sns|pubsub|nats|eventbridge|signalr)"),
}


def _parse_env_vars_from_text(raw: str) -> set[str]:
    """Extract likely env var names from config file text."""
    env_vars: set[str] = set()
    env_vars.update(re.findall(r"process\.env\.([A-Z_][A-Z0-9_]+)", raw))
    env_vars.update(re.findall(r"\$\{([A-Z_][A-Z0-9_]+)\}", raw))
    env_vars.update(re.findall(r"^([A-Z_][A-Z0-9_]{2,})\s*=", raw, re.MULTILINE))
    for key in re.findall(r'"([A-Za-z][A-Za-z0-9]{2,})"\s*:', raw):
        screaming = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).upper()
        if re.match(r"^[A-Z][A-Z0-9_]{2,}$", screaming):
            env_vars.add(screaming)
    return env_vars


def _build_config_structured(
    inventory: list[dict[str, Any]],
    cache: dict[str, Any],
) -> dict[str, Any]:
    """
    Build structured config inventory from key-only files.
    Returns dict for codebase_map.json "config" field.
    """
    config_files_data: list[dict[str, Any]] = []
    all_env_vars: set[str] = set()
    all_services: set[str] = set()

    eligible = [
        f for f in inventory
        if f["mode"] == "key-only" and _should_include_in_config_inventory(f["rel_path"])
    ]

    for cf in eligible[:30]:  # cap at 30 to avoid token explosion
        rel          = cf["rel_path"]
        cached_entry = cache.get(rel, {})
        redacted     = cached_entry.get("content", "")

        abs_path = Path(cf.get("abs_path", ""))
        raw      = ""
        if abs_path.exists():
            try:
                track_read(abs_path)
                raw = abs_path.read_text(errors="replace")
            except Exception:
                pass
        raw = raw or redacted

        file_env_vars = _parse_env_vars_from_text(raw)
        all_env_vars.update(file_env_vars)

        file_services: list[str] = []
        scan_text = raw or redacted
        for svc_name, pattern in _SERVICE_PATTERNS.items():
            if pattern.search(scan_text):
                file_services.append(svc_name)
                all_services.add(svc_name)

        config_files_data.append({
            "path":     rel,
            "env_vars": sorted(file_env_vars),
            "services": sorted(file_services),
        })

    return {
        "total_configs":     len(config_files_data),
        "services_detected": sorted(all_services),
        "env_vars_detected": sorted(all_env_vars),
        "files":             config_files_data,
    }


def _config_structured_to_prompt(config: dict[str, Any]) -> str:
    """Serialize structured config dict to compact string for the LLM prompt."""
    if not config or not config.get("files"):
        return ""
    lines = [
        f"Config files: {config['total_configs']}",
        f"Services detected: {', '.join(config['services_detected'])}",
        f"Env vars ({len(config['env_vars_detected'])}): {', '.join(config['env_vars_detected'][:60])}",
        "",
    ]
    for f in config["files"]:
        parts = []
        if f["env_vars"]:
            parts.append(f"env={','.join(f['env_vars'][:10])}")
        if f["services"]:
            parts.append(f"svc={','.join(f['services'])}")
        if parts:
            lines.append(f"  {f['path']}: {' | '.join(parts)}")
    return "\n".join(lines)


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
        unit_map  = {"m": "months", "w": "weeks", "d": "days"}
        return [f"--since={num}.{unit_map.get(unit, 'months')}.ago"]

    if scope.isdigit():
        return ["-n", scope]

    return ["--since=6.months.ago"]


def _git_log_stats(target: Path, scope: str) -> dict[str, Any]:
    """
    Run git log and return structured git data for codebase_map.json "git" field.

    Returns {} if not a git repo or git command fails.
    """
    if not (target / ".git").exists():
        return {}

    scope_args = _resolve_git_scope(scope)

    # File churn with per-file author tracking
    file_counts:  dict[str, int]      = {}
    file_authors: dict[str, set[str]] = {}
    try:
        result = subprocess.run(
            ["git", "log", "--format=%ae", "--name-only", *scope_args],
            capture_output=True, text=True, cwd=target, timeout=60,
        )
        if result.returncode != 0:
            return {}
        current_author = ""
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if "@" in line or line.startswith("github-"):
                current_author = line
            else:
                file_counts[line] = file_counts.get(line, 0) + 1
                file_authors.setdefault(line, set()).add(current_author)
    except Exception:
        return {}

    # Total commit count
    try:
        r = subprocess.run(
            ["git", "rev-list", "--count", "HEAD", *scope_args],
            capture_output=True, text=True, cwd=target, timeout=10,
        )
        total_commits = int(r.stdout.strip()) if r.returncode == 0 else 0
    except Exception:
        total_commits = 0

    # Contributors
    try:
        r = subprocess.run(
            ["git", "shortlog", "-sn", "--no-merges", *scope_args],
            capture_output=True, text=True, cwd=target, timeout=30,
        )
        authors: list[str] = []
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines()[:15]:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    authors.append(f"{parts[1]} ({parts[0].strip()} commits)")
    except Exception:
        authors = []

    # Hotspots tiered
    all_sorted = sorted(file_counts.items(), key=lambda x: -x[1])
    high = [
        {"file": f, "changes": c, "authors": sorted(file_authors.get(f, set()))}
        for f, c in all_sorted if c >= 10
    ][:20]
    medium = [
        {"file": f, "changes": c}
        for f, c in all_sorted if 5 <= c < 10
    ][:20]

    # Module activity
    total_changes  = sum(file_counts.values())
    module_counts: dict[str, int] = {}
    for fname, count in file_counts.items():
        parts  = fname.split("/")
        module = parts[0] if len(parts) > 1 else "(root)"
        module_counts[module] = module_counts.get(module, 0) + count

    module_activity = [
        {
            "module":  mod,
            "changes": cnt,
            "pct":     round(cnt / total_changes * 100, 1) if total_changes else 0,
        }
        for mod, cnt in sorted(module_counts.items(), key=lambda x: -x[1])[:10]
    ]

    return {
        "scope":           scope,
        "total_commits":   total_commits,
        "authors":         authors,
        "hotspots":        {"high": high, "medium": medium},
        "module_activity": module_activity,
    }


def _git_structured_to_prompt(git: dict[str, Any]) -> str:
    """Serialize structured git dict to compact string for the LLM prompt."""
    if not git:
        return ""
    lines = [
        f"Total commits (scope: {git.get('scope', '?')}): {git.get('total_commits', '?')}",
        "",
    ]
    high   = git.get("hotspots", {}).get("high", [])
    medium = git.get("hotspots", {}).get("medium", [])
    if high:
        lines += ["High-churn files (>=10 changes):", "| File | Changes | Authors |", "|------|---------|---------|"]
        for h in high:
            lines.append(f"| {h['file']} | {h['changes']} | {', '.join(h.get('authors', []))} |")
        lines.append("")
    if medium:
        lines += ["Medium-churn files (5-9 changes):", "| File | Changes |", "|------|---------|"]
        for m in medium:
            lines.append(f"| {m['file']} | {m['changes']} |")
        lines.append("")
    for m in git.get("module_activity", []):
        lines.append(f"  {m['module']}: {m['pct']}% ({m['changes']} changes)")
    if git.get("authors"):
        lines += ["", "Contributors:"] + [f"  {a}" for a in git["authors"]]
    return "\n".join(lines)


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
    context:     str,
    target_name: str,
    config:      dict[str, Any] | None = None,
    git:         dict[str, Any] | None = None,
) -> tuple[str, float]:
    """
    Call LLM to produce codebase_map.md content.

    Returns:
      (markdown_content, call_cost)
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system   = _MAP_SYSTEM.replace("{date}", date_str)

    user_parts = [f"Codebase: {target_name}\n\nExtracted content:\n\n{context}"]

    config_prompt = _config_structured_to_prompt(config or {})
    if config_prompt:
        user_parts.append(f"\n\n--- CONFIG INVENTORY DATA ---\n{config_prompt}")

    git_prompt = _git_structured_to_prompt(git or {})
    if git_prompt:
        user_parts.append(f"\n\n--- GIT/BLAME DATA ---\n{git_prompt}")

    user = "".join(user_parts)

    tokens_est = len(user) // 4
    print(f"[absorber] LLM call: {ROLE} | ~{tokens_est:,} input tokens")

    try:
        resp  = call_model(
            ROLE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=_MAX_TOKENS_MAP,
            temperature=0.2,
        )
        choice    = resp.choices[0]
        usage     = getattr(resp, "usage", None)
        call_cost = 0.0
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=get_model(ROLE), provider=get_provider(ROLE))
            print_call(__file__, pt, ct, call_cost)

        content       = choice.message.content
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
        return _fallback_map(context, target_name, config, git), 0.0


def _fallback_map(
    context:     str,
    target_name: str,
    config:      dict[str, Any] | None = None,
    git:         dict[str, Any] | None = None,
) -> str:
    """Produce minimal codebase_map.md without LLM when the call fails."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts    = [
        "# Codebase Map",
        f"_Generated: {date_str} | Absorber v2 (fallback — LLM unavailable)_\n",
        "## Project Overview",
        f"Target: {target_name}\n",
        "LLM call failed. Below is the raw extracted context for manual review.\n",
        "## Raw Extraction\n",
        context[:8000],
    ]
    config_prompt = _config_structured_to_prompt(config or {})
    if config_prompt:
        parts.append("\n## Config\n")
        parts.append(config_prompt[:4000])
    git_prompt = _git_structured_to_prompt(git or {})
    if git_prompt:
        parts.append("\n## Git/Blame\n")
        parts.append(git_prompt[:4000])
    parts.append("\n## Absorber Notes\n")
    parts.append("- This map was generated in fallback mode (no LLM). Re-run when model is available.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Append codebase_log.json (long-term audit trail)
# ─────────────────────────────────────────────────────────────────────────────

def _append_codebase_log(
    target:          Path,
    inventory:       list[dict[str, Any]],
    cached_count:    int,
    extracted_count: int,
    map_size:        int,
    *,
    call_cost:       float                       = 0.0,
    git_scope:       str                         = "",
    hotspot_summary: list[tuple[str, int]] | None = None,
) -> None:
    """
    Append a terse entry to absorber/codebase_log.json for audit trail.

    Args:
      call_cost        — USD cost from record_usage(); 0.0 if unavailable.
      git_scope        — raw --git-scope flag value (e.g. "6m", "all").
      hotspot_summary  — top-N [(file, change_count)]; None if git skipped.
    """
    entry = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "target":          str(target),
        "total_files":     len(inventory),
        "cached_files":    cached_count,
        "extracted_files": extracted_count,
        "modes": {
            "full":           sum(1 for f in inventory if f["mode"] == "full"),
            "key_only":       sum(1 for f in inventory if f["mode"] == "key-only"),
            "signature_only": sum(1 for f in inventory if f["mode"] == "signature-only"),
        },
        "languages":      _count_languages(inventory),
        "map_size_bytes": map_size,
        "cost":           round(call_cost or 0.0, 6),
        "git_scope":      git_scope or None,
        "hotspot_summary": (
            [{"file": f, "changes": c} for f, c in hotspot_summary]
            if hotspot_summary else None
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
# Context builder — assemble extraction results for LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

_MAX_PER_FILE      = 2_000    # chars per file — every file present, just capped
_MAX_CONTEXT_CHARS = 800_000  # ~200k tokens total budget


def _build_context(
    inventory: list[dict[str, Any]],
    cache:     dict[str, Any],
    force:     bool,
) -> tuple[str, int, int]:
    """
    Extract content from all files and build the context string for the LLM.

    Strategy:
      - Per-file cap _MAX_PER_FILE: every file present, content truncated not dropped
      - Grouped by top-level directory for structured context
      - Total budget _MAX_CONTEXT_CHARS: whole groups dropped only when exhausted

    Returns:
      (context_str, cached_count, extracted_count)
    """
    cached_count    = 0
    extracted_count = 0

    # Group by top-level directory
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in inventory:
        rel    = entry["rel_path"]
        parts  = rel.split("/")
        group  = parts[0] if len(parts) > 1 else "(root)"
        groups.setdefault(group, []).append(entry)

    sections:      list[str] = []
    total_chars              = 0
    omitted_groups: list[str] = []

    for group, entries in groups.items():
        group_lines: list[str] = [f"\n## {group}/"]
        group_chars             = len(group_lines[0])

        for entry in entries:
            raw_content, from_cache = extract_content(entry, cache, force)
            if from_cache:
                cached_count += 1
            else:
                extracted_count += 1

            rel  = entry["rel_path"]
            mode = entry["mode"]
            lang = entry["lang"] or "text"

            if len(raw_content) > _MAX_PER_FILE:
                display = raw_content[:_MAX_PER_FILE] + f"\n... ({len(raw_content) - _MAX_PER_FILE} chars truncated)"
            else:
                display = raw_content

            block = f"--- {rel} [{mode}] ({lang}) ---\n{display}\n"
            group_lines.append(block)
            group_chars += len(block)

        if total_chars + group_chars > _MAX_CONTEXT_CHARS:
            omitted_groups.append(group)
            continue

        sections.extend(group_lines)
        total_chars += group_chars

    if omitted_groups:
        omitted_files = sum(len(groups[g]) for g in omitted_groups)
        sections.append(
            f"\n... (context budget reached — {len(omitted_groups)} groups / "
            f"{omitted_files} files omitted: {', '.join(omitted_groups[:5])})"
        )

    return "\n".join(sections), cached_count, extracted_count


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_target(args: argparse.Namespace) -> Path:
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

    target      = _resolve_target(args)
    target_name = target.name

    print(f"[absorber] Target:    {target}")
    print(f"[absorber] Force:     {args.force}")
    print(f"[absorber] Git scope: {args.git_scope}")
    print(f"[absorber] Dry run:   {args.dry_run}")
    print()

    # ── Phase 1: File tree scan ───────────────────────────────────────────────
    rules_path = target / _IGNORED_FILE
    rules      = AbsorberIgnoreRules(rules_path)

    art_root      = Path(str(artifact_root()))
    skip_artifact = target == art_root.parent or art_root.is_relative_to(target)

    inventory = scan_files(target, rules, skip_artifact_control_dirs=skip_artifact)

    print(f"[absorber] Phase 1 — Scanned: {len(inventory)} files")
    mode_counts: dict[str, int] = {}
    for f in inventory:
        mode_counts[f["mode"]] = mode_counts.get(f["mode"], 0) + 1
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")
    print()

    if not inventory:
        print("[absorber][warn] No files found. Check target directory and absorber.ignored rules.")
        return

    # ── Phase 2: Content extraction ───────────────────────────────────────────
    cache                               = _load_cache()
    context, cached_count, extracted_count = _build_context(inventory, cache, args.force)

    print(f"[absorber] Phase 2 — Extracted: {extracted_count} new, {cached_count} from cache")
    print()

    _save_cache(cache)

    if args.dry_run:
        print("[absorber] Dry run — skipping LLM call and writes.")
        print(f"  Context size: {len(context):,} chars (~{len(context)//4:,} tokens)")
        print_artifact_summary()
        return

    # ── Phase 4: Git crawl (before LLM — data injected into prompt) ──────────
    git_data: dict[str, Any] = {}
    print(f"[absorber] Phase 4 — Git crawl (scope: {args.git_scope})")
    git_data = _git_log_stats(target, args.git_scope)
    if git_data:
        total_c = git_data.get("total_commits", 0)
        high_n  = len(git_data.get("hotspots", {}).get("high", []))
        med_n   = len(git_data.get("hotspots", {}).get("medium", []))
        print(f"  Commits: {total_c} | High-churn: {high_n} | Medium-churn: {med_n}")
    else:
        print("  No git data (not a git repo or no history in scope)")
    print()

    # ── Build structured config ───────────────────────────────────────────────
    config_data = _build_config_structured(inventory, cache)

    # ── Phase 3: LLM → codebase_map.md ───────────────────────────────────────
    print("[absorber] Phase 3 — LLM semantic compression")
    map_text, call_cost = call_llm_for_map(context, target_name, config_data, git_data)

    # ── Write codebase_map.md (LLM narrative — human/LLM readable) ───────────
    md_path = Path(str(CODEBASE_MD))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(map_text, encoding="utf-8")
    track_write(md_path)
    print(f"[absorber] Wrote: {md_path} ({md_path.stat().st_size:,} bytes)")

    # ── Write codebase_map.json (structured data — machine readable) ──────────
    codebase_map_json = {
        "meta": {
            "generated_at":     datetime.now(timezone.utc).isoformat(),
            "target":           str(target),
            "git_scope":        args.git_scope,
            "absorber_version": 2,
            "total_files":      len(inventory),
            "cached_files":     cached_count,
            "extracted_files":  extracted_count,
            "map_md":           str(md_path),
            "map_size_bytes":   len(map_text.encode()),
            "cost":             round(call_cost or 0.0, 6),
            "modes": {
                "full":           sum(1 for f in inventory if f["mode"] == "full"),
                "key_only":       sum(1 for f in inventory if f["mode"] == "key-only"),
                "signature_only": sum(1 for f in inventory if f["mode"] == "signature-only"),
            },
            "languages": _count_languages(inventory),
        },
        "config": config_data,
        "git":    git_data,
    }

    map_path = Path(str(CODEBASE_MAP))
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(codebase_map_json, indent=2, ensure_ascii=False))
    track_write(map_path)
    print(f"[absorber] Wrote: {map_path} ({map_path.stat().st_size:,} bytes)")

    # ── Phase 5: Append codebase_log ─────────────────────────────────────────
    hotspot_pairs = [
        (h["file"], h["changes"])
        for h in (
            git_data.get("hotspots", {}).get("high", []) +
            git_data.get("hotspots", {}).get("medium", [])
        )
    ] if git_data else None

    _append_codebase_log(
        target,
        inventory,
        cached_count,
        extracted_count,
        len(map_text.encode()),
        call_cost=call_cost,
        git_scope=args.git_scope,
        hotspot_summary=hotspot_pairs,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
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
        default="all",
        help="Git history scope: 6m, 3m, 500, all (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass cache, re-extract all files",
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