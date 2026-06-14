"""
pipeline/01_absorber.py
=======================
Step 1 — Absorb an existing codebase into the knowledge layer.

Design goal: always-fresh agent-ready artifacts. codebase_map.md and
codebase_map.json must reflect the current repo state so every downstream
pipeline step operates on accurate context.

────────────────────────────────────────────────────────────────
Artifacts produced
────────────────────────────────────────────────────────────────

  codebase_map.md        LLM-generated narrative. Sections:
                           Project Overview · Module Inventory · Entry Points
                           Data Flow · Config · ## Git/Blame · Tech Debt · Absorber Notes
                         Written once per full run. ## Git/Blame section can be
                         updated in-place by patch mode without touching other sections.

  codebase_map.json      Structured machine-readable data. Top-level keys:
                           meta   — run provenance (generated_at, patched_at, run_mode,
                                    total_files, cost, stale_since, …)
                           config — env vars (keys only, values redacted), detected services
                           git    — hotspots, contributors, module activity
                           staleness — written by --check-stale (stale, days_old, …)
                         Fully overwritten on full runs. Keys updated in-place on patch.

  codebase_log.json      Append-only audit trail. One entry per run (full or patch).

  cache/codebase_snapshot.json
                         Internal SHA256 per-file cache. Read+written every run.
                         Not consumed by downstream agents.

External integrations (optional, graceful fallback if absent):
  vfs CLI    — signature extraction
  Serena MCP — symbol-level call graph (future, via subprocess)

────────────────────────────────────────────────────────────────
Full run — phases
────────────────────────────────────────────────────────────────

  Phase 1  File tree scan      apply absorber.ignored rules → file inventory
  Phase 2  Content extraction  full / key-only / signature-only per file type
                               SHA256 cache skips files unchanged since last run
  Phase 3  Semantic compression  1 LLM call → full codebase_map.md narrative
  Phase 4  Git crawl           git log → codebase_map.json["git"]
  Phase 5  Write artifacts     codebase_map.md (OVERWRITE), codebase_map.json (OVERWRITE)
  Phase 6  Append log          1 entry appended to codebase_log.json

────────────────────────────────────────────────────────────────
Usage A — Full run  (first-time onboarding or major refactor)
────────────────────────────────────────────────────────────────

  python 01_absorber.py
  python 01_absorber.py --project my-app
  python 01_absorber.py --target /path/to/repo
  python 01_absorber.py --git-scope 6m|3m|500|all
  python 01_absorber.py --force

  All commands above run all 6 phases.

  Phases 1→2→3→4→5→6
    ├─ Phase 2: unchanged files served from SHA256 cache (--force bypasses)
    ├─ Phase 3: 1 LLM call, full context (~800k chars), rewrites entire codebase_map.md
    └─ Phase 4: git history depth controlled by --git-scope (default: all)

  Artifact impact:
    codebase_map.md          OVERWRITE  — entire file replaced by LLM output
    codebase_map.json        OVERWRITE  — entire file replaced; meta.stale_since = null
    codebase_log.json        APPEND     — 1 new entry {mode: "full", …}
    cache/snapshot.json      OVERWRITE  — updated SHA256s for all scanned files

  Cost: 1 LLM call. Runtime: 30s–5min depending on codebase size.
  When to use: first run, after a major refactor, after changing absorber config.

  ── --dry-run ──
  python 01_absorber.py --dry-run
    Runs Phase 1 + Phase 2 (scan + extract/cache) only. Stops before LLM.
    Prints context size (chars + estimated tokens). No artifact writes.
    Use to preview LLM input size or warm the cache without spending tokens.

    Artifact impact:
      cache/snapshot.json    OVERWRITE  — cache warmed
      everything else        UNCHANGED

────────────────────────────────────────────────────────────────
Usage B — --install-hook  (one-time setup, enables auto patch on every commit)
────────────────────────────────────────────────────────────────

  python 01_absorber.py --install-hook

  Writes a post-commit hook to .git/hooks/post-commit (appends if hook exists,
  never overwrites). The installed hook runs on every subsequent `git commit`:

    python 01_absorber.py --changed-since HEAD~1 --mode patch

  After --install-hook is run once, no further manual intervention is needed.
  Every commit automatically triggers a patch run (see Pattern 2 below).

  Artifact impact of --install-hook itself:
    .git/hooks/post-commit   WRITE/APPEND  — hook script added
    everything else          UNCHANGED     — no scan, no LLM

────────────────────────────────────────────────────────────────
Pattern 2 — --mode patch  (0 LLM · git-only update · auto or manual)
────────────────────────────────────────────────────────────────

  Triggered automatically by the post-commit hook after every commit.
  Can also be called manually at any time.

  python 01_absorber.py --mode patch                    ← manual
  python 01_absorber.py --changed-since HEAD~1 --mode patch  ← what hook runs

  Requires a prior full run (codebase_map.json must already exist).
  Skips Phase 1 (scan), Phase 2 (extraction), Phase 3 (LLM).
  Runs Phase 4 (git crawl) only, then writes results.

  commit
    └─► hook fires
          └─► Phase 4: git log → new git data
                ├─► codebase_map.json["git"]          IN-PLACE OVERWRITE
                ├─► codebase_map.json["meta"]
                │     patched_at  = now               IN-PLACE UPDATE
                │     stale_since = null              IN-PLACE UPDATE
                ├─► codebase_map.md ## Git/Blame      REGEX REPLACE (section only)
                │     (all other sections untouched)
                └─► codebase_log.json                 APPEND {mode:"patch", …}

  Cost: 0 LLM calls. Runtime: <1s.

  Note on --changed-since in patch mode:
    The hook combines --changed-since HEAD~1 with --mode patch.
    --changed-since invalidates cache entries for files in the diff.
    In patch mode extraction is skipped, so those invalidated cache entries
    will be re-extracted on the next full run or --changed-since full run.
    It is harmless and intentional: the cache is kept honest for whenever
    the next LLM run happens.

────────────────────────────────────────────────────────────────
Pattern 3 — --check-stale  (passive CI gate · 0 LLM · 0 extraction)
────────────────────────────────────────────────────────────────

  python 01_absorber.py --check-stale
  python 01_absorber.py --check-stale --stale-threshold 3

  Does NOT check for uncommitted/unstaged files.
  Checks the AGE of the artifact itself:
    reads codebase_map.json["meta"]["patched_at"]  (or "generated_at" if never patched)
    computes days since that timestamp
    if days > threshold (default 7) → marks as stale

  Also counts commits that landed after the artifact was last updated:
    git log --since=<patched_at> --name-only → changed_files count
  This reflects commits that the hook may have patched but a full LLM
  re-compression has not yet captured.

  why --check-stale is still useful even when hook is installed:
    - Hook only runs when YOU commit. If the repo is shared, a teammate's
      commits don't trigger your hook.
    - If the hook was installed recently on an existing repo, early commits
      were never patched.
    - Long-running branches may go days without a commit.
    - CI environments clone fresh — no hook, no patch history.

  Artifact impact:
    codebase_map.json["staleness"]   IN-PLACE UPSERT:
      {
        "checked_at":     "<iso>",
        "stale":          true | false,
        "reason":         "age" | "fresh" | "no_artifacts" | "unreadable_timestamp",
        "days_old":       N,
        "changed_files":  N,   ← commits since last absorb, not uncommitted files
        "threshold_days": N
      }
    everything else                  UNCHANGED

  Exit codes: 0 = fresh · 2 = stale
  Downstream agents read codebase_map.json["staleness"]["stale"] directly.

  Recommended CI gate:
    python 01_absorber.py --check-stale || python 01_absorber.py --mode patch
    # fresh → exit 0, nothing runs
    # stale → patch mode refreshes git data (0 LLM)

    For full narrative refresh on stale:
    python 01_absorber.py --check-stale || python 01_absorber.py --changed-since HEAD~20

────────────────────────────────────────────────────────────────
Pattern 4 — --changed-since  (manual · 1 LLM · partial extraction)
────────────────────────────────────────────────────────────────

  python 01_absorber.py --changed-since HEAD~1
  python 01_absorber.py --changed-since HEAD~20
  python 01_absorber.py --changed-since abc1234

  Human manual command. Runs all 6 phases (full run) but limits Phase 2
  extraction to only the files that changed since the given git ref.

  git diff --name-only <ref> HEAD
    └─► for each changed file: delete its SHA256 cache entry
          └─► Phase 2 re-extracts only those files from disk
                └─► unchanged files served from cache as before
                      └─► Phase 3: 1 LLM call
                            full context = re-extracted files + cached unchanged files
                            → codebase_map.md        OVERWRITE  (full narrative rewritten)
                            → codebase_map.json      OVERWRITE  (meta + git + config)
                            → codebase_log.json      APPEND
                            → cache/snapshot.json    OVERWRITE

  This is the primary way to refresh the full LLM narrative after commits
  have accumulated. The hook keeps git data current per-commit (patch mode),
  and --changed-since re-compresses the narrative when you decide it's needed.

  Typical trigger points:
    after merging a feature branch    → --changed-since HEAD~20
    after a sprint's worth of commits → --changed-since HEAD~50
    after a specific refactor commit  → --changed-since abc1234

  Cost: 1 LLM call. Extraction I/O proportional to number of changed files only.

  Relationship to Pattern 2:
    --mode patch     → per-commit, automatic, 0 LLM, only Git/Blame updated
    --changed-since  → manual, 1 LLM, full narrative rewritten with fresh context

────────────────────────────────────────────────────────────────
Recommended operational cadence
────────────────────────────────────────────────────────────────

  Day 0:
    python 01_absorber.py --target /path/to/repo   ← full run, build initial artifacts
    python 01_absorber.py --install-hook           ← enable auto patch on every commit

  Every commit (automatic, no action needed after Day 0):
    hook → --changed-since HEAD~1 --mode patch
    effect: codebase_map.json["git"] + ## Git/Blame stay current · 0 LLM

  Periodically / after feature branch merges:
    python 01_absorber.py --changed-since HEAD~N   ← refresh full narrative · 1 LLM
    (N = number of commits since last full LLM run)

  In CI before any agent pipeline run:
    python 01_absorber.py --check-stale --stale-threshold 3
    exit 2 → trigger --mode patch or --changed-since before agents consume artifacts

  After major refactor or dependency upgrade:
    python 01_absorber.py --force                  ← full re-extract + full LLM

────────────────────────────────────────────────────────────────
Artifact impact summary by command
────────────────────────────────────────────────────────────────

  Command                         md (full) md (Git/Blame) json (git) json (meta) json (stale) log    cache
  ──────────────────────────────  ───────── ────────────── ────────── ─────────── ──────────── ────── ─────
  (full run)                      OVERWRITE incl. above    OVERWRITE  OVERWRITE   cleared      APPEND OVERWRITE
  --force                         OVERWRITE incl. above    OVERWRITE  OVERWRITE   cleared      APPEND OVERWRITE
  --dry-run                       –         –              –          –           –            –      OVERWRITE
  --install-hook                  –         –              –          –           –            –      –
  --mode patch                    –         REGEX REPLACE  OVERWRITE  UPDATE      cleared      APPEND –
  --changed-since <ref>           OVERWRITE incl. above    OVERWRITE  OVERWRITE   cleared      APPEND OVERWRITE (partial)
  --check-stale                   –         –              –          –           UPSERT       –      –

  OVERWRITE = entire file replaced
  REGEX REPLACE = only ## Git/Blame section replaced, rest of file untouched
  UPDATE = specific keys updated in-place, rest of JSON untouched
  UPSERT = staleness key added/replaced in-place
  cleared = meta.stale_since set to null
  – = not touched

────────────────────────────────────────────────────────────────
Artifact paths (owner: absorber)
────────────────────────────────────────────────────────────────

  artifacts_<slug>/absorber/codebase_map.md
  artifacts_<slug>/absorber/codebase_map.json
  artifacts_<slug>/absorber/codebase_log.json
  artifacts_<slug>/absorber/cache/codebase_snapshot.json

  Reads: project source files + cache/codebase_snapshot.json (if present)
  At end of each run: prints files read and files written/appended.
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
# Pattern 1 — Git-hook installer
# ─────────────────────────────────────────────────────────────────────────────

_HOOK_SCRIPT = """\
#!/bin/sh
# absorber post-commit hook — invalidates cache for files changed in this commit
# then re-runs absorber in patch mode (git+staleness update, no LLM call).
# Installed by: python 01_absorber.py --install-hook
ABSORBER_SCRIPT="$(git rev-parse --show-toplevel)/pipeline/01_absorber.py"
if [ ! -f "$ABSORBER_SCRIPT" ]; then
    # fallback: look relative to hook location
    ABSORBER_SCRIPT="$(dirname "$0")/../../pipeline/01_absorber.py"
fi
if [ -f "$ABSORBER_SCRIPT" ]; then
    python "$ABSORBER_SCRIPT" --changed-since HEAD~1 --mode patch
fi
"""


def install_git_hook(target: Path) -> None:
    """
    Install a post-commit hook that runs absorber --changed-since HEAD~1 --mode patch
    after every commit. Appends to existing hook if one exists; does not overwrite.
    """
    git_dir = target / ".git"
    if not git_dir.is_dir():
        print("[absorber][error] Not a git repository — cannot install hook.", file=sys.stderr)
        sys.exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_path = hooks_dir / "post-commit"

    marker = "# absorber post-commit hook"

    if hook_path.exists():
        existing = hook_path.read_text()
        if marker in existing:
            print(f"[absorber] Hook already installed at {hook_path} — skipping.")
            return
        # Append to existing hook
        updated = existing.rstrip("\n") + "\n\n" + _HOOK_SCRIPT
        hook_path.write_text(updated)
        print(f"[absorber] Appended absorber hook to existing {hook_path}")
    else:
        hook_path.write_text(_HOOK_SCRIPT)
        print(f"[absorber] Installed hook at {hook_path}")

    hook_path.chmod(0o755)
    print("[absorber] Hook will run: absorber --changed-since HEAD~1 --mode patch on every commit")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 — Changed-since: invalidate cache for files touched since a ref
# ─────────────────────────────────────────────────────────────────────────────

def _invalidate_changed_files(target: Path, since_ref: str, cache: dict[str, Any]) -> list[str]:
    """
    Get files changed since git ref and delete their cache entries so
    extract_content() will re-extract them on the next _build_context() call.

    Returns list of invalidated rel_paths (for logging).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", since_ref, "HEAD"],
            capture_output=True, text=True, cwd=target, timeout=15,
        )
        if result.returncode != 0:
            print(f"[absorber][warn] git diff failed for ref '{since_ref}': {result.stderr.strip()}")
            return []
    except Exception as e:
        print(f"[absorber][warn] Could not run git diff: {e}")
        return []

    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    invalidated: list[str] = []

    for rel_path in changed:
        if rel_path in cache:
            del cache[rel_path]
            invalidated.append(rel_path)

    if invalidated:
        print(f"[absorber] Changed-since '{since_ref}': invalidated {len(invalidated)} cache entries")
        for p in invalidated[:10]:
            print(f"  - {p}")
        if len(invalidated) > 10:
            print(f"  ... and {len(invalidated) - 10} more")
    else:
        print(f"[absorber] Changed-since '{since_ref}': no cached files affected ({len(changed)} changed total)")

    return invalidated


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2 — Patch mode: update git + staleness, skip LLM re-compression
# ─────────────────────────────────────────────────────────────────────────────

_PATCH_GIT_SECTION_RE = re.compile(
    r"^## Git/Blame\s*\n.*?(?=^## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def run_patch_mode(target: Path, args: argparse.Namespace) -> None:
    """
    Patch mode: re-run git crawl and update staleness in codebase_map.json
    without calling the LLM. Also surgically replaces the ## Git/Blame section
    in codebase_map.md so it stays accurate after every commit.

    Appropriate for post-commit hooks and frequent incremental updates.
    Cost: 0 LLM calls, <1s typically.
    """
    print("[absorber] Mode: patch — updating git data and staleness (no LLM call)")

    md_path  = Path(str(CODEBASE_MD))
    map_path = Path(str(CODEBASE_MAP))

    if not map_path.exists():
        print("[absorber][warn] No codebase_map.json found — patch mode requires a prior full run.")
        print("[absorber]       Run without --mode patch first to generate initial artifacts.")
        sys.exit(1)

    # Load existing JSON
    try:
        track_read(map_path)
        existing_json: dict[str, Any] = json.loads(map_path.read_text())
    except Exception as e:
        print(f"[absorber][error] Could not read codebase_map.json: {e}", file=sys.stderr)
        sys.exit(1)

    # Re-run git crawl
    print(f"[absorber] Patch — git crawl (scope: {args.git_scope})")
    new_git = _git_log_stats(target, args.git_scope)
    if new_git:
        total_c = new_git.get("total_commits", 0)
        high_n  = len(new_git.get("hotspots", {}).get("high", []))
        print(f"  Commits: {total_c} | High-churn files: {high_n}")

    # Update JSON in-place
    existing_json["git"] = new_git
    existing_json.setdefault("meta", {})["patched_at"] = datetime.now(timezone.utc).isoformat()
    existing_json["meta"]["stale_since"] = None  # cleared on patch

    map_path.write_text(json.dumps(existing_json, indent=2, ensure_ascii=False))
    track_write(map_path)
    print(f"[absorber] Patched: {map_path}")

    # Surgically update ## Git/Blame section in .md
    if md_path.exists():
        try:
            track_read(md_path)
            md_text = md_path.read_text(encoding="utf-8")

            new_git_section = (
                "## Git/Blame\n"
                + _git_structured_to_prompt(new_git)
                + "\n\n"
            )

            if _PATCH_GIT_SECTION_RE.search(md_text):
                updated_md = _PATCH_GIT_SECTION_RE.sub(new_git_section, md_text)
            else:
                # Section not found — append
                updated_md = md_text.rstrip("\n") + "\n\n" + new_git_section

            md_path.write_text(updated_md, encoding="utf-8")
            track_write(md_path)
            print(f"[absorber] Patched Git/Blame section in: {md_path}")
        except Exception as e:
            print(f"[absorber][warn] Could not patch codebase_map.md: {e}")

    # Append a terse patch entry to codebase_log
    log_path = Path(str(CODEBASE_LOG))
    existing_log: list[dict[str, Any]] = []
    if log_path.exists():
        try:
            track_read(log_path)
            data = json.loads(log_path.read_text())
            existing_log = data.get("entries", data) if isinstance(data, dict) else data
        except Exception:
            pass

    patch_entry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode":         "patch",
        "target":       str(target),
        "git_scope":    args.git_scope,
        "total_commits": new_git.get("total_commits", 0) if new_git else 0,
    }
    existing_log.append(patch_entry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"entries": existing_log}, indent=2))
    track_write(log_path)
    print(f"[absorber] Appended patch entry to codebase_log (total: {len(existing_log)})")

    print()
    print_artifact_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3 — Staleness signal: detect drift without running absorber
# ─────────────────────────────────────────────────────────────────────────────

def check_stale(target: Path, threshold_days: int = 7) -> None:
    """
    Check if codebase_map artifacts are stale relative to the current git HEAD.

    Prints a machine-readable staleness report and sets exit code:
      0 — fresh (within threshold)
      2 — stale (exceeds threshold or no artifacts found)

    Downstream agents / CI can call this before consuming artifacts:
      python 01_absorber.py --check-stale || python 01_absorber.py --mode patch
    """
    map_path = Path(str(CODEBASE_MAP))

    print("[absorber] Staleness check")

    if not map_path.exists():
        print("[absorber][stale] No codebase_map.json found — never absorbed.")
        _write_staleness_to_json(map_path, stale=True, reason="no_artifacts", days_old=None)
        sys.exit(2)

    # Load generated_at from JSON
    try:
        track_read(map_path)
        meta      = json.loads(map_path.read_text()).get("meta", {})
        # prefer patched_at if available (patch mode refreshes this)
        ts_str    = meta.get("patched_at") or meta.get("generated_at", "")
        generated = datetime.fromisoformat(ts_str) if ts_str else None
    except Exception:
        generated = None

    if not generated:
        print("[absorber][stale] Could not parse generated_at from codebase_map.json.")
        _write_staleness_to_json(map_path, stale=True, reason="unreadable_timestamp", days_old=None)
        sys.exit(2)

    now      = datetime.now(timezone.utc)
    days_old = (now - generated.replace(tzinfo=timezone.utc)).days

    # Count files changed since artifact was generated (git-based drift signal)
    changed_count = 0
    if (target / ".git").is_dir():
        try:
            since_iso = generated.strftime("%Y-%m-%dT%H:%M:%S")
            result    = subprocess.run(
                ["git", "log", f"--since={since_iso}", "--name-only", "--format="],
                capture_output=True, text=True, cwd=target, timeout=15,
            )
            if result.returncode == 0:
                changed_count = len([l for l in result.stdout.splitlines() if l.strip()])
        except Exception:
            pass

    stale  = days_old > threshold_days
    reason = "age" if stale else "fresh"

    status = "STALE" if stale else "FRESH"
    print(f"  Status:        {status}")
    print(f"  Generated:     {generated.isoformat()}")
    print(f"  Age:           {days_old} day(s)")
    print(f"  Threshold:     {threshold_days} day(s)")
    print(f"  Files drifted: {changed_count} (since last absorb)")

    # Write staleness signal into codebase_map.json so agents can read it
    _write_staleness_to_json(map_path, stale=stale, reason=reason, days_old=days_old,
                             changed_count=changed_count, threshold_days=threshold_days)

    if stale:
        print(f"\n[absorber][stale] Artifacts are {days_old}d old (>{threshold_days}d threshold).")
        print("  Suggested: python 01_absorber.py --mode patch")
        print("  Or full:   python 01_absorber.py")
        sys.exit(2)
    else:
        print("\n[absorber] Artifacts are fresh.")
        sys.exit(0)


def _write_staleness_to_json(
    map_path: Path,
    *,
    stale:          bool,
    reason:         str,
    days_old:       int | None,
    changed_count:  int  = 0,
    threshold_days: int  = 7,
) -> None:
    """
    Upsert a 'staleness' key into codebase_map.json so downstream agents
    can read it without shelling out to git.
    """
    if not map_path.exists():
        return
    try:
        data = json.loads(map_path.read_text())
        data["staleness"] = {
            "checked_at":     datetime.now(timezone.utc).isoformat(),
            "stale":          stale,
            "reason":         reason,
            "days_old":       days_old,
            "changed_files":  changed_count,
            "threshold_days": threshold_days,
        }
        map_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[absorber][warn] Could not write staleness to JSON: {e}")


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

    # ── Phase 2: Content extraction (with optional changed-since invalidation) ──
    cache = _load_cache()

    # Pattern 1+4 — invalidate only files touched since a git ref
    if getattr(args, "changed_since", None):
        _invalidate_changed_files(target, args.changed_since, cache)
        print()

    context, cached_count, extracted_count = _build_context(inventory, cache, args.force)

    print(f"[absorber] Phase 2 — Extracted: {extracted_count} new, {cached_count} from cache")
    if getattr(args, "changed_since", None):
        print(f"  (incremental run — only files changed since '{args.changed_since}' were re-extracted)")
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
            "run_mode":         getattr(args, "mode", "full"),
            "changed_since":    getattr(args, "changed_since", None),
            "total_files":      len(inventory),
            "cached_files":     cached_count,
            "extracted_files":  extracted_count,
            "map_md":           str(md_path),
            "map_size_bytes":   len(map_text.encode()),
            "cost":             round(call_cost or 0.0, 6),
            "stale_since":      None,   # cleared on every full run; set by --check-stale
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

    # ── Pattern 1+4 — Incremental: invalidate cache for files changed since ref ──
    parser.add_argument(
        "--changed-since",
        default=None,
        metavar="GIT_REF",
        help=(
            "Invalidate cache only for files changed since this git ref "
            "(e.g. HEAD~1, HEAD~5, abc1234). "
            "Combines with --mode patch for zero-LLM incremental updates."
        ),
    )

    # ── Pattern 1 — Install post-commit git hook ──────────────────────────────
    parser.add_argument(
        "--install-hook",
        action="store_true",
        help=(
            "Install a post-commit git hook that runs "
            "'absorber --changed-since HEAD~1 --mode patch' after every commit."
        ),
    )

    # ── Pattern 2 — Run mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "patch"],
        help=(
            "full  — full scan + LLM semantic compression (default). "
            "patch — re-run git crawl + update staleness only, no LLM call. "
            "Designed for post-commit hooks and frequent incremental updates."
        ),
    )

    # ── Pattern 3 — Staleness check ──────────────────────────────────────────
    parser.add_argument(
        "--check-stale",
        action="store_true",
        help=(
            "Check if artifacts are stale without running absorber. "
            "Exits 0 if fresh, 2 if stale. Writes staleness signal to codebase_map.json."
        ),
    )
    parser.add_argument(
        "--stale-threshold",
        type=int,
        default=7,
        metavar="DAYS",
        help="Days before artifacts are considered stale (default: 7, used with --check-stale)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.project:
        os.environ["PIPELINE_PROJECT"] = args.project

    target = _resolve_target(args)

    print("=" * 60)
    print("  STEP 1 — ABSORBER")
    print("=" * 60)
    print()

    try:
        # ── Pattern 1 — Install git hook ─────────────────────────────────────
        if args.install_hook:
            install_git_hook(target)
            return

        # ── Pattern 3 — Staleness check (no extraction) ──────────────────────
        if args.check_stale:
            check_stale(target, threshold_days=args.stale_threshold)
            return  # check_stale calls sys.exit itself; this is a safety fallback

        # ── Pattern 2 — Patch mode (no LLM) ──────────────────────────────────
        if args.mode == "patch":
            run_patch_mode(target, args)
            return

        # ── Full run ──────────────────────────────────────────────────────────
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