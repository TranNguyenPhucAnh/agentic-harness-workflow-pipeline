"""
pipeline/mini_mode.py
=====================
Mini mode — lightweight targeted task runner for the LLM pipeline.

This module is the daily driver for small tasks that don't warrant the full
spec → scaffold → generate → judge pipeline.

TASK TYPES
──────────
  code        Patch local source files (TypeScript/Python/etc.).
              Verification: vitest (frontend) or ruff+py_compile (Python).
              This is the original mini mode behaviour.

  sql         Optimize or fix a SQL file (Athena, Spark SQL, dbt model).
              Verification: sqlfluff lint → parse-clean.

  python      Standalone Python script / Spark job / Airflow DAG.
              Verification: py_compile + ruff check.

  config      YAML / JSON / TOML config files.
              Verification: parse-clean (no schema beyond well-formed).

  text        Generic text / markdown / prose transformation.
              Verification: LLM self-review pass (no external tool).

  auto        (default) Detect type from --context-file extension or prompt
              keywords, then route to the appropriate verifier.

USAGE
──────────
  # Patch local code files (original behaviour)
  python harness.py --mini "fix button color" --files src/Header.tsx

  # Optimize a SQL file with context
  python harness.py --mini "optimize for partition pruning" \\
      --context-file queries/daily_agg.sql \\
      --output-file  queries/daily_agg.sql

  # Refactor a Python DAG
  python harness.py --mini "split into two tasks" \\
      --context-file dags/ingest_orders.py \\
      --output-file  dags/ingest_orders.py \\
      --task-type python

  # Multi-file code patch (no context-file needed — LLM suggests)
  python harness.py --mini "extract theme tokens to constants file"

  # Dry-run: print what would happen, don't write anything
  python harness.py --mini "..." --context-file ... --dry-run

ARTIFACT OWNERSHIP
──────────────────
  OWNS  : artifacts/run/mini_log.json        (append-only)
  WRITES: --output-file path (if provided)   (overwrite on success only)
  READS : knowledge/current/base.md
          knowledge/current/findings_notes.md
          knowledge/current/spec_addendum.md
          --context-file (user-supplied, read-only)
  NEVER TOUCHES: spec.md, scaffold.json, any state/ artifact
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# mini_mode.py lives in pipeline/ — ROOT is one level up.
_HERE = Path(__file__).parent
ROOT  = _HERE.parent

sys.path.insert(0, str(ROOT))
from artifacts.paths import (
    KNOWLEDGE_BASE,
    FINDINGS_NOTES,
    SPEC_ADDENDUM,
    RUN_DIR,
    ensure_dirs,
)
ensure_dirs()

MINI_LOG = RUN_DIR / "mini_log.json"

# ── Task type registry ────────────────────────────────────────────────────────
# Maps task type → (file extensions that auto-detect to this type)
_EXT_TO_TYPE: dict[str, str] = {
    ".sql":     "sql",
    ".py":      "python",
    ".yaml":    "config",
    ".yml":     "config",
    ".json":    "config",
    ".toml":    "config",
    ".ts":      "code",
    ".tsx":     "code",
    ".js":      "code",
    ".jsx":     "code",
    ".md":      "text",
    ".txt":     "text",
}

_PROMPT_KEYWORDS_TO_TYPE: dict[str, list[str]] = {
    "sql":    ["query", "athena", "sql", "select", "dbt", "spark sql", "presto"],
    "python": ["dag", "airflow", "spark", "glue", "script", "python"],
    "config": ["yaml", "config", "json", "toml", "env"],
}


# ════════════════════════════════════════════════════════════════════════════
# LLM call
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(system: str, user_message: str, label: str = "mini") -> str:
    """
    Single Qwen call via OpenRouter.
    Raises RuntimeError after 2 failed attempts.
    """
    import httpx

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen/qwen3.6-plus",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.15,
        "max_tokens": 32768,
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=180) as client:
        for attempt in range(2):
            try:
                r = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {})
                print(f"[{label}] tokens: prompt={usage.get('prompt_tokens','?')} "
                      f"completion={usage.get('completion_tokens','?')}")
                choice  = data["choices"][0]
                content = choice["message"].get("content", "").strip()
                if not content:
                    raise RuntimeError(
                        f"LLM returned empty content. "
                        f"finish_reason={choice.get('finish_reason')}"
                    )
                return content
            except Exception as e:
                last_error = e
                print(f"[{label}] LLM error: {e}", file=sys.stderr)
                if attempt == 0:
                    print(f"[{label}] Retrying in 3s …", file=sys.stderr)
                    time.sleep(3)

    raise RuntimeError(f"LLM call failed after retries: {last_error}")


# ════════════════════════════════════════════════════════════════════════════
# JSON helpers
# ════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str) -> dict:
    """Strip markdown fences, parse JSON. Raises RuntimeError on failure."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$",        "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"JSON parse failed: {e}\nRaw (first 500):\n{raw[:500]}"
            )
    raise RuntimeError(
        f"No JSON object found in LLM response.\nRaw (first 500):\n{raw[:500]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Task type detection
# ════════════════════════════════════════════════════════════════════════════

def detect_task_type(
    prompt: str,
    context_file: Path | None,
    files: list[str] | None,
) -> str:
    """
    Infer task type from context_file extension, then --files extensions,
    then prompt keywords. Falls back to 'code'.
    """
    # 1. context_file extension is the strongest signal
    if context_file is not None:
        ext = context_file.suffix.lower()
        if ext in _EXT_TO_TYPE:
            return _EXT_TO_TYPE[ext]

    # 2. --files extensions (use first match)
    if files:
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in _EXT_TO_TYPE:
                return _EXT_TO_TYPE[ext]

    # 3. Prompt keyword scan
    prompt_lower = prompt.lower()
    for task_type, keywords in _PROMPT_KEYWORDS_TO_TYPE.items():
        if any(kw in prompt_lower for kw in keywords):
            return task_type

    return "code"  # default


# ════════════════════════════════════════════════════════════════════════════
# Verifiers — one per task type
# ════════════════════════════════════════════════════════════════════════════

def _verify_code(files_written: list[str]) -> tuple[bool, str]:
    """
    Run full vitest suite for frontend/TypeScript code changes.
    Falls back to npm test if vitest binary not found.
    """
    try:
        result = subprocess.run(
            ["npx", "vitest", "run", "--reporter=verbose"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Test run timed out after 120s"
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ["npm", "test", "--", "--run"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = (result.stdout + result.stderr).strip()
            return result.returncode == 0, output
        except Exception as e:
            return False, f"Test runner not found: {e}"
    except Exception as e:
        return False, f"Test execution error: {e}"


def _verify_sql(file_path: Path) -> tuple[bool, str]:
    """
    Verify SQL using sqlfluff lint.
    Falls back to a basic parse check if sqlfluff not installed.
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # Try sqlfluff first (best-in-class SQL linter, handles Athena/Spark dialect)
    try:
        result = subprocess.run(
            ["sqlfluff", "lint", "--dialect=ansi", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        # sqlfluff exits 0 = clean, 1 = lint violations, 65 = parse error
        if result.returncode == 65:
            return False, f"SQL parse error:\n{output}"
        if result.returncode == 1:
            # Lint violations — warn but don't fail (style, not correctness)
            print(f"[mini/verify] sqlfluff lint warnings (non-blocking):\n{output}")
            return True, output
        return True, output
    except FileNotFoundError:
        pass  # sqlfluff not installed — fall through to basic check

    # Basic fallback: just confirm it's non-empty valid UTF-8 text
    try:
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            return False, "SQL file is empty after patch."
        # Heuristic: must contain SELECT, INSERT, UPDATE, CREATE, or WITH
        if not re.search(
            r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|WITH|MERGE)\b",
            content, re.IGNORECASE
        ):
            return False, "Output does not look like SQL (no DML/DDL keyword found)."
        return True, "Basic SQL structure check passed (sqlfluff not installed)."
    except Exception as e:
        return False, f"Could not read output file: {e}"


def _verify_python(file_path: Path) -> tuple[bool, str]:
    """
    Verify Python using py_compile (syntax) + ruff check (fast linter).
    Both must pass.
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # 1. Syntax check via py_compile
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return False, f"Python syntax error:\n{result.stderr.strip()}"
    except Exception as e:
        return False, f"py_compile failed: {e}"

    # 2. Ruff check (optional — warn only if not installed)
    try:
        result = subprocess.run(
            ["ruff", "check", str(file_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            # Ruff errors are non-blocking — warn but pass
            print(f"[mini/verify] ruff warnings (non-blocking):\n{output}")
        return True, output or "Python syntax OK"
    except FileNotFoundError:
        return True, "Python syntax OK (ruff not installed — skipped lint)"
    except Exception as e:
        return True, f"Python syntax OK (ruff error: {e})"


def _verify_config(file_path: Path) -> tuple[bool, str]:
    """
    Verify config files are well-formed (YAML / JSON / TOML).
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    ext = file_path.suffix.lower()
    content = file_path.read_text(encoding="utf-8")

    try:
        if ext == ".json":
            json.loads(content)
            return True, "JSON parse OK"

        if ext in (".yaml", ".yml"):
            import yaml  # type: ignore
            yaml.safe_load(content)
            return True, "YAML parse OK"

        if ext == ".toml":
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                import tomli as tomllib  # type: ignore  # pip install tomli
            tomllib.loads(content)
            return True, "TOML parse OK"

        # Unknown config extension — basic non-empty check
        if content.strip():
            return True, f"Config file non-empty (no parser for {ext})"
        return False, "Config file is empty after patch."

    except ImportError as e:
        # Parser library not available — warn and pass
        print(f"[mini/verify] WARNING: {e} — skipping strict parse check.")
        return True, f"Config parse skipped ({e})"
    except Exception as e:
        return False, f"Config parse error: {e}"


def _verify_text_llm(
    prompt: str,
    original: str,
    result_text: str,
) -> tuple[bool, str]:
    """
    For generic text tasks: ask LLM to self-review whether the output
    satisfies the original task. Returns (True, reason) or (False, reason).
    This is a best-effort check — not a hard gate.
    """
    system = """\
You are a QA reviewer. Given a task, original text, and a proposed output,
decide if the output correctly and completely fulfils the task.

Return ONLY a JSON object:
{
  "pass": true | false,
  "reason": "one sentence explanation"
}
No markdown fences. Raw JSON only."""

    user_msg = (
        f"Task: {prompt}\n\n"
        f"Original:\n{original[:3000]}\n\n"
        f"Output:\n{result_text[:3000]}"
    )
    try:
        raw = _call_llm(system, user_msg, label="mini/verify")
        verdict = _parse_json(raw)
        passed = bool(verdict.get("pass", False))
        reason = verdict.get("reason", "no reason given")
        return passed, reason
    except Exception as e:
        # LLM review failed — pass with warning rather than blocking
        print(f"[mini/verify] LLM review failed: {e}", file=sys.stderr)
        return True, f"LLM review unavailable: {e}"


def run_verifier(
    task_type: str,
    files_written: list[str],
    output_path: Path | None,
    prompt: str,
    original_content: str,
    result_text: str,
) -> tuple[bool, str]:
    """
    Dispatch to the correct verifier based on task_type.
    Returns (passed: bool, output: str).
    """
    if task_type == "sql":
        target = output_path or (ROOT / files_written[0] if files_written else None)
        if target is None:
            return False, "No output file to verify."
        return _verify_sql(target)

    if task_type == "python":
        target = output_path or (ROOT / files_written[0] if files_written else None)
        if target is None:
            return False, "No output file to verify."
        return _verify_python(target)

    if task_type == "config":
        target = output_path or (ROOT / files_written[0] if files_written else None)
        if target is None:
            return False, "No output file to verify."
        return _verify_config(target)

    if task_type == "text":
        return _verify_text_llm(prompt, original_content, result_text)

    # Default: code (vitest)
    return _verify_code(files_written)


# ════════════════════════════════════════════════════════════════════════════
# Knowledge context
# ════════════════════════════════════════════════════════════════════════════

def load_knowledge_context() -> str:
    """
    Concatenate available knowledge layer files into one context string.
    Returns '' if none exist (standalone mode).
    """
    sections: list[str] = []
    sources = [
        (KNOWLEDGE_BASE,  "base.md"),
        (FINDINGS_NOTES,  "findings_notes.md"),
        (SPEC_ADDENDUM,   "spec_addendum.md"),
    ]
    for path, label in sources:
        if path.exists():
            text = path.read_text().strip()
            if text:
                sections.append(f"### {label}\n{text}")
    return "\n\n".join(sections) if sections else ""


# ════════════════════════════════════════════════════════════════════════════
# Context file handling (Layer 2)
# ════════════════════════════════════════════════════════════════════════════

def load_context_file(path: Path) -> str:
    """Read --context-file. Raises FileNotFoundError if missing."""
    return path.read_text(encoding="utf-8")


def write_output_file(path: Path, content: str) -> None:
    """Write LLM result to --output-file. Creates parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[mini] Written → {path}")


# ════════════════════════════════════════════════════════════════════════════
# File targeting (code mode only)
# ════════════════════════════════════════════════════════════════════════════

# ── Skip lists (borrowed + extended from vfs) ─────────────────────────────────
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", "artifacts",
    ".venv", "venv", "dist", "build", ".next",
    "target", ".terraform", "testdata", ".tox",
    "coverage", ".pytest_cache", ".mypy_cache",
})

_SKIP_SUFFIXES: frozenset[str] = frozenset({
    ".min.js", ".min.css", ".d.ts", ".pyc", ".pyo",
    ".lock",   # package-lock.json, poetry.lock, etc.
    ".map",    # source maps
})

_SKIP_PATTERNS: frozenset[str] = frozenset({
    ".spec.", ".test.", "_test.", "test_",
})

# Extensions that carry meaningful signatures
_SIG_EXTS: frozenset[str] = frozenset({
    ".ts", ".tsx", ".js", ".jsx",
    ".py", ".go", ".rs", ".java",
    ".sql", ".yaml", ".yml", ".toml", ".json",
    ".tf", ".hcl",
})


def _should_skip(path: Path) -> bool:
    """Return True if this path should be excluded from file scanning."""
    # Skip any file inside a blocked directory (at any depth)
    if any(part in _SKIP_DIRS for part in path.parts):
        return True
    name = path.name
    # Skip files matching blocked suffix combinations
    name_lower = name.lower()
    if any(name_lower.endswith(s) for s in _SKIP_SUFFIXES):
        return True
    # Skip test/spec files
    if any(pat in name_lower for pat in _SKIP_PATTERNS):
        return True
    return False


def _extract_signatures(path: Path) -> list[str]:
    """
    Lightweight signature extraction — no AST, regex-based.
    Returns a list of short signature strings for the file.
    Covers: TS/JS exports, Python defs/classes, SQL DDL, Go funcs.

    Intentionally simple: the goal is navigation context for the LLM,
    not a complete API surface. Full content is injected separately
    by load_file_context() when the file is confirmed as a patch target.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    ext  = path.suffix.lower()
    sigs: list[str] = []

    if ext in (".ts", ".tsx", ".js", ".jsx"):
        # Exported functions, consts, classes, interfaces, types, enums
        pattern = re.compile(
            r"^export\s+(?:default\s+)?"
            r"(?:async\s+)?"
            r"(function\s+\w+|const\s+\w+|class\s+\w+|interface\s+\w+"
            r"|type\s+\w+|enum\s+\w+)",
            re.MULTILINE,
        )
        sigs = [m.group(0).replace("export default ", "export ")
                           .replace("export ", "").strip()
                for m in pattern.finditer(text)]

    elif ext == ".py":
        # Top-level def, async def, class (non-private)
        pattern = re.compile(
            r"^(?:async\s+)?def\s+([A-Za-z][A-Za-z0-9_]*)\s*\(|"
            r"^class\s+([A-Za-z][A-Za-z0-9_]*)",
            re.MULTILINE,
        )
        for m in pattern.finditer(text):
            name = m.group(1) or m.group(2)
            if not name.startswith("_"):
                sigs.append(m.group(0).strip().rstrip("(").strip())

    elif ext == ".go":
        pattern = re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?([A-Z]\w*)\s*\(", re.MULTILINE)
        sigs = [m.group(0).strip().rstrip("(").strip() for m in pattern.finditer(text)]

    elif ext == ".sql":
        pattern = re.compile(
            r"^(?:CREATE|ALTER|DROP)\s+(?:TABLE|VIEW|FUNCTION|PROCEDURE|INDEX)\s+\S+",
            re.MULTILINE | re.IGNORECASE,
        )
        sigs = [m.group(0).strip() for m in pattern.finditer(text)]

    elif ext in (".yaml", ".yml"):
        # Top-level keys only (e.g. DAG id, job name, pipeline stage)
        pattern = re.compile(r"^([A-Za-z_][\w-]*):", re.MULTILINE)
        sigs = [m.group(1) for m in pattern.finditer(text)][:10]  # cap at 10

    # Deduplicate, cap at 15 signatures per file
    seen: set[str] = set()
    result: list[str] = []
    for s in sigs:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
        if len(result) >= 15:
            break
    return result


def _build_file_tree_with_sigs(cap_chars: int = 10_000) -> str:
    """
    Walk the project, build a file tree where source files include
    their exported signatures inline. Non-source files get paths only.

    Format:
        src/hooks/useSensorData.ts: useSensorData, SensorConfig
        src/types/sensor.ts: SensorPoint, AnomalyCluster, DecisionScore
        package.json
        tsconfig.json

    This gives the LLM navigation-quality context at a fraction of the
    token cost of injecting full file contents.
    """
    lines: list[str] = []
    total = 0

    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or _should_skip(p):
            continue
        if total >= cap_chars:
            lines.append("  ... (truncated)")
            break

        rel = str(p.relative_to(ROOT))
        ext = p.suffix.lower()

        if ext in _SIG_EXTS:
            sigs = _extract_signatures(p)
            if sigs:
                line = f"  {rel}: {', '.join(sigs)}"
            else:
                line = f"  {rel}"
        else:
            line = f"  {rel}"

        lines.append(line)
        total += len(line)

    return "\n".join(lines)


def suggest_files(prompt: str, knowledge_context: str) -> list[str]:
    """
    Ask LLM to suggest which files need patching.

    Injects a signature-aware file tree (paths + exported names, bodies
    stripped) instead of either raw file contents or paths-only.
    Saves ~60-80% tokens vs full content while giving the LLM enough
    signal to make an informed file selection.
    """
    print("[mini/suggest] Building signature index …", end=" ", flush=True)
    sig_tree = _build_file_tree_with_sigs()
    print(f"{len(sig_tree)} chars")

    system = """\
You are a code assistant. Given a task and a project signature index \
(file paths with their exported names), identify which files need to \
change to complete the task.

The signature index format is:
  path/to/file.ts: ExportedName1, ExportedName2, ...

Return ONLY:
{
  "files": ["path/to/file.tsx", "path/to/other.ts"]
}
Raw JSON only. No markdown fences. No explanation."""

    tree_block = (
        f"\n\n### Project signature index\n```\n{sig_tree}\n```"
        if sig_tree else ""
    )
    know_block = (
        f"\n\n### Knowledge context\n{knowledge_context}"
        if knowledge_context else ""
    )
    user_msg = f"Task: {prompt}{tree_block}{know_block}"

    try:
        raw    = _call_llm(system, user_msg, label="mini/suggest")
        result = _parse_json(raw)
        files  = result.get("files", [])
        return [str(f) for f in files if isinstance(f, str) and f.strip()]
    except Exception as e:
        print(f"[mini] Could not get file suggestions: {e}", file=sys.stderr)
        return []


def confirm_files(suggested: list[str]) -> list[str] | None:
    """
    Show suggested files, ask user to confirm / override / abort.
    Returns confirmed list, or None to abort.
    """
    if not suggested:
        print("[mini] LLM could not suggest files. Specify with --files.")
        return None

    print("\n[mini] Suggested files to patch:")
    for i, f in enumerate(suggested, 1):
        print(f"  {i}. {f}")
    print("\n  Enter to confirm | type paths to override | 'q' to quit:")
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if user_input.lower() in ("q", "quit", "abort"):
        return None
    if user_input == "":
        return suggested
    return [p.strip() for p in re.split(r"[\s,]+", user_input) if p.strip()]


def apply_patch(files_changed: list[dict]) -> list[str]:
    """
    Write patched files to disk.
    Returns list of paths actually written.
    """
    written: list[str] = []
    for entry in files_changed:
        path_str = entry.get("path", "").strip()
        content  = entry.get("content", "")
        if not path_str:
            print("[mini] WARNING: patch entry missing 'path' — skipped.", file=sys.stderr)
            continue
        dest = ROOT / path_str
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            print(f"[mini] Patched: {path_str}")
            written.append(path_str)
        except OSError as e:
            print(f"[mini] ERROR writing {path_str}: {e}", file=sys.stderr)
    return written


# ── Character budget for file content injection ───────────────────────────────
# Keeps total prompt size reasonable. Per-file cap prevents one large file
# from crowding out the others. Adjust if your model supports longer context.
_FILE_CONTENT_TOTAL_CAP = 40_000   # chars across all files combined
_FILE_CONTENT_PER_FILE  = 12_000   # chars per individual file
_DEP_CONTEXT_CAP        =  8_000   # chars for Layer B dep signatures combined


def load_file_context(target_files: list[str]) -> str:
    """
    Layer A — Read current content of target files from disk and format
    them as a fenced code block section for injection into the LLM prompt.

    Only reads files that exist. Truncates files exceeding _FILE_CONTENT_PER_FILE
    chars with a clear marker. Stops adding files once _FILE_CONTENT_TOTAL_CAP
    is reached (preserves most-important-first ordering).

    Returns '' if no files could be read (e.g. all new files).
    """
    sections: list[str] = []
    total_chars = 0

    for rel_path in target_files:
        if total_chars >= _FILE_CONTENT_TOTAL_CAP:
            remaining = len(target_files) - len(sections)
            if remaining:
                sections.append(
                    f"<!-- {remaining} more file(s) omitted — total context cap reached -->"
                )
            break

        abs_path = ROOT / rel_path
        if not abs_path.exists():
            sections.append(f"### {rel_path}\n*(new file — does not exist yet)*")
            continue

        try:
            raw = abs_path.read_text(encoding="utf-8")
        except Exception as e:
            sections.append(f"### {rel_path}\n*(could not read: {e})*")
            continue

        truncated = False
        if len(raw) > _FILE_CONTENT_PER_FILE:
            raw = raw[:_FILE_CONTENT_PER_FILE]
            truncated = True

        # Infer language for fenced block syntax highlighting
        ext = abs_path.suffix.lstrip(".")
        lang = {"ts": "typescript", "tsx": "typescript", "js": "javascript",
                "jsx": "javascript", "py": "python", "sql": "sql",
                "yaml": "yaml", "yml": "yaml", "json": "json",
                "toml": "toml", "md": "markdown"}.get(ext, ext)

        block = f"### {rel_path}\n```{lang}\n{raw}\n```"
        if truncated:
            block += f"\n*... (truncated at {_FILE_CONTENT_PER_FILE} chars)*"

        sections.append(block)
        total_chars += len(raw)

    if not sections:
        return ""

    header = f"### Current file contents ({len(sections)} file(s) loaded)"
    return header + "\n\n" + "\n\n".join(sections)


# ════════════════════════════════════════════════════════════════════════════
# Layer B — Dependency context (depth-1 local imports → signatures only)
# ════════════════════════════════════════════════════════════════════════════

def scan_local_imports(file_path: Path) -> list[str]:
    """
    Parse import statements in a single file and return relative paths
    of local (non-node_modules) dependencies.

    Covers:
      TS/JS  — import ... from './x' | '../x' | await import('./x')
      Python — from .mod import X  |  from ..mod import X
               from src.mod import X  |  import src.mod
      SQL / YAML / TOML / JSON — no import concept → returns []

    Depth-1 only: does NOT recurse into the dependencies found.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    ext     = file_path.suffix.lower()
    results: list[str] = []

    # ── TypeScript / JavaScript ───────────────────────────────────────────
    if ext in (".ts", ".tsx", ".js", ".jsx"):
        pattern = re.compile(
            r"""(?:import\s+.*?\s+from\s+|import\s*\(\s*)['"](\./[^'"]+|\.\.\/[^'"]+)['"]""",
            re.DOTALL,
        )
        base = file_path.parent
        for m in pattern.finditer(text):
            raw = m.group(1)
            # Resolve the specifier relative to the importing file
            resolved = (base / raw).resolve()
            # Try with common extensions if no extension given
            candidate: Path | None = None
            if resolved.exists():
                candidate = resolved
            else:
                for try_ext in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx"):
                    attempt = Path(str(resolved) + try_ext) if not try_ext.startswith("/") \
                              else resolved / f"index{try_ext.split('/')[-1]}"
                    if attempt.exists():
                        candidate = attempt
                        break
            if candidate:
                try:
                    results.append(str(candidate.relative_to(ROOT)))
                except ValueError:
                    pass  # outside ROOT — skip

    # ── Python ───────────────────────────────────────────────────────────
    elif ext == ".py":
        base = file_path.parent

        # Relative imports: from .mod import X  /  from ..mod import X
        rel_pattern = re.compile(r"^\s*from\s+(\.+)([\w.]*)\s+import", re.MULTILINE)
        for m in rel_pattern.finditer(text):
            dots  = m.group(1)      # e.g. "." or ".."
            mod   = m.group(2)      # e.g. "utils" or "utils.helpers"
            level = len(dots)       # 1 = same package, 2 = parent, …
            anchor = base
            for _ in range(level - 1):
                anchor = anchor.parent
            if mod:
                candidate = anchor / Path(mod.replace(".", "/"))
                for try_ext in (".py", "/__init__.py"):
                    attempt = Path(str(candidate) + try_ext) if not try_ext.startswith("/") \
                              else candidate / "__init__.py"
                    if attempt.exists():
                        try:
                            results.append(str(attempt.relative_to(ROOT)))
                        except ValueError:
                            pass
                        break

        # Absolute-local imports with src/ prefix:
        # from src.utils import X  /  import src.utils
        abs_pattern = re.compile(
            r"^\s*(?:from\s+(src(?:[\w.]+)?)\s+import|import\s+(src(?:[\w.]+)?))",
            re.MULTILINE,
        )
        for m in abs_pattern.finditer(text):
            mod_str = (m.group(1) or m.group(2)).replace(".", "/")
            candidate = ROOT / mod_str
            for try_ext in (".py", "/__init__.py"):
                attempt = Path(str(candidate) + try_ext) if not try_ext.startswith("/") \
                          else candidate / "__init__.py"
                if attempt.exists():
                    try:
                        results.append(str(attempt.relative_to(ROOT)))
                    except ValueError:
                        pass
                    break

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for r in results:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def resolve_deps(target_files: list[str]) -> list[str]:
    """
    Scan all target_files for local imports (depth 1).
    Returns sorted list of dependency paths that are NOT already in target_files,
    exist on disk, and are not skipped by _should_skip.
    """
    target_set = set(target_files)
    dep_set: set[str] = set()

    for rel in target_files:
        abs_path = ROOT / rel
        if not abs_path.exists():
            continue
        for dep in scan_local_imports(abs_path):
            if dep not in target_set:
                dep_path = ROOT / dep
                if dep_path.exists() and not _should_skip(dep_path):
                    dep_set.add(dep)

    return sorted(dep_set)


def build_dep_context(dep_files: list[str]) -> str:
    """
    Layer B — Build a compact dependency context string from dep_files.

    Uses _extract_signatures() (already exists) to get API surface only.
    Files are sorted by import-count (most-referenced first) for truncation priority.
    Total output is capped at _DEP_CONTEXT_CAP chars.

    Format injected into prompt:
        ### Dependency context (read-only — do not modify these files)
        src/types/sensor.ts: SensorPoint, AnomalyCluster, DecisionScore
        src/utils/anomaly.ts: buildAnomalyCluster, detectAnomaly
    """
    if not dep_files:
        return ""

    lines: list[str] = []
    total = 0

    for rel in dep_files:
        abs_path = ROOT / rel
        sigs = _extract_signatures(abs_path)
        if sigs:
            line = f"  {rel}: {', '.join(sigs)}"
        else:
            line = f"  {rel}"

        if total + len(line) > _DEP_CONTEXT_CAP:
            remaining = len(dep_files) - len(lines)
            if remaining:
                lines.append(f"  ... ({remaining} more dep file(s) omitted — cap reached)")
            break

        lines.append(line)
        total += len(line)

    if not lines:
        return ""

    header = "### Dependency context (read-only — do not modify these files)"
    return header + "\n" + "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════

def append_mini_log(
    prompt: str,
    task_type: str,
    files_changed: list[str],
    context_file: str | None,
    output_file: str | None,
    verify_result: str,
    retry_count: int,
) -> None:
    """Append one entry to run/mini_log.json."""
    entry: dict = {
        "timestamp":    datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_type":    task_type,
        "prompt":       prompt,
        "context_file": context_file,
        "output_file":  output_file,
        "files_changed": files_changed,
        "verify_result": verify_result,
        "retry_count":  retry_count,
    }
    existing: list[dict] = []
    if MINI_LOG.exists():
        try:
            existing = json.loads(MINI_LOG.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(entry)
    try:
        MINI_LOG.parent.mkdir(parents=True, exist_ok=True)
        MINI_LOG.write_text(json.dumps(existing, indent=2))
    except OSError as e:
        print(f"[mini] WARNING: could not write mini_log.json: {e}", file=sys.stderr)


def append_findings_note(
    prompt: str,
    task_type: str,
    error_output: str,
    retry_count: int,
) -> None:
    """Append failure pattern to findings_notes.md (only on fail/retry > 1)."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    note = (
        f"\n---\n"
        f"## Mini run failure — {ts}\n\n"
        f"**Task type:** `{task_type}`  \n"
        f"**Task:** {prompt}\n\n"
        f"**Retries:** {retry_count}\n\n"
        f"**Failure pattern:**\n"
        f"```\n{error_output[-1500:].strip()}\n```\n"
    )
    try:
        FINDINGS_NOTES.parent.mkdir(parents=True, exist_ok=True)
        with FINDINGS_NOTES.open("a") as f:
            f.write(note)
        print(f"[mini] Failure pattern → {FINDINGS_NOTES.relative_to(ROOT)}")
    except OSError as e:
        print(f"[mini] WARNING: could not write findings_notes.md: {e}", file=sys.stderr)


# ════════════════════════════════════════════════════════════════════════════
# Prompt builders — per task type
# ════════════════════════════════════════════════════════════════════════════

def _build_context_file_prompt(
    prompt: str,
    task_type: str,
    context_content: str,
    knowledge_context: str,
    prev_error: str,
) -> tuple[str, str]:
    """
    Build (system, user) prompt for context-file mode (Layer 2).
    LLM returns plain text output, not JSON.
    """
    type_instructions: dict[str, str] = {
        "sql": (
            "You are a senior data engineer specializing in SQL optimization.\n"
            "Return ONLY the complete rewritten SQL. No explanation, no markdown fences."
        ),
        "python": (
            "You are a senior Python/data-engineering developer.\n"
            "Return ONLY the complete rewritten Python file. No explanation, no markdown fences."
        ),
        "config": (
            "You are a DevOps/infrastructure engineer.\n"
            "Return ONLY the complete updated config file. No explanation, no markdown fences."
        ),
        "text": (
            "You are a technical writer / document editor.\n"
            "Return ONLY the complete rewritten text. No explanation, no preamble."
        ),
    }
    system = type_instructions.get(
        task_type,
        "You are a senior developer. Return ONLY the complete updated file content."
    )

    knowledge_block = (
        f"\n\n### Project knowledge context\n{knowledge_context}"
        if knowledge_context else ""
    )
    error_block = (
        f"\n\n### Previous attempt failed\nErrors:\n{prev_error[-2000:]}"
        f"\n\nFix those issues in this attempt."
        if prev_error else ""
    )
    user_msg = (
        f"Task: {prompt}\n\n"
        f"### Current file content\n{context_content}"
        f"{knowledge_block}"
        f"{error_block}"
    )
    return system, user_msg


def _build_code_patch_prompt(
    prompt: str,
    target_files: list[str],
    file_context: str,
    dep_context: str,
    knowledge_context: str,
    prev_error: str,
) -> tuple[str, str]:
    """
    Build (system, user) prompt for code-patch mode.
    LLM returns JSON {files_changed: [...]}.
    file_context  — Layer A: full content of target files.
    dep_context   — Layer B: signatures of local imports (read-only context).
    """
    files_constraint = (
        f"You MUST only modify these files: {json.dumps(target_files)}\n"
        if target_files else
        "Modify only the files necessary for this task.\n"
    )
    system = (
        f"You are a senior developer performing a targeted code patch.\n\n"
        f"{files_constraint}"
        f"Rules:\n"
        f"- DO NOT regenerate the full project.\n"
        f"- DO NOT create unrelated files.\n"
        f"- DO NOT modify spec.md or any artifact files.\n"
        f"- Return ONLY a JSON object:\n"
        f"  {{\"files_changed\": [{{\"path\": \"src/...\", \"content\": \"<full file>\"}}]}}\n"
        f"- Each 'content' must be the COMPLETE updated file — not a diff, not a snippet.\n"
        f"- Raw JSON only. No markdown fences."
    )
    file_block  = f"\n\n{file_context}"                                if file_context        else ""
    dep_block   = f"\n\n{dep_context}"                                 if dep_context         else ""
    know_block  = f"\n\n### Knowledge context\n{knowledge_context}"   if knowledge_context   else ""
    error_block = (
        f"\n\n### Previous attempt failed\nErrors:\n{prev_error[-2000:]}"
        f"\n\nFix those issues."
        if prev_error else ""
    )
    user_msg = (
        f"Task: {prompt}"
        f"{file_block}"
        f"{dep_block}"
        f"{know_block}"
        f"\n\nFiles to patch: {json.dumps(target_files)}"
        f"{error_block}"
    )
    return system, user_msg


# ════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════

def run_mini(
    prompt: str,
    files: list[str] | None,
    context_file: Path | None,
    output_file: Path | None,
    task_type_override: str | None,
    dry_run: bool = False,
) -> None:
    """
    Mini mode entry point.

    Two execution paths:
      A) context-file mode  — when --context-file is provided.
         LLM rewrites the file content; result written to --output-file
         (defaults to overwriting context-file if not specified).
         Verifier is type-specific (SQL, Python, config, text).

      B) code-patch mode    — when no --context-file.
         LLM patches local project files; returns JSON {files_changed: [...]}.
         Verifier is vitest (or ruff for Python).
    """
    print(f"\n{'='*60}")
    print(f"  MINI MODE")
    print(f"{'='*60}")
    print(f"[mini] Task: {prompt}")

    # ── Detect task type ─────────────────────────────────────────────────────
    task_type = task_type_override or detect_task_type(prompt, context_file, files)
    if task_type_override == "auto" or task_type_override is None:
        print(f"[mini] Task type: {task_type} (auto-detected)")
    else:
        print(f"[mini] Task type: {task_type} (explicit)")

    # ── Load knowledge context ───────────────────────────────────────────────
    knowledge_context = load_knowledge_context()
    if knowledge_context:
        print(f"[mini] Knowledge context: {len(knowledge_context)} chars")
    else:
        print("[mini] No knowledge context — standalone mode.")

    # ── Load context file (Layer 2) ──────────────────────────────────────────
    context_content: str = ""
    if context_file is not None:
        if not context_file.exists():
            print(f"[mini] ERROR: --context-file not found: {context_file}",
                  file=sys.stderr)
            sys.exit(1)
        context_content = load_context_file(context_file)
        print(f"[mini] Context file: {context_file}  ({len(context_content)} chars)")

    # ── Determine output target ──────────────────────────────────────────────
    # context-file mode: output_file defaults to context_file (overwrite in place)
    effective_output = output_file or (context_file if context_file else None)
    if context_file and effective_output == context_file:
        print(f"[mini] Output: overwriting {context_file} (no --output-file specified)")
    elif effective_output:
        print(f"[mini] Output: {effective_output}")

    # ── File targeting (code-patch mode only) ────────────────────────────────
    target_files: list[str] = []
    if context_file is None:
        if files:
            target_files = list(files)
        else:
            print("[mini] No --files — asking LLM to suggest …")
            suggested    = suggest_files(prompt, knowledge_context)
            confirmed    = confirm_files(suggested)
            if confirmed is None:
                print("[mini] Aborted.")
                sys.exit(0)
            target_files = confirmed
        print(f"[mini] Target files: {target_files}")

    # ── Layer A: load current file contents ──────────────────────────────────
    # Only in code-patch mode (context-file mode already has explicit content).
    file_context = ""
    if context_file is None and target_files:
        file_context = load_file_context(target_files)
        if file_context:
            loaded = [f for f in target_files if (ROOT / f).exists()]
            print(f"[mini] File context loaded: {len(loaded)} file(s)  "
                  f"({len(file_context)} chars injected into prompt)")
        else:
            print("[mini] File context: no existing files found (all new).")

    # ── Layer B: dependency context ───────────────────────────────────────────
    # Depth-1 local imports of target files → signatures only (read-only).
    # Skipped entirely in context-file mode (no target_files to scan).
    dep_files: list[str] = []
    dep_context = ""
    if context_file is None and target_files:
        dep_files = resolve_deps(target_files)
        if dep_files:
            dep_context = build_dep_context(dep_files)
            print(f"[mini] Dep context: {len(dep_files)} file(s)  "
                  f"({len(dep_context)} chars injected into prompt)")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n[mini] DRY RUN — nothing will be written.")
        print(f"  task_type    : {task_type}")
        print(f"  context_file : {context_file or '(none)'}")
        print(f"  output_file  : {effective_output or '(none)'}")
        print(f"  target_files : {target_files or '(none — context-file mode)'}")
        print(f"  file_context : {len(file_context)} chars" if file_context else
              f"  file_context : (none)")
        if dep_files:
            print(f"  dep_context  : {len(dep_files)} file(s)  ({len(dep_context)} chars)")
            for d in dep_files:
                print(f"    └─ {d}")
        else:
            print(f"  dep_context  : (none)")
        print(f"  verifier     : "
              f"{'vitest' if task_type == 'code' else task_type + ' linter/parser'}")
        return

    # ── Retry loop ────────────────────────────────────────────────────────────
    retry_count      = 0
    last_error       = ""
    files_written: list[str] = []
    result_text      = ""
    final_passed     = False

    for attempt in range(3):  # 1 initial + 2 retries
        if attempt > 0:
            retry_count += 1
            print(f"\n[mini] Retry {retry_count}/2 …")

        # ── Build prompt ─────────────────────────────────────────────────────
        if context_file is not None:
            # Layer 2: context-file mode
            system, user_msg = _build_context_file_prompt(
                prompt, task_type, context_content,
                knowledge_context, last_error if attempt > 0 else "",
            )
        else:
            # Original: code-patch mode
            system, user_msg = _build_code_patch_prompt(
                prompt, target_files,
                file_context,
                dep_context,
                knowledge_context, last_error if attempt > 0 else "",
            )

        # ── LLM call ─────────────────────────────────────────────────────────
        try:
            raw = _call_llm(system, user_msg)
        except Exception as e:
            print(f"[mini] LLM error: {e}", file=sys.stderr)
            last_error = str(e)
            continue

        # ── Apply result ─────────────────────────────────────────────────────
        if context_file is not None:
            # Layer 2: raw text output → write to output file
            result_text = raw.strip()
            if not result_text:
                print("[mini] WARNING: LLM returned empty content.", file=sys.stderr)
                last_error = "LLM returned empty content."
                continue
            assert effective_output is not None
            write_output_file(effective_output, result_text)
            files_written = [str(effective_output)]
        else:
            # Code-patch mode: JSON output → apply to project files
            try:
                patch = _parse_json(raw)
            except RuntimeError as e:
                print(f"[mini] Parse error: {e}", file=sys.stderr)
                last_error = str(e)
                continue

            entries = patch.get("files_changed", [])
            if not entries:
                print("[mini] WARNING: LLM returned empty files_changed.", file=sys.stderr)
                last_error = "LLM returned empty files_changed."
                continue

            files_written = apply_patch(entries)
            result_text   = raw  # keep for potential text-verifier fallback

        # ── Verify ───────────────────────────────────────────────────────────
        print(f"\n[mini] Verifying ({task_type}) …")
        passed, verify_output = run_verifier(
            task_type    = task_type,
            files_written= files_written,
            output_path  = effective_output,
            prompt       = prompt,
            original_content = context_content,
            result_text  = result_text,
        )
        last_error = verify_output

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"[mini] Verify: {status}")
        if not passed:
            tail = "\n".join(verify_output.splitlines()[-20:])
            print(f"\n[mini] Verify output:\n{tail}")

        if passed:
            final_passed = True
            break

    # ── Log ───────────────────────────────────────────────────────────────────
    append_mini_log(
        prompt       = prompt,
        task_type    = task_type,
        files_changed= files_written,
        context_file = str(context_file) if context_file else None,
        output_file  = str(effective_output) if effective_output else None,
        verify_result= "pass" if final_passed else "fail",
        retry_count  = retry_count,
    )
    print(f"[mini] Logged → {MINI_LOG.relative_to(ROOT)}")

    # ── Knowledge contribution ────────────────────────────────────────────────
    if not final_passed or retry_count > 1:
        append_findings_note(prompt, task_type, last_error, retry_count)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
  
    if final_passed:
        print(f"  MINI ✅  PASS  (task_type={task_type}, retries={retry_count})")
        if files_written:
            print(f"  Files written: {files_written}")
        if retry_count > 1:
            print(f"  Retry pattern logged → findings_notes.md")
    else:
        print(f"  MINI ❌  FAIL  (gave up after {retry_count} retries)")
        print(f"\n  Failure pattern logged → knowledge/current/findings_notes.md")
        print(f"\n  Next steps:")
        print(f"    1. Fix manually:  edit {files_written or 'the relevant files'}")
        print(f"    2. Capture fix:   python pipeline/07_update_knowledge.py --capture-human-fix")
        print(f"       └─ git diff src/ → distills pattern into knowledge/current/base.md")
        print(f"       └─ next mini run will have this context injected automatically")
    print(f"{'='*60}\n")

    sys.exit(0 if final_passed else 1)
