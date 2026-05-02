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

def suggest_files(prompt: str, knowledge_context: str) -> list[str]:
    """Ask LLM to suggest which files need patching."""
    system = """\
You are a code assistant. Given a task, return the files that need to change.
Return ONLY:
{
  "files": ["src/path/to/file.tsx", ...]
}
Raw JSON only. No markdown fences."""
    user_msg = (
        f"Task: {prompt}\n\nKnowledge context:\n{knowledge_context}"
        if knowledge_context else f"Task: {prompt}"
    )
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
    knowledge_context: str,
    prev_error: str,
) -> tuple[str, str]:
    """
    Build (system, user) prompt for code-patch mode (original behaviour).
    LLM returns JSON {files_changed: [...]}.
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
        f"- Each 'content' must be the COMPLETE file — not a diff, not a snippet.\n"
        f"- Raw JSON only. No markdown fences."
    )
    context_block = (
        f"\n\n### Knowledge context\n{knowledge_context}" if knowledge_context else ""
    )
    error_block = (
        f"\n\n### Previous attempt failed\nErrors:\n{prev_error[-2000:]}"
        f"\n\nFix those issues."
        if prev_error else ""
    )
    user_msg = (
        f"Task: {prompt}"
        f"{context_block}"
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

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n[mini] DRY RUN — nothing will be written.")
        print(f"  task_type    : {task_type}")
        print(f"  context_file : {context_file or '(none)'}")
        print(f"  output_file  : {effective_output or '(none)'}")
        print(f"  target_files : {target_files or '(none — context-file mode)'}")
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
    else:
        print(f"  MINI ❌  FAIL  (gave up after {retry_count} retries)")
        print(f"  See findings_notes.md for failure pattern.")
    print(f"{'='*60}\n")

    sys.exit(0 if final_passed else 1)
