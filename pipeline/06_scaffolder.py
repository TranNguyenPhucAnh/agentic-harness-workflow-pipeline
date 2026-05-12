"""
pipeline/06_scaffolder.py
=========================
Step 6 — Generate scaffold JSON from canonical spec and materialize source/test stubs.

This step is intended for the FULL pipeline. Mini runs should normally skip this
step and use the mini planner/implementer flow instead.

Writes, owner: scaffolder:
  artifacts_<slug>/state/scaffolder_codebase_skeleton.json
  artifacts_<slug>/src/**
  artifacts_<slug>/tests/**

Reads:
  artifacts_<slug>/specwright_spec_<slug>.md

Direct execution:
  python 06_scaffolder.py --project my-app
  PIPELINE_PROJECT=my-app python 06_scaffolder.py

Required environment:
  Xem artifacts/models.py — API key được lấy tự động theo role "scaffolder".

At the end of each run, prints:
  - artifacts/files read
  - artifacts/files created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# === WRITE AUTHORITY: scaffolder ===
# OWNS  : artifacts_<slug>/state/scaffolder_codebase_skeleton.json
#         artifacts_<slug>/src/**
#         artifacts_<slug>/tests/**
# READS : artifacts_<slug>/specwright_spec_<slug>.md

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from artifacts.paths import (  # noqa: E402
    SCAFFOLD_JSON,
    SRC_DIR,
    TESTS_DIR,
    ensure_dirs,
    get_spec_path,
)


ROLE = "scaffolder"
MAX_OUTPUT_TOKENS = 32768  # Reduced from 16384 — most providers cap at 16k output tokens


# ─────────────────────────────────────────────────────────────────────────────
# Artifact/file access tracking
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_READ: set[str] = set()
_ARTIFACTS_WRITTEN: set[str] = set()


def _track_read(path: Any) -> None:
    _ARTIFACTS_READ.add(str(path))


def _track_write(path: Any) -> None:
    _ARTIFACTS_WRITTEN.add(str(path))


def _print_artifact_access_summary() -> None:
    print("[06] Artifacts/files read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[06]   READ  {item}")
    else:
        print("[06]   READ  (none)")

    print("[06] Artifacts/files created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[06]   WRITE {item}")
    else:
        print("[06]   WRITE (none)")


SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software architect generating a scaffold from a technical spec.

    You will receive the canonical spec. The spec is the source of truth. Follow
    its stack, file tree, output schema, naming, testing framework, and acceptance
    criteria.

    Your task:
    1. Read the spec carefully, especially the file tree and output schema sections.
    2. Produce a SINGLE valid JSON object matching the schema requested by the spec.
    3. The JSON MUST be valid and parseable by JSON.parse / json.loads.

    JSON requirements:
    - Use double quotes " for all JSON strings.
    - Escape any internal " characters as \\".
    - If you output code in a "code" field, it MUST be a single JSON string value
      with all newlines as \\n and all quotes properly escaped.
    - Do NOT use single quotes ' as JSON string delimiters.
    - Do NOT include comments.
    - Do NOT include trailing commas.
    - Do NOT wrap the response in markdown fences.
    - Output raw JSON only.

    Scaffold requirements:
    - Do NOT add files not listed by the spec's file tree.
    - For non-test files, generate interfaces/types/function signatures/stubs as
      requested by the spec. If the spec asks for stubs, use explicit
      "not implemented" placeholders.
    - For test files, generate complete runnable tests using the framework
      specified by the spec.
    - Preserve file paths exactly as specified by the spec unless the output
      schema explicitly says otherwise.
      
    Output schema for each file entry (STRICT — no other key names allowed):
    {
      "file_path": "relative/path/to/file.py",   // NOT "path", NOT "filepath"
      "code": "...",
      "is_test": false
    }
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI / project setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="06_scaffolder.py",
        description="Generate scaffold JSON/files from the canonical spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 06_scaffolder.py --project my-app
              PIPELINE_PROJECT=my-app python 06_scaffolder.py

              python 06_scaffolder.py --project my-app --dry-run

            Model/provider config: xem artifacts/models.py, role "scaffolder".
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
        "--dry-run",
        action="store_true",
        help="Call model and validate scaffold, but do not write files.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for transient model/API failures. Default: 5.",
    )
    return parser


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
        "PIPELINE_PROJECT=<name> before running 06_scaffolder.py directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model call
# ─────────────────────────────────────────────────────────────────────────────

def call_scaffolder(
    spec_content: str,
    *,
    max_retries: int = 5,
) -> dict[str, Any]:
    """
    Call the scaffolder role (config in artifacts/models.py) with the canonical spec.

    Retries on transient failures with exponential backoff.
    Returns the parsed scaffold as a dict.
    """
    import random
    import time

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the canonical spec:\n\n{spec_content}",
        },
    ]

    model    = get_model(ROLE)
    provider = get_provider(ROLE)
    print(f"[06] Calling model: {model} (provider: {provider}) …")

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = call_model(
                ROLE,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.2,
            )
            text = resp.choices[0].message.content or ""

            # Detect likely token-limit truncation before attempting JSON parse
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            if finish_reason == "max_tokens":
                raise ValueError(
                    f"Model hit max_tokens={MAX_OUTPUT_TOKENS} limit — response truncated. "
                    "Reduce scaffold scope or split into smaller requests."
                )

            return _parse_json(text)

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries:
                wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(
                    f"[06][warn] Call failed (attempt {attempt}/{max_retries}), "
                    f"retry in {wait:.1f}s: {exc}"
                )
                time.sleep(wait)
            else:
                print(f"[06][error] Model call failed: {exc}", file=sys.stderr)

    raise RuntimeError(f"Scaffolder call failed after {max_retries} retries") from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing / validation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict[str, Any]:
    """
    Robust JSON extraction.

    Handles accidental markdown fences and, as a fallback, extracts the outermost
    JSON object.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        if "Unterminated string" in str(exc) or cleaned.rstrip()[-1:] not in ("}", "]"):
            print(
                f"[06][warn] Primary JSON parse failed (TRUNCATED RESPONSE — model likely "
                f"hit max_tokens={MAX_OUTPUT_TOKENS} limit): {exc}",
                file=sys.stderr,
            )
            print(
                f"[06][warn] Response is {len(cleaned)} chars. Reduce scaffold scope or "
                f"verify your model supports {MAX_OUTPUT_TOKENS} output tokens.",
                file=sys.stderr,
            )
        else:
            print(f"[06][warn] Primary JSON parse failed: {exc}", file=sys.stderr)
    else:
        if not isinstance(parsed, dict):
            raise ValueError("Model returned JSON, but top-level value is not an object.")
        return parsed

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            print(
                f"[06][error] JSON parse failed after extracting outer object: {exc}",
                file=sys.stderr,
            )
            print(f"[06][error] Raw output first 1000 chars:\n{cleaned[:1000]}", file=sys.stderr)
            raise SystemExit(1) from exc

        if not isinstance(parsed, dict):
            raise ValueError("Extracted JSON top-level value is not an object.")
        return parsed

    print("[06][error] No JSON object found in model response.", file=sys.stderr)
    print(f"[06][error] Raw output first 1000 chars:\n{cleaned[:1000]}", file=sys.stderr)
    raise SystemExit(1)


_FILE_PATH_ALIASES = ("path", "filepath", "filename", "file", "name")

def _normalize_scaffold_entry(entry: dict[str, Any], idx: int) -> dict[str, Any]:
    if "file_path" not in entry:
        for alias in _FILE_PATH_ALIASES:
            if alias in entry:
                print(f"[06][warn] files[{idx}]: aliased '{alias}' → 'file_path'")
                entry = {**entry, "file_path": entry[alias]}
                break
    return entry


def _validate_scaffold(scaffold: dict[str, Any]) -> None:
    required = {"files"}
    missing = required - set(scaffold.keys())
    if missing:
        raise ValueError(f"scaffold JSON missing required keys: {sorted(missing)}")

    files = scaffold.get("files")
    if not isinstance(files, list):
        raise ValueError('scaffold["files"] must be a list')

    normalized: list[dict] = []
    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(f"scaffold files[{idx}] must be an object")

        entry = _normalize_scaffold_entry(entry, idx)

        file_path = entry.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(f"scaffold files[{idx}].file_path must be a non-empty string")

        code = entry.get("code")
        if not isinstance(code, str):
            raise ValueError(f"scaffold files[{idx}].code must be a string")

        is_test = entry.get("is_test", False)
        if not isinstance(is_test, bool):
            raise ValueError(f"scaffold files[{idx}].is_test must be boolean when present")

        normalized.append(entry)

    scaffold["files"] = normalized


# ─────────────────────────────────────────────────────────────────────────────
# File writer
# ─────────────────────────────────────────────────────────────────────────────

def _safe_relative_path(raw_path: str) -> Path:
    """
    Convert model-provided path to safe relative path.

    Rejects:
      - absolute paths
      - path traversal using ..
      - empty paths
    """
    normalized = raw_path.replace("\\", "/").strip()
    rel = Path(normalized)

    if not normalized:
        raise ValueError("empty file_path is not allowed")

    if rel.is_absolute():
        raise ValueError(f"absolute file_path is not allowed: {raw_path}")

    if any(part == ".." for part in rel.parts):
        raise ValueError(f"path traversal is not allowed: {raw_path}")

    return rel


def _destination_for_entry(entry: dict[str, Any]) -> Path:
    file_path = entry["file_path"]
    is_test = entry.get("is_test", False)

    rel = _safe_relative_path(file_path)

    if is_test:
        parts = rel.parts
        if parts and parts[0] == "tests":
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(rel.name)
        return TESTS_DIR / rel

    parts = rel.parts
    if parts and parts[0] == "src":
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(rel.name)
    return SRC_DIR / rel


def write_files(scaffold: dict[str, Any]) -> None:
    SCAFFOLD_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCAFFOLD_JSON.write_text(json.dumps(scaffold, indent=2))
    _track_write(SCAFFOLD_JSON)
    print(f"[06] Scaffold JSON → {SCAFFOLD_JSON}")

    for entry in scaffold["files"]:
        dest = _destination_for_entry(entry)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(entry["code"])
        _track_write(dest)

        tag = "TEST" if entry.get("is_test", False) else "SRC "
        print(f"[06] [{tag}] {dest}")

    print("[06] Done.")


def preview_files(scaffold: dict[str, Any]) -> None:
    print("[06] --dry-run: scaffold validated. No files written.")
    print(f"[06] Would write scaffold JSON → {SCAFFOLD_JSON}")

    for entry in scaffold["files"]:
        dest = _destination_for_entry(entry)
        tag = "TEST" if entry.get("is_test", False) else "SRC "
        print(f"[06] Would write [{tag}] {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    exit_code = 0

    try:
        parser = _build_parser()
        args = parser.parse_args()

        _configure_project(args.project, parser)

        # Important: do not call ensure_dirs() at import-time.
        # PIPELINE_PROJECT must be available before artifact paths are resolved.
        ensure_dirs()

        spec_path = get_spec_path()
        if not spec_path.exists():
            print(f"[06][error] canonical spec not found: {spec_path}", file=sys.stderr)
            print(
                "[06][hint] This is a full-only scaffold step. Ensure specwright "
                "created the canonical spec before running 06_scaffolder.py.",
                file=sys.stderr,
            )
            sys.exit(1)

        _track_read(spec_path)
        spec = spec_path.read_text(errors="replace")

        scaffold = call_scaffolder(
            spec,
            max_retries=args.max_retries,
        )

        # DEBUG
        if scaffold.get("files"):
            print(f"[06][debug] files[0] keys: {list(scaffold['files'][0].keys())}", file=sys.stderr)

        try:
            _validate_scaffold(scaffold)
        except ValueError as exc:
            print(f"[06][error] Invalid scaffold JSON: {exc}", file=sys.stderr)
            print(
                f"[06][debug] Scaffold first 1000 chars:\n"
                f"{json.dumps(scaffold, indent=2)[:1000]}",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.dry_run:
            preview_files(scaffold)
            return

        write_files(scaffold)

    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[06][error] Scaffolder failed: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
