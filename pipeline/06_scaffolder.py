"""
pipeline/06_scaffolder.py
=========================
Step 6 — Generate scaffold JSON from spec and materialize source/test stubs.

This step is intended for the FULL pipeline. Mini runs should normally skip this
step and use the mini planner/implementer flow instead.

Writes, owner: scaffolder:
  artifacts_<slug>/state/scaffolder_codebase_skeleton.json
  artifacts_<slug>/cache/scaffolder_compressed_spec.md
  artifacts_<slug>/src/**
  artifacts_<slug>/tests/**

Reads:
  artifacts_<slug>/specwright_spec_<slug>.md

Direct execution:
  python 06_scaffolder.py --project my-app
  PIPELINE_PROJECT=my-app python 06_scaffolder.py

Required environment:
  GEMINI_API_KEY=<your-key>

Optional environment:
  GEMINI_MODEL=gemini-2.5-flash

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import httpx

# === WRITE AUTHORITY: scaffolder ===
# OWNS  : artifacts_<slug>/state/scaffolder_codebase_skeleton.json
#         artifacts_<slug>/cache/scaffolder_compressed_spec.md
#         artifacts_<slug>/src/**
#         artifacts_<slug>/tests/**
# READS : artifacts_<slug>/specwright_spec_<slug>.md

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    SCAFFOLD_JSON,
    SCAFFOLDER_COMPRESSED_SPEC,
    SRC_DIR,
    TESTS_DIR,
    ensure_dirs,
    get_spec_path,
)


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 32768


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
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI / project setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate scaffold JSON/files from the canonical spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 06_scaffolder.py --project my-app
              PIPELINE_PROJECT=my-app python 06_scaffolder.py

              python 06_scaffolder.py --project my-app --model gemini-2.5-flash
              python 06_scaffolder.py --project my-app --dry-run
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
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help=f"Gemini model name. Default: env GEMINI_MODEL or {DEFAULT_GEMINI_MODEL}.",
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


def _require_api_key(parser: argparse.ArgumentParser) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        parser.error(
            "GEMINI_API_KEY is not set. Export GEMINI_API_KEY=<your-key> and retry."
        )
    return api_key


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_url(model: str, api_key: str) -> str:
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )


def _extract_gemini_text(raw: dict[str, Any]) -> str:
    try:
        parts = raw["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {raw}") from exc

    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()

    if not text:
        raise ValueError(f"Gemini returned no text parts: {raw}")

    return text


def call_gemini(
    spec_content: str,
    *,
    api_key: str,
    model: str,
    max_retries: int = 5,
) -> dict[str, Any]:
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Here is the canonical spec:\n\n{spec_content}"}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }

    print(f"[06] Calling Gemini model: {model} …")

    timeout = httpx.Timeout(120.0, connect=30.0)
    url = _gemini_url(model, api_key)

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.post(url, json=payload)
                resp.raise_for_status()

                raw = resp.json()
                text = _extract_gemini_text(raw)
                return _parse_json(text)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response else None

                if status in {429, 500, 502, 503, 504} and attempt < max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(
                        f"[06][warn] API status {status}, retry "
                        f"{attempt}/{max_retries} in {wait:.1f}s …"
                    )
                    time.sleep(wait)
                    continue

                print(f"[06][error] API call failed: {exc}", file=sys.stderr)
                raise

            except httpx.TransportError as exc:
                if attempt < max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(
                        f"[06][warn] Transport error, retry "
                        f"{attempt}/{max_retries} in {wait:.1f}s: {exc}"
                    )
                    time.sleep(wait)
                    continue

                print(f"[06][error] Transport error: {exc}", file=sys.stderr)
                raise

    raise RuntimeError("Gemini call failed after retries")


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


def _validate_scaffold(scaffold: dict[str, Any]) -> None:
    required = {"scaffold_version", "files", "implementation_instructions"}
    missing = required - set(scaffold.keys())
    if missing:
        raise ValueError(f"scaffold JSON missing required keys: {sorted(missing)}")

    files = scaffold.get("files")
    if not isinstance(files, list):
        raise ValueError('scaffold["files"] must be a list')

    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(f"scaffold files[{idx}] must be an object")

        file_path = entry.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(f"scaffold files[{idx}].file_path must be a non-empty string")

        code = entry.get("code")
        if not isinstance(code, str):
            raise ValueError(f"scaffold files[{idx}].code must be a string")

        is_test = entry.get("is_test", False)
        if not isinstance(is_test, bool):
            raise ValueError(f"scaffold files[{idx}].is_test must be boolean when present")


# ─────────────────────────────────────────────────────────────────────────────
# Spec compression
# ─────────────────────────────────────────────────────────────────────────────

def _compress_spec(spec: str) -> str:
    """
    Create compressed version of the canonical spec for downstream models.

    Removes sections commonly useful only for scaffold-generation instructions:
      - ## 0.
      - ## 8.

    Keeps the rest of the spec intact.
    """
    lines = spec.splitlines()
    out: list[str] = []

    skip = False
    skip_headers = ("## 0.", "## 8.")
    resume_prefix = "## "

    for line in lines:
        if any(line.startswith(h) for h in skip_headers):
            skip = True
        elif (
            skip
            and line.startswith(resume_prefix)
            and not any(line.startswith(h) for h in skip_headers)
        ):
            skip = False

        if not skip:
            out.append(line)

    return "\n".join(out)


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


def write_files(scaffold: dict[str, Any], spec: str) -> None:
    SCAFFOLD_JSON.parent.mkdir(parents=True, exist_ok=True)
    SCAFFOLD_JSON.write_text(json.dumps(scaffold, indent=2))
    print(f"[06] Scaffold JSON → {SCAFFOLD_JSON}")

    for entry in scaffold["files"]:
        dest = _destination_for_entry(entry)

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(entry["code"])

        tag = "TEST" if entry.get("is_test", False) else "SRC "
        print(f"[06] [{tag}] {dest}")

    compressed = _compress_spec(spec)
    SCAFFOLDER_COMPRESSED_SPEC.parent.mkdir(parents=True, exist_ok=True)
    SCAFFOLDER_COMPRESSED_SPEC.write_text(compressed)

    savings = 0
    if spec:
        savings = round((1 - len(compressed) / len(spec)) * 100)

    print(f"[06] Compressed spec → {SCAFFOLDER_COMPRESSED_SPEC} ({savings}% smaller)")
    print("[06] Done.")


def preview_files(scaffold: dict[str, Any], spec: str) -> None:
    print("[06] --dry-run: scaffold validated. No files written.")
    print(f"[06] Would write scaffold JSON → {SCAFFOLD_JSON}")

    for entry in scaffold["files"]:
        dest = _destination_for_entry(entry)
        tag = "TEST" if entry.get("is_test", False) else "SRC "
        print(f"[06] Would write [{tag}] {dest}")

    compressed = _compress_spec(spec)
    savings = round((1 - len(compressed) / len(spec)) * 100) if spec else 0
    print(f"[06] Would write compressed spec → {SCAFFOLDER_COMPRESSED_SPEC} ({savings}% smaller)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: do not call ensure_dirs() at import-time.
    # PIPELINE_PROJECT must be available before artifact paths are resolved.
    ensure_dirs()

    api_key = _require_api_key(parser)

    spec_path = get_spec_path()
    if not spec_path.exists():
        print(f"[06][error] canonical spec not found: {spec_path}", file=sys.stderr)
        print(
            "[06][hint] This is a full-only scaffold step. Ensure specwright "
            "created the canonical spec before running 06_scaffolder.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = spec_path.read_text(errors="replace")

    scaffold = call_gemini(
        spec,
        api_key=api_key,
        model=args.model,
        max_retries=args.max_retries,
    )

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
        preview_files(scaffold, spec)
        return

    write_files(scaffold, spec)


if __name__ == "__main__":
    main()
