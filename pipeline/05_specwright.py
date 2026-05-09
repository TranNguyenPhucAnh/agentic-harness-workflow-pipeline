#!/usr/bin/env python3
"""
05_specwright.py
================
Spec Agent — nhận enriched prompt từ 04_enricher, gửi cho model xịn,
nhận về spec markdown, cho user review/edit, rồi hỏi có muốn kích hoạt
full harness không.

Vị trí trong luồng:
    04_enricher → [05_specwright] → artifacts_<slug>/specwright_spec_<slug>.md → (optional) harness

Inputs:
    execution/enricher_session_enriched_prompt.md      — output của 04_enricher (bắt buộc)
    state/clarificator_requirement_synthesis.md        — fallback nếu enriched_prompt thiếu
    execution/clarificator_session_raw.json            — để lấy project metadata

Output:
    artifacts_<slug>/specwright_spec_<slug>.md         — technical spec tại get_spec_path()
                                                         (đây là input canonical cho toàn bộ harness:
                                                          spectracker, scaffolder, planner, executor, judge, v.v.)

Usage:
    python 05_specwright.py --project my-app
    python 05_specwright.py --project my-app --model anthropic/claude-opus-4-5
    python 05_specwright.py --project my-app --dry-run
    python 05_specwright.py --project my-app --no-review

    # Thường được gọi tự động từ 04_enricher.py.

Artifacts produced (owner: specwright):
    artifacts_<slug>/specwright_spec_<slug>.md         — get_spec_path(), input cho harness.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # type: ignore
    get_spec_path,
    CLARIFIED_REQ,
    CLARIFICATOR_SESSION_RAW,
    ENRICHER_SESSION_PROMPT,
    ensure_dirs,
)

# Local aliases — map canonical constants to the short names used internally
CLARIFICATION_REPORT = CLARIFICATOR_SESSION_RAW
ENRICHED_PROMPT      = ENRICHER_SESSION_PROMPT

# === WRITE AUTHORITY: specwright ===
# OWNS  : artifacts_<slug>/specwright_spec_<slug>.md   (dynamic path via get_spec_path())
# READS : enricher_session_enriched_prompt.md,
#         clarificator_requirement_synthesis.md (fallback),
#         clarificator_session_raw.json


# ── Model config ──────────────────────────────────────────────────────────────
# Spec agent dùng model xịn — spec phải đủ chất lượng để harness chạy được.
# Default: claude-sonnet (cân bằng chất/giá). Override bằng --model.
_SPEC_MODEL_DEFAULT = "anthropic/claude-sonnet-4-5"
_MAX_TOKENS_SPEC    = 8192


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _print_banner(msg: str) -> None:
    width = min(80, len(msg) + 4)
    print("\n" + "─" * width)
    print(f"  {msg}")
    print("─" * width)


def _wrap(text: str, indent: int = 0) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix, subsequent_indent=prefix)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    system: str,
    user: str,
    model: str = _SPEC_MODEL_DEFAULT,
    max_tokens: int = _MAX_TOKENS_SPEC,
) -> str:
    if "/" in model:
        api_key  = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        model_id = model
    else:
        api_key  = os.environ.get("OPENAI_API_KEY", "")
        base_url = "https://api.openai.com/v1"
        model_id = model

    if not api_key:
        print("\n[specwright][offline] No API key found. Paste LLM response then EOF (Ctrl-D):")
        return sys.stdin.read()

    try:
        payload = {
            "model":      model_id,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        print(f"[specwright] Using model: {model_id}")
        with httpx.Client(timeout=180) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[specwright][error] LLM call failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Context loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_enriched_prompt() -> str:
    if ENRICHED_PROMPT.exists():
        return ENRICHED_PROMPT.read_text(encoding="utf-8")
    return ""


def _load_clarified_req() -> str:
    if CLARIFIED_REQ.exists():
        return CLARIFIED_REQ.read_text(encoding="utf-8")
    return ""


def _load_clarification_report() -> dict:
    if CLARIFICATION_REPORT.exists():
        try:
            return json.loads(CLARIFICATION_REPORT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Spec generation
# ─────────────────────────────────────────────────────────────────────────────

_SPEC_SYSTEM = """
You are a senior software architect writing a technical specification document.

Your output will be used directly as input to an AI coding pipeline that will
implement the system. The spec must be precise, complete, and unambiguous.

REQUIRED SECTIONS (use H2 headings, in this order):

## Overview
One paragraph: what the system does, who uses it, and why it exists.

## Goals
Bulleted list of concrete, measurable objectives.

## Non-Goals
Bulleted list of what is explicitly out of scope for this iteration.

## Architecture
Describe the overall system structure: components, layers, data flow.
Include a text-based diagram if it aids clarity (ASCII or mermaid fenced block).

## Tech Stack
Table or bulleted list: language, framework, database, infrastructure, key libraries.
Be specific about versions where relevant.

## Data Models
For each entity: fields, types, constraints, relationships.
Use a consistent format (e.g. TypeScript interface, SQL DDL, or structured prose).

## API Contracts
For each endpoint or interface:
  - Method + path (or function signature)
  - Request schema
  - Response schema
  - Error cases
If not applicable (e.g. pure backend job), write "N/A — no external API surface."

## Workflow & State Machine
Describe the main workflows as numbered steps.
If the system has stateful entities (orders, approvals, jobs), include a state
transition table: State | Trigger | Next State | Side Effects.

## Error Handling
How the system handles: validation errors, external service failures, 
data integrity violations, unexpected states.

## Non-Functional Requirements
Performance targets, scalability assumptions, availability SLA,
security requirements, observability (logging, metrics, alerts).

## Out of Scope
Features or concerns deferred to future iterations. Be explicit.

## Acceptance Criteria
Numbered list of testable conditions that define "done".
Each criterion must be verifiable by an automated test or a manual check procedure.

## Open Questions
List any remaining ambiguities as:
  - `<!-- TODO: clarify -->` inline in the relevant section, AND
  - A numbered list here with a suggested resolution for each.

QUALITY RULES (strictly enforced):
1. Every section must be present. Do not skip any, even if brief.
2. Every Acceptance Criterion must be independently testable.
3. Data model field types must be concrete (string, uuid, integer, boolean,
   timestamp — not "some value" or "relevant data").
4. API request/response schemas must include required vs optional fields.
5. No vague language: avoid "should", "might", "as needed", "etc.", "and so on".
   Replace with exact behavior.
6. If the enriched prompt contains <!-- TODO: clarify --> markers, preserve them
   in the relevant section and list them in Open Questions.
7. Output only the spec markdown. No preamble, no postamble.
""".strip()


def _generate_spec(enriched_prompt: str, model: str) -> str:
    """Call spec model with enriched prompt, return raw spec markdown."""
    user_msg = f"""Use the enriched prompt below to write the technical specification.
Follow all instructions in the prompt exactly.

{enriched_prompt}

Write the specification now."""

    return _call_llm(_SPEC_SYSTEM, user_msg, model=model)


# ─────────────────────────────────────────────────────────────────────────────
# Spec validation — lightweight structural check before showing to user
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = [
    "## Overview",
    "## Goals",
    "## Non-Goals",
    "## Architecture",
    "## Tech Stack",
    "## Data Models",
    "## API Contracts",
    "## Workflow",
    "## Error Handling",
    "## Non-Functional",
    "## Out of Scope",
    "## Acceptance Criteria",
]


def _validate_spec(spec: str) -> list[str]:
    """
    Check that required H2 sections are present.
    Returns list of missing section names (empty = all good).
    """
    missing: list[str] = []
    for section in _REQUIRED_SECTIONS:
        # Case-insensitive prefix match — allow minor heading variations
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if not pattern.search(spec):
            missing.append(section)
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# User review
# ─────────────────────────────────────────────────────────────────────────────

def _review_spec(spec: str, spec_file: Path) -> tuple[str, bool]:
    """
    Show spec summary (first 60 lines) to user, ask to confirm / edit / abort.
    Returns (final_spec_text, should_continue_to_harness).
    """
    _print_banner(f"Spec generated — review before activating harness")

    # Show first 60 lines as preview
    lines = spec.strip().splitlines()
    preview_lines = lines[:60]
    print("\n" + "─" * 72)
    print("\n".join(preview_lines))
    if len(lines) > 60:
        print(f"\n  ... [{len(lines) - 60} more lines] — full spec written to {spec_file.name}")
    print("─" * 72)

    print(f"\n  Full spec: {spec_file}")
    print()
    print("  [1] confirm — activate full harness pipeline")
    print("  [2] edit    — open $EDITOR to modify spec before running harness")
    print("  [3] stop    — keep spec, do not run harness now\n")

    while True:
        choice = input("  → Choose 1 / 2 / 3: ").strip()
        if choice in ("1", "confirm"):
            return spec, True
        if choice in ("2", "edit"):
            edited = _open_in_editor(spec, spec_file)
            if edited and edited.strip():
                print("\n[specwright] Updated spec loaded.")
                return edited, True
            print("[specwright] Editor returned empty — keeping generated spec.")
            return spec, True
        if choice in ("3", "stop"):
            return spec, False
        print("  Please enter 1, 2, or 3.")


def _open_in_editor(content: str, hint_path: Path) -> str:
    """
    Write content to a temp file named after the spec, open $EDITOR,
    return modified content.
    """
    import tempfile
    editor = os.environ.get("EDITOR", "nano")
    # Use spec filename as suffix hint so editor shows correct syntax highlight
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{hint_path.name}",
        delete=False, encoding="utf-8"
    ) as tf:
        tf.write(content)
        tmp_path = tf.name

    try:
        subprocess.run([editor, tmp_path], check=True)
        return Path(tmp_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[specwright][warn] Editor '{editor}' not found. Set $EDITOR env var.")
        return content
    except subprocess.CalledProcessError as exc:
        print(f"[specwright][warn] Editor exited with error: {exc}")
        return content
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Harness launcher
# ─────────────────────────────────────────────────────────────────────────────

def _launch_harness(project_name: str) -> None:
    """
    Launch harness.py for the current project.
    harness.py reads PIPELINE_PROJECT from env (already set in main).
    harness.py picks up the spec from get_spec_path() → specwright_spec_<slug>.md.
    """
    script = Path(__file__).parent / "harness.py"
    if not script.exists():
        print(f"\n[specwright][warn] harness.py not found at {script}")
        print(f"[specwright]       Run manually: PIPELINE_PROJECT={project_name!r} python harness.py")
        return

    print(f"\n[specwright] Launching harness → {script.name}  [project: {project_name!r}]")
    print(f"[specwright] Spec: {get_spec_path()}")
    print(f"[specwright] Press Ctrl-C to abort harness at any time.\n")

    env = os.environ.copy()
    env["PIPELINE_PROJECT"] = project_name

    try:
        subprocess.run(
            [sys.executable, str(script)],
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print("\n[specwright] Harness interrupted by user.")


# ─────────────────────────────────────────────────────────────────────────────
# Resolve project
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_project(arg_project: str | None) -> str:
    from artifacts.paths import get_project_name  # type: ignore
    if arg_project:
        return arg_project.strip()
    try:
        return get_project_name()
    except RuntimeError:
        pass
    print()
    print("[specwright] No --project specified and PIPELINE_PROJECT not set.")
    name = input("  Enter project name: ").strip()
    if not name:
        print("[specwright] Project name cannot be empty.")
        sys.exit(1)
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="05_specwright — generate artifacts_<slug>/specwright_spec_<slug>.md from enriched prompt"
    )
    parser.add_argument("--project",   metavar="NAME",
                        help="Project workspace name. Prompted if omitted.")
    parser.add_argument("--model",     metavar="MODEL", default=_SPEC_MODEL_DEFAULT,
                        help=f"Model to use for spec generation. Default: {_SPEC_MODEL_DEFAULT}")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Generate spec, print to stdout, do not write file or launch harness.")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip interactive review — write spec and ask about harness directly.")
    parser.add_argument("--no-harness", action="store_true",
                        help="Write spec but do not offer to launch harness.")
    args = parser.parse_args()

    # ── Resolve project ───────────────────────────────────────────────────────
    project_name = _resolve_project(args.project)
    os.environ["PIPELINE_PROJECT"] = project_name
    ensure_dirs()

    # get_spec_path() resolves lazily to artifacts_<slug>/specwright_spec_<slug>.md —
    # the canonical location every downstream pipeline script reads.
    spec_file = get_spec_path()

    print(f"[specwright] Workspace:  {project_name!r}")
    print(f"[specwright] Spec target: {spec_file}")
    print(f"[specwright] Model:       {args.model}")

    # ── Load enriched prompt ──────────────────────────────────────────────────
    enriched_prompt = _load_enriched_prompt()
    if enriched_prompt.strip():
        print(f"[specwright] Loaded enricher_session_enriched_prompt.md ({len(enriched_prompt)} chars)")
    else:
        # Fallback: use clarified_req directly (less optimal but functional)
        print("[specwright][warn] enricher_session_enriched_prompt.md not found — falling back to clarificator_requirement_synthesis.md")
        enriched_prompt = _load_clarified_req()
        if not enriched_prompt.strip():
            print(
                "[specwright][error] Neither enricher_session_enriched_prompt.md nor clarificator_requirement_synthesis.md found.\n"
                "            Run 03_clarificator.py → 04_enricher.py first."
            )
            sys.exit(1)
        print(f"[specwright] Loaded clarificator_requirement_synthesis.md as fallback ({len(enriched_prompt)} chars)")

    # ── Check if spec already exists — warn user ──────────────────────────────
    if spec_file.exists():
        existing_lines = spec_file.read_text(encoding="utf-8").splitlines()
        print(f"\n[specwright][warn] {spec_file.name} already exists ({len(existing_lines)} lines).")
        overwrite = input("  Overwrite? [y/N]: ").strip().lower()
        if overwrite not in ("y", "yes"):
            print("[specwright] Aborted — existing spec preserved.")
            sys.exit(0)

    # ── Generate spec ─────────────────────────────────────────────────────────
    _print_banner(f"Generating spec — {project_name}")
    print("[specwright] Calling spec model (this may take 30–60s for complex specs) ...")

    try:
        raw_spec = _generate_spec(enriched_prompt, model=args.model)
    except Exception as exc:
        print(f"[specwright][error] Spec generation failed: {exc}")
        sys.exit(1)

    # Strip markdown fences if model wrapped output
    spec = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", raw_spec.strip(), flags=re.MULTILINE)

    # ── Validate structure ────────────────────────────────────────────────────
    missing = _validate_spec(spec)
    if missing:
        print(f"\n[specwright][warn] {len(missing)} expected section(s) not found in generated spec:")
        for s in missing:
            print(f"  ✗ {s}")
        print("  The spec may be incomplete. Review carefully before running harness.")
    else:
        print("[specwright] ✓ All required sections present.")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        _print_banner("Dry run — spec (not written)")
        print(spec)
        return

    # ── Write spec file ───────────────────────────────────────────────────────
    header = (
        f"<!-- specwright_spec_{os.environ.get('PIPELINE_PROJECT', 'unknown')} — generated by 05_specwright on {_now_iso()} -->\n"
        f"<!-- project: {project_name} | model: {args.model} -->\n\n"
    )
    final_spec = header + spec.strip() + "\n"
    spec_file.write_text(final_spec, encoding="utf-8")
    print(f"[specwright] ✓ Spec written → {spec_file}")

    # ── Review ────────────────────────────────────────────────────────────────
    if args.no_review or args.no_harness:
        # Still show summary but skip full review flow
        lines = spec.strip().splitlines()
        _print_banner(f"Spec ready — {len(lines)} lines")
        print(f"  File:  {spec_file}")
        if not args.no_harness:
            launch_choice = input("\n  Launch harness now? [y/N]: ").strip().lower()
            if launch_choice in ("y", "yes"):
                _launch_harness(project_name)
            else:
                _print_harness_instructions(project_name, spec_file)
        else:
            _print_harness_instructions(project_name, spec_file)
        return

    # Full interactive review
    final_spec_content, run_harness = _review_spec(spec, spec_file)

    # If user edited in review, overwrite file with updated content
    if final_spec_content != spec:
        updated = header + final_spec_content.strip() + "\n"
        spec_file.write_text(updated, encoding="utf-8")
        print(f"[specwright] ✓ Spec updated → {spec_file}")

    if run_harness:
        _launch_harness(project_name)
    else:
        _print_harness_instructions(project_name, spec_file)


def _print_harness_instructions(project_name: str, spec_file: Path) -> None:
    _print_banner("Spec saved — harness not activated")
    print(f"  Spec:   {spec_file}")
    print(f"\n  When ready to build:")
    print(f"    PIPELINE_PROJECT={project_name!r} python harness.py")
    print(f"\n  Or with flags:")
    print(f"    PIPELINE_PROJECT={project_name!r} python harness.py --dry-run")
    print(f"    PIPELINE_PROJECT={project_name!r} python harness.py --skip-judge\n")


if __name__ == "__main__":
    main()
