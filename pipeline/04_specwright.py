"""
04_specwright.py
================
Spec Agent — nhận enriched prompt từ 03_enricher, gửi cho model xịn,
nhận về spec markdown, cho user review/edit, rồi hỏi có muốn kích hoạt
full harness không.

Vị trí trong luồng:
    03_enricher → [04_specwright] → 05_spectracker → spec/specwright_spec_<slug>.md → (optional) harness

Inputs:
    enricher/enriched_prompt.md       — output của 03_enricher (bắt buộc)
    clarificator/session.json         — fallback: field requirement_synthesis
                                        + project metadata

Output:
    spec/specwright_spec_<slug>.md    — technical spec tại get_spec_path()
                                        (đây là input canonical cho toàn bộ harness:
                                         spectracker, scaffolder, planner, executor, judge, v.v.)

Usage:
    python 04_specwright.py --project my-app
    python 04_specwright.py --project my-app --dry-run
    python 04_specwright.py --project my-app --no-review

    # Model được resolve từ artifacts/models.py role "specwright".
    # Để đổi model: sửa ROLES["specwright"] trong models.py.

Artifacts produced (owner: specwright):
    spec/specwright_spec_<slug>.md    — get_spec_path(), input cho harness.py

At the end of each run, prints:
    - artifacts read
    - artifacts created/updated/overwritten/appended
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
from typing import Any

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # type: ignore
    CLARIFICATOR_SESSION,
    ENRICHER_OVERWRITE_PROMPT,
    ensure_dirs,
    get_spec_path,
)
from artifacts.models import call_model, get_model, get_provider  # type: ignore
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.call_llm import call_llm
from modules.post_interactive import prompt_next_step  # noqa: E402

# === WRITE AUTHORITY: specwright ===
# OWNS  : spec/specwright_spec_<slug>.md   (dynamic path via get_spec_path())
# READS : enricher/enriched_prompt.md (upstream-aware, enricher)
#         clarificator/session.json        (fallback + metadata)
#         spec/specwright_spec_<slug>.md   (self-read: exists check + overwrite guard)


# ── Model config ──────────────────────────────────────────────────────────────
# Model identity resolved from artifacts/models.py role "specwright".
# Để đổi model: sửa ROLES["specwright"] trong models.py — không sửa file này.
ROLE             = "specwright"
_MAX_TOKENS_SPEC = 32768


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
# Version management
# ─────────────────────────────────────────────────────────────────────────────
#
# Version scheme:  MAJOR.MINOR.PATCH
#   MAJOR — incremented when spec is regenerated from scratch (full LLM call,
#            user confirmed overwrite of an existing spec)
#   MINOR — incremented when user edits the spec in $EDITOR during review
#           (content changed post-generation, not a full regeneration)
#   PATCH — reserved for patcher (12_patcher.py); specwright always writes 0
#
# spectracker reads version via:
#   re.search(r"^# Version:\s*(\S+)", text, re.MULTILINE)
# The version line MUST be the first line of the written file so the
# `^` anchor in MULTILINE mode matches it reliably.
#
# First-run (no existing spec): starts at 1.0.0
# Subsequent full regeneration: MAJOR += 1, MINOR = 0, PATCH = 0
# Post-generation editor edit:  MINOR += 1, PATCH = 0

_VERSION_RE = re.compile(r"^#\s*Version:\s*(\S+)", re.MULTILINE)


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """
    Extract (major, minor, patch) from spec text.
    Returns None if no valid version line found.
    """
    m = _VERSION_RE.search(text)
    if not m:
        return None
    parts = m.group(1).split(".")
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        return major, minor, patch
    except (IndexError, ValueError):
        return None


def _read_current_version(spec_file: Path) -> tuple[int, int, int] | None:
    """
    Read the version embedded in the existing spec file.
    Returns None if file doesn't exist or has no valid version.
    """
    if not spec_file.exists():
        return None
    try:
        text = spec_file.read_text(encoding="utf-8")
        return _parse_version(text)
    except OSError:
        return None


def _next_version_regenerate(current: tuple[int, int, int] | None) -> str:
    """
    Compute next version for a full regeneration (MAJOR bump).
    First run → 1.0.0
    Existing  → (MAJOR+1).0.0
    """
    if current is None:
        return "1.0.0"
    major, _, _ = current
    return f"{major + 1}.0.0"


def _format_version_line(version: str) -> str:
    """Return the canonical version header line spectracker expects."""
    return f"# Version: {version}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

# _call_llm removed — use call_llm() from modules.call_llm



# ─────────────────────────────────────────────────────────────────────────────
# Context loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_enriched_prompt() -> str:
    """Load enricher/enriched_prompt.md — primary input for spec generation."""
    if ENRICHER_OVERWRITE_PROMPT.exists():
        track_read(ENRICHER_OVERWRITE_PROMPT)
        return ENRICHER_OVERWRITE_PROMPT.read_text(encoding="utf-8")
    return ""


def _load_clarificator_session() -> dict:
    """
    Load clarificator/session.json.
    Returns full session dict with fields: decisions, conflicts, unresolved,
    tier_counts, input_sources, req_hash, requirement_synthesis.
    """
    if CLARIFICATOR_SESSION.exists():
        track_read(CLARIFICATOR_SESSION)
        try:
            return json.loads(CLARIFICATOR_SESSION.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_requirement_synthesis(session: dict) -> str:
    """Extract requirement_synthesis text field from clarificator session."""
    return session.get("requirement_synthesis", "")


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


def _generate_spec(enriched_prompt: str) -> str:
    """Call specwright model with enriched prompt, return raw spec markdown."""
    user_msg = f"""Use the enriched prompt below to write the technical specification.
Follow all instructions in the prompt exactly.

{enriched_prompt}

Write the specification now."""

    print(f"[specwright] Using model: {get_model(ROLE)}")
    content, _ = call_llm(ROLE, _SPEC_SYSTEM, user_msg, caller_file=__file__)
    return content


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
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if not pattern.search(spec):
            missing.append(section)
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# User review
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
# Harness instructions
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="04_specwright — generate spec/specwright_spec_<slug>.md from enriched prompt"
        )
        parser.add_argument("--project",   metavar="NAME",
                            help="Project workspace name. Prompted if omitted.")
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

        # get_spec_path() resolves lazily to spec/specwright_spec_<slug>.md —
        # the canonical location every downstream pipeline script reads.
        spec_file = get_spec_path()

        print(f"[specwright] Workspace:   {project_name!r}")
        print(f"[specwright] Spec target: {spec_file}")
        print(f"[specwright] Model:       {get_model(ROLE)}")

        # ── Load enriched prompt ──────────────────────────────────────────────────
        enriched_prompt = _load_enriched_prompt()
        if enriched_prompt.strip():
            print(f"[specwright] Loaded enricher/enriched_prompt.md ({len(enriched_prompt)} chars)")
        else:
            # Fallback: extract requirement_synthesis from clarificator/session.json
            print("[specwright][warn] enricher/enriched_prompt.md not found — falling back to clarificator/session.json")
            session = _load_clarificator_session()
            enriched_prompt = _extract_requirement_synthesis(session)
            if not enriched_prompt.strip():
                print(
                    "[specwright][error] Neither enricher/enriched_prompt.md nor "
                    "clarificator/session.json[requirement_synthesis] found.\n"
                    "            Run 02_clarificator.py → 03_enricher.py first."
                )
                sys.exit(1)
            print(f"[specwright] Loaded requirement_synthesis from session.json as fallback ({len(enriched_prompt)} chars)")

        # Load clarificator session for metadata (project context, decisions count, etc.)
        # This is a non-critical read — used for logging/diagnostics only.
        clarificator_session = _load_clarificator_session()
        if clarificator_session:
            decisions_count = len(clarificator_session.get("decisions", []))
            print(f"[specwright] Loaded clarificator/session.json metadata ({decisions_count} decisions)")

        # ── Check if spec already exists — read version before overwrite ────────────
        existing_version = _read_current_version(spec_file)
        if spec_file.exists():
            track_read(spec_file)
            existing_lines = spec_file.read_text(encoding="utf-8").splitlines()
            ver_display = f"v{'.'.join(str(x) for x in existing_version)}" if existing_version else "no version"
            print(f"\n[specwright][warn] {spec_file.name} already exists ({len(existing_lines)} lines, {ver_display}).")
            overwrite = input("  Overwrite? [y/N]: ").strip().lower()
            if overwrite not in ("y", "yes"):
                print("[specwright] Aborted — existing spec preserved.")
                sys.exit(0)

        # ── Generate spec ─────────────────────────────────────────────────────────
        _print_banner(f"Generating spec — {project_name}")
        print("[specwright] Calling spec model (this may take 30–60s for complex specs) ...")

        try:
            raw_spec = _generate_spec(enriched_prompt)
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

        # ── Compute new version ───────────────────────────────────────────────────
        new_version = _next_version_regenerate(existing_version)
        print(f"[specwright] Version: {('none' if existing_version is None else 'v' + '.'.join(str(x) for x in existing_version))} → v{new_version}")

        # ── Write spec file ───────────────────────────────────────────────────────
        # Version line MUST be first — spectracker regex uses ^# Version: with MULTILINE.
        header = (
            f"{_format_version_line(new_version)}\n"
            f"<!-- specwright_spec_{os.environ.get('PIPELINE_PROJECT', 'unknown')} — generated by 04_specwright on {_now_iso()} -->\n"
            f"<!-- project: {project_name} | model: {get_model(ROLE)} -->\n\n"
        )
        final_spec = header + spec.strip() + "\n"
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(final_spec, encoding="utf-8")
        # NOTE: track_write is intentionally deferred to after review.
        # If the user edits the spec, only the final written version is tracked
        # (one track_write per file per run, not one per write() call).
        print(f"[specwright] ✓ Spec written → {spec_file}")

        # ── Done — track write, hand off to post_interactive ─────────────────────
        track_write(spec_file)
        lines = spec.strip().splitlines()
        _print_banner(f"Spec ready — {len(lines)} lines | v{new_version}")
        print(f"  File: {spec_file}")

    finally:
        print_summary("[04]")
        print_artifact_summary("[04]")
        prompt_next_step(ROLE, prefix="[04]")


if __name__ == "__main__":
    main()
