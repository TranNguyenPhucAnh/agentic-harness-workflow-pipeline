"""
04_specwright.py
================
Spec Agent — nhận enriched prompt từ 03_enricher, gọi model, ghi spec markdown.

Vị trí trong luồng:
    03_enricher → [04_specwright] → 05_spectracker → spec/specwright_spec_<slug>.md

Inputs:
    enricher/enriched_prompt.md              — output của 03_enricher (bắt buộc)
    clarificator/requirement_synthesis.md    — fallback nếu enriched_prompt.md không có
    absorber/codebase_map.md                 — codebase context (optional)
    archivist/knowledge_log.md               — accumulated knowledge (optional)

Output:
    spec/specwright_spec_<slug>.md           — technical spec tại get_spec_path()

Usage:
    python 04_specwright.py --project my-app
    python 04_specwright.py --project my-app --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # type: ignore
    ABSORBER_CODEBASE_MD,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    CLARIFICATOR_REQUIREMENT_SYNTHESIS,
    ENRICHER_OVERWRITE_PROMPT,
    ensure_dirs,
    get_spec_path,
)
from artifacts.models import get_model  # type: ignore
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_summary  # noqa: E402
from modules.call_llm import call_llm
from modules.post_interactive import prompt_next_step  # noqa: E402

# === WRITE AUTHORITY: specwright ===
# OWNS  : spec/specwright_spec_<slug>.md                (dynamic path via get_spec_path())
# READS : enricher/enriched_prompt.md                   (primary input)
#         clarificator/requirement_synthesis.md         (fallback input)
#         absorber/codebase_map.md                      (codebase context, optional)
#         archivist/knowledge_log.md                    (accumulated knowledge, optional)
#         archivist/spec_gaps.md                      (spec gap awareness, optional)
#         spec/specwright_spec_<slug>.md                (self-read: exists check + version)


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


# ─────────────────────────────────────────────────────────────────────────────
# Version management
# ─────────────────────────────────────────────────────────────────────────────

_VERSION_RE = re.compile(r"^#\s*Version:\s*(\S+)", re.MULTILINE)


def _parse_version(text: str) -> tuple[int, int, int] | None:
    m = _VERSION_RE.search(text)
    if not m:
        return None
    parts = m.group(1).split(".")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return None


def _read_current_version(spec_file: Path) -> tuple[int, int, int] | None:
    if not spec_file.exists():
        return None
    try:
        return _parse_version(spec_file.read_text(encoding="utf-8"))
    except OSError:
        return None


def _next_version_regenerate(current: tuple[int, int, int] | None) -> str:
    if current is None:
        return "1.0.0"
    major, _, _ = current
    return f"{major + 1}.0.0"


def _format_version_line(version: str) -> str:
    return f"# Version: {version}"


# ─────────────────────────────────────────────────────────────────────────────
# Context loaders — all read .md files directly
# ─────────────────────────────────────────────────────────────────────────────

def _load_enriched_prompt() -> str:
    """Load enricher/enriched_prompt.md — primary input."""
    if not ENRICHER_OVERWRITE_PROMPT.exists():
        return ""
    try:
        track_read(ENRICHER_OVERWRITE_PROMPT)
        return ENRICHER_OVERWRITE_PROMPT.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_requirement_synthesis() -> str:
    """Load clarificator/requirement_synthesis.md — fallback input."""
    if not CLARIFICATOR_REQUIREMENT_SYNTHESIS.exists():
        return ""
    try:
        track_read(CLARIFICATOR_REQUIREMENT_SYNTHESIS)
        return CLARIFICATOR_REQUIREMENT_SYNTHESIS.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _load_codebase_map() -> str:
    """Load absorber/codebase_map.md — codebase context, optional."""
    if not ABSORBER_CODEBASE_MD.exists():
        return ""
    try:
        track_read(ABSORBER_CODEBASE_MD)
        return ABSORBER_CODEBASE_MD.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[specwright][warn] Could not read codebase_map.md: {exc}")
        return ""


def _load_knowledge_log() -> str:
    """Load archivist/knowledge_log.md — accumulated knowledge, optional."""
    if not ARCHIVIST_KNOWLEDGE_LOG.exists():
        return ""
    try:
        track_read(ARCHIVIST_KNOWLEDGE_LOG)
        return ARCHIVIST_KNOWLEDGE_LOG.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[specwright][warn] Could not read knowledge_log.md: {exc}")
        return ""


def _load_spec_gaps() -> str:
    """Load archivist/spec_gaps.md — edge cases surfaced by previous runs, optional."""
    if not ARCHIVIST_SPEC_GAPS.exists():
        return ""
    try:
        track_read(ARCHIVIST_SPEC_GAPS)
        return ARCHIVIST_SPEC_GAPS.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[specwright][warn] Could not read spec_gaps.md: {exc}")
        return ""


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


def _generate_spec(
    enriched_prompt: str,
    codebase_map: str = "",
    knowledge_log: str = "",
    spec_gaps: str = "",
) -> str:
    """Call specwright model, return raw spec markdown."""
    parts = [
        "Use the enriched prompt below to write the technical specification.",
        "Follow all instructions in the prompt exactly.",
        "",
    ]

    if codebase_map:
        parts += [
            "--- EXISTING CODEBASE CONTEXT ---",
            "The following is a map of the existing codebase.",
            "Use it to ensure the spec integrates correctly with existing architecture,",
            "file structure, patterns, and conventions.",
            "",
            codebase_map,
            "--- END CODEBASE CONTEXT ---",
            "",
        ]

    if knowledge_log:
        parts += [
            "--- ACCUMULATED KNOWLEDGE ---",
            "The following is accumulated knowledge from previous pipeline runs.",
            "Use it to inform decisions, avoid known pitfalls, and build on prior context.",
            "",
            knowledge_log,
            "--- END ACCUMULATED KNOWLEDGE ---",
            "",
        ]

    if spec_gaps:
        parts += [
            "--- KNOWN SPEC GAPS ---",
            "The following edge cases and spec gaps were surfaced by previous pipeline runs.",
            "Address them explicitly in the relevant sections of the new spec.",
            "",
            spec_gaps,
            "--- END KNOWN SPEC GAPS ---",
            "",
        ]

    parts += [
        "--- ENRICHED PROMPT ---",
        enriched_prompt,
        "--- END ENRICHED PROMPT ---",
        "",
        "Write the specification now.",
    ]

    user_msg = "\n".join(parts)
    print(f"[specwright] Model            : {get_model(ROLE)}")
    if codebase_map:
        print(f"[specwright] Codebase context : {len(codebase_map):,} chars")
    if knowledge_log:
        print(f"[specwright] Knowledge log    : {len(knowledge_log):,} chars")
    if spec_gaps:
        print(f"[specwright] Spec gaps        : {len(spec_gaps):,} chars")
    print(f"[specwright] Enriched prompt  : {len(enriched_prompt):,} chars")

    result, _ = call_llm(
        ROLE,
        _SPEC_SYSTEM,
        user_msg,
        max_tokens=_MAX_TOKENS_SPEC,
        caller_file=__file__,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Spec validation
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
    return [
        s for s in _REQUIRED_SECTIONS
        if not re.compile(re.escape(s), re.IGNORECASE).search(spec)
    ]


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
    try:
        parser = argparse.ArgumentParser(
            description="04_specwright — generate spec/specwright_spec_<slug>.md"
        )
        parser.add_argument("--project", metavar="NAME",
                            help="Project workspace name. Prompted if omitted.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Generate spec, print to stdout, do not write file.")
        args = parser.parse_args()

        # ── Resolve project ───────────────────────────────────────────────────
        project_name = _resolve_project(args.project)
        os.environ["PIPELINE_PROJECT"] = project_name
        ensure_dirs()

        spec_file = get_spec_path()
        print(f"[specwright] Project:     {project_name!r}")
        print(f"[specwright] Spec target: {spec_file}")

        # ── Load primary input ────────────────────────────────────────────────
        enriched_prompt = _load_enriched_prompt()
        if enriched_prompt:
            print(f"[specwright] Loaded enriched_prompt.md ({len(enriched_prompt):,} chars)")
        else:
            print("[specwright][warn] enriched_prompt.md not found — trying requirement_synthesis.md")
            enriched_prompt = _load_requirement_synthesis()
            if not enriched_prompt:
                print(
                    "[specwright][error] Neither enriched_prompt.md nor "
                    "requirement_synthesis.md found.\n"
                    "           Run 02_clarificator.py → 03_enricher.py first."
                )
                sys.exit(1)
            print(f"[specwright] Loaded requirement_synthesis.md ({len(enriched_prompt):,} chars)")

        # ── Load optional context ─────────────────────────────────────────────
        codebase_map = _load_codebase_map()
        if codebase_map:
            print(f"[specwright] Loaded codebase_map.md ({len(codebase_map):,} chars)")
        else:
            print("[specwright] No codebase_map.md — greenfield mode")

        knowledge_log = _load_knowledge_log()
        if knowledge_log:
            print(f"[specwright] Loaded knowledge_log.md ({len(knowledge_log):,} chars)")

        spec_gaps = _load_spec_gaps()
        if spec_gaps:
            print(f"[specwright] Loaded spec_gaps.md ({len(spec_gaps):,} chars)")

        # ── Check existing spec ───────────────────────────────────────────────
        existing_version = _read_current_version(spec_file)
        if spec_file.exists():
            track_read(spec_file)
            existing_lines = spec_file.read_text(encoding="utf-8").splitlines()
            ver_display = (
                f"v{'.'.join(str(x) for x in existing_version)}"
                if existing_version else "no version"
            )
            print(f"\n[specwright][warn] {spec_file.name} already exists "
                  f"({len(existing_lines)} lines, {ver_display}).")
            if input("  Overwrite? [y/N]: ").strip().lower() not in ("y", "yes"):
                print("[specwright] Aborted — existing spec preserved.")
                sys.exit(0)

        # ── Generate ──────────────────────────────────────────────────────────
        _print_banner(f"Generating spec — {project_name}")
        print("[specwright] Calling spec model ...")

        try:
            raw_spec = _generate_spec(enriched_prompt, codebase_map, knowledge_log, spec_gaps)
        except Exception as exc:
            print(f"[specwright][error] Spec generation failed: {exc}")
            sys.exit(1)

        spec = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", raw_spec.strip(), flags=re.MULTILINE)

        # ── Validate ──────────────────────────────────────────────────────────
        missing = _validate_spec(spec)
        if missing:
            print(f"\n[specwright][warn] {len(missing)} section(s) missing:")
            for s in missing:
                print(f"  ✗ {s}")
        else:
            print("[specwright] ✓ All required sections present.")

        # ── Dry run ───────────────────────────────────────────────────────────
        if args.dry_run:
            _print_banner("Dry run — spec not written")
            print(spec)
            return

        # ── Write ─────────────────────────────────────────────────────────────
        new_version = _next_version_regenerate(existing_version)
        ver_before = ("none" if not existing_version
                      else "v" + ".".join(str(x) for x in existing_version))
        print(f"[specwright] Version: {ver_before} → v{new_version}")

        header = (
            f"{_format_version_line(new_version)}\n"
            f"<!-- specwright_spec_{project_name} — generated by 04_specwright on {_now_iso()} -->\n"
            f"<!-- project: {project_name} | model: {get_model(ROLE)} -->\n\n"
        )
        final_spec = header + spec.strip() + "\n"
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(final_spec, encoding="utf-8")
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