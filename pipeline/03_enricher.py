"""
03_enricher.py
==============
Prompt Agent — nhận clarified artifacts + knowledge layer + raw input của user,
enrich thành một structured prompt đủ context cho model xịn downstream (spec agent).

Vị trí trong luồng:
    02_clarificator → [03_enricher] → 04_specwright → 05_spectracker → specwright_spec_<slug>.md → harness

Inputs (đọc từ artifacts của project hiện tại):
    state/clarificator_requirement_synthesis.md        — output chính của clarificator
    execution/clarificator_overwrite_raw.json          — decisions, conflicts, metadata
    knowledge/current/archivist_knowledge_log.md       — knowledge base (nếu có)
    knowledge/current/absorber_codebase_map.md         — absorber output (nếu có)
    knowledge/current/absorber_config_map.json         — absorber output (nếu có)
    knowledge/current/absorber_blame_map.md            — absorber output (nếu có)

Output (ghi vào artifacts của project):
    execution/enricher_overwrite_enriched_prompt.md    — enriched prompt, user review trước khi gửi spec agent

Usage:
    python 03_enricher.py --project my-app
    python 03_enricher.py --project my-app --extra-context "Focus on backend only"
    python 03_enricher.py --project my-app --dry-run

    # Thường được gọi tự động từ 02_clarificator.py khi user chọn mode full.

Artifacts produced (owner: enricher):
    artifacts_<slug>/execution/enricher_overwrite_enriched_prompt.md

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

import httpx

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # type: ignore
    ABSORBER_BLAME_MAP,
    ABSORBER_CODEBASE_MAP,
    ABSORBER_CONFIG_MAP,
    ARCHIVIST_KNOWLEDGE_LOG,
    CLARIFICATOR_OVERWRITE_RAW,
    CLARIFIED_REQ,
    ENRICHER_OVERWRITE_PROMPT,
    ensure_dirs,
)

# Local aliases — map canonical constants to the short names used internally
CLARIFICATION_REPORT = CLARIFICATOR_OVERWRITE_RAW
KNOWLEDGE_BASE       = ARCHIVIST_KNOWLEDGE_LOG
CODEBASE_MAP         = ABSORBER_CODEBASE_MAP
CONFIG_MAP           = ABSORBER_CONFIG_MAP
BLAME_MAP            = ABSORBER_BLAME_MAP
ENRICHED_PROMPT      = ENRICHER_OVERWRITE_PROMPT

# NOTE: run/mini_analysis.md (MINI_ANALYSIS) has been removed — deprecated with mini_mode.py.
# enricher no longer reads it.

# === WRITE AUTHORITY: enricher ===
# OWNS  : artifacts_<slug>/execution/enricher_overwrite_enriched_prompt.md
# READS : artifacts_<slug>/state/clarificator_requirement_synthesis.md
#         artifacts_<slug>/execution/clarificator_overwrite_raw.json
#         artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
#         artifacts_<slug>/knowledge/current/absorber_codebase_map.md
#         artifacts_<slug>/knowledge/current/absorber_config_map.json
#         artifacts_<slug>/knowledge/current/absorber_blame_map.md


# ─────────────────────────────────────────────────────────────────────────────
# Artifact access tracking
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_READ: set[str] = set()
_ARTIFACTS_WRITTEN: set[str] = set()


def _track_read(path: Any) -> None:
    _ARTIFACTS_READ.add(str(path))


def _track_write(path: Any) -> None:
    _ARTIFACTS_WRITTEN.add(str(path))


def _print_artifact_access_summary() -> None:
    print("[03] Artifacts read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[03]   READ  {item}")
    else:
        print("[03]   READ  (none)")

    print("[03] Artifacts created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[03]   WRITE {item}")
    else:
        print("[03]   WRITE (none)")


# ── Model config ──────────────────────────────────────────────────────────────
# Prompt agent chạy model nhanh/rẻ — nhiệm vụ là enrich context, không cần Opus.
_ENRICH_MODEL      = "deepseek/deepseek-v4-pro"
_MAX_TOKENS_ENRICH = 4096


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
# LLM call — same thin wrapper pattern as clarificator
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    system: str,
    user: str,
    model: str = _ENRICH_MODEL,
    max_tokens: int = _MAX_TOKENS_ENRICH,
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
        print("\n[enricher][offline] No API key found. Paste LLM response then EOF (Ctrl-D):")
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
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[enricher][error] LLM call failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Context loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_clarified_req() -> str:
    if CLARIFIED_REQ.exists():
        _track_read(CLARIFIED_REQ)
        return CLARIFIED_REQ.read_text(encoding="utf-8")
    return ""


def _load_clarification_report() -> dict:
    if CLARIFICATION_REPORT.exists():
        _track_read(CLARIFICATION_REPORT)
        try:
            return json.loads(CLARIFICATION_REPORT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _load_knowledge_layer() -> str:
    """Load all available knowledge layer artifacts for the current project."""
    parts: list[str] = []

    if CODEBASE_MAP.exists():
        _track_read(CODEBASE_MAP)
        parts.append(f"=== absorber_codebase_map.md ===\n{CODEBASE_MAP.read_text(encoding='utf-8')}")

    if CONFIG_MAP.exists():
        _track_read(CONFIG_MAP)
        parts.append(f"=== absorber_config_map.json ===\n{CONFIG_MAP.read_text(encoding='utf-8')}")

    if BLAME_MAP.exists():
        _track_read(BLAME_MAP)
        parts.append(f"=== absorber_blame_map.md ===\n{BLAME_MAP.read_text(encoding='utf-8')}")

    if KNOWLEDGE_BASE.exists():
        _track_read(KNOWLEDGE_BASE)
        parts.append(f"=== archivist_knowledge_log.md ===\n{KNOWLEDGE_BASE.read_text(encoding='utf-8')}")

    return "\n\n".join(parts)


def _summarize_decisions(report: dict) -> str:
    """Format decisions from clarification report into a compact summary block."""
    decisions = report.get("decisions", [])
    if not decisions:
        return "(no clarification decisions recorded)"

    lines: list[str] = []
    for d in decisions:
        pri  = d.get("priority", "").upper()
        tier = d.get("tier", "?")
        lines.append(f"  [{d['id']}] T{tier} {pri}: {d['question']}")
        lines.append(f"    → {d['answer']}")
        if d.get("impact"):
            lines.append(f"    impact: {d['impact']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt enrichment
# ─────────────────────────────────────────────────────────────────────────────

_ENRICH_SYSTEM = """
You are a senior software architect and prompt engineer.

Your task: given a clarified requirement document, its clarification decisions,
an optional mini analysis, and an optional knowledge layer of the existing codebase,
produce ONE structured "Enriched Prompt" document that a spec-writing model can use
directly to write a complete, unambiguous technical specification.

The Enriched Prompt is not the spec itself — it is the fully-loaded input prompt
that will be sent to a spec model. Think of it as: "everything the spec model
needs to know, organized for maximum clarity."

OUTPUT FORMAT — output the enriched prompt as a markdown document with these
sections in order. Do not add extra sections or change the order.

---

# Enriched Prompt — <project name>

## Role
One sentence: what role should the spec model adopt?
Example: "You are a senior backend engineer writing a technical spec for a
production-grade <domain> system."

## Task
One clear paragraph: what the spec model must produce.
Be explicit about output format, length expectation, and audience
(e.g. "a markdown spec targeting mid-level engineers building this system").

## Context: What We're Building
3–6 bullet points summarizing the core system being built.
Pull from the clarified requirement. Be concrete — tech stack, user roles,
scale, integrations if known.

## Context: Existing Codebase
If knowledge layer is available: summarize relevant existing structure —
key files, modules, patterns, constraints the spec must respect.
If not available: write "No existing codebase — greenfield project."

## Context: Clarification Decisions
List every clarification decision that shapes the spec.
Format: "- **[CLR-XXX]** <decision summary in one line>"
These are non-negotiable — the spec must incorporate all of them.

## Context: Warnings & Risks
Pull from mini_analysis warnings (if available) + any conflicts from the
clarification report. If none: "(none identified)"

## Constraints
Technical, business, and process constraints the spec must respect.
Examples: auth method, deployment target, SLA, compliance requirements,
existing API contracts that cannot break.

## Clarified Requirement (full text)
Paste the full clarified_requirement.md content here verbatim.
Do not summarize — the spec model needs the full text.

## Instructions for the Spec Model
Numbered list of explicit instructions for how to write the spec.
Always include:
  1. Write in markdown. Use H2 for major sections.
  2. Be implementation-ready: every section should be specific enough for a
     developer to start coding without asking follow-up questions.
  3. Include: Overview, Goals, Non-Goals, Architecture, Data Models,
     API Contracts (if applicable), Workflow & State Machine (if applicable),
     Error Handling, NFRs, Out of Scope, Acceptance Criteria.
  4. Incorporate every decision in the Clarification Decisions section above.
  5. Flag any remaining ambiguities as <!-- TODO: clarify --> inline comments.

---

RULES:
- Do not write the spec itself. Only write the enriched prompt.
- Do not truncate the clarified requirement — paste it in full.
- If mini_analysis is available, mine it for warnings and optimization hints.
- If knowledge layer is available, extract constraints from config_map.json
  and file-level context from codebase_map.md.
- Be precise. Vague instructions produce vague specs.
- Output only the markdown document. No preamble, no postamble.
""".strip()


def _enrich(
    project_name: str,
    clarified_req: str,
    report: dict,
    knowledge_layer: str,
    extra_context: str,
) -> str:
    decisions_block = _summarize_decisions(report)

    conflicts = report.get("conflicts", [])
    conflicts_block = (
        "\n".join(f"  [{c['id']}] {c['description']}" for c in conflicts)
        if conflicts else "(none detected)"
    )

    user_msg = f"""PROJECT NAME: {project_name}

CLARIFIED REQUIREMENT:
{clarified_req}

CLARIFICATION DECISIONS:
{decisions_block}

CONFLICTS:
{conflicts_block}

KNOWLEDGE LAYER (existing codebase artifacts, if available):
{knowledge_layer if knowledge_layer.strip() else "(not available — absorber has not run for this project)"}

EXTRA CONTEXT FROM USER:
{extra_context.strip() if extra_context.strip() else "(none)"}

Produce the enriched prompt document now."""

    return _call_llm(_ENRICH_SYSTEM, user_msg)


# ─────────────────────────────────────────────────────────────────────────────
# User review: show enriched prompt, ask to confirm / edit / abort
# ─────────────────────────────────────────────────────────────────────────────

def _review_prompt(enriched: str) -> tuple[str, bool]:
    """
    Show enriched prompt to user, ask: confirm / edit / abort.
    Returns (final_prompt_text, should_continue).
    """
    _print_banner("Enriched prompt — review before sending to spec agent")
    print("\n" + "─" * 72)
    print(enriched.strip())
    print("─" * 72)

    print("\n  [1] confirm — send this prompt to spec agent")
    print("  [2] edit    — open $EDITOR to modify (writes to temp file)")
    print("  [3] abort   — stop here, enricher_overwrite_enriched_prompt.md saved for manual review\n")

    while True:
        choice = input("  → Choose 1 / 2 / 3: ").strip()
        if choice in ("1", "confirm"):
            return enriched, True
        if choice in ("2", "edit"):
            edited = _open_in_editor(enriched)
            if edited and edited.strip():
                print("\n[enricher] Updated prompt loaded.")
                return edited, True
            print("[enricher] Editor returned empty content — keeping original.")
            return enriched, True
        if choice in ("3", "abort"):
            return enriched, False
        print("  Please enter 1, 2, or 3.")


def _open_in_editor(content: str) -> str:
    """Write content to a temp file, open $EDITOR, return modified content."""
    import tempfile
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="enriched_prompt_",
        delete=False, encoding="utf-8"
    ) as tf:
        tf.write(content)
        tmp_path = tf.name

    try:
        subprocess.run([editor, tmp_path], check=True)
        return Path(tmp_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[enricher][warn] Editor '{editor}' not found. Set $EDITOR env var.")
        return content
    except subprocess.CalledProcessError as exc:
        print(f"[enricher][warn] Editor exited with error: {exc}")
        return content
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Launch spec agent
# ─────────────────────────────────────────────────────────────────────────────

def _launch_spec_agent(project_name: str) -> None:
    script = Path(__file__).parent / "05_specwright.py"
    if not script.exists():
        print(f"\n[enricher][warn] 05_specwright.py not found at {script}")
        print(f"[enricher]       Create it first, then run:")
        print(f"           python 05_specwright.py --project {project_name!r}")
        return

    print(f"\n[enricher] Launching specwright → {script.name}")
    try:
        subprocess.run(
            [sys.executable, str(script), "--project", project_name],
            check=False,
        )
    except KeyboardInterrupt:
        print("\n[enricher] Specwright interrupted.")


# ─────────────────────────────────────────────────────────────────────────────
# Resolve project (same pattern as clarificator)
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
    print("[enricher] No --project specified and PIPELINE_PROJECT not set.")
    name = input("  Enter project name: ").strip()
    if not name:
        print("[enricher] Project name cannot be empty.")
        sys.exit(1)
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="03_enricher — enrich clarified requirement into a structured spec prompt"
        )
        parser.add_argument("--project",       metavar="NAME",
                            help="Project workspace name. Prompted if omitted.")
        parser.add_argument("--extra-context", metavar="TEXT", default="",
                            help="Additional free-text context to include in the enriched prompt.")
        parser.add_argument("--dry-run",       action="store_true",
                            help="Run enrichment, print result, do not write files or launch spec agent.")
        parser.add_argument("--no-review",     action="store_true",
                            help="Skip interactive review — write and forward enriched prompt automatically.")
        args = parser.parse_args()

        # ── Resolve project ───────────────────────────────────────────────────────
        project_name = _resolve_project(args.project)
        os.environ["PIPELINE_PROJECT"] = project_name
        ensure_dirs()
        print(f"[enricher] Workspace: {project_name!r}")

        # ── Load inputs ───────────────────────────────────────────────────────────
        clarified_req = _load_clarified_req()
        if not clarified_req.strip():
            print(
                "[enricher][error] clarificator_requirement_synthesis.md not found or empty.\n"
                "           Run 02_clarificator.py first."
            )
            sys.exit(1)
        print(f"[enricher] Loaded clarificator_requirement_synthesis.md ({len(clarified_req)} chars)")

        report = _load_clarification_report()
        n_decisions = len(report.get("decisions", []))
        print(f"[enricher] Loaded clarificator_overwrite_raw.json ({n_decisions} decisions)")

        knowledge_layer = _load_knowledge_layer()
        if knowledge_layer.strip():
            parts = []
            if CODEBASE_MAP.exists():   parts.append("absorber_codebase_map")
            if CONFIG_MAP.exists():     parts.append("absorber_config_map")
            if BLAME_MAP.exists():      parts.append("absorber_blame_map")
            if KNOWLEDGE_BASE.exists(): parts.append("archivist_knowledge_log")
            print(f"[enricher] Knowledge layer loaded: {', '.join(parts)}")
        else:
            print("[enricher] No knowledge layer found — proceeding in greenfield mode")

        # ── Enrich ────────────────────────────────────────────────────────────────
        _print_banner(f"Enriching prompt — {project_name}")
        print("[enricher] Calling LLM to build enriched prompt ...")

        try:
            enriched = _enrich(
                project_name   = project_name,
                clarified_req  = clarified_req,
                report         = report,
                knowledge_layer= knowledge_layer,
                extra_context  = args.extra_context,
            )
        except Exception as exc:
            print(f"[enricher][error] Enrichment failed: {exc}")
            sys.exit(1)

        # ── Dry run: print and exit ───────────────────────────────────────────────
        if args.dry_run:
            _print_banner("Dry run — enriched prompt (not written)")
            print(enriched.strip())
            return

        # ── Write enricher_overwrite_enriched_prompt.md ──────────────────────────
        header = (
            f"# Enriched Prompt — {project_name}\n"
            f"Generated: {_now_iso()}\n\n"
            f"---\n\n"
        )
        final_content = header + enriched.strip() + "\n"
        ENRICHED_PROMPT.write_text(final_content, encoding="utf-8")
        _track_write(ENRICHED_PROMPT)
        print(f"[enricher] ✓ Enriched prompt → {ENRICHED_PROMPT}")

        # ── Review ────────────────────────────────────────────────────────────────
        if args.no_review:
            should_continue = True
            final_prompt = enriched
        else:
            final_prompt, should_continue = _review_prompt(enriched)
            # If user edited, overwrite the written file with updated content
            if final_prompt != enriched:
                updated = header + final_prompt.strip() + "\n"
                ENRICHED_PROMPT.write_text(updated, encoding="utf-8")
                _track_write(ENRICHED_PROMPT)
                print(f"[enricher] ✓ Enriched prompt updated → {ENRICHED_PROMPT}")

        if not should_continue:
            _print_banner("Stopped — enricher_overwrite_enriched_prompt.md saved for manual review")
            print(f"  Review:   {ENRICHED_PROMPT}")
            print(f"  Continue: python 05_specwright.py --project {project_name!r}\n")
            return

        # ── Launch spec agent ─────────────────────────────────────────────────────
        _launch_spec_agent(project_name)

    finally:
        _print_artifact_access_summary()


if __name__ == "__main__":
    main()
