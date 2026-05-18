"""
03_enricher.py
==============
Prompt Agent — nhận clarified artifacts + knowledge layer + raw input của user,
enrich thành một structured prompt đủ context cho model xịn downstream (spec agent).

Vị trí trong luồng:
    02_clarificator → [03_enricher] → 04_specwright → 05_spectracker → specwright_spec_<slug>.md → harness

Inputs (đọc từ artifacts của project hiện tại):
    clarificator/session.json                — output chính của clarificator (synthesis embedded)
    absorber/codebase_map.md                 — knowledge base (nếu có)
    archivist/knowledge_log.md               — archivist knowledge (nếu có)

Output (ghi vào artifacts của project):
    enricher/enriched_prompt.md              — short-term, overwrite per run
    enricher/prompt_log.json                 — long-term, append-only audit

Usage:
    python 03_enricher.py --project my-app
    python 03_enricher.py --project my-app --extra-context "Focus on backend only"
    python 03_enricher.py --project my-app --dry-run

    # Thường được gọi tự động từ 02_clarificator.py khi user chọn mode full.

Artifacts produced (owner: enricher):
    artifacts_<slug>/enricher/enriched_prompt.md
    artifacts_<slug>/enricher/prompt_log.json

At the end of each run, prints:
    - artifacts read
    - artifacts created/updated/overwritten/appended
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # type: ignore
    ABSORBER_CODEBASE_MAP,
    ARCHIVIST_KNOWLEDGE_LOG,
    CLARIFICATOR_SESSION,
    ENRICHER_OVERWRITE_PROMPT,
    ENRICHER_PROMPT_LOG,
    ensure_dirs,
)
from artifacts.models import call_model, get_model, get_provider  # type: ignore
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402

# === WRITE AUTHORITY: enricher ===
# OWNS  : artifacts_<slug>/enricher/enriched_prompt.md
#          artifacts_<slug>/enricher/prompt_log.json
# READS : artifacts_<slug>/clarificator/session.json
#          artifacts_<slug>/absorber/codebase_map.md
#          artifacts_<slug>/archivist/knowledge_log.md


# ── Model config ──────────────────────────────────────────────────────────────
ROLE               = "enricher"
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
# LLM call — thin wrapper delegating to central model registry
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    system: str,
    user: str,
    max_tokens: int = _MAX_TOKENS_ENRICH,
) -> tuple[str, float]:
    """
    Call the enricher model via the central model registry.
    Model identity and provider are resolved from artifacts/models.py role "enricher".
    Falls back to stdin mock when API key is missing (offline dev).

    Returns:
      (content, call_cost) — call_cost is 0.0 if usage unavailable or offline fallback.
    """
    try:
        resp = call_model(
            ROLE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        call_cost = 0.0
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=get_model(ROLE), provider=get_provider(ROLE))
            print_call(__file__, pt, ct, call_cost)
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Model returned empty content.")
        return content, call_cost
    except RuntimeError as exc:
        if "not set" in str(exc):
            print("\n[enricher][offline] No API key found. Paste LLM response then EOF (Ctrl-D):")
            return sys.stdin.read(), 0.0
        raise
    except Exception as exc:
        print(f"[enricher][error] LLM call failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Context loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_session() -> dict:
    """Load clarificator/session.json — the single source for clarification output."""
    if CLARIFICATOR_SESSION.exists():
        track_read(CLARIFICATOR_SESSION)
        try:
            return json.loads(CLARIFICATOR_SESSION.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_requirement_synthesis(session: dict) -> str:
    """Extract the requirement_synthesis text field from session.json."""
    return session.get("requirement_synthesis", "")


def _load_knowledge_layer() -> str:
    """Load all available knowledge layer artifacts for the current project."""
    parts: list[str] = []

    if ABSORBER_CODEBASE_MAP.exists():
        track_read(ABSORBER_CODEBASE_MAP)
        parts.append(f"=== absorber/codebase_map.md ===\n{ABSORBER_CODEBASE_MAP.read_text(encoding='utf-8')}")

    if ARCHIVIST_KNOWLEDGE_LOG.exists():
        track_read(ARCHIVIST_KNOWLEDGE_LOG)
        parts.append(f"=== archivist/knowledge_log.md ===\n{ARCHIVIST_KNOWLEDGE_LOG.read_text(encoding='utf-8')}")

    return "\n\n".join(parts)


def _summarize_decisions(session: dict) -> str:
    """Format decisions from session.json into a compact summary block."""
    decisions = session.get("decisions", [])
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
# Append to long-term log
# ─────────────────────────────────────────────────────────────────────────────

def _append_prompt_log(
    project_name: str,
    enriched_text: str,
    session: dict,
    extra_context: str,
    call_cost: float = 0.0,
) -> None:
    """Append an entry to enricher/prompt_log.json (long-term, append-only)."""
    entry = {
        "generated_at": _now_iso(),
        "project": project_name,
        "model": get_model(ROLE),
        "input_session_hash": session.get("req_hash", ""),
        "decisions_count": len(session.get("decisions", [])),
        "extra_context": extra_context.strip() if extra_context.strip() else None,
        "enriched_prompt_length": len(enriched_text),
        "cost": round(call_cost, 6),
    }

    entries: list[dict] = []
    if ENRICHER_PROMPT_LOG.exists():
        try:
            track_read(ENRICHER_PROMPT_LOG)
            data = json.loads(ENRICHER_PROMPT_LOG.read_text(encoding="utf-8"))
            entries = data.get("entries", [])
        except (json.JSONDecodeError, KeyError):
            entries = []

    entries.append(entry)
    ENRICHER_PROMPT_LOG.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    track_write(ENRICHER_PROMPT_LOG)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt enrichment
# ─────────────────────────────────────────────────────────────────────────────

_ENRICH_SYSTEM = """
You are a senior software architect and prompt engineer.

Your task: given a clarified requirement document, its clarification decisions,
and an optional knowledge layer of the existing codebase,
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
Pull from any conflicts in the clarification session. If none: "(none identified)"

## Constraints
Technical, business, and process constraints the spec must respect.
Examples: auth method, deployment target, SLA, compliance requirements,
existing API contracts that cannot break.

## Clarified Requirement (full text)
Paste the full requirement_synthesis content here verbatim.
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
- If knowledge layer is available, extract constraints from codebase_map.md
  and file-level context.
- Be precise. Vague instructions produce vague specs.
- Output only the markdown document. No preamble, no postamble.
""".strip()


def _enrich(
    project_name: str,
    requirement_synthesis: str,
    session: dict,
    knowledge_layer: str,
    extra_context: str,
) -> tuple[str, float]:
    decisions_block = _summarize_decisions(session)

    conflicts = session.get("conflicts", [])
    conflicts_block = (
        "\n".join(f"  [{c['id']}] {c['description']}" for c in conflicts)
        if conflicts else "(none detected)"
    )

    user_msg = f"""PROJECT NAME: {project_name}

CLARIFIED REQUIREMENT (requirement_synthesis):
{requirement_synthesis}

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
    print("  [3] abort   — stop here, enriched_prompt.md saved for manual review\n")

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
    script = Path(__file__).parent / "04_specwright.py"
    if not script.exists():
        print(f"\n[enricher][warn] 04_specwright.py not found at {script}")
        print(f"[enricher]       Create it first, then run:")
        print(f"           python 04_specwright.py --project {project_name!r}")
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
        session = _load_session()
        requirement_synthesis = _extract_requirement_synthesis(session)

        if not requirement_synthesis.strip():
            print(
                "[enricher][error] clarificator/session.json not found or missing requirement_synthesis.\n"
                "           Run 02_clarificator.py first."
            )
            sys.exit(1)
        print(f"[enricher] Loaded clarificator/session.json ({len(requirement_synthesis)} chars synthesis)")

        n_decisions = len(session.get("decisions", []))
        print(f"[enricher] Session contains {n_decisions} decisions")

        knowledge_layer = _load_knowledge_layer()
        if knowledge_layer.strip():
            parts = []
            if ABSORBER_CODEBASE_MAP.exists():    parts.append("absorber/codebase_map")
            if ARCHIVIST_KNOWLEDGE_LOG.exists():   parts.append("archivist/knowledge_log")
            print(f"[enricher] Knowledge layer loaded: {', '.join(parts)}")
        else:
            print("[enricher] No knowledge layer found — proceeding in greenfield mode")

        # ── Enrich ────────────────────────────────────────────────────────────────
        _print_banner(f"Enriching prompt — {project_name}")
        print("[enricher] Calling LLM to build enriched prompt ...")

        try:
            enriched, call_cost = _enrich(
                project_name          = project_name,
                requirement_synthesis = requirement_synthesis,
                session               = session,
                knowledge_layer       = knowledge_layer,
                extra_context         = args.extra_context,
            )
        except Exception as exc:
            print(f"[enricher][error] Enrichment failed: {exc}")
            sys.exit(1)

        # ── Dry run: print and exit ───────────────────────────────────────────────
        if args.dry_run:
            _print_banner("Dry run — enriched prompt (not written)")
            print(enriched.strip())
            return

        # ── Write enricher/enriched_prompt.md (short-term, overwrite) ─────────────
        header = (
            f"# Enriched Prompt — {project_name}\n"
            f"Generated: {_now_iso()}\n\n"
            f"---\n\n"
        )
        final_content = apply_md_header(
            header + enriched.strip() + "\n",
            ENRICHER_OVERWRITE_PROMPT,
            owner="03_enricher.py",
            model=get_model(ROLE),
        )
        ENRICHER_OVERWRITE_PROMPT.write_text(final_content, encoding="utf-8")
        track_write(ENRICHER_OVERWRITE_PROMPT)
        print(f"[enricher] ✓ Enriched prompt → {ENRICHER_OVERWRITE_PROMPT}")

        # ── Review ────────────────────────────────────────────────────────────────
        if args.no_review:
            should_continue = True
            final_prompt = enriched
        else:
            final_prompt, should_continue = _review_prompt(enriched)
            # If user edited, overwrite the written file with updated content
            if final_prompt != enriched:
                updated = apply_md_header(
                    header + final_prompt.strip() + "\n",
                    ENRICHER_OVERWRITE_PROMPT,
                    owner="03_enricher.py",
                    model=get_model(ROLE),
                )
                ENRICHER_OVERWRITE_PROMPT.write_text(updated, encoding="utf-8")
                track_write(ENRICHER_OVERWRITE_PROMPT)
                print(f"[enricher] ✓ Enriched prompt updated → {ENRICHER_OVERWRITE_PROMPT}")

        # ── Append to long-term log ──────────────────────────────────────────────
        _append_prompt_log(
            project_name  = project_name,
            enriched_text = final_prompt,
            session       = session,
            extra_context = args.extra_context,
            call_cost     = call_cost,
        )
        print(f"[enricher] ✓ Prompt log appended → {ENRICHER_PROMPT_LOG}")

        if not should_continue:
            _print_banner("Stopped — enricher/enriched_prompt.md saved for manual review")
            print(f"  Review:   {ENRICHER_OVERWRITE_PROMPT}")
            print(f"  Continue: python 04_specwright.py --project {project_name!r}\n")
            return

        # ── Launch spec agent ─────────────────────────────────────────────────────
        _launch_spec_agent(project_name)

    finally:
        print_summary("[03]")
        print_artifact_summary("[03]")
        prompt_next_step(ROLE, prefix="[03]")


if __name__ == "__main__":
    main()
