"""
pipeline/04c_spec_impact.py
============================
Spec Impact & Risk Assessor — reads a written spec against the existing
codebase and surfaces risks, breaking changes, missing coverage, and
high-confidence corrections before spectracker locks the version.

Position in pipeline:
    specwright → [04b_spec_validator] → [04c_spec_impact] → spectracker
                   npm facts            risk / impact        version lock

Why a separate script from spec_validator (04b):
    spec_validator uses npm registry as ground truth — deterministic, no LLM.
    spec_impact uses LLM semantic analysis — probabilistic, needs codebase context.
    Mixing them would break spec_validator's core invariant ("no LLM for facts").

────────────────────────────────────────────────────────────────
What this script assesses
────────────────────────────────────────────────────────────────

  1. Impact on existing codebase
     Which files / modules will change? What is the blast radius?
     Does the spec add, modify, or remove existing behaviour?

  2. Risk classification  (per FEATURE_INTAKE pattern)
     TINY   — isolated addition, no existing behaviour touched
     NORMAL — extends existing module, limited blast radius
     HIGH   — touches shared state, auth, data model, or public API surface

  3. Breaking changes
     Spec requirements that contradict or remove current behaviour
     described in codebase_map.md.

  4. Missing coverage in spec
     Requirements that are mentioned but lack AC, lack error handling,
     or lack data model detail — gaps that will cause ambiguity downstream.

  5. Internal consistency
     Conflicts between spec sections (e.g. Tech Stack says X, Architecture
     says Y; AC references an endpoint not in API Contracts).

────────────────────────────────────────────────────────────────
Two types of findings, two UX paths
────────────────────────────────────────────────────────────────

  HIGH CONFIDENCE — auto-patchable suggestions
    The issue is clear, the fix is deterministic, and the patch is small.
    Examples: AC references an undefined term, missing error case,
    section says "TBD", NFR has no concrete value.
    UX: print suggestion + ask y/n → if y, patch spec in-place.

  LOW CONFIDENCE / MANUAL REVIEW — suggest only, human decides
    The issue requires architectural judgement or business context.
    Examples: risk=HIGH breaking changes, ambiguous scope boundary,
    conflicting data model assumptions.
    UX: highlight section + line + suggest directions. No y/n. Human edits.

────────────────────────────────────────────────────────────────
Inputs consumed
────────────────────────────────────────────────────────────────

  spec/<slug>.md          Canonical spec (specwright output, post-validator)
  absorber/codebase_map.md  LLM narrative of existing codebase
                            If absent: impact assessment runs spec-only (degraded)

────────────────────────────────────────────────────────────────
Outputs written
────────────────────────────────────────────────────────────────

  spec/spec_impact_report.md      (short-term, OVERWRITE)
    Human-readable structured report. Sections:
      Risk Level, Blast Radius, Breaking Changes, Missing Coverage,
      Internal Conflicts, High-confidence Patches Applied,
      Manual Review Required.

  spec/spec_impact_log.json       (long-term, APPEND)
    One entry per run: timestamp, spec_version, risk_level,
    finding counts, patches_applied, manual_review_count.

  spec/<slug>.md                  (patched in-place if patches accepted)
    Only modified if human accepts ≥1 high-confidence suggestion.
    Sections patched in-place via regex replace — structure preserved.
    No version bump — spectracker has not run yet.

────────────────────────────────────────────────────────────────
Artifact impact by command
────────────────────────────────────────────────────────────────

  Command                    spec_impact_report.md  log.json  spec patched?
  ─────────────────────────  ─────────────────────  ────────  ─────────────
  (normal run)               OVERWRITE              APPEND    if y accepted
  --no-patch                 OVERWRITE              APPEND    never
  --accept-all               OVERWRITE              APPEND    all high-conf
  --dry-run                  –                      –         never

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python 04c_spec_impact.py --project my-app
    Full assessment. Reads spec + codebase_map.md.
    Writes report + log. Prompts y/n for high-confidence patches.

  python 04c_spec_impact.py --project my-app --no-patch
    Assessment only. Never patches spec even if high-confidence fixes found.
    Use when you want to review the report before deciding.

  python 04c_spec_impact.py --project my-app --accept-all
    Accept all high-confidence patches without prompting.
    Use in automated / non-interactive runs (CI pre-check).

  python 04c_spec_impact.py --project my-app --dry-run
    Builds LLM briefing and prints it. No LLM call, no writes.

  python 04c_spec_impact.py --project my-app --show-last
    Print most recent spec_impact_report.md without re-running.

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_MD,
    SPEC_DIR,
    artifact_root,
    ensure_dirs,
    get_project_slug,
    get_spec_path,
)
from modules.call_llm import call_llm_json  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_summary as print_cost_summary  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402
from modules.post_interactive import prompt_next_step          # noqa: E402


ROLE = "spec_impact_assessor"
# NOTE: add "spec_impact_assessor" to artifacts/models.py (same model as
# "spec_validator" initially; can diverge independently).

MAX_SPEC_CHARS     = 100_000
MAX_CODEBASE_CHARS =  80_000
MAX_BRIEFING_CHARS = 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths  (all under spec/ per pipeline convention)
# ─────────────────────────────────────────────────────────────────────────────

def _spec_dir() -> Path:
    return artifact_root() / "spec"

def _report_path() -> Path:
    return _spec_dir() / "spec_impact_report.md"

def _log_path() -> Path:
    return _spec_dir() / "spec_impact_log.json"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="04c_spec_impact.py",
        description="Assess spec impact, risk, and gaps against existing codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python 04c_spec_impact.py --project my-app
              python 04c_spec_impact.py --project my-app --no-patch
              python 04c_spec_impact.py --project my-app --accept-all
              python 04c_spec_impact.py --project my-app --dry-run
              python 04c_spec_impact.py --project my-app --show-last
        """),
    )
    p.add_argument("--project",    default=None,
                   help="Project name. Sets PIPELINE_PROJECT.")
    p.add_argument("--no-patch",   action="store_true",
                   help="Never patch spec even when high-confidence fixes are found.")
    p.add_argument("--accept-all", action="store_true",
                   help="Accept all high-confidence patches without prompting (non-interactive).")
    p.add_argument("--dry-run",    action="store_true",
                   help="Build briefing and print it. No LLM call, no writes.")
    p.add_argument("--show-last",  action="store_true",
                   help="Print most recent spec_impact_report.md and exit.")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error(
            "PIPELINE_PROJECT is not set. "
            "Use --project <name> or export PIPELINE_PROJECT=<name>."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safe readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path: Path, label: str) -> str:
    if not path.exists():
        print(f"[spec_impact][warn] {label} not found: {path}")
        return ""
    track_read(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[spec_impact][warn] Could not read {label}: {e}")
        return ""


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    print(f"[spec_impact] Truncating {label}: {len(text):,} → {limit:,} chars")
    return text[:limit] + f"\n\n[truncated — {len(text):,} chars total]"


# ─────────────────────────────────────────────────────────────────────────────
# Spec helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_spec_version(text: str) -> str:
    m = re.search(r"^#\s*Version:\s*(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def _list_sections(text: str) -> list[str]:
    """Return list of section heading texts (## only, for briefing display)."""
    return re.findall(r"^##\s+(.+)$", text, re.MULTILINE)


def _list_sections_with_lines(text: str) -> list[tuple[int, str]]:
    """
    Return [(line_number, heading_text), ...] for ALL headings # through ###.
    Used to build the section index injected into the LLM briefing and to
    resolve spec_section references in findings to approximate line numbers.
    """
    result = []
    for i, line in enumerate(text.splitlines(), start=1):
        m = re.match(r"^#{1,3}\s+(.+)$", line)
        if m:
            result.append((i, m.group(1).strip()))
    return result


def _find_line_for_section(spec_text: str, section_heading: str) -> int | None:
    """Return approximate 1-based line number for a section heading (fuzzy match)."""
    heading_lower = section_heading.lower()
    for i, line in enumerate(spec_text.splitlines(), start=1):
        if re.match(r"^#{1,3}\s+", line) and heading_lower in line.lower():
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Briefing builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_briefing(
    spec_text:    str,
    codebase_map: str,
    spec_version: str,
    sections:     list[str],
) -> str:
    parts: list[str] = []

    # Section index with line numbers (# through ###)
    sections_with_lines = _list_sections_with_lines(spec_text)
    section_index = "\n".join(
        f"  L{ln}: {h}" for ln, h in sections_with_lines
    ) or "  (none found)"

    # Context block
    has_codebase = bool(codebase_map.strip())
    parts.append(textwrap.dedent(f"""\
        ## 0. Assessment context

        **Spec version:** {spec_version}
        **Spec sections ({len(sections)}):** {', '.join(sections)}
        **Codebase map available:** {'yes' if has_codebase else 'no (degraded — spec-only analysis)'}

        **Section index with line numbers (use spec_line in findings):**
        {section_index}

        Your task: assess the spec for impact on the existing codebase, risks,
        missing coverage, and internal conflicts. Identify which findings are
        high-confidence auto-patchable vs which require manual human review.
    """).strip())

    # Spec
    parts.append(
        "## 1. Spec\n\n"
        "```markdown\n"
        + _truncate(spec_text, MAX_SPEC_CHARS, "spec")
        + "\n```"
    )

    # Codebase map
    if has_codebase:
        parts.append(
            "## 2. Existing codebase map (absorber narrative)\n\n"
            + _truncate(codebase_map, MAX_CODEBASE_CHARS, "codebase_map.md")
        )
    else:
        parts.append(
            "## 2. Existing codebase map\n\n"
            "_[Not available — impact assessment will be spec-only. "
            "Run absorber for full analysis.]_"
        )

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a senior architect reviewing a technical spec before it enters an
AI coding pipeline. You have access to a narrative description of the existing
codebase. Your job is to surface risks, gaps, and conflicts — not to rewrite
the spec.

────────────────────────────────────────────────────────────────
Assess the following dimensions:

1. RISK LEVEL  (pick one for the spec as a whole)
   TINY   — isolated addition, no existing behaviour touched, no shared state
   NORMAL — extends an existing module, limited blast radius, well-scoped
   HIGH   — touches auth, shared data model, public API surface, or removes
             existing behaviour

2. BLAST RADIUS
   Which existing files / modules / components will change?
   Derive from codebase_map.md if available, else infer from spec architecture.

3. BREAKING CHANGES
   Requirements that contradict, modify, or remove behaviour currently
   described in codebase_map.md. Flag any "this replaces X" or implicit
   removal of an existing interface.

4. MISSING COVERAGE in spec
   AC items that are not independently testable (vague, no threshold)
   Sections that say "TBD", "as needed", "etc.", or "similar to X"
   Error cases described in spec sections but missing from Error Handling
   Data model fields referenced in API Contracts but not defined in Data Models
   NFR values with no concrete number (e.g. "fast" instead of "≤200ms")

5. INTERNAL CONFLICTS
   Tech Stack vs Architecture disagreements
   API Contracts referencing endpoints not in Workflow section
   AC testing behaviour not described anywhere in the spec
   Acceptance Criteria that contradict each other

────────────────────────────────────────────────────────────────
Two finding types — classify each finding as one of:

  HIGH_CONFIDENCE — the issue is clear, the fix is a small deterministic
    text change that does not require architectural judgement.
    You MUST supply an exact patch: the original_text (verbatim from spec,
    ≤5 lines) and the replacement_text.
    Examples:
      - NFR says "should be fast" → patch to "≤200ms p99 under 100 concurrent"
      - AC-05 says "user sees feedback" → patch to "user sees a toast within 300ms"
      - Missing error case in Error Handling → add bullet point
      - "TBD" placeholder → replace with inferred concrete value

  MANUAL_REVIEW — requires architectural judgement, business context,
    or significant rewrite. Do NOT supply a patch.
    Supply 2-3 concrete directional suggestions instead.
    Examples:
      - Spec adds new user role that touches auth middleware
      - Data model change that would require migration
      - Scope boundary ambiguity that could double implementation cost

────────────────────────────────────────────────────────────────
Return raw JSON only (no markdown fences):

{
  "spec_version": "<version>",
  "risk_level": "TINY" | "NORMAL" | "HIGH",
  "risk_rationale": "<1-2 sentences explaining risk classification>",
  "blast_radius": {
    "files":   ["src/lib/auth.ts", "src/components/Sidebar.tsx"],
    "modules": ["auth module", "settings page"],
    "summary": "<1 sentence>"
  },
  "breaking_changes": [
    {
      "description": "<what existing behaviour is affected>",
      "spec_section": "<section heading>",
      "existing_behaviour": "<what codebase currently does per codebase_map>",
      "severity": "HIGH" | "MEDIUM" | "LOW"
    }
  ],
  "findings": [
    {
      "id":              "F-01",
      "type":            "MISSING_COVERAGE" | "INTERNAL_CONFLICT" | "BREAKING_CHANGE" | "RISK",
      "finding_class":   "HIGH_CONFIDENCE" | "MANUAL_REVIEW",
      "confidence":      "high" | "medium" | "low",
      "spec_section":    "<section heading>",
      "spec_line":       <approximate 1-based line number from the section index, or null>,
      "line_hint":       "<verbatim short excerpt from spec near the issue, ≤20 words>",
      "description":     "<what is wrong or missing>",
      "impact":          "<why this matters for implementation>",

      // HIGH_CONFIDENCE only — omit both keys for MANUAL_REVIEW:
      "original_text":    "<exact text to replace, verbatim from spec>",
      "replacement_text": "<replacement text>",

      // MANUAL_REVIEW only — omit for HIGH_CONFIDENCE:
      "suggestions": [
        "<direction 1>",
        "<direction 2>"
      ],
      // MANUAL_REVIEW medium-confidence — provide 2-3 concrete options:
      "options": [
        "<option A — concrete action>",
        "<option B — concrete action>",
        "<option C — concrete action>"
      ]
    }
  ],
  "summary": "<3-4 sentence executive summary>",
  "proceed_recommendation": "PROCEED" | "REVIEW_FIRST" | "BLOCK",
  "proceed_rationale": "<1-2 sentences>"
}

proceed_recommendation:
  PROCEED      — no blockers, safe to run spectracker
  REVIEW_FIRST — manual review items exist but not blocking; human should read report
  BLOCK        — HIGH breaking changes or HIGH risk without mitigation; spectracker should wait

Be direct. Do not invent issues. If the spec is clean, say so with an empty findings array.

confidence field:
  high   — evidence clearly visible in both spec and codebase map; fix is unambiguous.
           HIGH_CONFIDENCE findings must be high confidence.
  medium — issue is real but fix involves tradeoffs or judgment.
           MANUAL_REVIEW findings should use medium or low.
  low    — speculative; depends on context not in the inputs. Flag as informational.

options field (MANUAL_REVIEW + medium confidence only):
  Provide 2-3 concrete, actionable options the human can choose between.
  Each option should be a complete sentence describing a specific action.
  Omit for low-confidence findings (no options needed — just flag the concern).

spec_line field:
  Use the section index (line numbers) provided in the briefing to give an
  approximate line number. This helps human navigate large specs. Use null
  if the finding is not tied to a specific section.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Spec patcher — in-place regex replace, section-aware
# ─────────────────────────────────────────────────────────────────────────────

def _apply_patch(spec_text: str, original: str, replacement: str) -> tuple[str, bool]:
    """
    Replace original_text with replacement_text in spec.
    Returns (new_text, success).
    Only patches if original is found exactly once — avoids ambiguous replacements.
    """
    if not original or not original.strip():
        return spec_text, False

    count = spec_text.count(original)
    if count == 0:
        print(f"  [patch] Original text not found in spec (may already be patched)")
        return spec_text, False
    if count > 1:
        print(f"  [patch] Original text found {count} times — skipping ambiguous patch")
        return spec_text, False

    return spec_text.replace(original, replacement, 1), True


def _interactive_patch_loop(
    spec_text:  str,
    findings:   list[dict[str, Any]],
    *,
    accept_all: bool,
) -> tuple[str, list[str]]:
    """
    Walk HIGH_CONFIDENCE findings. For each:
      - If accept_all: apply without prompting
      - Else: show diff and ask y/n

    Returns (patched_spec_text, list_of_applied_finding_ids).
    """
    patchable = [
        f for f in findings
        if f.get("finding_class") == "HIGH_CONFIDENCE"
        and f.get("original_text")
        and f.get("replacement_text")
    ]

    if not patchable:
        return spec_text, []

    applied: list[str] = []

    print()
    print(f"  {len(patchable)} high-confidence patch(es) available:")
    print()

    for finding in patchable:
        fid         = finding.get("id", "?")
        section     = finding.get("spec_section", "?")
        description = finding.get("description", "")
        original    = finding.get("original_text", "")
        replacement = finding.get("replacement_text", "")

        confidence = finding.get("confidence", "")
        spec_line  = finding.get("spec_line")
        loc = f"§ {section}"
        if spec_line:
            loc += f"  (line ~{spec_line})"
        conf_str = f"  [confidence: {confidence}]" if confidence else ""
        print(f"  ── {fid}: {loc}{conf_str} ──")
        print(f"  Issue: {description}")
        print(f"  Replace:")
        for line in original.splitlines():
            print(f"    - {line}")
        print(f"  With:")
        for line in replacement.splitlines():
            print(f"    + {line}")
        print()

        if accept_all:
            do_apply = True
            print(f"  [--accept-all] Applying {fid}.")
        else:
            try:
                choice = input(f"  Apply patch {fid}? [y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "n"
            do_apply = (choice == "y")

        if do_apply:
            spec_text, ok = _apply_patch(spec_text, original, replacement)
            if ok:
                applied.append(fid)
                print(f"  ✓ Applied {fid}")
            else:
                print(f"  ✗ Could not apply {fid} (text not found exactly)")
        else:
            print(f"  Skipped {fid}")
        print()

    return spec_text, applied


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

_RISK_EMOJI = {"TINY": "🟢", "NORMAL": "🟡", "HIGH": "🔴"}
_PROCEED_EMOJI = {"PROCEED": "✅", "REVIEW_FIRST": "⚠️", "BLOCK": "🚫"}


def _write_report_md(
    result:           dict[str, Any],
    spec_version:     str,
    patches_applied:  list[str],
) -> str:
    risk        = result.get("risk_level", "NORMAL")
    risk_emoji  = _RISK_EMOJI.get(risk, "")
    proceed     = result.get("proceed_recommendation", "REVIEW_FIRST")
    p_emoji     = _PROCEED_EMOJI.get(proceed, "")
    summary     = result.get("summary", "")
    rationale   = result.get("risk_rationale", "")
    p_rationale = result.get("proceed_rationale", "")
    blast       = result.get("blast_radius", {})
    breaking    = result.get("breaking_changes", [])
    findings    = result.get("findings", [])
    ts          = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    slug = os.environ.get("PIPELINE_PROJECT", "<name>")

    L: list[str] = []
    L += [
        "# Spec Impact Report", "",
        f"**Spec version:** {spec_version}",
        f"**Assessed at:** {ts}",
        f"**Risk level:** {risk_emoji} {risk} — {rationale}",
        f"**Recommendation:** {p_emoji} {proceed} — {p_rationale}",
        "",
        "## Summary", "",
        summary, "",
    ]

    # Blast radius
    if blast:
        files   = blast.get("files", [])
        modules = blast.get("modules", [])
        b_sum   = blast.get("summary", "")
        L += ["## Blast Radius", ""]
        if b_sum:
            L += [b_sum, ""]
        if files:
            L += ["**Likely affected files:**"]
            L += [f"- `{f}`" for f in files]
            L += [""]
        if modules:
            L += ["**Affected modules/components:**"]
            L += [f"- {m}" for m in modules]
            L += [""]

    # Breaking changes
    if breaking:
        L += ["## Breaking Changes", ""]
        for bc in breaking:
            sev   = bc.get("severity", "?")
            desc  = bc.get("description", "")
            sec   = bc.get("spec_section", "")
            exist = bc.get("existing_behaviour", "")
            L += [
                f"### [{sev}] {desc}", "",
                f"**Spec section:** {sec}",
                f"**Existing behaviour:** {exist}",
                "",
            ]

    # Manual review findings
    manual = [f for f in findings if f.get("finding_class") == "MANUAL_REVIEW"]
    if manual:
        L += ["## Manual Review Required", "",
              "_These findings require architectural judgement. "
              "Human decides — no auto-patch._", ""]
        for finding in manual:
            fid        = finding.get("id", "?")
            ftype      = finding.get("type", "")
            sec        = finding.get("spec_section", "")
            spec_line  = finding.get("spec_line")
            confidence = finding.get("confidence", "")
            hint       = finding.get("line_hint", "")
            desc       = finding.get("description", "")
            impact     = finding.get("impact", "")
            suggs      = finding.get("suggestions", [])
            opts       = finding.get("options") or []

            # Location string with line number
            loc = f"§ **{sec}**" if sec else ""
            if loc and spec_line:
                loc += f" (line ~{spec_line})"
            conf_badge = f" · confidence: {confidence}" if confidence else ""

            L += [f"### {fid} — {ftype}: {sec}  _(risk{conf_badge})_", ""]
            if loc:
                L += [f"**Location:** {loc}", ""]
            if hint:
                L += [f"> …{hint}…", ""]
            L += [
                f"**Issue:** {desc}",
                f"**Impact:** {impact}",
            ]
            if suggs:
                L += ["**Suggested directions:**"]
                L += [f"- {s}" for s in suggs]
            if opts:
                L += ["**Options:**"]
                for i, opt in enumerate(opts, 1):
                    L += [f"  {i}. {opt}"]
            L += [""]

    # High-confidence patches
    high_conf = [f for f in findings if f.get("finding_class") == "HIGH_CONFIDENCE"]
    if high_conf:
        L += ["## High-confidence Findings", ""]
        for finding in high_conf:
            fid    = finding.get("id", "?")
            sec    = finding.get("spec_section", "")
            desc   = finding.get("description", "")
            status = "✓ PATCHED" if fid in patches_applied else "— not applied"
            L += [f"- **{fid}** [{sec}]: {desc}  {status}"]
        L += [""]

    # Next step
    L += ["---", ""]
    if proceed == "BLOCK":
        L += [
            "## 🚫 Recommendation: Review before proceeding",
            "",
            "Address the **Breaking Changes** and **Manual Review** items above,",
            "then re-run spec_impact before spectracker.",
            "```",
            f"python pipeline/04c_spec_impact.py --project {slug}",
            "```", "",
        ]
    elif proceed == "REVIEW_FIRST":
        L += [
            "## ⚠️ Recommendation: Read report, then proceed",
            "",
            "Review **Manual Review Required** items above.",
            "If acceptable, proceed with spectracker:",
            "```",
            f"python pipeline/05_spectracker.py --project {slug}",
            "```", "",
        ]
    else:
        L += [
            "## ✅ Clear to proceed",
            "",
            "No blocking issues found. Run spectracker when ready:",
            "```",
            f"python pipeline/05_spectracker.py --project {slug}",
            "```", "",
        ]

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(
    result:          dict[str, Any],
    spec_version:    str,
    patches_applied: list[str],
) -> None:
    log = _log_path()
    existing: list[dict[str, Any]] = []

    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text())
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    findings       = result.get("findings", [])
    manual_count   = sum(1 for f in findings if f.get("finding_class") == "MANUAL_REVIEW")
    high_conf_count = sum(1 for f in findings if f.get("finding_class") == "HIGH_CONFIDENCE")
    breaking_count = len(result.get("breaking_changes", []))

    high_conf_count_by_confidence = {
        lvl: sum(1 for f in findings
                 if f.get("finding_class") == "HIGH_CONFIDENCE"
                 and f.get("confidence") == lvl)
        for lvl in ("high", "medium", "low")
    }
    entry = {
        "assessed_at":            datetime.now(timezone.utc).isoformat(),
        "spec_version":           spec_version,
        "risk_level":             result.get("risk_level"),
        "proceed_recommendation": result.get("proceed_recommendation"),
        "breaking_changes":       breaking_count,
        "manual_review_count":    manual_count,
        "high_confidence_count":  high_conf_count,
        "confidence_breakdown":   high_conf_count_by_confidence,
        "patches_applied":        patches_applied,
    }
    existing.append(entry)

    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"entries": existing}, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# Long-term artifact commit — y/n keep log entry
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_commit_log() -> None:
    """Ask user whether to keep the log entry just appended to spec_impact_log.json."""
    log = _log_path()
    if not log.exists():
        return
    try:
        data    = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        return
    if not entries:
        return
    try:
        ans = input(f"  Keep this entry in {log.name}? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"  [04c] Entry kept in {log.name} (non-interactive).")
        return
    if ans in ("n", "no"):
        entries.pop()
        try:
            log.write_text(
                json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [04c] Entry discarded — {log.name} unchanged.")
        except Exception as exc:
            print(f"  [04c][warn] Could not revert {log.name}: {exc}")
    else:
        print(f"  [04c] Entry kept in {log.name} (total: {len(entries)}).")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()

    print("=" * 60)
    print("  SPEC IMPACT ASSESSOR")
    print("=" * 60)
    print()

    # ── --show-last ──────────────────────────────────────────────────────────
    if args.show_last:
        p = _report_path()
        if not p.exists():
            print("[spec_impact] No previous report found.")
            sys.exit(2)
        print(p.read_text())
        sys.exit(0)

    # ── Load inputs ──────────────────────────────────────────────────────────
    spec_path    = get_spec_path()
    spec_text    = _read_text(spec_path, "spec")
    codebase_map = _read_text(Path(str(ABSORBER_CODEBASE_MD)), "codebase_map.md")

    if not spec_text:
        print(f"[spec_impact][error] Spec not found: {spec_path}", file=sys.stderr)
        sys.exit(2)

    if not codebase_map:
        print("[spec_impact][warn] codebase_map.md not found.")
        print("  Impact assessment will be spec-only (degraded quality).")
        print("  Run absorber for full analysis.\n")

    spec_version = _parse_spec_version(spec_text)
    sections     = _list_sections(spec_text)

    print(f"  Spec version:   {spec_version}")
    print(f"  Sections found: {len(sections)}: {', '.join(sections)}")
    print(f"  Codebase map:   {'found' if codebase_map else 'MISSING (degraded)'}")
    print()

    # ── Briefing ─────────────────────────────────────────────────────────────
    briefing = _build_briefing(
        spec_text    = spec_text,
        codebase_map = codebase_map,
        spec_version = spec_version,
        sections     = sections,
    )
    briefing = briefing[:MAX_BRIEFING_CHARS]
    print(f"  Briefing size: {len(briefing):,} chars")
    print()

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[spec_impact] DRY RUN — briefing follows:\n")
        print(briefing)
        sys.exit(0)

    # ── LLM call ─────────────────────────────────────────────────────────────
    print("[spec_impact] Calling LLM for assessment …")
    t0 = time.time()

    try:
        result, cost = call_llm_json(
            role        = ROLE,
            system      = _SYSTEM,
            user        = briefing,
            max_tokens  = 8192,
            caller_file = __file__,
            label       = "spec_impact",
        )
    except Exception as e:
        print(f"[spec_impact][error] LLM call failed: {e}", file=sys.stderr)
        sys.exit(2)

    elapsed = time.time() - t0
    risk    = result.get("risk_level", "?")
    proceed = result.get("proceed_recommendation", "?")
    n_find  = len(result.get("findings", []))
    n_break = len(result.get("breaking_changes", []))

    print(f"  Elapsed: {elapsed:.1f}s  |  cost: ${cost:.4f}")
    print(f"  Risk level:     {risk}")
    print(f"  Recommendation: {proceed}")
    print(f"  Findings:       {n_find}")
    print(f"  Breaking:       {n_break}")
    print()

    # ── Patches (interactive or --accept-all) ─────────────────────────────────
    patches_applied: list[str] = []
    patched_spec    = spec_text

    if not args.no_patch:
        findings = result.get("findings", [])
        high_conf = [f for f in findings if f.get("finding_class") == "HIGH_CONFIDENCE"]

        if high_conf:
            patched_spec, patches_applied = _interactive_patch_loop(
                spec_text  = spec_text,
                findings   = findings,
                accept_all = args.accept_all,
            )

            if patches_applied:
                spec_path.write_text(patched_spec, encoding="utf-8")
                track_write(spec_path)
                print(f"  Spec patched ({len(patches_applied)} change(s)): {spec_path}")
                print()
        else:
            print("  No high-confidence patches available.")
            print()
    else:
        print("  --no-patch: skipping all patches.")
        print()

    # ── Write artifacts ───────────────────────────────────────────────────────
    exit_code = 0
    try:
        _spec_dir().mkdir(parents=True, exist_ok=True)

        report_md   = _write_report_md(result, spec_version, patches_applied)
        report_path = _report_path()

        report_with_header = apply_md_header(
            content = report_md,
            path    = report_path,
            owner   = "04c_spec_impact.py",
        )
        report_path.write_text(report_with_header, encoding="utf-8")
        track_write(report_path)
        print(f"  Written:  {report_path}")

        _append_log(result, spec_version, patches_applied)
        print(f"  Appended: {_log_path()}")

        # ── Final summary ─────────────────────────────────────────────────────
        emoji = _PROCEED_EMOJI.get(proceed, "")
        print(f"\n  {emoji} {proceed} — {result.get('proceed_rationale', '')}")
        print()

        if proceed == "BLOCK":
            exit_code = 1

    except Exception as exc:
        print(f"[04c][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 2

    finally:
        print()
        print_artifact_summary("[04c]")
        print()
        print_cost_summary("[04c]")
        prompt_next_step(ROLE, prefix="[04c]")

    # Long-term artifact commit (after summary, before exit)
    if exit_code in (0, 1):
        _maybe_commit_log()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()