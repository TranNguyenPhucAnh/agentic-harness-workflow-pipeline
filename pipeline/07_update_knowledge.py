"""
pipeline/07_update_knowledge.py
Step 7b — Long-term knowledge distillation after human review.

Run this MANUALLY after reviewing judge_report.md and deciding what to
carry forward permanently.  Unlike 07_fix_from_judge.py (which runs
automatically and fixes code), this script updates the pipeline's
"institutional memory":

  1. spec_addendum.md  — spec clarifications that don't require a full version
                         bump (edge cases, invariants the judge surfaced)
  2. judge_findings.md — refresh with human-curated notes for future runs
  3. GLM global_notes  — patch glm_plan.json with lessons learned so the
                         planner doesn't miss the same things next time
  4. stdout advice     — for findings that DO require a spec version bump,
                         prints the exact spec sections to update manually

Does NOT
────────
  - Touch spec.md directly (spec is human-controlled)
  - Re-run any pipeline steps
  - Call any LLM (this is deterministic — human decision gets persisted)

Usage
─────
  # Interactive mode: script prompts you for each finding
  python pipeline/07_update_knowledge.py

  # Non-interactive: accept all suggested actions
  python pipeline/07_update_knowledge.py --accept-all

  # Dry-run: print what would be done without writing anything
  python pipeline/07_update_knowledge.py --dry-run

  # Process only specific severity
  python pipeline/07_update_knowledge.py --only-blocking
  python pipeline/07_update_knowledge.py --only-non-blocking

Reads
─────
  reports/judge_raw.json
  reports/judge_fix_report.json    (optional — fix status from 07_fix_from_judge)
  scaffold/glm_plan.json           (optional — to patch global_notes)
  scaffold/judge_findings.md       (existing findings to merge with)

Writes
──────
  scaffold/judge_findings.md       ← refreshed with human-approved notes
  scaffold/spec_addendum.md        ← edge cases / invariants to inject downstream
  scaffold/glm_plan.json           ← global_notes patched (if glm_plan exists)
  reports/knowledge_update_log.json ← audit trail of all decisions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent

ROOT         = Path(__file__).parent.parent
SCAFFOLD_DIR = ROOT / "scaffold"
REPORTS_DIR  = ROOT / "reports"

JUDGE_RAW_PATH    = REPORTS_DIR / "judge_raw.json"
FIX_REPORT_PATH   = REPORTS_DIR / "judge_fix_report.json"
GLM_PLAN_PATH     = SCAFFOLD_DIR / "glm_plan.json"
FINDINGS_PATH     = SCAFFOLD_DIR / "judge_findings.md"
ADDENDUM_PATH     = SCAFFOLD_DIR / "spec_addendum.md"
UPDATE_LOG_PATH   = REPORTS_DIR / "knowledge_update_log.json"


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeAction:
    """One decision: what to do with a judge finding."""
    finding:        str
    severity:       str           # "blocking" | "non_blocking" | "gap_risk"
    action:         str           # see ACTION_* constants below
    target:         str           # which artifact is updated
    content:        str           # what gets written
    human_approved: bool = False
    note:           str  = ""


# Action types
ACTION_ADDENDUM      = "spec_addendum"     # edge case → spec_addendum.md
ACTION_GLM_NOTE      = "glm_global_note"   # invariant → glm_plan.json global_notes
ACTION_FINDINGS_ADD  = "findings_add"      # lesson → judge_findings.md
ACTION_SPEC_BUMP     = "spec_bump_needed"  # requires human spec.md edit (just prints advice)
ACTION_SKIP          = "skip"              # human chose to skip


# ════════════════════════════════════════════════════════════════════════════
# Load & parse judge verdict
# ════════════════════════════════════════════════════════════════════════════

def _load_verdict() -> dict:
    if not JUDGE_RAW_PATH.exists():
        print(f"[07b] ERROR: {JUDGE_RAW_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(JUDGE_RAW_PATH.read_text()).get("response", "")
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$",        "", raw.strip())
    return json.loads(raw)


def _load_fix_report() -> dict:
    if not FIX_REPORT_PATH.exists():
        return {}
    return json.loads(FIX_REPORT_PATH.read_text())


# ════════════════════════════════════════════════════════════════════════════
# Classify findings → suggested actions
# ════════════════════════════════════════════════════════════════════════════

# Keywords that indicate spec needs an explicit edge-case definition
_SPEC_EDGE_KEYWORDS = {
    "edge case", "undefined", "not defined", "spec doesn't define",
    "spec doesn't specify", "ambiguous", "no spec for",
}

# Keywords that are good GLM global_note candidates
_GLM_NOTE_KEYWORDS = {
    "requestAnimationFrame", "rAF", "useMemo", "useCallback",
    "dark theme", "dependency order", "hook", "circular", "architecture",
    "performance", "memo", "duplicate", "singleton",
}

# Keywords that suggest a full spec bump is needed
_SPEC_BUMP_KEYWORDS = {
    "contradiction", "incorrect spec", "spec is wrong", "should be changed",
    "spec should", "spec needs to", "update spec",
}


def _suggest_action(finding: str, severity: str, section_notes: str) -> tuple[str, str, str]:
    """
    Return (action_type, target_artifact, suggested_content).
    Heuristic: different finding types map to different artifacts.
    """
    text = (finding + " " + section_notes).lower()

    # Spec bump needed?
    if any(kw in text for kw in _SPEC_BUMP_KEYWORDS):
        content = (
            f"MANUAL ACTION REQUIRED — update spec.md:\n"
            f"Finding: {finding}\n"
            f"Suggestion: define behaviour explicitly in the relevant section."
        )
        return ACTION_SPEC_BUMP, "spec.md (manual)", content

    # Edge case → spec_addendum
    if any(kw in text for kw in _SPEC_EDGE_KEYWORDS):
        content = (
            f"## Edge case: {finding[:80]}\n\n"
            f"Behaviour: {_derive_edge_behaviour(finding)}\n"
        )
        return ACTION_ADDENDUM, "scaffold/spec_addendum.md", content

    # Architectural invariant → GLM global_notes
    if any(kw in text for kw in _GLM_NOTE_KEYWORDS) or severity == "blocking":
        content = _derive_glm_note(finding)
        return ACTION_GLM_NOTE, "scaffold/glm_plan.json (global_notes)", content

    # Fallback: general lesson → judge_findings.md
    content = f"- {finding}"
    return ACTION_FINDINGS_ADD, "scaffold/judge_findings.md", content


def _derive_edge_behaviour(finding: str) -> str:
    """Extract a behaviour definition from a finding description."""
    # Heuristic: if finding says "X when Y", suggest "When Y, X should..."
    lower = finding.lower()
    if "when" in lower:
        parts = re.split(r"\bwhen\b", finding, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            return f"When {parts[1].strip()}, behaviour should be explicitly defined."
    return f"Define exact behaviour for: {finding}"


def _derive_glm_note(finding: str) -> str:
    """Convert a finding into a GLM global_notes bullet."""
    # Strip common judge phrasing to get a direct instruction
    instruction = finding
    replacements = [
        ("uses setInterval instead of requestAnimationFrame",
         "MUST use requestAnimationFrame, NOT setInterval"),
        ("missing useMemo wrapper",
         "MUST wrap data generation in useMemo([]) to prevent unnecessary re-renders"),
        ("duplicate.*implementations?",
         "one canonical implementation only — no duplicate hooks across directories"),
        ("light theme",
         "dark theme REQUIRED: bg-gray-800/900, text-gray-100/200/300 — never bg-white"),
    ]
    for pattern, replacement in replacements:
        instruction = re.sub(pattern, replacement, instruction, flags=re.IGNORECASE)

    return instruction.strip().rstrip(".")


# ════════════════════════════════════════════════════════════════════════════
# Interactive prompt
# ════════════════════════════════════════════════════════════════════════════

def _prompt_action(
    finding:   str,
    severity:  str,
    suggested: tuple[str, str, str],
    idx:       int,
    total:     int,
) -> tuple[str, str, bool]:
    """
    Ask human what to do with this finding.
    Returns (final_action, final_content, approved).
    """
    action_type, target, content = suggested
    sep = "─" * 60

    print(f"\n{sep}")
    print(f"Finding [{idx}/{total}]  ({severity.upper()})")
    print(f"{sep}")
    print(f"  {finding}")
    print(f"\nSuggested action : {action_type}")
    print(f"Target artifact  : {target}")
    print(f"Content to write :")
    print(indent(content, "  "))
    print()
    print("  [a] Accept suggested action")
    print("  [e] Edit content before saving")
    print("  [g] Force → glm_global_note")
    print("  [x] Force → spec_addendum")
    print("  [f] Force → judge_findings only")
    print("  [s] Skip this finding")

    while True:
        choice = input("  Choice [a/e/g/x/f/s]: ").strip().lower()
        if choice in ("a", ""):
            return action_type, content, True
        elif choice == "e":
            print("  Enter new content (blank line to finish):")
            lines: list[str] = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            return action_type, "\n".join(lines), True
        elif choice == "g":
            return ACTION_GLM_NOTE, _derive_glm_note(finding), True
        elif choice == "x":
            edge_content = (
                f"## Edge case: {finding[:80]}\n\n"
                f"Behaviour: {_derive_edge_behaviour(finding)}\n"
            )
            return ACTION_ADDENDUM, edge_content, True
        elif choice == "f":
            return ACTION_FINDINGS_ADD, f"- {finding}", True
        elif choice == "s":
            return ACTION_SKIP, "", False
        print("  Invalid choice. Please enter a/e/g/x/f/s.")


# ════════════════════════════════════════════════════════════════════════════
# Apply actions to artifacts
# ════════════════════════════════════════════════════════════════════════════

def _apply_addendum(content: str, dry_run: bool) -> None:
    """Append to spec_addendum.md (create if missing)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = f"\n<!-- added {now} -->\n{content}\n"
    if dry_run:
        print(f"  [DRY] Would append to {ADDENDUM_PATH}:\n{indent(block, '    ')}")
        return
    with open(ADDENDUM_PATH, "a") as f:
        if ADDENDUM_PATH.stat().st_size == 0 if ADDENDUM_PATH.exists() else True:
            f.write(f"# Spec Addendum — edge cases & invariants\n"
                    f"_Auto-generated by 07_update_knowledge.py. "
                    f"Review before next scaffold run._\n")
        f.write(block)
    print(f"  ✓ Appended to {ADDENDUM_PATH}")


def _apply_glm_note(content: str, dry_run: bool) -> None:
    """Append content to glm_plan.json global_notes."""
    if not GLM_PLAN_PATH.exists():
        print(f"  [warn] {GLM_PLAN_PATH} not found — note saved to judge_findings.md instead")
        _apply_findings(f"[GLM note] {content}", dry_run)
        return

    plan = json.loads(GLM_PLAN_PATH.read_text())
    existing = plan.get("global_notes", "")
    separator = "\n" if existing and not existing.endswith("\n") else ""
    plan["global_notes"] = existing + separator + content

    if dry_run:
        print(f"  [DRY] Would patch {GLM_PLAN_PATH} global_notes with:\n"
              f"    {content}")
        return

    GLM_PLAN_PATH.write_text(json.dumps(plan, indent=2))
    print(f"  ✓ Patched {GLM_PLAN_PATH} global_notes")


def _apply_findings(content: str, dry_run: bool) -> None:
    """Append to judge_findings.md."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = f"\n<!-- {now} -->\n{content}\n"
    if dry_run:
        print(f"  [DRY] Would append to {FINDINGS_PATH}:\n{indent(block, '    ')}")
        return

    mode = "a" if FINDINGS_PATH.exists() else "w"
    with open(FINDINGS_PATH, mode) as f:
        if mode == "w":
            f.write("# Judge findings\n_Auto-managed — do not edit manually._\n")
        f.write(block)
    print(f"  ✓ Appended to {FINDINGS_PATH}")


def _print_spec_bump_advice(content: str) -> None:
    """Print actionable advice for findings that require manual spec.md edits."""
    print(f"\n  {'!'*50}")
    print(f"  MANUAL SPEC EDIT REQUIRED")
    print(f"  {'!'*50}")
    print(indent(content, "  "))
    print()


APPLY_MAP = {
    ACTION_ADDENDUM:     _apply_addendum,
    ACTION_GLM_NOTE:     _apply_glm_note,
    ACTION_FINDINGS_ADD: _apply_findings,
}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-term knowledge update after human review of judge report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--accept-all",        action="store_true",
                        help="Accept all suggested actions without prompting")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Print what would be done without writing anything")
    parser.add_argument("--only-blocking",     action="store_true",
                        help="Process only blocking issues")
    parser.add_argument("--only-non-blocking", action="store_true",
                        help="Process only non-blocking notes")
    args = parser.parse_args()
    interactive = not args.accept_all and not args.dry_run

    # ── Load ──────────────────────────────────────────────────────────────────
    verdict    = _load_verdict()
    fix_report = _load_fix_report()

    if verdict.get("verdict") not in ("NEEDS_REVISION", "APPROVED_WITH_NOTES"):
        print(f"[07b] Judge verdict is {verdict.get('verdict')} — "
              f"no knowledge update needed for APPROVED runs.")
        sys.exit(0)

    print(f"[07b] Knowledge update for verdict: {verdict['verdict']}")
    print(f"[07b] Dry-run: {args.dry_run}  |  Interactive: {interactive}")

    # ── Gather findings ───────────────────────────────────────────────────────
    sections = verdict.get("sections", {})
    section_notes_map = {k: v.get("notes", "") for k, v in sections.items()}

    all_findings: list[tuple[str, str]] = []   # (description, severity)

    if not args.only_non_blocking:
        for desc in verdict.get("blocking_issues", []):
            all_findings.append((desc, "blocking"))

    if not args.only_blocking:
        for desc in verdict.get("non_blocking_notes", []):
            all_findings.append((desc, "non_blocking"))

        # Also surface gaps/risks from the gaps_risks section
        gaps_notes = section_notes_map.get("gaps_risks", "")
        if gaps_notes:
            # Split numbered items like "1) ... 2) ..."
            items = re.split(r"\d+\)", gaps_notes)
            for item in items[1:]:   # skip empty first element
                item = item.strip()
                if item and len(item) > 20:
                    all_findings.append((item, "gap_risk"))

    if not all_findings:
        print("[07b] No findings to process.")
        sys.exit(0)

    print(f"\n[07b] Processing {len(all_findings)} finding(s) …\n")

    # ── Process each finding ──────────────────────────────────────────────────
    actions: list[KnowledgeAction] = []
    now = datetime.now(timezone.utc).isoformat()

    for idx, (finding, severity) in enumerate(all_findings, 1):
        # Find relevant section notes for this finding
        sec_notes = ""
        for sec_name, notes in section_notes_map.items():
            if any(w in notes.lower() for w in finding.lower().split()[:5]):
                sec_notes = notes
                break

        suggested = _suggest_action(finding, severity, sec_notes)

        if interactive:
            final_action, final_content, approved = _prompt_action(
                finding, severity, suggested, idx, len(all_findings)
            )
        else:
            final_action, final_content, approved = suggested[0], suggested[2], True
            print(f"  [{idx}/{len(all_findings)}] {severity.upper()}: "
                  f"{finding[:60]}… → {final_action}")

        ka = KnowledgeAction(
            finding=finding, severity=severity,
            action=final_action, target=suggested[1],
            content=final_content, human_approved=approved,
        )
        actions.append(ka)

        if not approved or final_action == ACTION_SKIP:
            print(f"  ↳ Skipped")
            continue

        if final_action == ACTION_SPEC_BUMP:
            _print_spec_bump_advice(final_content)
            continue

        apply_fn = APPLY_MAP.get(final_action)
        if apply_fn:
            apply_fn(final_content, args.dry_run)
        else:
            print(f"  [warn] Unknown action: {final_action}")

    # ── Inject spec_addendum into pipeline_context if it was updated ──────────
    if not args.dry_run:
        addendum_was_updated = any(a.action == ACTION_ADDENDUM for a in actions
                                   if a.human_approved and a.action != ACTION_SKIP)
        if addendum_was_updated and ADDENDUM_PATH.exists():
            ctx_path = SCAFFOLD_DIR / "pipeline_context.json"
            if ctx_path.exists():
                ctx = json.loads(ctx_path.read_text())
                ctx["spec_addendum_path"] = "scaffold/spec_addendum.md"
                ctx_path.write_text(json.dumps(ctx, indent=2))
                print(f"\n[07b] pipeline_context.json updated with spec_addendum_path")

    # ── Audit log ─────────────────────────────────────────────────────────────
    if not args.dry_run:
        log_entry = {
            "timestamp":       now,
            "judge_verdict":   verdict.get("verdict"),
            "findings_total":  len(all_findings),
            "actions_taken":   sum(1 for a in actions
                                   if a.human_approved and a.action != ACTION_SKIP),
            "skipped":         sum(1 for a in actions if a.action == ACTION_SKIP),
            "spec_bumps":      sum(1 for a in actions if a.action == ACTION_SPEC_BUMP),
            "details":         [asdict(a) for a in actions],
        }

        existing_log: list[dict] = []
        if UPDATE_LOG_PATH.exists():
            try:
                existing_log = json.loads(UPDATE_LOG_PATH.read_text())
            except Exception:
                existing_log = []

        existing_log.append(log_entry)
        UPDATE_LOG_PATH.write_text(json.dumps(existing_log, indent=2))
        print(f"\n[07b] Audit log → {UPDATE_LOG_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    applied   = sum(1 for a in actions if a.human_approved and a.action not in (ACTION_SKIP, ACTION_SPEC_BUMP))
    skipped   = sum(1 for a in actions if a.action == ACTION_SKIP)
    spec_bumps= sum(1 for a in actions if a.action == ACTION_SPEC_BUMP)

    print(f"\n{'='*50}")
    print(f"  KNOWLEDGE UPDATE SUMMARY")
    print(f"{'='*50}")
    print(f"  Findings processed : {len(all_findings)}")
    print(f"  Actions applied    : {applied}")
    print(f"  Skipped            : {skipped}")
    print(f"  Spec bumps needed  : {spec_bumps}  ← edit spec.md manually")

    if spec_bumps:
        print(f"\n  Next steps for spec bumps:")
        for a in actions:
            if a.action == ACTION_SPEC_BUMP:
                print(f"    • {a.finding[:70]}")
        print(f"\n  After editing spec.md, run:")
        print(f"    python harness.py  (full re-run with new spec version)")

    if not args.dry_run and applied > 0:
        print(f"\n  Artifacts updated:")
        for a in actions:
            if a.human_approved and a.action not in (ACTION_SKIP, ACTION_SPEC_BUMP):
                print(f"    • {a.target}")
        print(f"\n  Next run will automatically use these learnings.")
        print(f"  To verify: python harness.py --test-only --skip-judge")


if __name__ == "__main__":
    main()
