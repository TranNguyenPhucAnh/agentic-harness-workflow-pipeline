"""
pipeline/07_update_knowledge.py
Step 7b — Long-term knowledge distillation after human review.

Two modes:

  A) JUDGE-DRIVEN (original behaviour):
     Run after reviewing judge_report.md. Processes judge findings → writes
     spec_addendum.md, glm_plan.json global_notes, judge_findings.md.

  B) HUMAN-FIX CAPTURE (new):
     Run after you manually fix code that AI couldn't fix.
     Uses `git diff` to capture what you changed, links it to escalated clusters,
     and distils a Pattern entry into scaffold/knowledge_base.md.
     On next run, knowledge_base.md is injected into Minimax L2 system prompt.

Usage
─────
  # After judge review (original mode):
  python pipeline/07_update_knowledge.py
  python pipeline/07_update_knowledge.py --accept-all
  python pipeline/07_update_knowledge.py --dry-run

  # After manual human fix (new mode):
  python pipeline/07_update_knowledge.py --capture-human-fix
  python pipeline/07_update_knowledge.py --capture-human-fix --dry-run

  # View accumulated knowledge base:
  python pipeline/07_update_knowledge.py --show-knowledge

Writes (judge mode)
───────────────────
  scaffold/judge_findings.md
  scaffold/spec_addendum.md
  scaffold/glm_plan.json        (global_notes patched)
  scaffold/knowledge_base.md    (pattern entries from judge blocking issues)
  reports/knowledge_update_log.json

Writes (human-fix capture mode)
────────────────────────────────
  reports/human_fix_record.json  ← diff + cluster context + root cause
  scaffold/knowledge_base.md     ← new Pattern entry appended
  scaffold/judge_findings.md     ← human fix note appended (regression prevention)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent

ROOT         = Path(__file__).parent.parent
SCAFFOLD_DIR = ROOT / "scaffold"
REPORTS_DIR  = ROOT / "reports"

JUDGE_RAW_PATH      = REPORTS_DIR / "judge_raw.json"
FIX_REPORT_PATH     = REPORTS_DIR / "judge_fix_report.json"
HUMAN_FIX_PATH      = REPORTS_DIR / "human_fix_record.json"
GLM_PLAN_PATH       = SCAFFOLD_DIR / "glm_plan.json"
FINDINGS_PATH       = SCAFFOLD_DIR / "judge_findings.md"
ADDENDUM_PATH       = SCAFFOLD_DIR / "spec_addendum.md"
KNOWLEDGE_BASE_PATH = SCAFFOLD_DIR / "knowledge_base.md"
UPDATE_LOG_PATH     = REPORTS_DIR / "knowledge_update_log.json"
ESCALATED_PATH      = REPORTS_DIR / "escalated_clusters.json"


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeAction:
    finding:        str
    severity:       str
    action:         str
    target:         str
    content:        str
    human_approved: bool = False
    note:           str  = ""


ACTION_ADDENDUM      = "spec_addendum"
ACTION_GLM_NOTE      = "glm_global_note"
ACTION_FINDINGS_ADD  = "findings_add"
ACTION_KNOWLEDGE     = "knowledge_base"
ACTION_SPEC_BUMP     = "spec_bump_needed"
ACTION_SKIP          = "skip"


# ════════════════════════════════════════════════════════════════════════════
# Tầng 1 — Human fix capture
# ════════════════════════════════════════════════════════════════════════════

def _git_diff_src() -> str:
    """Get unstaged + staged diff for src/ files."""
    try:
        staged   = subprocess.run(
            ["git", "diff", "--cached", "--", "src/"],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout
        unstaged = subprocess.run(
            ["git", "diff", "--", "src/"],
            cwd=ROOT, capture_output=True, text=True,
        ).stdout
        return (staged + unstaged).strip()
    except Exception as e:
        return f"(git diff failed: {e})"


def _parse_changed_files_from_diff(diff: str) -> list[str]:
    """Extract list of changed src/ files from a unified diff."""
    return re.findall(r"^\+\+\+ b/(src/[^\n]+)", diff, re.MULTILINE)


def _load_escalated_clusters() -> list[dict]:
    if not ESCALATED_PATH.exists():
        return []
    try:
        return json.loads(ESCALATED_PATH.read_text()).get("clusters", [])
    except Exception:
        return []


def _match_clusters_to_files(
    changed_files: list[str],
    escalated: list[dict],
) -> list[dict]:
    """
    Find escalated clusters whose src_file matches one of the changed files.
    These are clusters the AI couldn't fix — human just fixed them.
    """
    matched = []
    for cluster in escalated:
        src = cluster.get("src_file", "")
        if src in changed_files:
            matched.append(cluster)
    return matched


def _build_knowledge_pattern(
    diff: str,
    changed_files: list[str],
    matched_clusters: list[dict],
    root_cause: str,
    spec_version: str,
) -> str:
    """
    Build a markdown Pattern entry for knowledge_base.md.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    files_str = "\n".join(f"- `{f}`" for f in changed_files)

    clusters_str = ""
    if matched_clusters:
        clusters_str = "\n**AI-escalated clusters fixed by human:**\n"
        for c in matched_clusters:
            note = c.get("note", "")
            attempts = c.get("attempts", "?")
            clusters_str += (
                f"- `{c.get('cluster', '?')}` "
                f"({attempts} AI attempt(s)) — {note}\n"
            )

    # Extract a short diff summary (only + lines, cap at 40 lines)
    added_lines = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    diff_preview = "\n".join(added_lines[:40])
    if len(added_lines) > 40:
        diff_preview += f"\n... ({len(added_lines) - 40} more lines)"

    return (
        f"## Pattern — {now} (spec {spec_version})\n\n"
        f"**Files changed by human:**\n{files_str}\n\n"
        + (f"**Root cause:** {root_cause}\n\n" if root_cause else "")
        + clusters_str
        + f"\n**Diff preview (added lines):**\n```diff\n{diff_preview}\n```\n\n"
        f"---\n\n"
    )


def _append_knowledge_base(entry: str, dry_run: bool) -> None:
    if dry_run:
        print(f"\n[DRY RUN] Would append to {KNOWLEDGE_BASE_PATH}:")
        print(indent(entry[:300] + ("…" if len(entry) > 300 else ""), "  "))
        return

    KNOWLEDGE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = "# Knowledge Base — Human Fix Patterns\n\n"
    if not KNOWLEDGE_BASE_PATH.exists():
        KNOWLEDGE_BASE_PATH.write_text(header)
    existing = KNOWLEDGE_BASE_PATH.read_text()
    KNOWLEDGE_BASE_PATH.write_text(existing + entry)
    print(f"  ✓ Appended pattern to {KNOWLEDGE_BASE_PATH}")


def capture_human_fix(dry_run: bool) -> None:
    """
    Tầng 1: capture human intervention.
    - git diff src/ → find changed files
    - match against escalated_clusters.json
    - prompt for root_cause
    - write human_fix_record.json + knowledge_base.md entry
    """
    print("\n[07b] HUMAN FIX CAPTURE MODE")
    print("[07b] Scanning git diff for src/ changes …")

    diff = _git_diff_src()
    if not diff:
        print("[07b] No staged/unstaged changes found in src/.")
        print("      Stage your changes with `git add src/` first, or check git status.")
        return

    changed_files = _parse_changed_files_from_diff(diff)
    if not changed_files:
        print("[07b] Could not parse changed files from diff.")
        return

    print(f"[07b] Changed files detected:")
    for f in changed_files:
        print(f"  {f}")

    escalated = _load_escalated_clusters()
    matched   = _match_clusters_to_files(changed_files, escalated)

    if matched:
        print(f"\n[07b] Matched {len(matched)} escalated cluster(s) to your fix:")
        for c in matched:
            print(f"  * {c.get('cluster')} — {c.get('note', '')}")
    else:
        print("\n[07b] No matching escalated clusters found "
              "(fix may be proactive or from judge review).")

    # Prompt for root cause
    print("\n[07b] Briefly describe the root cause of the bug you fixed.")
    print("      (Press Enter to skip)")
    try:
        root_cause = input("  Root cause: ").strip()
    except (EOFError, KeyboardInterrupt):
        root_cause = ""

    # Load current spec version
    spec_version = "unknown"
    spec_path = ROOT / "spec.md"
    if spec_path.exists():
        m = re.search(r"^#\s*Version:\s*(\S+)", spec_path.read_text(), re.MULTILINE)
        if m:
            spec_version = m.group(1)

    # Build knowledge pattern
    pattern = _build_knowledge_pattern(
        diff=diff,
        changed_files=changed_files,
        matched_clusters=matched,
        root_cause=root_cause,
        spec_version=spec_version,
    )

    # Write knowledge_base.md
    _append_knowledge_base(pattern, dry_run)

    # Append short note to judge_findings.md for regression prevention
    if not dry_run and matched:
        regression_note = (
            f"\n## Human fix — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        )
        for c in matched:
            regression_note += f"- Fixed `{c.get('src_file', '?')}`: {root_cause or c.get('note', '')}\n"
        _apply_findings(regression_note, dry_run=False)
        print(f"  ✓ Regression note appended to {FINDINGS_PATH}")

    # Write human_fix_record.json
    if not dry_run:
        record = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "spec_version":     spec_version,
            "changed_files":    changed_files,
            "root_cause":       root_cause,
            "matched_clusters": matched,
            "diff_size_lines":  len(diff.splitlines()),
        }
        existing_records: list[dict] = []
        if HUMAN_FIX_PATH.exists():
            try:
                existing_records = json.loads(HUMAN_FIX_PATH.read_text())
            except Exception:
                pass
        existing_records.append(record)
        HUMAN_FIX_PATH.write_text(json.dumps(existing_records, indent=2))
        print(f"  ✓ Fix record → {HUMAN_FIX_PATH}")

    print(f"\n[07b] Human fix captured. Minimax will use this pattern on next run.")
    print(f"[07b] To verify: python harness.py --test-only --skip-judge")


# ════════════════════════════════════════════════════════════════════════════
# Tầng 2 — knowledge_base.md writer from judge findings
# ════════════════════════════════════════════════════════════════════════════

def _blocking_to_knowledge_pattern(finding: str, spec_version: str) -> str:
    """Convert a judge blocking issue into a knowledge_base.md Pattern entry."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"## Pattern — {now} (spec {spec_version})\n\n"
        f"**Source:** Judge blocking issue\n\n"
        f"**Finding:** {finding}\n\n"
        f"**Inject into:** Minimax L2 system prompt (do NOT reintroduce)\n\n"
        f"---\n\n"
    )


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
# Classify findings → suggested actions (unchanged from original)
# ════════════════════════════════════════════════════════════════════════════

_SPEC_EDGE_KEYWORDS = {
    "edge case", "undefined", "not defined", "spec doesn't define",
    "spec doesn't specify", "ambiguous", "no spec for",
}
_GLM_NOTE_KEYWORDS = {
    "requestAnimationFrame", "rAF", "useMemo", "useCallback",
    "dark theme", "dependency order", "hook", "circular", "architecture",
    "performance", "memo", "duplicate", "singleton",
}
_SPEC_BUMP_KEYWORDS = {
    "contradiction", "incorrect spec", "spec is wrong", "should be changed",
    "spec should", "spec needs to", "update spec",
}


def _suggest_action(finding: str, severity: str, section_notes: str) -> tuple[str, str, str]:
    text = (finding + " " + section_notes).lower()

    if any(kw in text for kw in _SPEC_BUMP_KEYWORDS):
        content = (
            f"MANUAL ACTION REQUIRED — update spec.md:\n"
            f"Finding: {finding}\n"
            f"Suggestion: define behaviour explicitly in the relevant section."
        )
        return ACTION_SPEC_BUMP, "spec.md (manual)", content

    if any(kw in text for kw in _SPEC_EDGE_KEYWORDS):
        content = f"## Edge case: {finding[:80]}\n\nBehaviour: define exact behaviour for: {finding}\n"
        return ACTION_ADDENDUM, "scaffold/spec_addendum.md", content

    if any(kw in text for kw in _GLM_NOTE_KEYWORDS) or severity == "blocking":
        content = finding
        return ACTION_GLM_NOTE, "scaffold/glm_plan.json (global_notes)", content

    content = f"- {finding}"
    return ACTION_FINDINGS_ADD, "scaffold/judge_findings.md", content


# ════════════════════════════════════════════════════════════════════════════
# Apply functions
# ════════════════════════════════════════════════════════════════════════════

def _apply_addendum(content: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would append to {ADDENDUM_PATH}:\n{indent(content, '    ')}")
        return
    mode = "a" if ADDENDUM_PATH.exists() else "w"
    with open(ADDENDUM_PATH, mode) as f:
        if mode == "w":
            f.write("# Spec addendum\n_Edge cases surfaced by judge — inject downstream._\n\n")
        f.write(content + "\n")
    print(f"  ✓ Appended to {ADDENDUM_PATH}")


def _apply_glm_note(content: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would patch glm_plan.json global_notes: {content[:80]}")
        return
    if not GLM_PLAN_PATH.exists():
        print(f"  [warn] glm_plan.json not found — skipping GLM note.")
        return
    plan = json.loads(GLM_PLAN_PATH.read_text())
    existing = plan.get("global_notes", "")
    separator = " | " if existing else ""
    plan["global_notes"] = existing + separator + content
    GLM_PLAN_PATH.write_text(json.dumps(plan, indent=2))
    print(f"  ✓ Patched glm_plan.json global_notes")


def _apply_findings(content: str, dry_run: bool) -> None:
    block = f"\n{content}\n"
    if dry_run:
        print(f"  [DRY RUN] Would append to {FINDINGS_PATH}:\n{indent(block, '    ')}")
        return
    mode = "a" if FINDINGS_PATH.exists() else "w"
    with open(FINDINGS_PATH, mode) as f:
        if mode == "w":
            f.write("# Judge findings\n_Auto-managed — do not edit manually._\n")
        f.write(block)
    print(f"  ✓ Appended to {FINDINGS_PATH}")


def _print_spec_bump_advice(content: str) -> None:
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
# Interactive prompt
# ════════════════════════════════════════════════════════════════════════════

def _prompt_action(
    finding: str,
    severity: str,
    suggested: tuple[str, str, str],
    idx: int,
    total: int,
) -> tuple[str, str, bool]:
    action, target, content = suggested
    print(f"\n[{idx}/{total}] {severity.upper()}: {finding[:100]}")
    print(f"  Suggested: {action} → {target}")
    print(f"  Content preview: {content[:80]}…" if len(content) > 80 else f"  Content: {content}")
    print("  Actions: [y] accept  [s] skip  [g] glm_note  [a] addendum  [f] findings  [k] knowledge_base")
    try:
        choice = input("  Choice [y]: ").strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        choice = "y"

    if choice == "s":
        return ACTION_SKIP, content, True
    elif choice == "g":
        return ACTION_GLM_NOTE, content, True
    elif choice == "a":
        return ACTION_ADDENDUM, content, True
    elif choice == "f":
        return ACTION_FINDINGS_ADD, content, True
    elif choice == "k":
        return ACTION_KNOWLEDGE, content, True
    return action, content, True


# ════════════════════════════════════════════════════════════════════════════
# Show knowledge base
# ════════════════════════════════════════════════════════════════════════════

def show_knowledge() -> None:
    if not KNOWLEDGE_BASE_PATH.exists():
        print("[07b] No knowledge_base.md yet.")
        return
    print(KNOWLEDGE_BASE_PATH.read_text())

    # Also show human_fix_record summary
    if HUMAN_FIX_PATH.exists():
        records = json.loads(HUMAN_FIX_PATH.read_text())
        print(f"\n── Human fix records: {len(records)} total ──")
        for r in records[-5:]:   # last 5
            print(f"  {r['timestamp'][:10]}  v{r['spec_version']}  "
                  f"{len(r['changed_files'])} file(s)  "
                  f"{r.get('root_cause', '')[:60]}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge update: judge-driven or human-fix capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--capture-human-fix", action="store_true",
                        help="Capture manual human fix via git diff → knowledge_base.md")
    parser.add_argument("--show-knowledge", action="store_true",
                        help="Print knowledge_base.md and human fix history, then exit")
    parser.add_argument("--accept-all",        action="store_true",
                        help="Accept all suggested actions without prompting")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Print what would be done without writing anything")
    parser.add_argument("--only-blocking",     action="store_true",
                        help="Process only blocking issues")
    parser.add_argument("--only-non-blocking", action="store_true",
                        help="Process only non-blocking notes")
    args = parser.parse_args()

    if args.show_knowledge:
        show_knowledge()
        return

    if args.capture_human_fix:
        capture_human_fix(dry_run=args.dry_run)
        return

    # ── Judge-driven mode (original) ─────────────────────────────────────────
    interactive = not args.accept_all and not args.dry_run

    verdict    = _load_verdict()
    fix_report = _load_fix_report()

    if verdict.get("verdict") not in ("NEEDS_REVISION", "APPROVED_WITH_NOTES"):
        print(f"[07b] Judge verdict is {verdict.get('verdict')} — "
              f"no knowledge update needed for APPROVED runs.")
        sys.exit(0)

    print(f"[07b] Knowledge update for verdict: {verdict['verdict']}")
    print(f"[07b] Dry-run: {args.dry_run}  |  Interactive: {interactive}")

    sections          = verdict.get("sections", {})
    section_notes_map = {k: v.get("notes", "") for k, v in sections.items()}

    all_findings: list[tuple[str, str]] = []

    if not args.only_non_blocking:
        for desc in verdict.get("blocking_issues", []):
            all_findings.append((desc, "blocking"))

    if not args.only_blocking:
        for desc in verdict.get("non_blocking_notes", []):
            all_findings.append((desc, "non_blocking"))
        gaps_notes = section_notes_map.get("gaps_risks", "")
        if gaps_notes:
            items = re.split(r"\d+\)", gaps_notes)
            for item in items[1:]:
                item = item.strip()
                if item and len(item) > 20:
                    all_findings.append((item, "gap_risk"))

    if not all_findings:
        print("[07b] No findings to process.")
        sys.exit(0)

    print(f"\n[07b] Processing {len(all_findings)} finding(s) …\n")

    # Load spec version for knowledge patterns
    spec_version = "unknown"
    spec_path = ROOT / "spec.md"
    if spec_path.exists():
        m = re.search(r"^#\s*Version:\s*(\S+)", spec_path.read_text(), re.MULTILINE)
        if m:
            spec_version = m.group(1)

    actions: list[KnowledgeAction] = []
    now = datetime.now(timezone.utc).isoformat()

    for idx, (finding, severity) in enumerate(all_findings, 1):
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
            print("  ↳ Skipped")
            continue

        if final_action == ACTION_SPEC_BUMP:
            _print_spec_bump_advice(final_content)
            continue

        # Blocking issues also get a knowledge_base.md entry
        if severity == "blocking" and final_action != ACTION_SKIP:
            kb_entry = _blocking_to_knowledge_pattern(finding, spec_version)
            _append_knowledge_base(kb_entry, dry_run=args.dry_run)

        if final_action == ACTION_KNOWLEDGE:
            kb_entry = _blocking_to_knowledge_pattern(finding, spec_version)
            _append_knowledge_base(kb_entry, dry_run=args.dry_run)
            continue

        apply_fn = APPLY_MAP.get(final_action)
        if apply_fn:
            apply_fn(final_content, args.dry_run)
        else:
            print(f"  [warn] Unknown action: {final_action}")

    # Inject spec_addendum into pipeline_context
    if not args.dry_run:
        if any(a.action == ACTION_ADDENDUM for a in actions
               if a.human_approved and a.action != ACTION_SKIP):
            ctx_path = SCAFFOLD_DIR / "pipeline_context.json"
            if ctx_path.exists():
                ctx = json.loads(ctx_path.read_text())
                ctx["spec_addendum_path"] = "scaffold/spec_addendum.md"
                ctx_path.write_text(json.dumps(ctx, indent=2))
                print(f"\n[07b] pipeline_context.json updated with spec_addendum_path")

    # Audit log
    if not args.dry_run:
        log_entry = {
            "timestamp":       now,
            "mode":            "judge_driven",
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

    applied    = sum(1 for a in actions if a.human_approved and a.action not in (ACTION_SKIP, ACTION_SPEC_BUMP))
    skipped    = sum(1 for a in actions if a.action == ACTION_SKIP)
    spec_bumps = sum(1 for a in actions if a.action == ACTION_SPEC_BUMP)

    print(f"\n{'='*50}")
    print(f"  KNOWLEDGE UPDATE SUMMARY")
    print(f"{'='*50}")
    print(f"  Findings processed : {len(all_findings)}")
    print(f"  Actions applied    : {applied}")
    print(f"  Skipped            : {skipped}")
    print(f"  Spec bumps needed  : {spec_bumps}  ← edit spec.md manually")

    if spec_bumps:
        print(f"\n  After editing spec.md:")
        print(f"    python harness.py")

    if not args.dry_run and applied > 0:
        print(f"\n  Next run will use these learnings automatically.")
        print(f"  To verify: python harness.py --test-only --skip-judge")


if __name__ == "__main__":
    main()
