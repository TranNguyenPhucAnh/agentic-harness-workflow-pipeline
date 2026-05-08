"""
pipeline/07_update_knowledge.py
Step 7b — Long-term knowledge distillation after human review.

Two modes:

  A) JUDGE-DRIVEN:
     Run after reviewing judge_report.md. Processes judge findings → writes
     spec_addendum.md, plan_notes.json, findings_notes.md, knowledge base.

  B) HUMAN-FIX CAPTURE:
     Run after you manually fix code that AI couldn't fix.
     Uses `git diff` to capture what you changed, links it to escalated clusters,
     and distills a Pattern entry into artifacts_<slug>/knowledge/current/base.md.
     On next run, base.md is injected into downstream prompts.

Mini-aware behaviour:
  - Reads plan_mini.json, analysis_mini.json, impl_record.json.
  - Human-fix capture diffs mini target files instead of hardcoding src/.
  - Knowledge patterns include mini scope/context where available.

Usage
─────
  # After judge review:
  python pipeline/07_update_knowledge.py
  python pipeline/07_update_knowledge.py --accept-all
  python pipeline/07_update_knowledge.py --dry-run

  # After manual human fix:
  python pipeline/07_update_knowledge.py --capture-human-fix
  python pipeline/07_update_knowledge.py --capture-human-fix --dry-run

  # View accumulated knowledge base:
  python pipeline/07_update_knowledge.py --show-knowledge

Writes (judge mode)
───────────────────
  artifacts_<slug>/knowledge/current/findings_notes.md
  artifacts_<slug>/knowledge/current/spec_addendum.md
  artifacts_<slug>/state/plan_notes.json
  artifacts_<slug>/knowledge/current/base.md
  artifacts_<slug>/knowledge/history/update_log.json

Writes (human-fix capture mode)
────────────────────────────────
  artifacts_<slug>/knowledge/history/update_log.json   ← diff + cluster context + root cause
  artifacts_<slug>/knowledge/current/base.md           ← new Pattern entry appended
  artifacts_<slug>/knowledge/current/findings_notes.md ← human fix note appended

For taxonomy details see docs/artifacts.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent

# === WRITE AUTHORITY: 07_update_knowledge ===
# OWNS  : artifacts_<slug>/knowledge/current/base.md
#         artifacts_<slug>/knowledge/current/findings_notes.md
#         artifacts_<slug>/knowledge/history/update_log.json
#         artifacts_<slug>/state/plan_notes.json
# READS : artifacts_<slug>/run/judge_raw.json
#         artifacts_<slug>/state/plan.json
#         artifacts_<slug>/state/plan_mini.json
#         artifacts_<slug>/run/analysis_mini.json
#         artifacts_<slug>/run/impl_record.json
#         artifacts_<slug>/knowledge/current/spec_addendum.md
#         artifacts_<slug>/run/test_report.json

import sys as _sys

_sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (
    SPEC_PATH,
    artifact_root,
    JUDGE_RAW as JUDGE_RAW_PATH,
    PLAN_JSON as GLM_PLAN_PATH,
    PLAN_MINI as PLAN_MINI_PATH,
    PLAN_NOTES as PLAN_NOTES_PATH,
    FINDINGS_NOTES as FINDINGS_NOTES_PATH,
    SPEC_ADDENDUM as ADDENDUM_PATH,
    KNOWLEDGE_BASE as KNOWLEDGE_BASE_PATH,
    UPDATE_LOG as UPDATE_LOG_PATH,
    TEST_REPORT as TEST_REPORT_PATH,
    ANALYSIS_MINI as ANALYSIS_MINI_PATH,
    IMPL_RECORD as IMPL_RECORD_PATH,
    CLARIFIED_REQ as CLARIFIED_REQ_PATH,
    ENRICHED_PROMPT as ENRICHED_PROMPT_PATH,
    ensure_dirs,
)

ensure_dirs()


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeAction:
    finding: str
    severity: str
    action: str
    target: str
    content: str
    human_approved: bool = False
    note: str = ""


ACTION_ADDENDUM = "spec_addendum"
ACTION_GLM_NOTE = "glm_global_note"
ACTION_FINDINGS_ADD = "findings_add"
ACTION_KNOWLEDGE = "knowledge_base"
ACTION_SPEC_BUMP = "spec_bump_needed"
ACTION_SKIP = "skip"


# ════════════════════════════════════════════════════════════════════════════
# Shared artifact helpers / mini context
# ════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _load_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        return path.read_text()
    except Exception:
        return default


def _load_impl_record() -> dict:
    return _load_json(IMPL_RECORD_PATH, {})


def _current_scope() -> str:
    rec = _load_impl_record()
    scope = rec.get("scope")
    if scope:
        return str(scope)
    if PLAN_MINI_PATH.exists() or ANALYSIS_MINI_PATH.exists():
        return "mini"
    return "full"


def _load_mini_context() -> dict:
    """Load mini planning/analysis/implementation context if present."""
    plan_mini = _load_json(PLAN_MINI_PATH, {})
    analysis_mini = _load_json(ANALYSIS_MINI_PATH, {})
    impl_record = _load_impl_record()

    return {
        "scope": _current_scope(),
        "clarified_requirement": _load_text(CLARIFIED_REQ_PATH).strip(),
        "enriched_prompt": _load_text(ENRICHED_PROMPT_PATH).strip(),
        "plan_mini": plan_mini,
        "analysis_mini": analysis_mini,
        "impl_record": impl_record,
    }


def _mini_target_files(ctx: dict | None = None) -> list[str]:
    """Return allowed/expected files for mini runs."""
    ctx = ctx or _load_mini_context()
    out: list[str] = []

    plan = ctx.get("plan_mini") or {}
    for item in plan.get("target_files", []) or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("path"):
            out.append(str(item["path"]))

    impl = ctx.get("impl_record") or {}
    for item in impl.get("files", []) or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("path"):
            out.append(str(item["path"]))

    seen = set()
    unique: list[str] = []
    for p in out:
        p = p.strip().lstrip("./")
        if p and p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def _run_context_summary(ctx: dict | None = None) -> str:
    ctx = ctx or _load_mini_context()
    scope = ctx.get("scope", "full")

    if scope != "mini":
        return "scope=full"

    plan = ctx.get("plan_mini") or {}
    analysis = ctx.get("analysis_mini") or {}
    target_files = _mini_target_files(ctx)

    task_summary = (
        plan.get("task_summary")
        or plan.get("summary")
        or plan.get("title")
        or ""
    )
    warnings = analysis.get("warnings") or []
    conflicts = analysis.get("conflicts") or []
    recommendations = analysis.get("recommendations") or []

    parts = ["scope=mini"]

    if task_summary:
        parts.append(f"task={task_summary}")

    if target_files:
        parts.append("target_files=" + ", ".join(target_files))

    if warnings:
        parts.append(f"warnings={len(warnings)}")

    if conflicts:
        parts.append(f"conflicts={len(conflicts)}")

    if recommendations:
        parts.append(f"recommendations={len(recommendations)}")

    return " | ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# Tầng 1 — Human fix capture
# ════════════════════════════════════════════════════════════════════════════

def _git_diff_for_paths(paths: list[str]) -> str:
    """Get unstaged + staged diff for selected paths.

    Note:
      This assumes artifact_root() is a git worktree root or inside one.
      If generated artifacts are not in a git repo, capture mode may not work.
    """
    try:
        root = artifact_root()
        args_paths = paths or ["src/"]

        staged = subprocess.run(
            ["git", "diff", "--cached", "--", *args_paths],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout

        unstaged = subprocess.run(
            ["git", "diff", "--", *args_paths],
            cwd=root,
            capture_output=True,
            text=True,
        ).stdout

        return (staged + unstaged).strip()
    except Exception as e:
        return f"(git diff failed: {e})"


def _parse_changed_files_from_diff(diff: str) -> list[str]:
    """Extract changed files from a unified diff."""
    return re.findall(r"^\+\+\+ b/([^\n]+)", diff, re.MULTILINE)


def _load_escalated_clusters() -> list[dict]:
    """Read escalated clusters from test_report.json."""
    if not TEST_REPORT_PATH.exists():
        return []

    try:
        report = json.loads(TEST_REPORT_PATH.read_text())
        return report.get("escalated", [])
    except Exception:
        return []


def _match_clusters_to_files(
    changed_files: list[str],
    escalated: list[dict],
) -> list[dict]:
    """Find escalated clusters whose file/src_file/path matches changed files."""
    matched = []
    changed = set(changed_files)

    for cluster in escalated:
        src = (
            cluster.get("src_file")
            or cluster.get("file")
            or cluster.get("path")
            or ""
        )
        if src in changed:
            matched.append(cluster)

    return matched


def _build_knowledge_pattern(
    diff: str,
    changed_files: list[str],
    matched_clusters: list[dict],
    root_cause: str,
    spec_version: str,
    run_context: str,
) -> str:
    """Build a markdown Pattern entry for knowledge_base.md."""
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

    added_lines = [
        line
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    diff_preview = "\n".join(added_lines[:40])

    if len(added_lines) > 40:
        diff_preview += f"\n... ({len(added_lines) - 40} more lines)"

    return (
        f"## Pattern — {now} (spec {spec_version})\n\n"
        f"**Run context:** {run_context}\n\n"
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
    """Capture human intervention via git diff, update knowledge base."""
    print("\n[07b] HUMAN FIX CAPTURE MODE")

    ctx = _load_mini_context()
    scope = ctx.get("scope", "full")

    diff_paths = ["src/"]

    if scope == "mini":
        mini_files = _mini_target_files(ctx)
        if mini_files:
            diff_paths = mini_files

    print(f"[07b] Scope: {scope}")
    print(f"[07b] Scanning git diff for: {', '.join(diff_paths)}")

    diff = _git_diff_for_paths(diff_paths)
    if not diff:
        print("[07b] No staged/unstaged changes found in selected path(s).")
        print("      Stage your changes first, or check git status.")
        return

    if diff.startswith("(git diff failed:"):
        print(f"[07b] {diff}")
        return

    changed_files = _parse_changed_files_from_diff(diff)
    if not changed_files:
        print("[07b] Could not parse changed files from diff.")
        return

    if scope == "mini":
        allowed = set(_mini_target_files(ctx))
        outside = [f for f in changed_files if allowed and f not in allowed]
        if outside:
            print("[07b] WARNING: changed files outside mini target scope:")
            for f in outside:
                print(f"  ! {f}")

    print("[07b] Changed files detected:")
    for f in changed_files:
        print(f"  {f}")

    escalated = _load_escalated_clusters()
    matched = _match_clusters_to_files(changed_files, escalated)

    if matched:
        print(f"\n[07b] Matched {len(matched)} escalated cluster(s) to your fix:")
        for c in matched:
            print(f"  * {c.get('cluster')} — {c.get('note', '')}")
    else:
        print(
            "\n[07b] No matching escalated clusters found "
            "(fix may be proactive or from judge review)."
        )

    print("\n[07b] Briefly describe the root cause of the bug you fixed.")
    print("      (Press Enter to skip)")

    try:
        root_cause = input("  Root cause: ").strip()
    except (EOFError, KeyboardInterrupt):
        root_cause = ""

    spec_version = "unknown"
    if SPEC_PATH.exists():
        m = re.search(
            r"^#\s*Version:\s*(\S+)",
            SPEC_PATH.read_text(),
            re.MULTILINE,
        )
        if m:
            spec_version = m.group(1)

    pattern = _build_knowledge_pattern(
        diff=diff,
        changed_files=changed_files,
        matched_clusters=matched,
        root_cause=root_cause,
        spec_version=spec_version,
        run_context=_run_context_summary(ctx),
    )

    _append_knowledge_base(pattern, dry_run)

    if not dry_run and matched:
        regression_note = (
            f"\n## Human fix — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
        )

        for c in matched:
            fixed_file = (
                c.get("src_file")
                or c.get("file")
                or c.get("path")
                or "?"
            )
            regression_note += (
                f"- Fixed `{fixed_file}`: "
                f"{root_cause or c.get('note', '')}\n"
            )

        _apply_findings(regression_note, dry_run=False)
        print(f"  ✓ Regression note appended to {FINDINGS_NOTES_PATH}")

    if not dry_run:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "human_fix_capture",
            "scope": scope,
            "run_context": _run_context_summary(ctx),
            "spec_version": spec_version,
            "changed_files": changed_files,
            "root_cause": root_cause,
            "matched_clusters": matched,
            "diff_size_lines": len(diff.splitlines()),
        }

        existing_records: list[dict] = []
        if UPDATE_LOG_PATH.exists():
            try:
                existing_records = json.loads(UPDATE_LOG_PATH.read_text())
            except Exception:
                existing_records = []

        existing_records.append(record)
        UPDATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_LOG_PATH.write_text(json.dumps(existing_records, indent=2))
        print(f"  ✓ Fix record appended to {UPDATE_LOG_PATH}")

    print("\n[07b] Human fix captured. Future runs will use this pattern.")


# ════════════════════════════════════════════════════════════════════════════
# Tầng 2 — knowledge_base.md writer from judge findings
# ════════════════════════════════════════════════════════════════════════════

def _blocking_to_knowledge_pattern(
    finding: str,
    spec_version: str,
    run_context: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return (
        f"## Pattern — {now} (spec {spec_version})\n\n"
        f"**Source:** Judge blocking issue\n\n"
        f"**Run context:** {run_context}\n\n"
        f"**Finding:** {finding}\n\n"
        f"**Inject into:** downstream implementation prompt "
        f"(do NOT reintroduce)\n\n"
        f"---\n\n"
    )


# ════════════════════════════════════════════════════════════════════════════
# Load & parse judge verdict
# ════════════════════════════════════════════════════════════════════════════

def _load_verdict() -> dict:
    if not JUDGE_RAW_PATH.exists():
        print(f"[07b] ERROR: {JUDGE_RAW_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    raw_obj = json.loads(JUDGE_RAW_PATH.read_text())
    raw = raw_obj.get("response", "")

    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    return json.loads(raw)


def _load_previous_fixes() -> dict:
    """Load existing fix records from update_log.json, not used for decisions."""
    if not UPDATE_LOG_PATH.exists():
        return {}

    try:
        logs = json.loads(UPDATE_LOG_PATH.read_text())
        return {"total_records": len(logs)}
    except Exception:
        return {}


# ════════════════════════════════════════════════════════════════════════════
# Classify findings → suggested actions
# ════════════════════════════════════════════════════════════════════════════

_SPEC_EDGE_KEYWORDS = {
    "edge case",
    "undefined",
    "not defined",
    "spec doesn't define",
    "spec doesn't specify",
    "ambiguous",
    "no spec for",
}

_GLM_NOTE_KEYWORDS = {
    "requestAnimationFrame",
    "raf",
    "usememo",
    "usecallback",
    "dark theme",
    "dependency order",
    "hook",
    "circular",
    "architecture",
    "performance",
    "memo",
    "duplicate",
    "singleton",
}

_SPEC_BUMP_KEYWORDS = {
    "contradiction",
    "incorrect spec",
    "spec is wrong",
    "should be changed",
    "spec should",
    "spec needs to",
    "update spec",
}


def _suggest_action(
    finding: str,
    severity: str,
    section_notes: str,
) -> tuple[str, str, str]:
    text = (finding + " " + section_notes).lower()

    if any(kw in text for kw in _SPEC_BUMP_KEYWORDS):
        content = (
            "MANUAL ACTION REQUIRED — update spec.md:\n"
            f"Finding: {finding}\n"
            "Suggestion: define behaviour explicitly in the relevant section."
        )
        return ACTION_SPEC_BUMP, "spec.md (manual)", content

    if any(kw in text for kw in _SPEC_EDGE_KEYWORDS):
        content = (
            f"## Edge case: {finding[:80]}\n\n"
            f"Behaviour: define exact behaviour for: {finding}\n"
        )
        return (
            ACTION_ADDENDUM,
            "artifacts_<slug>/knowledge/current/spec_addendum.md",
            content,
        )

    if any(kw in text for kw in _GLM_NOTE_KEYWORDS) or severity == "blocking":
        content = finding
        return ACTION_GLM_NOTE, "artifacts_<slug>/state/plan_notes.json", content

    content = f"- {finding}"
    return (
        ACTION_FINDINGS_ADD,
        "artifacts_<slug>/knowledge/current/findings_notes.md",
        content,
    )


# ════════════════════════════════════════════════════════════════════════════
# Apply functions
# ════════════════════════════════════════════════════════════════════════════

def _apply_addendum(content: str, dry_run: bool) -> None:
    if dry_run:
        print(
            f"  [DRY RUN] Would append to {ADDENDUM_PATH}:\n"
            f"{indent(content, '    ')}"
        )
        return

    ADDENDUM_PATH.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if ADDENDUM_PATH.exists() else "w"
    with open(ADDENDUM_PATH, mode) as f:
        if mode == "w":
            f.write(
                "# Spec addendum\n"
                "_Edge cases surfaced by judge — inject downstream._\n\n"
            )
        f.write(content + "\n")

    print(f"  ✓ Appended to {ADDENDUM_PATH}")


def _apply_glm_note(content: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would append to plan_notes.json: {content[:80]}")
        return

    if not GLM_PLAN_PATH.exists() and not PLAN_MINI_PATH.exists():
        print("  [warn] No plan.json or plan_mini.json found — skipping planner note.")
        return

    notes: list = []
    if PLAN_NOTES_PATH.exists():
        try:
            notes = json.loads(PLAN_NOTES_PATH.read_text())
        except Exception:
            notes = []

    notes.append(
        {
            "note": content,
            "added": datetime.now(timezone.utc).isoformat(),
            "scope": _current_scope(),
        }
    )

    PLAN_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_NOTES_PATH.write_text(json.dumps(notes, indent=2))

    print("  ✓ Appended to plan_notes.json")


def _apply_findings(content: str, dry_run: bool) -> None:
    block = f"\n{content}\n"

    if dry_run:
        print(
            f"  [DRY RUN] Would append to {FINDINGS_NOTES_PATH}:\n"
            f"{indent(block, '    ')}"
        )
        return

    FINDINGS_NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if FINDINGS_NOTES_PATH.exists() else "w"
    with open(FINDINGS_NOTES_PATH, mode) as f:
        if mode == "w":
            f.write("# Judge findings\n_Auto-managed — do not edit manually._\n")
        f.write(block)

    print(f"  ✓ Appended to {FINDINGS_NOTES_PATH}")


def _print_spec_bump_advice(content: str) -> None:
    print(f"\n  {'!' * 50}")
    print("  MANUAL SPEC EDIT REQUIRED")
    print(f"  {'!' * 50}")
    print(indent(content, "  "))
    print()


APPLY_MAP = {
    ACTION_ADDENDUM: _apply_addendum,
    ACTION_GLM_NOTE: _apply_glm_note,
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

    if len(content) > 80:
        print(f"  Content preview: {content[:80]}…")
    else:
        print(f"  Content: {content}")

    print(
        "  Actions: [y] accept  [s] skip  [g] glm_note  "
        "[a] addendum  [f] findings  [k] knowledge_base"
    )

    try:
        choice = input("  Choice [y]: ").strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        choice = "y"

    if choice == "s":
        return ACTION_SKIP, content, True
    if choice == "g":
        return ACTION_GLM_NOTE, content, True
    if choice == "a":
        return ACTION_ADDENDUM, content, True
    if choice == "f":
        return ACTION_FINDINGS_ADD, content, True
    if choice == "k":
        return ACTION_KNOWLEDGE, content, True

    return action, content, True


# ════════════════════════════════════════════════════════════════════════════
# Show knowledge base
# ════════════════════════════════════════════════════════════════════════════

def show_knowledge() -> None:
    if not KNOWLEDGE_BASE_PATH.exists():
        print("[07b] No knowledge base yet.")
    else:
        print(KNOWLEDGE_BASE_PATH.read_text())

    if UPDATE_LOG_PATH.exists():
        try:
            logs = json.loads(UPDATE_LOG_PATH.read_text())
        except Exception:
            logs = []

        print(f"\n── Update log: {len(logs)} total records ──")

        for r in logs[-5:]:
            mode = r.get("mode", "unknown")
            ts = r.get("timestamp", "")[:10] if "timestamp" in r else "?"
            scope = r.get("scope", "?")

            if mode == "human_fix_capture":
                print(
                    f"  {ts}  {mode}  scope={scope}  "
                    f"{len(r.get('changed_files', []))} file(s)  "
                    f"{r.get('root_cause', '')[:60]}"
                )
            else:
                print(
                    f"  {ts}  {mode}  scope={scope}  "
                    f"{r.get('judge_verdict', '?')}"
                )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge update: judge-driven or human-fix capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--capture-human-fix",
        action="store_true",
        help="Capture manual human fix via git diff → knowledge_base.md",
    )
    parser.add_argument(
        "--show-knowledge",
        action="store_true",
        help="Print knowledge_base.md and update log, then exit",
    )
    parser.add_argument(
        "--accept-all",
        action="store_true",
        help="Accept all suggested actions without prompting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing anything",
    )
    parser.add_argument(
        "--only-blocking",
        action="store_true",
        help="Process only blocking issues",
    )
    parser.add_argument(
        "--only-non-blocking",
        action="store_true",
        help="Process only non-blocking notes",
    )

    args = parser.parse_args()

    if args.show_knowledge:
        show_knowledge()
        return

    if args.capture_human_fix:
        capture_human_fix(dry_run=args.dry_run)
        return

    # ── Judge-driven mode ───────────────────────────────────────────────────
    interactive = not args.accept_all and not args.dry_run

    verdict = _load_verdict()
    _ = _load_previous_fixes()

    run_ctx = _load_mini_context()
    scope = run_ctx.get("scope", "full")
    run_summary = _run_context_summary(run_ctx)

    if verdict.get("verdict") not in ("NEEDS_REVISION", "APPROVED_WITH_NOTES"):
        print(
            f"[07b] Judge verdict is {verdict.get('verdict')} — "
            "no knowledge update needed for APPROVED runs."
        )
        sys.exit(0)

    print(f"[07b] Knowledge update for verdict: {verdict['verdict']}")
    print(f"[07b] Run context: {run_summary}")
    print(f"[07b] Dry-run: {args.dry_run}  |  Interactive: {interactive}")

    sections = verdict.get("sections", {})
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

    spec_version = "unknown"
    if SPEC_PATH.exists():
        m = re.search(
            r"^#\s*Version:\s*(\S+)",
            SPEC_PATH.read_text(),
            re.MULTILINE,
        )
        if m:
            spec_version = m.group(1)

    actions: list[KnowledgeAction] = []
    now = datetime.now(timezone.utc).isoformat()

    for idx, (finding, severity) in enumerate(all_findings, 1):
        sec_notes = ""

        for _sec_name, notes in section_notes_map.items():
            if any(w in notes.lower() for w in finding.lower().split()[:5]):
                sec_notes = notes
                break

        suggested = _suggest_action(finding, severity, sec_notes)

        if interactive:
            final_action, final_content, approved = _prompt_action(
                finding,
                severity,
                suggested,
                idx,
                len(all_findings),
            )
        else:
            final_action, final_content, approved = suggested[0], suggested[2], True
            print(
                f"  [{idx}/{len(all_findings)}] {severity.upper()}: "
                f"{finding[:60]}… → {final_action}"
            )

        ka = KnowledgeAction(
            finding=finding,
            severity=severity,
            action=final_action,
            target=suggested[1],
            content=final_content,
            human_approved=approved,
        )
        actions.append(ka)

        if not approved or final_action == ACTION_SKIP:
            print("  ↳ Skipped")
            continue

        if final_action == ACTION_SPEC_BUMP:
            _print_spec_bump_advice(final_content)
            continue

        if severity == "blocking" and final_action != ACTION_SKIP:
            kb_entry = _blocking_to_knowledge_pattern(
                finding,
                spec_version,
                run_summary,
            )
            _append_knowledge_base(kb_entry, dry_run=args.dry_run)

        if final_action == ACTION_KNOWLEDGE:
            kb_entry = _blocking_to_knowledge_pattern(
                finding,
                spec_version,
                run_summary,
            )
            _append_knowledge_base(kb_entry, dry_run=args.dry_run)
            continue

        apply_fn = APPLY_MAP.get(final_action)
        if apply_fn:
            apply_fn(final_content, args.dry_run)
        else:
            print(f"  [warn] Unknown action: {final_action}")

    if not args.dry_run:
        log_entry = {
            "timestamp": now,
            "mode": "judge_driven",
            "scope": scope,
            "run_context": run_summary,
            "judge_verdict": verdict.get("verdict"),
            "findings_total": len(all_findings),
            "actions_taken": sum(
                1
                for a in actions
                if a.human_approved and a.action != ACTION_SKIP
            ),
            "skipped": sum(1 for a in actions if a.action == ACTION_SKIP),
            "spec_bumps": sum(1 for a in actions if a.action == ACTION_SPEC_BUMP),
            "details": [asdict(a) for a in actions],
        }

        existing_log: list[dict] = []
        if UPDATE_LOG_PATH.exists():
            try:
                existing_log = json.loads(UPDATE_LOG_PATH.read_text())
            except Exception:
                existing_log = []

        existing_log.append(log_entry)
        UPDATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        UPDATE_LOG_PATH.write_text(json.dumps(existing_log, indent=2))

        print(f"\n[07b] Audit log appended to {UPDATE_LOG_PATH}")

    applied = sum(
        1
        for a in actions
        if a.human_approved and a.action not in (ACTION_SKIP, ACTION_SPEC_BUMP)
    )
    skipped = sum(1 for a in actions if a.action == ACTION_SKIP)
    spec_bumps = sum(1 for a in actions if a.action == ACTION_SPEC_BUMP)

    print(f"\n{'=' * 50}")
    print("  KNOWLEDGE UPDATE SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Scope              : {scope}")
    print(f"  Findings processed : {len(all_findings)}")
    print(f"  Actions applied    : {applied}")
    print(f"  Skipped            : {skipped}")
    print(f"  Spec bumps needed  : {spec_bumps}  ← edit spec.md manually")

    if spec_bumps:
        print("\n  Spec bumps detected — edit spec.md manually before next run.")


if __name__ == "__main__":
    main()
