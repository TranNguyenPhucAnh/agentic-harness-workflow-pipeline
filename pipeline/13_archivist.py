"""
pipeline/13_archivist.py
========================
Step 13 — Long-term knowledge distillation after human review.

Two modes:

  A) JUDGE-DRIVEN:
     Run after reviewing judge_verdict_summary.md / judge_session_verdict_raw.json.
     Processes judge findings and writes/updates:
       - archivist_knowledge_log.md
       - archivist_spec_gaps.md
       - archivist_curation_log.json

  B) HUMAN-FIX CAPTURE:
     Run after you manually fix code that AI couldn't fix.
     Uses `git diff` to capture what you changed, links it to escalated clusters,
     and distills a Pattern entry into archivist_knowledge_log.md.
     On next run, archivist_knowledge_log.md is injected into downstream prompts.

Mini-aware behaviour:
  - Reads planner_mini_execution_plan.json, planner_mini_impact_analysis.json,
    executor_session_manifest.json.
  - Human-fix capture diffs mini target files instead of hardcoding src/.
  - Knowledge patterns include mini scope/context where available.

Usage
─────
  # After judge review:
  python pipeline/13_archivist.py --project my-app
  python pipeline/13_archivist.py --project my-app --accept-all
  python pipeline/13_archivist.py --project my-app --dry-run

  # After manual human fix:
  python pipeline/13_archivist.py --project my-app --capture-human-fix
  python pipeline/13_archivist.py --project my-app --capture-human-fix --dry-run

  # View accumulated knowledge:
  python pipeline/13_archivist.py --project my-app --show-knowledge

Writes
──────
  artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
  artifacts_<slug>/knowledge/current/archivist_spec_gaps.md
  artifacts_<slug>/knowledge/history/archivist_curation_log.json

Reads
─────
  artifacts_<slug>/execution/judge_session_verdict_raw.json
  artifacts_<slug>/reports/judge_verdict_summary.md
  artifacts_<slug>/state/planner_full_execution_plan.json
  artifacts_<slug>/state/planner_mini_execution_plan.json
  artifacts_<slug>/state/planner_mini_impact_analysis.json
  artifacts_<slug>/execution/executor_session_manifest.json
  artifacts_<slug>/execution/debugger_session_test_summary.json
  artifacts_<slug>/state/clarificator_requirement_synthesis.md
  artifacts_<slug>/execution/enricher_session_enriched_prompt.md
  artifacts_<slug>/knowledge/current/patcher_findings_snapshot.md
  artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
  artifacts_<slug>/knowledge/current/archivist_spec_gaps.md
  artifacts_<slug>/specwright_spec_<slug>.md

At the end of each run, prints:
  - artifacts/files read
  - artifacts/files created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent
from typing import Any

# === WRITE AUTHORITY: archivist ===
# OWNS  : artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
#         artifacts_<slug>/knowledge/current/archivist_spec_gaps.md
#         artifacts_<slug>/knowledge/history/archivist_curation_log.json
# READS : artifacts_<slug>/execution/judge_session_verdict_raw.json
#         artifacts_<slug>/reports/judge_verdict_summary.md
#         artifacts_<slug>/state/planner_full_execution_plan.json
#         artifacts_<slug>/state/planner_mini_execution_plan.json
#         artifacts_<slug>/state/planner_mini_impact_analysis.json
#         artifacts_<slug>/execution/executor_session_manifest.json
#         artifacts_<slug>/execution/debugger_session_test_summary.json
#         artifacts_<slug>/state/clarificator_requirement_synthesis.md
#         artifacts_<slug>/execution/enricher_session_enriched_prompt.md
#         artifacts_<slug>/knowledge/current/patcher_findings_snapshot.md
#         artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
#         artifacts_<slug>/knowledge/current/archivist_spec_gaps.md
#         artifacts_<slug>/specwright_spec_<slug>.md

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    ARCHIVIST_CURATION_LOG,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    CLARIFIED_REQ,
    DEBUGGER_SESSION_TEST_SUMMARY,
    ENRICHER_SESSION_PROMPT,
    EXECUTOR_SESSION_MANIFEST,
    JUDGE_SESSION_VERDICT_RAW,
    JUDGE_VERDICT_SUMMARY,
    PATCHER_FINDINGS_SNAPSHOT,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_IMPACT,
    PLANNER_MINI_PLAN,
    artifact_root,
    ensure_dirs,
    get_spec_path,
)


# ════════════════════════════════════════════════════════════════════════════
# Artifact/file access tracking
# ════════════════════════════════════════════════════════════════════════════

_ARTIFACTS_READ: set[str] = set()
_ARTIFACTS_WRITTEN: set[str] = set()


def _track_read(path: Any) -> None:
    _ARTIFACTS_READ.add(str(path))


def _track_write(path: Any) -> None:
    _ARTIFACTS_WRITTEN.add(str(path))


def _print_artifact_access_summary() -> None:
    print("[13] Artifacts/files read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[13]   READ  {item}")
    else:
        print("[13]   READ  (none)")

    print("[13] Artifacts/files created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[13]   WRITE {item}")
    else:
        print("[13]   WRITE (none)")


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


ACTION_SPEC_GAP = "spec_gap"
ACTION_ARCHITECTURE_PATTERN = "architecture_pattern"
ACTION_FINDING_PATTERN = "finding_pattern"
ACTION_KNOWLEDGE = "knowledge_log"
ACTION_SPEC_BUMP = "spec_bump_needed"
ACTION_SKIP = "skip"


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="13_archivist.py",
        description="Knowledge update: judge-driven or human-fix capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
    )
    parser.add_argument(
        "--capture-human-fix",
        action="store_true",
        help="Capture manual human fix via git diff → archivist_knowledge_log.md",
    )
    parser.add_argument(
        "--show-knowledge",
        action="store_true",
        help="Print archivist_knowledge_log.md and archivist_curation_log.json, then exit",
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

    return parser


def _configure_project(
    project: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 13_archivist.py directly."
    )


# ════════════════════════════════════════════════════════════════════════════
# Shared artifact helpers / mini context
# ════════════════════════════════════════════════════════════════════════════

def _load_json(path: Any, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        _track_read(path)
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f"[13][warn] Could not parse JSON {path}: {exc}", file=sys.stderr)
        return default


def _load_text(path: Any, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        _track_read(path)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[13][warn] Could not read {path}: {exc}", file=sys.stderr)
        return default


def _load_impl_record() -> dict[str, Any]:
    rec = _load_json(EXECUTOR_SESSION_MANIFEST, {})
    return rec if isinstance(rec, dict) else {}


def _current_scope() -> str:
    rec = _load_impl_record()
    scope = rec.get("scope")
    if scope:
        return str(scope)
    if PLANNER_MINI_PLAN.exists() or PLANNER_MINI_IMPACT.exists():
        return "mini"
    return "full"


def _load_mini_context() -> dict[str, Any]:
    """Load mini planning/analysis/implementation context if present."""
    plan_mini = _load_json(PLANNER_MINI_PLAN, {})
    analysis_mini = _load_json(PLANNER_MINI_IMPACT, {})
    impl_record = _load_impl_record()

    return {
        "scope": _current_scope(),
        "clarified_requirement": _load_text(CLARIFIED_REQ).strip(),
        "enriched_prompt": _load_text(ENRICHER_SESSION_PROMPT).strip(),
        "plan_mini": plan_mini,
        "analysis_mini": analysis_mini,
        "impl_record": impl_record,
    }


def _mini_target_files(ctx: dict[str, Any] | None = None) -> list[str]:
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


def _run_context_summary(ctx: dict[str, Any] | None = None) -> str:
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


def _extract_spec_version() -> str:
    spec_path = get_spec_path()
    if not spec_path.exists():
        return "unknown"

    text = _load_text(spec_path)
    for pattern in (
        r"^#\s*Version:\s*(\S+)",
        r"^version:\s*(\S+)",
        r"^Spec Version:\s*(\S+)",
    ):
        m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1)

    return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Human fix capture
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


def _load_escalated_clusters() -> list[dict[str, Any]]:
    """Read escalated clusters from debugger_session_test_summary.json."""
    if not DEBUGGER_SESSION_TEST_SUMMARY.exists():
        return []

    try:
        report = _load_json(DEBUGGER_SESSION_TEST_SUMMARY, {})
        if isinstance(report, dict):
            escalated = report.get("escalated", [])
            return escalated if isinstance(escalated, list) else []
    except Exception:
        pass

    return []


def _match_clusters_to_files(
    changed_files: list[str],
    escalated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    matched_clusters: list[dict[str, Any]],
    root_cause: str,
    spec_version: str,
    run_context: str,
) -> str:
    """Build a markdown Pattern entry for archivist_knowledge_log.md."""
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
        f"**Source:** Human fix capture\n\n"
        f"**Run context:** {run_context}\n\n"
        f"**Files changed by human:**\n{files_str}\n\n"
        + (f"**Root cause:** {root_cause}\n\n" if root_cause else "")
        + clusters_str
        + f"\n**Diff preview (added lines):**\n```diff\n{diff_preview}\n```\n\n"
        f"---\n\n"
    )


def _append_knowledge_log(entry: str, dry_run: bool) -> None:
    if dry_run:
        print(f"\n[DRY RUN] Would append to {ARCHIVIST_KNOWLEDGE_LOG}:")
        print(indent(entry[:300] + ("…" if len(entry) > 300 else ""), "  "))
        return

    ARCHIVIST_KNOWLEDGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    header = "# Archivist Knowledge Log\n\n"
    if not ARCHIVIST_KNOWLEDGE_LOG.exists():
        ARCHIVIST_KNOWLEDGE_LOG.write_text(header, encoding="utf-8")
        _track_write(ARCHIVIST_KNOWLEDGE_LOG)

    existing = _load_text(ARCHIVIST_KNOWLEDGE_LOG)
    ARCHIVIST_KNOWLEDGE_LOG.write_text(existing.rstrip() + "\n\n" + entry, encoding="utf-8")
    _track_write(ARCHIVIST_KNOWLEDGE_LOG)

    print(f"  ✓ Appended pattern to {ARCHIVIST_KNOWLEDGE_LOG}")


def capture_human_fix(dry_run: bool) -> None:
    """Capture human intervention via git diff, update archivist knowledge."""
    print("\n[13] HUMAN FIX CAPTURE MODE")

    ctx = _load_mini_context()
    scope = ctx.get("scope", "full")

    diff_paths = ["src/"]

    if scope == "mini":
        mini_files = _mini_target_files(ctx)
        if mini_files:
            diff_paths = mini_files

    print(f"[13] Scope: {scope}")
    print(f"[13] Scanning git diff for: {', '.join(diff_paths)}")

    diff = _git_diff_for_paths(diff_paths)
    if not diff:
        print("[13] No staged/unstaged changes found in selected path(s).")
        print("     Stage your changes first, or check git status.")
        return

    if diff.startswith("(git diff failed:"):
        print(f"[13] {diff}")
        return

    changed_files = _parse_changed_files_from_diff(diff)
    if not changed_files:
        print("[13] Could not parse changed files from diff.")
        return

    if scope == "mini":
        allowed = set(_mini_target_files(ctx))
        outside = [f for f in changed_files if allowed and f not in allowed]
        if outside:
            print("[13] WARNING: changed files outside mini target scope:")
            for f in outside:
                print(f"  ! {f}")

    print("[13] Changed files detected:")
    for f in changed_files:
        print(f"  {f}")

    escalated = _load_escalated_clusters()
    matched = _match_clusters_to_files(changed_files, escalated)

    if matched:
        print(f"\n[13] Matched {len(matched)} escalated cluster(s) to your fix:")
        for c in matched:
            print(f"  * {c.get('cluster')} — {c.get('note', '')}")
    else:
        print(
            "\n[13] No matching escalated clusters found "
            "(fix may be proactive or from judge review)."
        )

    print("\n[13] Briefly describe the root cause of the bug you fixed.")
    print("     (Press Enter to skip)")

    try:
        root_cause = input("  Root cause: ").strip()
    except (EOFError, KeyboardInterrupt):
        root_cause = ""

    spec_version = _extract_spec_version()

    pattern = _build_knowledge_pattern(
        diff=diff,
        changed_files=changed_files,
        matched_clusters=matched,
        root_cause=root_cause,
        spec_version=spec_version,
        run_context=_run_context_summary(ctx),
    )

    _append_knowledge_log(pattern, dry_run)

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

        _apply_knowledge_log(regression_note, dry_run=False)
        print(f"  ✓ Regression note appended to {ARCHIVIST_KNOWLEDGE_LOG}")

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
        append_curation_log(record)

    print("\n[13] Human fix captured. Future runs will use this pattern.")


# ════════════════════════════════════════════════════════════════════════════
# Knowledge pattern helpers
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
        f"**Inject into:** downstream planning / implementation / debugging prompts "
        f"(do NOT reintroduce)\n\n"
        f"---\n\n"
    )


# ════════════════════════════════════════════════════════════════════════════
# Load & parse judge verdict
# ════════════════════════════════════════════════════════════════════════════

def _load_verdict() -> dict[str, Any]:
    if not JUDGE_SESSION_VERDICT_RAW.exists():
        print(f"[13] ERROR: {JUDGE_SESSION_VERDICT_RAW} not found.", file=sys.stderr)
        sys.exit(1)

    raw_obj = _load_json(JUDGE_SESSION_VERDICT_RAW, {})
    if not isinstance(raw_obj, dict):
        print(f"[13] ERROR: invalid judge raw shape: {JUDGE_SESSION_VERDICT_RAW}", file=sys.stderr)
        sys.exit(1)

    raw = raw_obj.get("response", "")
    if not isinstance(raw, str) or not raw.strip():
        print("[13] ERROR: judge raw has empty response.", file=sys.stderr)
        sys.exit(1)

    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group())


def _load_previous_curation_summary() -> dict[str, Any]:
    """Load existing curation records, not used for decisions."""
    if not ARCHIVIST_CURATION_LOG.exists():
        return {}

    try:
        logs = _load_json(ARCHIVIST_CURATION_LOG, [])
        if isinstance(logs, list):
            return {"total_records": len(logs)}
    except Exception:
        pass

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

_ARCHITECTURE_PATTERN_KEYWORDS = {
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
            "MANUAL ACTION REQUIRED — update canonical spec:\n"
            f"Finding: {finding}\n"
            "Suggestion: define behaviour explicitly in the relevant section."
        )
        return ACTION_SPEC_BUMP, "specwright_spec_<slug>.md (manual)", content

    if any(kw in text for kw in _SPEC_EDGE_KEYWORDS):
        content = (
            f"## Edge case: {finding[:80]}\n\n"
            f"Behaviour: define exact behaviour for: {finding}\n"
        )
        return (
            ACTION_SPEC_GAP,
            "artifacts_<slug>/knowledge/current/archivist_spec_gaps.md",
            content,
        )

    if any(kw in text for kw in _ARCHITECTURE_PATTERN_KEYWORDS) or severity == "blocking":
        content = finding
        return (
            ACTION_ARCHITECTURE_PATTERN,
            "artifacts_<slug>/knowledge/current/archivist_knowledge_log.md",
            content,
        )

    content = f"- {finding}"
    return (
        ACTION_FINDING_PATTERN,
        "artifacts_<slug>/knowledge/current/archivist_knowledge_log.md",
        content,
    )


# ════════════════════════════════════════════════════════════════════════════
# Apply functions
# ════════════════════════════════════════════════════════════════════════════

def _apply_spec_gap(content: str, dry_run: bool) -> None:
    if dry_run:
        print(
            f"  [DRY RUN] Would append to {ARCHIVIST_SPEC_GAPS}:\n"
            f"{indent(content, '    ')}"
        )
        return

    ARCHIVIST_SPEC_GAPS.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Archivist Spec Gaps\n"
        "_Edge cases and spec gaps surfaced by judge/human review._\n\n"
    )

    if not ARCHIVIST_SPEC_GAPS.exists():
        ARCHIVIST_SPEC_GAPS.write_text(header, encoding="utf-8")
        _track_write(ARCHIVIST_SPEC_GAPS)

    existing = _load_text(ARCHIVIST_SPEC_GAPS)
    ARCHIVIST_SPEC_GAPS.write_text(existing.rstrip() + "\n\n" + content + "\n", encoding="utf-8")
    _track_write(ARCHIVIST_SPEC_GAPS)

    print(f"  ✓ Appended to {ARCHIVIST_SPEC_GAPS}")


def _apply_knowledge_log(content: str, dry_run: bool) -> None:
    block = f"\n{content}\n"

    if dry_run:
        print(
            f"  [DRY RUN] Would append to {ARCHIVIST_KNOWLEDGE_LOG}:\n"
            f"{indent(block, '    ')}"
        )
        return

    ARCHIVIST_KNOWLEDGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    header = "# Archivist Knowledge Log\n_Accumulated architecture decisions, bug patterns, and lessons learned._\n"
    if not ARCHIVIST_KNOWLEDGE_LOG.exists():
        ARCHIVIST_KNOWLEDGE_LOG.write_text(header, encoding="utf-8")
        _track_write(ARCHIVIST_KNOWLEDGE_LOG)

    existing = _load_text(ARCHIVIST_KNOWLEDGE_LOG)
    ARCHIVIST_KNOWLEDGE_LOG.write_text(existing.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")
    _track_write(ARCHIVIST_KNOWLEDGE_LOG)

    print(f"  ✓ Appended to {ARCHIVIST_KNOWLEDGE_LOG}")


def _print_spec_bump_advice(content: str) -> None:
    print(f"\n  {'!' * 50}")
    print("  MANUAL SPEC EDIT REQUIRED")
    print(f"  {'!' * 50}")
    print(indent(content, "  "))
    print()


APPLY_MAP = {
    ACTION_SPEC_GAP: _apply_spec_gap,
    ACTION_ARCHITECTURE_PATTERN: _apply_knowledge_log,
    ACTION_FINDING_PATTERN: _apply_knowledge_log,
    ACTION_KNOWLEDGE: _apply_knowledge_log,
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
        "  Actions: [y] accept  [s] skip  [g] spec_gap  "
        "[a] architecture_pattern  [f] finding_pattern  [k] knowledge_log"
    )

    try:
        choice = input("  Choice [y]: ").strip().lower() or "y"
    except (EOFError, KeyboardInterrupt):
        choice = "y"

    if choice == "s":
        return ACTION_SKIP, content, True
    if choice == "g":
        return ACTION_SPEC_GAP, content, True
    if choice == "a":
        return ACTION_ARCHITECTURE_PATTERN, content, True
    if choice == "f":
        return ACTION_FINDING_PATTERN, content, True
    if choice == "k":
        return ACTION_KNOWLEDGE, content, True

    return action, content, True


# ════════════════════════════════════════════════════════════════════════════
# Show knowledge
# ════════════════════════════════════════════════════════════════════════════

def show_knowledge() -> None:
    if not ARCHIVIST_KNOWLEDGE_LOG.exists():
        print("[13] No archivist knowledge log yet.")
    else:
        print(_load_text(ARCHIVIST_KNOWLEDGE_LOG))

    if ARCHIVIST_SPEC_GAPS.exists():
        print("\n── Spec gaps ──")
        print(_load_text(ARCHIVIST_SPEC_GAPS))

    if ARCHIVIST_CURATION_LOG.exists():
        try:
            logs = _load_json(ARCHIVIST_CURATION_LOG, [])
        except Exception:
            logs = []

        if not isinstance(logs, list):
            logs = []

        print(f"\n── Curation log: {len(logs)} total records ──")

        for r in logs[-5:]:
            if not isinstance(r, dict):
                continue

            mode = r.get("mode", "unknown")
            ts = r.get("timestamp", "")[:10] if "timestamp" in r else "?"
            scope = r.get("scope", "?")

            if mode == "human_fix_capture":
                print(
                    f"  {ts}  {mode}  scope={scope}  "
                    f"{len(r.get('changed_files', []))} file(s)  "
                    f"{str(r.get('root_cause', ''))[:60]}"
                )
            else:
                print(
                    f"  {ts}  {mode}  scope={scope}  "
                    f"{r.get('judge_verdict', '?')}"
                )


# ════════════════════════════════════════════════════════════════════════════
# Curation log
# ════════════════════════════════════════════════════════════════════════════

def append_curation_log(record: dict[str, Any]) -> None:
    ARCHIVIST_CURATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing_log: list[Any] = []
    if ARCHIVIST_CURATION_LOG.exists():
        try:
            loaded = _load_json(ARCHIVIST_CURATION_LOG, [])
            if isinstance(loaded, list):
                existing_log = loaded
        except Exception:
            existing_log = []

    existing_log.append(record)
    ARCHIVIST_CURATION_LOG.write_text(
        json.dumps(existing_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _track_write(ARCHIVIST_CURATION_LOG)

    print(f"\n[13] Curation log appended to {ARCHIVIST_CURATION_LOG}")


# ════════════════════════════════════════════════════════════════════════════
# Main judge-driven mode
# ════════════════════════════════════════════════════════════════════════════

def run_judge_driven(args: argparse.Namespace) -> int:
    interactive = not args.accept_all and not args.dry_run

    verdict = _load_verdict()
    _ = _load_previous_curation_summary()

    # Optional context reads for richer access trace and future extension.
    _ = _load_text(JUDGE_VERDICT_SUMMARY)
    _ = _load_json(PLANNER_FULL_PLAN, {})
    _ = _load_text(PATCHER_FINDINGS_SNAPSHOT)

    run_ctx = _load_mini_context()
    scope = run_ctx.get("scope", "full")
    run_summary = _run_context_summary(run_ctx)

    if verdict.get("verdict") not in ("NEEDS_REVISION", "APPROVED_WITH_NOTES"):
        print(
            f"[13] Judge verdict is {verdict.get('verdict')} — "
            "no knowledge update needed for APPROVED runs."
        )
        return 0

    print(f"[13] Knowledge update for verdict: {verdict['verdict']}")
    print(f"[13] Run context: {run_summary}")
    print(f"[13] Dry-run: {args.dry_run}  |  Interactive: {interactive}")

    sections = verdict.get("sections", {})
    if not isinstance(sections, dict):
        sections = {}

    section_notes_map = {
        k: v.get("notes", "")
        for k, v in sections.items()
        if isinstance(v, dict)
    }

    all_findings: list[tuple[str, str]] = []

    if not args.only_non_blocking:
        for desc in verdict.get("blocking_issues", []):
            all_findings.append((str(desc), "blocking"))

    if not args.only_blocking:
        for desc in verdict.get("non_blocking_notes", []):
            all_findings.append((str(desc), "non_blocking"))

        gaps_notes = section_notes_map.get("gaps_risks", "")
        if gaps_notes:
            items = re.split(r"\d+\)", gaps_notes)
            for item in items[1:]:
                item = item.strip()
                if item and len(item) > 20:
                    all_findings.append((item, "gap_risk"))

    if not all_findings:
        print("[13] No findings to process.")
        return 0

    print(f"\n[13] Processing {len(all_findings)} finding(s) …\n")

    spec_version = _extract_spec_version()

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
            _append_knowledge_log(kb_entry, dry_run=args.dry_run)

        if final_action == ACTION_KNOWLEDGE:
            kb_entry = _blocking_to_knowledge_pattern(
                finding,
                spec_version,
                run_summary,
            )
            _append_knowledge_log(kb_entry, dry_run=args.dry_run)
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
        append_curation_log(log_entry)

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
    print(f"  Spec bumps needed  : {spec_bumps}  ← edit canonical spec manually")

    if spec_bumps:
        print("\n  Spec bumps detected — edit canonical spec manually before next run.")

    return 0


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: project env must be configured before ensure_dirs().
    ensure_dirs()

    exit_code = 0

    try:
        if args.show_knowledge:
            show_knowledge()
            return

        if args.capture_human_fix:
            capture_human_fix(dry_run=args.dry_run)
            return

        exit_code = run_judge_driven(args)

    except Exception as exc:
        print(f"[13][error] Archivist failed: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
