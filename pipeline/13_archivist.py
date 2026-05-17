"""
pipeline/13_archivist.py
========================
Step 13 — Long-term knowledge distillation after human review.

Two modes:

  A) JUDGE-DRIVEN:
     Run after reviewing judge/verdict_summary.md / judge/verdict_raw.json.
     Processes judge findings and writes/updates:
       - archivist/knowledge_log.md
       - archivist/spec_gaps.md
       - archivist/curation_log.json

  B) HUMAN-FIX CAPTURE:
     Run after you manually fix code that AI couldn't fix.
     Uses `git diff` to capture what you changed, links it to escalated clusters,
     and distills a Pattern entry into archivist/knowledge_log.md.
     On next run, knowledge_log.md is injected into downstream prompts.

Mini-aware behaviour:
  - Reads planner/mini_plan.json (includes impact field),
    executor/manifest.json.
  - Human-fix capture diffs mini target files instead of hardcoding output/src/.
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
  artifacts_<slug>/archivist/knowledge_log.md     (append)
  artifacts_<slug>/archivist/spec_gaps.md         (append)
  artifacts_<slug>/archivist/curation_log.json    (append)

Reads
─────
  artifacts_<slug>/judge/verdict_raw.json
  artifacts_<slug>/judge/verdict_summary.md
  artifacts_<slug>/planner/full_plan.json
  artifacts_<slug>/planner/mini_plan.json
  artifacts_<slug>/executor/manifest.json
  artifacts_<slug>/debugger/test_summary.json
  artifacts_<slug>/clarificator/session.json
  artifacts_<slug>/enricher/enriched_prompt.md
  artifacts_<slug>/patcher/attempt_log.json       (last entry, replaces snapshot)
  artifacts_<slug>/archivist/knowledge_log.md
  artifacts_<slug>/archivist/spec_gaps.md
  artifacts_<slug>/spec/specwright_spec_<slug>.md

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
# OWNS  : artifacts_<slug>/archivist/knowledge_log.md
#         artifacts_<slug>/archivist/spec_gaps.md
#         artifacts_<slug>/archivist/curation_log.json
# READS : artifacts_<slug>/judge/verdict_raw.json
#         artifacts_<slug>/judge/verdict_summary.md
#         artifacts_<slug>/planner/full_plan.json
#         artifacts_<slug>/planner/mini_plan.json
#         artifacts_<slug>/executor/manifest.json
#         artifacts_<slug>/debugger/test_summary.json
#         artifacts_<slug>/clarificator/session.json
#         artifacts_<slug>/enricher/enriched_prompt.md
#         artifacts_<slug>/patcher/attempt_log.json
#         artifacts_<slug>/archivist/knowledge_log.md
#         artifacts_<slug>/archivist/spec_gaps.md
#         artifacts_<slug>/spec/specwright_spec_<slug>.md

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    ARCHIVIST_CURATION_LOG,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    CLARIFIED_REQ,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    ENRICHER_OVERWRITE_PROMPT,
    EXECUTOR_OVERWRITE_MANIFEST,
    JUDGE_OVERWRITE_VERDICT_RAW,
    JUDGE_VERDICT_SUMMARY,
    PATCHER_ATTEMPT_LOG,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    artifact_root,
    ensure_dirs,
    get_project_name,
    get_project_slug,
    get_spec_path,
)
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

ROLE = "archivist"

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
# CLI / project/session setup
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
        track_read(path)
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f"[13][warn] Could not parse JSON {path}: {exc}", file=sys.stderr)
        return default


def _load_text(path: Any, default: str = "") -> str:
    if not path.exists():
        return default
    try:
        track_read(path)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"[13][warn] Could not read {path}: {exc}", file=sys.stderr)
        return default


def _load_impl_record() -> dict[str, Any]:
    rec = _load_json(EXECUTOR_OVERWRITE_MANIFEST, {})
    return rec if isinstance(rec, dict) else {}


def _load_test_report() -> dict[str, Any]:
    rec = _load_json(DEBUGGER_OVERWRITE_TEST_SUMMARY, {})
    return rec if isinstance(rec, dict) else {}


def _current_scope() -> str:
    rec = _load_impl_record()
    scope = rec.get("scope")
    if scope in {"full", "mini"}:
        return str(scope)

    report = _load_test_report()
    scope = report.get("scope")
    if scope in {"full", "mini"}:
        return str(scope)

    if PLANNER_MINI_PLAN.exists():
        return "mini"

    return "full"


def _extract_file_list(value: Any) -> list[str]:
    files: list[str] = []

    if not isinstance(value, list):
        return files

    for item in value:
        if isinstance(item, str):
            files.append(item)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("file_path") or item.get("file")
            if isinstance(path, str):
                files.append(path)

    seen: set[str] = set()
    out: list[str] = []
    for fp in files:
        fp = fp.strip().lstrip("./")
        if fp and fp not in seen:
            seen.add(fp)
            out.append(fp)

    return out


def _load_mini_context() -> dict[str, Any]:
    """Load mini planning/analysis/implementation context if present."""
    plan_mini = _load_json(PLANNER_MINI_PLAN, {})
    if not isinstance(plan_mini, dict):
        plan_mini = {}
    analysis_mini = plan_mini.get("impact", {})
    if not isinstance(analysis_mini, dict):
        analysis_mini = {}
    impl_record = _load_impl_record()

    return {
        "scope": _current_scope(),
        "clarified_requirement": _load_text(CLARIFIED_REQ).strip(),
        "enriched_prompt": _load_text(ENRICHER_OVERWRITE_PROMPT).strip(),
        "plan_mini": plan_mini,
        "analysis_mini": analysis_mini,
        "impl_record": impl_record,
    }


def _mini_target_files(ctx: dict[str, Any] | None = None) -> list[str]:
    """Return allowed/expected files for mini runs."""
    ctx = ctx or _load_mini_context()

    plan = ctx.get("plan_mini") or {}
    impl = ctx.get("impl_record") or {}

    out: list[str] = []
    if isinstance(plan, dict):
        out.extend(_extract_file_list(plan.get("target_files", [])))
    if isinstance(impl, dict):
        out.extend(_extract_file_list(impl.get("files", [])))

    seen: set[str] = set()
    unique: list[str] = []
    for fp in out:
        fp = fp.strip().lstrip("./")
        if fp and fp not in seen:
            seen.add(fp)
            unique.append(fp)

    return unique


def _run_context_summary(ctx: dict[str, Any] | None = None) -> str:
    ctx = ctx or _load_mini_context()
    scope = ctx.get("scope", "full")

    if scope != "mini":
        return "scope=full"

    plan = ctx.get("plan_mini") or {}
    analysis = ctx.get("analysis_mini") or {}
    target_files = _mini_target_files(ctx)

    task_summary = ""
    if isinstance(plan, dict):
        task_summary = (
            plan.get("task_summary")
            or plan.get("summary")
            or plan.get("title")
            or plan.get("goal")
            or ""
        )

    warnings = analysis.get("warnings") if isinstance(analysis, dict) else []
    conflicts = analysis.get("conflicts") if isinstance(analysis, dict) else []
    recommendations = analysis.get("recommendations") if isinstance(analysis, dict) else []

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
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1)

    return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# Human fix capture
# ════════════════════════════════════════════════════════════════════════════

def _git_diff_for_paths(paths: list[str]) -> str:
    """
    Get unstaged + staged diff for selected paths.

    Note:
      This assumes artifact_root() is a git worktree root or inside one.
      If generated artifacts are not in a git repo, capture mode may not work.
    """
    try:
        root = artifact_root()
        args_paths = paths or ["output/src/"]

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
    except Exception as exc:
        return f"(git diff failed: {exc})"


def _parse_changed_files_from_diff(diff: str) -> list[str]:
    """Extract changed files from a unified diff."""
    return re.findall(r"^\+\+\+ b/([^\n]+)", diff, re.MULTILINE)


def _load_escalated_clusters() -> list[dict[str, Any]]:
    """Read escalated clusters from debugger_overwrite_test_summary.json."""
    if not DEBUGGER_OVERWRITE_TEST_SUMMARY.exists():
        return []

    report = _load_json(DEBUGGER_OVERWRITE_TEST_SUMMARY, {})
    if isinstance(report, dict):
        escalated = report.get("escalated", [])
        return escalated if isinstance(escalated, list) else []

    return []


def _match_clusters_to_files(
    changed_files: list[str],
    escalated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find escalated clusters whose file/src_file/path matches changed files."""
    matched: list[dict[str, Any]] = []
    changed = set(changed_files)

    for cluster in escalated:
        if not isinstance(cluster, dict):
            continue

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
    files_str = "\n".join(f"- `{fp}`" for fp in changed_files)

    clusters_str = ""
    if matched_clusters:
        clusters_str = "\n**AI-escalated clusters fixed by human:**\n"
        for cluster in matched_clusters:
            note = cluster.get("note", "")
            attempts = cluster.get("attempts", "?")
            clusters_str += (
                f"- `{cluster.get('cluster', '?')}` "
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
        f"**Project:** {get_project_name()} (`{get_project_slug()}`)\n\n"
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
        track_write(ARCHIVIST_KNOWLEDGE_LOG)

    existing = _load_text(ARCHIVIST_KNOWLEDGE_LOG)
    ARCHIVIST_KNOWLEDGE_LOG.write_text(
        existing.rstrip() + "\n\n" + entry,
        encoding="utf-8",
    )
    track_write(ARCHIVIST_KNOWLEDGE_LOG)

    print(f"  ✓ Appended pattern to {ARCHIVIST_KNOWLEDGE_LOG}")


def capture_human_fix(dry_run: bool) -> None:
    """Capture human intervention via git diff, update archivist knowledge."""
    print("\n[13] HUMAN FIX CAPTURE MODE")

    ctx = _load_mini_context()
    scope = ctx.get("scope", "full")

    diff_paths = ["output/src/"]

    if scope == "mini":
        mini_files = _mini_target_files(ctx)
        if mini_files:
            diff_paths = mini_files

    print(f"[13] Project: {get_project_name()} ({get_project_slug()})")
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
        outside = [fp for fp in changed_files if allowed and fp not in allowed]
        if outside:
            print("[13] WARNING: changed files outside mini target scope:")
            for fp in outside:
                print(f"  ! {fp}")

    print("[13] Changed files detected:")
    for fp in changed_files:
        print(f"  {fp}")

    escalated = _load_escalated_clusters()
    matched = _match_clusters_to_files(changed_files, escalated)

    if matched:
        print(f"\n[13] Matched {len(matched)} escalated cluster(s) to your fix:")
        for cluster in matched:
            print(f"  * {cluster.get('cluster')} — {cluster.get('note', '')}")
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

        for cluster in matched:
            fixed_file = (
                cluster.get("src_file")
                or cluster.get("file")
                or cluster.get("path")
                or "?"
            )
            regression_note += (
                f"- Fixed `{fixed_file}`: "
                f"{root_cause or cluster.get('note', '')}\n"
            )

        _apply_knowledge_log(regression_note, dry_run=False)
        print(f"  ✓ Regression note appended to {ARCHIVIST_KNOWLEDGE_LOG}")

    if not dry_run:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "human_fix_capture",
            "project": get_project_name(),
            "project_slug": get_project_slug(),
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
        f"**Project:** {get_project_name()} (`{get_project_slug()}`)\n\n"
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
    if not JUDGE_OVERWRITE_VERDICT_RAW.exists():
        print(f"[13] ERROR: {JUDGE_OVERWRITE_VERDICT_RAW} not found.", file=sys.stderr)
        print("[13] Run 11_judge.py first.", file=sys.stderr)
        sys.exit(1)

    raw_obj = _load_json(JUDGE_OVERWRITE_VERDICT_RAW, {})
    if not isinstance(raw_obj, dict):
        print(f"[13] ERROR: invalid judge raw shape: {JUDGE_OVERWRITE_VERDICT_RAW}", file=sys.stderr)
        sys.exit(1)

    raw = raw_obj.get("response", "")
    if not isinstance(raw, str) or not raw.strip():
        print("[13] ERROR: judge_overwrite_verdict_raw.json has empty response.", file=sys.stderr)
        sys.exit(1)

    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())

    if not isinstance(parsed, dict):
        print("[13] ERROR: judge verdict JSON top-level is not an object.", file=sys.stderr)
        sys.exit(1)

    return parsed


def _load_previous_curation_summary() -> dict[str, Any]:
    """Load existing curation records, not used for decisions."""
    if not ARCHIVIST_CURATION_LOG.exists():
        return {}

    loaded = _load_json(ARCHIVIST_CURATION_LOG, {})
    if isinstance(loaded, dict):
        entries = loaded.get("entries", [])
    elif isinstance(loaded, list):
        entries = loaded
    else:
        entries = []

    return {"total_records": len(entries)}


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

    if any(keyword in text for keyword in _SPEC_BUMP_KEYWORDS):
        content = (
            "MANUAL ACTION REQUIRED — update canonical spec:\n"
            f"Finding: {finding}\n"
            "Suggestion: define behaviour explicitly in the relevant section."
        )
        return ACTION_SPEC_BUMP, "specwright_spec_<slug>.md (manual)", content

    if any(keyword in text for keyword in _SPEC_EDGE_KEYWORDS):
        content = (
            f"## Edge case: {finding[:80]}\n\n"
            f"Behaviour: define exact behaviour for: {finding}\n"
        )
        return (
            ACTION_SPEC_GAP,
            "archivist/spec_gaps.md",
            content,
        )

    if any(keyword in text for keyword in _ARCHITECTURE_PATTERN_KEYWORDS) or severity == "blocking":
        content = finding
        return (
            ACTION_ARCHITECTURE_PATTERN,
            "archivist/knowledge_log.md",
            content,
        )

    content = f"- {finding}"
    return (
        ACTION_FINDING_PATTERN,
        "archivist/knowledge_log.md",
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
        track_write(ARCHIVIST_SPEC_GAPS)

    existing = _load_text(ARCHIVIST_SPEC_GAPS)
    ARCHIVIST_SPEC_GAPS.write_text(
        existing.rstrip() + "\n\n" + content + "\n",
        encoding="utf-8",
    )
    track_write(ARCHIVIST_SPEC_GAPS)

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
        track_write(ARCHIVIST_KNOWLEDGE_LOG)

    existing = _load_text(ARCHIVIST_KNOWLEDGE_LOG)
    ARCHIVIST_KNOWLEDGE_LOG.write_text(
        existing.rstrip() + "\n\n" + block.strip() + "\n",
        encoding="utf-8",
    )
    track_write(ARCHIVIST_KNOWLEDGE_LOG)

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
    print(f"[13] Project: {get_project_name()} ({get_project_slug()})")

    if not ARCHIVIST_KNOWLEDGE_LOG.exists():
        print("[13] No archivist knowledge log yet.")
    else:
        print(_load_text(ARCHIVIST_KNOWLEDGE_LOG))

    if ARCHIVIST_SPEC_GAPS.exists():
        print("\n── Spec gaps ──")
        print(_load_text(ARCHIVIST_SPEC_GAPS))

    if ARCHIVIST_CURATION_LOG.exists():
        log_data = _load_json(ARCHIVIST_CURATION_LOG, {})
        if isinstance(log_data, dict):
            logs = log_data.get("entries", [])
        elif isinstance(log_data, list):
            logs = log_data  # migrate legacy bare-list
        else:
            logs = []

        print(f"\n── Curation log: {len(logs)} total records ──")

        for record in logs[-5:]:
            if not isinstance(record, dict):
                continue

            mode = record.get("mode", "unknown")
            ts = record.get("timestamp", "")[:10] if "timestamp" in record else "?"
            scope = record.get("scope", "?")

            if mode == "human_fix_capture":
                print(
                    f"  {ts}  {mode}  scope={scope}  "
                    f"{len(record.get('changed_files', []))} file(s)  "
                    f"{str(record.get('root_cause', ''))[:60]}"
                )
            else:
                print(
                    f"  {ts}  {mode}  scope={scope}  "
                    f"{record.get('judge_verdict', '?')}"
                )


# ════════════════════════════════════════════════════════════════════════════
# Curation log
# ════════════════════════════════════════════════════════════════════════════

def append_curation_log(record: dict[str, Any]) -> None:
    ARCHIVIST_CURATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing_entries: list[Any] = []
    if ARCHIVIST_CURATION_LOG.exists():
        loaded = _load_json(ARCHIVIST_CURATION_LOG, {})
        if isinstance(loaded, dict):
            existing_entries = loaded.get("entries", [])
        elif isinstance(loaded, list):
            # migrate legacy bare-list format
            existing_entries = loaded

    existing_entries.append(record)
    ARCHIVIST_CURATION_LOG.write_text(
        json.dumps({"entries": existing_entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(ARCHIVIST_CURATION_LOG)

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

    # Read last entry of patcher attempt log (replaces removed snapshot file).
    _patcher_log = _load_json(PATCHER_ATTEMPT_LOG, {})
    _patcher_entries = _patcher_log.get("entries", []) if isinstance(_patcher_log, dict) else []
    _ = _patcher_entries[-1] if _patcher_entries else {}

    run_ctx = _load_mini_context()
    scope = run_ctx.get("scope", "full")
    run_summary = _run_context_summary(run_ctx)

    if verdict.get("verdict") not in ("NEEDS_REVISION", "APPROVED_WITH_NOTES"):
        print(
            f"[13] Judge verdict is {verdict.get('verdict')} — "
            "no knowledge update needed for APPROVED runs."
        )
        return 0

    print(f"[13] Project: {get_project_name()} ({get_project_slug()})")
    print(f"[13] Knowledge update for verdict: {verdict['verdict']}")
    print(f"[13] Run context: {run_summary}")
    print(f"[13] Dry-run: {args.dry_run}  |  Interactive: {interactive}")

    sections = verdict.get("sections", {})
    if not isinstance(sections, dict):
        sections = {}

    section_notes_map = {
        key: value.get("notes", "")
        for key, value in sections.items()
        if isinstance(value, dict)
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
            first_words = finding.lower().split()[:5]
            if first_words and any(word in notes.lower() for word in first_words):
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

        knowledge_action = KnowledgeAction(
            finding=finding,
            severity=severity,
            action=final_action,
            target=suggested[1],
            content=final_content,
            human_approved=approved,
        )
        actions.append(knowledge_action)

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
            "project": get_project_name(),
            "project_slug": get_project_slug(),
            "scope": scope,
            "run_context": run_summary,
            "judge_verdict": verdict.get("verdict"),
            "findings_total": len(all_findings),
            "actions_taken": sum(
                1
                for action in actions
                if action.human_approved and action.action != ACTION_SKIP
            ),
            "skipped": sum(1 for action in actions if action.action == ACTION_SKIP),
            "spec_bumps": sum(1 for action in actions if action.action == ACTION_SPEC_BUMP),
            "details": [asdict(action) for action in actions],
        }
        append_curation_log(log_entry)

    applied = sum(
        1
        for action in actions
        if action.human_approved and action.action not in (ACTION_SKIP, ACTION_SPEC_BUMP)
    )
    skipped = sum(1 for action in actions if action.action == ACTION_SKIP)
    spec_bumps = sum(1 for action in actions if action.action == ACTION_SPEC_BUMP)

    print(f"\n{'=' * 50}")
    print("  KNOWLEDGE UPDATE SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Project            : {get_project_name()} ({get_project_slug()})")
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
        print_artifact_summary("[13]")
        prompt_next_step(ROLE, prefix="[13]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
