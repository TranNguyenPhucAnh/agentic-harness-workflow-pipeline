"""
pipeline/10_reporter.py
=======================
Step 10 — Aggregate pipeline artifacts into reports/reporter_execution_summary.md.

Session model:
  When PIPELINE_SESSION is set, session-local artifacts resolve under:
    artifacts_<slug>/sessions/<NNN>/

  When PIPELINE_SESSION is not set, paths.py falls back to the old layout:
    artifacts_<slug>/

Supports both:
  - FULL flow:
      planner_full_execution_plan.json
      scaffolder_codebase_skeleton.json
      spectracker_overwrite_version_delta.json
      debugger_overwrite_test_summary.json
      executor_overwrite_manifest.json

  - MINI flow:
      planner_mini_execution_plan.json
      planner_mini_impact_analysis.json
      clarificator_requirement_synthesis.md
      enricher_overwrite_enriched_prompt.md
      executor_overwrite_manifest.json
      debugger_overwrite_test_summary.json

Writes:
  artifacts_<slug>/sessions/<NNN>/reports/reporter_execution_summary.md
  or, without PIPELINE_SESSION:
  artifacts_<slug>/reports/reporter_execution_summary.md

Reads:
  artifacts_<slug>/sessions/<NNN>/state/planner_full_execution_plan.json
  artifacts_<slug>/sessions/<NNN>/state/planner_mini_execution_plan.json
  artifacts_<slug>/sessions/<NNN>/state/planner_mini_impact_analysis.json
  artifacts_<slug>/sessions/<NNN>/execution/debugger_overwrite_test_summary.json
  artifacts_<slug>/sessions/<NNN>/state/scaffolder_codebase_skeleton.json
  artifacts_<slug>/sessions/<NNN>/cache/spectracker_overwrite_version_delta.json
  artifacts_<slug>/sessions/<NNN>/execution/executor_overwrite_manifest.json
  artifacts_<slug>/sessions/<NNN>/reports/judge_verdict_summary.md
  artifacts_<slug>/sessions/<NNN>/state/clarificator_requirement_synthesis.md
  artifacts_<slug>/sessions/<NNN>/execution/enricher_overwrite_enriched_prompt.md

Direct execution:
  python 10_reporter.py --project my-app
  PIPELINE_PROJECT=my-app python 10_reporter.py
  PIPELINE_PROJECT=my-app PIPELINE_SESSION=001 python 10_reporter.py

At the end of each run, prints:
  - artifacts read
  - artifacts created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# === WRITE AUTHORITY: reporter ===
# OWNS  : artifacts_<slug>/sessions/<NNN>/reports/reporter_execution_summary.md
# READS : artifacts_<slug>/sessions/<NNN>/state/planner_full_execution_plan.json
#         artifacts_<slug>/sessions/<NNN>/state/planner_mini_execution_plan.json
#         artifacts_<slug>/sessions/<NNN>/state/planner_mini_impact_analysis.json
#         artifacts_<slug>/sessions/<NNN>/execution/debugger_overwrite_test_summary.json
#         artifacts_<slug>/sessions/<NNN>/state/scaffolder_codebase_skeleton.json
#         artifacts_<slug>/sessions/<NNN>/cache/spectracker_overwrite_version_delta.json
#         artifacts_<slug>/sessions/<NNN>/execution/executor_overwrite_manifest.json
#         artifacts_<slug>/sessions/<NNN>/reports/judge_verdict_summary.md
#         artifacts_<slug>/sessions/<NNN>/state/clarificator_requirement_synthesis.md
#         artifacts_<slug>/sessions/<NNN>/execution/enricher_overwrite_enriched_prompt.md

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    CLARIFIED_REQ,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    ENRICHER_OVERWRITE_PROMPT,
    EXECUTOR_OVERWRITE_MANIFEST,
    JUDGE_VERDICT_SUMMARY,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_IMPACT,
    PLANNER_MINI_PLAN,
    REPORTER_EXECUTION_SUMMARY,
    SCAFFOLD_JSON,
    SPECTRACKER_VERSION_DELTA,
    ensure_dirs,
    get_project_name,
    get_project_slug,
    get_session_id,
)


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
    print("[10] Artifacts read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[10]   READ  {item}")
    else:
        print("[10]   READ  (none)")

    print("[10] Artifacts created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[10]   WRITE {item}")
    else:
        print("[10]   WRITE (none)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI / project setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="10_reporter.py",
        description="Aggregate pipeline artifacts into reports/reporter_execution_summary.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 10_reporter.py --project my-app
              PIPELINE_PROJECT=my-app python 10_reporter.py
              PIPELINE_PROJECT=my-app PIPELINE_SESSION=001 python 10_reporter.py
        """),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Optional session id for direct execution. Sets PIPELINE_SESSION. "
            "Example: --session 1 resolves to sessions/001."
        ),
    )
    return parser


def _configure_project(
    project: str | None,
    session: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project

    if session is not None:
        raw = str(session).strip()
        if not raw:
            parser.error("--session cannot be empty.")
        try:
            os.environ["PIPELINE_SESSION"] = f"{int(raw):03d}"
        except ValueError:
            parser.error("--session must be an integer, e.g. --session 1.")

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 10_reporter.py directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe artifact loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Any) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        _track_read(path)
        data = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        print(f"[10][warn] Could not parse JSON artifact {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print(f"[10][warn] JSON artifact is not an object: {path}", file=sys.stderr)
        return None

    return data


def _load_text(path: Any) -> str:
    if not path.exists():
        return ""

    try:
        _track_read(path)
        return path.read_text(errors="replace").strip()
    except Exception as exc:
        print(f"[10][warn] Could not read artifact {path}: {exc}", file=sys.stderr)
        return ""


def load_test_report() -> dict[str, Any] | None:
    return _load_json(DEBUGGER_OVERWRITE_TEST_SUMMARY)


def load_planner_full_plan() -> dict[str, Any] | None:
    return _load_json(PLANNER_FULL_PLAN)


def load_plan_mini() -> dict[str, Any] | None:
    return _load_json(PLANNER_MINI_PLAN)


def load_analysis_mini() -> dict[str, Any] | None:
    return _load_json(PLANNER_MINI_IMPACT)


def load_impl_record() -> dict[str, Any] | None:
    return _load_json(EXECUTOR_OVERWRITE_MANIFEST)


def _detect_scope(impl_record: dict[str, Any] | None) -> str:
    if impl_record:
        scope = impl_record.get("scope")
        if scope in {"full", "mini"}:
            return scope

    test_report = load_test_report()
    if test_report:
        scope = test_report.get("scope")
        if scope in {"full", "mini"}:
            return scope

    if PLANNER_MINI_PLAN.exists() or PLANNER_MINI_IMPACT.exists():
        return "mini"

    return "full"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_status(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def _fmt_code(value: Any, default: str = "?") -> str:
    if value is None:
        value = default
    return f"`{value}`"


def _shorten(text: str, limit: int = 400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _extract_file_list(value: Any) -> list[str]:
    """
    Normalize file list from either:
      ["src/a.ts"]
    or:
      [{"path": "src/a.ts", ...}]
    or:
      [{"file_path": "src/a.ts", ...}]
    or:
      [{"file": "src/a.ts", ...}]
    """
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

    return files


# ─────────────────────────────────────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────────────────────────────────────

def append_full_plan_section(lines: list[str], planner_full_plan: dict[str, Any] | None) -> None:
    lines += [
        "## Planner output",
        "",
    ]

    if not planner_full_plan:
        lines += [
            "_No planner_full_execution_plan.json found._",
            "",
        ]
        return

    tasks = planner_full_plan.get("tasks", [])
    impl_order = planner_full_plan.get("implementation_order", [])
    global_notes = planner_full_plan.get("global_notes", "")

    tasks_count = len(tasks) if isinstance(tasks, list) else 0
    order_count = len(impl_order) if isinstance(impl_order, list) else 0

    lines += [
        "| Plan version | Scope | Tasks decomposed | Files in order |",
        "|---|---|---:|---:|",
        (
            f"| {_fmt_code(planner_full_plan.get('plan_version'))} "
            f"| {_fmt_code(planner_full_plan.get('scope', 'full'))} "
            f"| {tasks_count} | {order_count} |"
        ),
        "",
    ]

    if global_notes:
        lines += [f"> **Global notes:** {_shorten(str(global_notes), 800)}", ""]

    stack = planner_full_plan.get("stack")
    if isinstance(stack, dict) and stack:
        lines += [
            "<details><summary>Detected stack</summary>",
            "",
            "```json",
            json.dumps(stack, indent=2, ensure_ascii=False),
            "```",
            "",
            "</details>",
            "",
        ]

    if isinstance(impl_order, list) and impl_order:
        lines += [
            "<details><summary>Implementation order</summary>",
            "",
        ]
        for index, file_path in enumerate(impl_order, 1):
            lines.append(f"{index}. `{file_path}`")
        lines += ["", "</details>", ""]


def append_mini_plan_section(
    lines: list[str],
    plan_mini: dict[str, Any] | None,
    analysis_mini: dict[str, Any] | None,
) -> None:
    lines += [
        "## Mini planner output",
        "",
    ]

    if not plan_mini and not analysis_mini:
        lines += [
            "_No planner_mini_execution_plan.json or planner_mini_impact_analysis.json found._",
            "",
        ]
        return

    task_summary = ""
    if plan_mini:
        task_summary = (
            plan_mini.get("task_summary")
            or plan_mini.get("summary")
            or plan_mini.get("goal")
            or ""
        )

    if task_summary:
        lines += [
            f"**Task summary:** {_shorten(str(task_summary), 800)}",
            "",
        ]

    target_files = _extract_file_list(plan_mini.get("target_files", []) if plan_mini else [])
    risks = plan_mini.get("risks", []) if plan_mini else []
    constraints = plan_mini.get("constraints", []) if plan_mini else []

    lines += [
        "| Artifact | Status | Count |",
        "|---|---|---:|",
        (
            f"| planner_mini_execution_plan.json "
            f"| {'✅ found' if plan_mini else '⚠️ missing'} "
            f"| {len(target_files)} target file(s) |"
        ),
        (
            f"| planner_mini_impact_analysis.json "
            f"| {'✅ found' if analysis_mini else '⚠️ missing'} "
            f"| — |"
        ),
        "",
    ]

    if target_files:
        lines += [
            "<details><summary>Target files</summary>",
            "",
        ]
        for fp in target_files:
            lines.append(f"- `{fp}`")
        lines += ["", "</details>", ""]

    if isinstance(risks, list) and risks:
        lines += ["**Risks:**"]
        for risk in risks:
            lines.append(f"- {risk}")
        lines.append("")

    if isinstance(constraints, list) and constraints:
        lines += ["**Constraints:**"]
        for constraint in constraints:
            lines.append(f"- {constraint}")
        lines.append("")

    if analysis_mini:
        warnings = analysis_mini.get("warnings", [])
        conflicts = analysis_mini.get("conflicts", [])
        recommendations = analysis_mini.get("recommendations", [])
        notes = analysis_mini.get("notes", [])

        if isinstance(warnings, list) and warnings:
            lines += ["**Impact analysis warnings:**"]
            for item in warnings:
                lines.append(f"- {item}")
            lines.append("")

        if isinstance(conflicts, list) and conflicts:
            lines += ["**Impact analysis conflicts:**"]
            for item in conflicts:
                lines.append(f"- {item}")
            lines.append("")

        if isinstance(recommendations, list) and recommendations:
            lines += ["**Impact analysis recommendations:**"]
            for item in recommendations:
                lines.append(f"- {item}")
            lines.append("")

        if isinstance(notes, list) and notes:
            lines += ["**Impact analysis notes:**"]
            for item in notes:
                lines.append(f"- {item}")
            lines.append("")


def append_mini_requirement_section(lines: list[str]) -> None:
    clarified = _load_text(CLARIFIED_REQ)
    enriched = _load_text(ENRICHER_OVERWRITE_PROMPT)

    if not clarified and not enriched:
        return

    lines += [
        "",
        "## Mini requirement context",
        "",
    ]

    if clarified:
        lines += [
            "**Clarified requirement synthesis:**",
            "",
            "> " + _shorten(clarified.replace("\n", "\n> "), 1200),
            "",
        ]

    if enriched:
        lines += [
            "<details><summary>Enriched prompt</summary>",
            "",
            enriched,
            "",
            "</details>",
            "",
        ]


def append_test_results_section(
    lines: list[str],
    test_report: dict[str, Any] | None,
) -> bool:
    """
    Append test/verifier results.

    Returns:
      True if final status is pass, else False.
    """
    lines += [
        "## Test / verifier results",
        "",
        "| Runner | Scope | Status | Iterations used | Final summary |",
        "|---|---|---|---:|---|",
    ]

    if test_report is None:
        lines.append("| — | — | ⚠️ No report | — | debugger_overwrite_test_summary.json missing |")
        lines.append("")
        return False

    final = test_report.get("final_status") == "PASS"
    scope = test_report.get("scope", "full")
    impl_label = test_report.get("impl", "unknown")
    iterations = test_report.get("iterations", [])
    iters = test_report.get(
        "total_iterations",
        len(iterations) if isinstance(iterations, list) else 0,
    )
    max_i = test_report.get("max_iter", "—")

    last = iterations[-1] if isinstance(iterations, list) and iterations else {}
    summary = last.get("summary", test_report.get("summary", "—")) if isinstance(last, dict) else "—"

    lines.append(
        f"| `{impl_label}` | `{scope}` | {render_status(final)} | {iters} / {max_i} | {_shorten(str(summary), 180)} |"
    )
    lines.append("")

    if scope == "mini" and test_report.get("mini_verification"):
        append_mini_verification_details(lines, test_report)
    else:
        append_iteration_details(lines, test_report)

    return final


def append_mini_verification_details(
    lines: list[str],
    test_report: dict[str, Any],
) -> None:
    mini = test_report.get("mini_verification", {})
    checks = mini.get("checks", []) if isinstance(mini, dict) else []

    lines += [
        "### Mini verification details",
        "",
    ]

    if not isinstance(checks, list) or not checks:
        lines += ["_No mini checks recorded._", ""]
        return

    lines += [
        "| File / Check | Kind | Status | Message |",
        "|---|---|---|---|",
    ]

    for check in checks:
        if not isinstance(check, dict):
            continue

        file_label = check.get("file", "—")
        kind = check.get("kind", "—")
        passed = bool(check.get("passed"))
        message = _shorten(str(check.get("message", "—")), 240)

        lines.append(
            f"| `{file_label}` | `{kind}` | {render_status(passed)} | {message} |"
        )

    lines.append("")


def append_iteration_details(
    lines: list[str],
    test_report: dict[str, Any],
) -> None:
    iterations = test_report.get("iterations", [])
    if not isinstance(iterations, list) or not iterations:
        return

    total_iterations = test_report.get("total_iterations", len(iterations))

    lines += [
        "### Iteration details",
        "",
    ]

    for item in iterations:
        if not isinstance(item, dict):
            continue

        iteration = item.get("iteration", "?")
        is_last = iteration == total_iterations
        passed = bool(item.get("passed"))
        icon = "✅" if passed else ("❌" if is_last else "🔄")

        lines.append(f"**Iteration {iteration}** {icon}")
        lines.append("")
        lines.append(f"```text\n{item.get('summary', '—')}\n```")

        clusters = item.get("cluster_details", [])
        if isinstance(clusters, list) and clusters:
            lines.append("")
            for cluster in clusters:
                if not isinstance(cluster, dict):
                    continue

                repaired = "✅" if cluster.get("repaired") else "❌"
                layer = cluster.get("layer_used", "")
                owner = cluster.get("owner", "")
                esc_to = cluster.get("escalated_to", "")
                failures = cluster.get("failures", 0)
                cluster_name = cluster.get("cluster", "—")
                note = cluster.get("note", "")

                layer_badge = f" `[{layer}]`" if layer else ""
                owner_badge = f" `{owner}`" if owner else ""
                esc_badge = (
                    f" ⚠️ ESCALATED→{esc_to}"
                    if cluster.get("escalated") and esc_to
                    else " ⚠️ ESCALATED"
                    if cluster.get("escalated")
                    else ""
                )
                note_str = f" — {_shorten(str(note), 240)}" if note else ""

                lines.append(
                    f"  {repaired}{layer_badge}{owner_badge}{esc_badge} "
                    f"`{cluster_name}` — {failures} failure(s){note_str}"
                )

        lines.append("")

    impl_label = test_report.get("impl", "unknown")
    max_i = test_report.get("max_iter", 3)
    max_ca = test_report.get("max_cluster_attempts", 2)

    lines.append(
        f"_Config: impl={impl_label}, max_iter={max_i}, "
        f"max_cluster_attempts={max_ca}_"
    )
    lines.append("")


def append_scaffold_section(lines: list[str]) -> None:
    scaffold = _load_json(SCAFFOLD_JSON)
    if not scaffold:
        return

    files = scaffold.get("files", [])
    if not isinstance(files, list):
        files = []

    n_src = sum(1 for item in files if isinstance(item, dict) and not item.get("is_test"))
    n_tests = sum(1 for item in files if isinstance(item, dict) and item.get("is_test"))

    lines += [
        "",
        "## Scaffold summary",
        "",
        f"- Scaffold version: {_fmt_code(scaffold.get('scaffold_version'))}",
        f"- Source stubs generated: {n_src}",
        f"- Test files generated: {n_tests}",
    ]


def append_escalated_section(
    lines: list[str],
    test_report: dict[str, Any] | None,
) -> None:
    if not test_report:
        return

    esc = test_report.get("escalated", [])
    if not isinstance(esc, list) or not esc:
        return

    lines += [
        "",
        "## ⚠️ Escalated clusters",
        "",
        f"**{len(esc)} cluster(s) require human review.**",
        "",
        "| Cluster | Failures | Note |",
        "|---|---:|---|",
    ]

    for item in esc:
        if not isinstance(item, dict):
            continue

        cluster = item.get("cluster", "—")
        failures = item.get("failures", "—")
        note = _shorten(str(item.get("note", "—")), 300)
        lines.append(f"| `{cluster}` | {failures} | {note} |")


def append_spec_delta_section(lines: list[str]) -> None:
    delta = _load_json(SPECTRACKER_VERSION_DELTA)
    if not delta:
        return

    fv = delta.get("from_version") or "(none)"
    tv = delta.get("to_version", "?")
    is_first = bool(delta.get("is_first_run", False))
    changed = delta.get("changed_sections", [])
    affected = delta.get("affected_files", [])
    skipped = delta.get("unaffected_files", [])
    summaries = delta.get("section_summaries", {})

    if not isinstance(changed, list):
        changed = []
    if not isinstance(affected, list):
        affected = []
    if not isinstance(skipped, list):
        skipped = []
    if not isinstance(summaries, dict):
        summaries = {}

    lines += [
        "",
        "## Spec delta",
        "",
        "| From | To | Mode | Changed § | Affected files |",
        "|---|---|---|---|---:|",
        f"| `{fv}` | `{tv}` | {'full' if is_first else 'partial'} "
        f"| {', '.join(f'§{sec}' for sec in changed) or '—'} | {len(affected)} |",
        "",
    ]

    if changed and not is_first:
        lines.append("**Changed sections:**")
        for sec in changed:
            note = summaries.get(str(sec), summaries.get(sec, ""))
            lines.append(f"- §{sec}{': ' + str(note) if note else ''}")
        lines.append("")

    if skipped:
        lines += [
            f"<details><summary>{len(skipped)} unaffected file(s) reused from previous run</summary>",
            "",
        ]
        for fp in skipped:
            lines.append(f"- `{fp}`")
        lines += ["", "</details>", ""]


def append_impl_record_section(
    lines: list[str],
    impl_record: dict[str, Any] | None,
    scope: str,
    plan_mini: dict[str, Any] | None,
) -> None:
    if not impl_record:
        return

    mode = impl_record.get("mode", "unknown")
    written = _extract_file_list(impl_record.get("files", []))
    failed = _extract_file_list(impl_record.get("failed_files", []))
    skipped_delta = _extract_file_list(impl_record.get("skipped_delta", []))
    is_delta = "delta" in str(mode)

    lines += [
        "",
        "## Implementation record",
        "",
        f"- Scope: `{scope}`",
        f"- Mode: `{mode}`",
        f"- Files implemented this run: {len(written)}",
    ]

    if failed:
        lines.append(f"- Failed files: {len(failed)}")

    if is_delta and skipped_delta:
        lines.append(f"- Files reused by delta mode: {len(skipped_delta)}")

    if scope == "mini" and plan_mini:
        target_files = _extract_file_list(plan_mini.get("target_files", []))
        if target_files:
            lines.append(f"- Mini target files: {len(target_files)}")

    if written:
        lines += [
            "",
            "<details><summary>Implemented files</summary>",
            "",
        ]
        for fp in written:
            lines.append(f"- `{fp}`")
        lines += ["", "</details>", ""]

    if failed:
        lines += [
            "",
            "<details><summary>Failed files</summary>",
            "",
        ]
        for fp in failed:
            lines.append(f"- `{fp}`")
        lines += ["", "</details>", ""]


def append_judge_section(lines: list[str]) -> None:
    judge = _load_text(JUDGE_VERDICT_SUMMARY)
    if not judge:
        return

    lines += [
        "",
        "## Judge verdict summary",
        "",
        "<details><summary>Open judge verdict summary</summary>",
        "",
        judge,
        "",
        "</details>",
        "",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def _write_github_step_summary(summary_md: str) -> None:
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not github_summary:
        return

    try:
        with open(github_summary, "a", encoding="utf-8") as handle:
            handle.write(summary_md)
            if not summary_md.endswith("\n"):
                handle.write("\n")
    except Exception as exc:
        print(f"[10][warn] Could not write GITHUB_STEP_SUMMARY: {exc}", file=sys.stderr)


def write_summary(summary_md: str) -> None:
    REPORTER_EXECUTION_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    REPORTER_EXECUTION_SUMMARY.write_text(summary_md, encoding="utf-8")
    _track_write(REPORTER_EXECUTION_SUMMARY)

    print(summary_md)
    print(f"[10] Report written → {REPORTER_EXECUTION_SUMMARY}")

    _write_github_step_summary(summary_md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, args.session, parser)

    # Important: project/session env must be available before ensure_dirs().
    ensure_dirs()

    exit_code = 0

    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        impl_record = load_impl_record()
        scope = _detect_scope(impl_record)

        test_report = load_test_report()
        planner_full_plan = load_planner_full_plan()
        plan_mini = load_plan_mini()
        analysis_mini = load_analysis_mini()

        session_id = get_session_id()
        session_label = session_id if session_id is not None else "legacy/no-session"

        lines: list[str] = [
            "# LLM Pipeline Report",
            f"_Generated: {now}_",
            "",
            f"**Project:** `{get_project_name()}`",
            f"**Project slug:** `{get_project_slug()}`",
            f"**Session:** `{session_label}`",
            f"**Scope:** `{scope}`",
            "",
        ]

        if scope == "mini":
            append_mini_plan_section(lines, plan_mini, analysis_mini)
            append_mini_requirement_section(lines)
        else:
            append_full_plan_section(lines, planner_full_plan)

        all_passed = append_test_results_section(lines, test_report)

        lines += [
            "",
            f"**Overall:** {'✅ Pipeline passed' if all_passed else '❌ Pipeline had failures'}",
            "",
        ]

        append_scaffold_section(lines)
        append_escalated_section(lines, test_report)
        append_spec_delta_section(lines)
        append_impl_record_section(lines, impl_record, scope, plan_mini)
        append_judge_section(lines)

        summary_md = "\n".join(lines).rstrip() + "\n"
        write_summary(summary_md)

    except Exception as exc:
        print(f"[10][error] Reporter failed: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
