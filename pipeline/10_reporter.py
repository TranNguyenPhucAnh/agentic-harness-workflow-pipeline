"""
pipeline/10_reporter.py
=====================
Step 5b — Aggregate pipeline artifacts into reports/summary.md.

Supports both:
  - FULL flow: plan.json, scaffold.json, spec_delta.json, test_report.json
  - MINI flow: plan_mini.json, analysis_mini.json, clarified_requirement.md,
               enriched_prompt.md, impl_record.json, test_report.json

Writes:
  artifacts_<slug>/reports/summary.md

Reads:
  artifacts_<slug>/state/plan.json
  artifacts_<slug>/state/plan_mini.json
  artifacts_<slug>/run/analysis_mini.json
  artifacts_<slug>/run/test_report.json
  artifacts_<slug>/state/scaffold.json
  artifacts_<slug>/cache/spec_delta.json
  artifacts_<slug>/run/impl_record.json
  artifacts_<slug>/reports/judge_report.md
  artifacts_<slug>/knowledge/current/clarified_requirement.md
  artifacts_<slug>/knowledge/current/enriched_prompt.md

Direct execution:
  python 05_report.py --project my-app
  PIPELINE_PROJECT=my-app python 05_report.py

For taxonomy details see docs/artifacts.md
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

# === WRITE AUTHORITY: 05_report ===
# OWNS  : artifacts_<slug>/reports/summary.md
# READS : planner/test/scaffold/delta/impl/judge artifacts

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ANALYSIS_MINI,
    CLARIFIED_REQ,
    ENRICHED_PROMPT,
    IMPL_RECORD,
    JUDGE_REPORT,
    PLAN_JSON as GLM_PLAN_PATH,
    PLAN_MINI,
    SCAFFOLD_JSON,
    SPEC_DELTA as DELTA_JSON,
    SUMMARY,
    TEST_REPORT,
    ensure_dirs,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / project setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate pipeline artifacts into reports/summary.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 05_report.py --project my-app
              PIPELINE_PROJECT=my-app python 05_report.py
        """),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
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
        "PIPELINE_PROJECT=<name> before running 05_report.py directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe artifact loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        print(f"[05][warn] Could not parse JSON artifact {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print(f"[05][warn] JSON artifact is not an object: {path}", file=sys.stderr)
        return None

    return data


def _load_text(path: Path) -> str:
    if not path.exists():
        return ""

    try:
        return path.read_text(errors="replace").strip()
    except Exception as exc:
        print(f"[05][warn] Could not read artifact {path}: {exc}", file=sys.stderr)
        return ""


def load_test_report() -> dict[str, Any] | None:
    return _load_json(TEST_REPORT)


def load_glm_plan() -> dict[str, Any] | None:
    return _load_json(GLM_PLAN_PATH)


def load_plan_mini() -> dict[str, Any] | None:
    return _load_json(PLAN_MINI)


def load_analysis_mini() -> dict[str, Any] | None:
    return _load_json(ANALYSIS_MINI)


def load_impl_record() -> dict[str, Any] | None:
    return _load_json(IMPL_RECORD)


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

    if PLAN_MINI.exists() or ANALYSIS_MINI.exists():
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


def _append_details_block(
    lines: list[str],
    title: str,
    items: list[str],
    *,
    empty: str = "—",
) -> None:
    if not items:
        lines.append(empty)
        return

    lines += [
        f"<details><summary>{title}</summary>",
        "",
    ]
    for item in items:
        lines.append(f"- `{item}`")
    lines += ["", "</details>", ""]


# ─────────────────────────────────────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────────────────────────────────────

def append_full_plan_section(lines: list[str], glm_plan: dict[str, Any] | None) -> None:
    lines += [
        "## GLM 5.1 — Planner output",
        "",
    ]

    if not glm_plan:
        lines += [
            "_No plan.json found._",
            "",
        ]
        return

    tasks = glm_plan.get("tasks", [])
    impl_order = glm_plan.get("implementation_order", [])
    global_notes = glm_plan.get("global_notes", "")

    tasks_count = len(tasks) if isinstance(tasks, list) else 0
    order_count = len(impl_order) if isinstance(impl_order, list) else 0

    lines += [
        "| Plan version | Tasks decomposed | Files in order |",
        "|---|---:|---:|",
        f"| {_fmt_code(glm_plan.get('plan_version'))} | {tasks_count} | {order_count} |",
        "",
    ]

    if global_notes:
        lines += [f"> **Global notes:** {_shorten(str(global_notes), 800)}", ""]

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
            "_No plan_mini.json or analysis_mini.json found._",
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
        f"| plan_mini.json | {'✅ found' if plan_mini else '⚠️ missing'} | {len(target_files)} target file(s) |",
        f"| analysis_mini.json | {'✅ found' if analysis_mini else '⚠️ missing'} | — |",
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
            lines += ["**Analysis warnings:**"]
            for item in warnings:
                lines.append(f"- {item}")
            lines.append("")

        if isinstance(conflicts, list) and conflicts:
            lines += ["**Analysis conflicts:**"]
            for item in conflicts:
                lines.append(f"- {item}")
            lines.append("")

        if isinstance(recommendations, list) and recommendations:
            lines += ["**Analysis recommendations:**"]
            for item in recommendations:
                lines.append(f"- {item}")
            lines.append("")

        if isinstance(notes, list) and notes:
            lines += ["**Analysis notes:**"]
            for item in notes:
                lines.append(f"- {item}")
            lines.append("")


def append_mini_requirement_section(lines: list[str]) -> None:
    clarified = _load_text(CLARIFIED_REQ)
    enriched = _load_text(ENRICHED_PROMPT)

    if not clarified and not enriched:
        return

    lines += [
        "",
        "## Mini requirement context",
        "",
    ]

    if clarified:
        lines += [
            "**Clarified requirement:**",
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
        lines.append("| — | — | ⚠️ No report | — | test_report.json missing |")
        lines.append("")
        return False

    final = test_report.get("final_status") == "PASS"
    scope = test_report.get("scope", "full")
    impl_label = test_report.get("impl", "qwen+minimax")
    iterations = test_report.get("iterations", [])
    iters = test_report.get("total_iterations", len(iterations) if isinstance(iterations, list) else 0)
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

    impl_label = test_report.get("impl", "qwen+minimax")
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
    delta = _load_json(DELTA_JSON)
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


def append_judge_section(lines: list[str]) -> None:
    judge = _load_text(JUDGE_REPORT)
    if not judge:
        return

    lines += [
        "",
        "## Judge report",
        "",
        "<details><summary>Open judge report</summary>",
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
        print(f"[05][warn] Could not write GITHUB_STEP_SUMMARY: {exc}", file=sys.stderr)


def write_summary(summary_md: str) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(summary_md)

    print(summary_md)
    print(f"[05] Report written → {SUMMARY}")

    _write_github_step_summary(summary_md)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: project env must be available before ensure_dirs().
    ensure_dirs()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    impl_record = load_impl_record()
    scope = _detect_scope(impl_record)

    test_report = load_test_report()
    glm_plan = load_glm_plan()
    plan_mini = load_plan_mini()
    analysis_mini = load_analysis_mini()

    lines: list[str] = [
        "# LLM Pipeline Report",
        f"_Generated: {now}_",
        "",
        f"**Scope:** `{scope}`",
        "",
    ]

    if scope == "mini":
        append_mini_plan_section(lines, plan_mini, analysis_mini)
        append_mini_requirement_section(lines)
    else:
        append_full_plan_section(lines, glm_plan)

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


if __name__ == "__main__":
    main()
