"""
pipeline/05_report.py
Step 5b — Aggregate iteration JSON + GLM plan metadata into reports/summary.md
          This file is printed to $GITHUB_STEP_SUMMARY for the Actions UI.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

GLM_PLAN_PATH = ROOT / "scaffold" / "glm_plan.json"


def load_report(impl: str) -> dict | None:
    p = REPORTS_DIR / f"{impl}_iterations.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_glm_plan() -> dict | None:
    if not GLM_PLAN_PATH.exists():
        return None
    return json.loads(GLM_PLAN_PATH.read_text())


def render_status(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# LLM Pipeline Report",
        f"_Generated: {now}_",
        "",
    ]

    # ── GLM plan summary ─────────────────────────────────────────────────────
    glm_plan = load_glm_plan()
    if glm_plan:
        tasks = glm_plan.get("tasks", [])
        impl_order = glm_plan.get("implementation_order", [])
        global_notes = glm_plan.get("global_notes", "")
        lines += [
            "## GLM 5.1 — Planner output",
            "",
            f"| Plan version | Tasks decomposed | Files in order |",
            f"|---|---|---|",
            f"| `{glm_plan.get('plan_version', '?')}` "
            f"| {len(tasks)} "
            f"| {len(impl_order)} |",
            "",
        ]
        if global_notes:
            lines += [f"> **Global notes:** {global_notes}", ""]
        lines += [
            "<details><summary>Implementation order</summary>",
            "",
        ]
        for i, fp in enumerate(impl_order, 1):
            lines.append(f"{i}. `{fp}`")
        lines += ["", "</details>", ""]
    else:
        lines += [
            "## GLM 5.1 — Planner output",
            "",
            "_No glm_plan.json found — pipeline ran in `--only-qwen` mode._",
            "",
        ]

    # ── Qwen test results ────────────────────────────────────────────────────
    lines += [
        "## Qwen 3.6+ (surface) + Minimax 2.7 (logic) — Executor results",
        "",
        "| Model | Status | Iterations used | Final test summary |",
        "|---|---|---|---|",
    ]

    qwen_report = load_report("qwen")
    all_passed = True
    details: list[str] = []

    if qwen_report is None:
        lines.append("| QWEN | ⚠️ No report | — | Report file missing |")
        all_passed = False
    else:
        final = qwen_report["final_status"] == "PASS"
        if not final:
            all_passed = False

        last    = qwen_report["iterations"][-1] if qwen_report["iterations"] else {}
        summary = last.get("summary", "—")
        iters   = qwen_report["total_iterations"]
        max_i   = qwen_report.get("max_iter", 3)
        max_ca  = qwen_report.get("max_cluster_attempts", 2)

        lines.append(
            f"| QWEN | {render_status(final)} | {iters} / {max_i} | {summary} |"
        )

        # Detailed iteration breakdown
        details.append("\n### QWEN — iteration detail\n")
        for it in qwen_report["iterations"]:
            is_last = it["iteration"] == iters
            icon = "✅" if it["passed"] else ("❌" if is_last else "🔄")
            details.append(f"**Iteration {it['iteration']}** {icon}  ")
            details.append(f"```\n{it['summary']}\n```")

            clusters = it.get("cluster_details", [])
            if clusters:
                details.append("")
                for c in clusters:
                    repaired    = "✅" if c.get("repaired") else "❌"
                    layer       = c.get("layer_used", "")
                    owner       = c.get("owner", "")
                    esc_to      = c.get("escalated_to", "")
                    layer_badge = f" `[{layer}]`" if layer else ""
                    owner_badge = f" `{owner}`" if owner else ""
                    esc_badge   = (
                        f" ⚠️ ESCALATED→{esc_to}" if c.get("escalated") and esc_to
                        else " ⚠️ ESCALATED" if c.get("escalated")
                        else ""
                    )
                    note_str    = f" — {c['note']}" if c.get("note") else ""
                    details.append(
                        f"  {repaired}{layer_badge}{owner_badge}{esc_badge} "
                        f"`{c['cluster']}` — {c['failures']} failure(s){note_str}"
                    )
            details.append("")

        impl_label = qwen_report.get("impl", "qwen+minimax")
        details.append(f"\n_Config: impl={impl_label}, max_iter={max_i}, max_cluster_attempts={max_ca}_\n")

    lines.append("")
    lines.append(
        f"**Overall:** {'✅ Pipeline passed' if all_passed else '❌ Pipeline had failures'}"
    )
    lines += details

    # ── Scaffold summary ─────────────────────────────────────────────────────
    scaffold_json = ROOT / "scaffold" / "scaffold.json"
    if scaffold_json.exists():
        scaffold = json.loads(scaffold_json.read_text())
        n_src   = sum(1 for f in scaffold["files"] if not f.get("is_test"))
        n_tests = sum(1 for f in scaffold["files"] if f.get("is_test"))
        lines += [
            "",
            "## Scaffold summary",
            f"- Scaffold version: `{scaffold.get('scaffold_version', '?')}`",
            f"- Source stubs generated: {n_src}",
            f"- Test files generated: {n_tests}",
        ]

    # ── Escalated clusters ───────────────────────────────────────────────────
    esc_json = REPORTS_DIR / "escalated_clusters.json"
    if esc_json.exists():
        esc = json.loads(esc_json.read_text())
        n = esc.get("total_escalated", 0)
        lines += [
            "",
            "## ⚠️ Escalated Clusters (require human review)",
            "",
            f"**{n} cluster(s) hit the give-up threshold.**",
            "",
            "| Cluster | Failures | Note |",
            "|---|---|---|",
        ]
        for c in esc.get("clusters", []):
            lines.append(
                f"| `{c['cluster']}` | {c['failures']} | {c.get('note', '—')} |"
            )

    # ── Impl record ──────────────────────────────────────────────────────────
    impl_record = ROOT / "scaffold" / "impl_qwen.json"
    if impl_record.exists():
        rec = json.loads(impl_record.read_text())
        mode = rec.get("mode", "unknown")
        written = rec.get("files", [])
        lines += [
            "",
            "## Implementation record",
            f"- Mode: `{mode}`",
            f"- Files written: {len(written)}",
        ]

    summary_md = "\n".join(lines) + "\n"
    out = REPORTS_DIR / "summary.md"
    out.write_text(summary_md)
    print(summary_md)
    print(f"[05] Report written → {out}")


if __name__ == "__main__":
    main()
