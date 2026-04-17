"""
pipeline/05_report.py
Step 6 — Aggregate iteration JSON files into reports/summary.md
         This file is printed to $GITHUB_STEP_SUMMARY for the Actions UI.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

IMPLS = ["qwen", "glm"]


def load_report(impl: str) -> dict | None:
    p = REPORTS_DIR / f"{impl}_iterations.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def render_status(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# LLM Pipeline Report",
        f"_Generated: {now}_",
        "",
        "## Results",
        "",
        "| Model | Status | Iterations used | Final test summary |",
        "|---|---|---|---|",
    ]

    all_passed = True
    details = []

    for impl in IMPLS:
        report = load_report(impl)
        if report is None:
            lines.append(f"| {impl.upper()} | ⚠️ No report | — | Report file missing |")
            all_passed = False
            continue

        final = report["final_status"] == "PASS"
        if not final:
            all_passed = False

        last = report["iterations"][-1] if report["iterations"] else {}
        summary = last.get("summary", "—")
        iters   = report["total_iterations"]

        lines.append(
            f"| {impl.upper()} | {render_status(final)} | {iters} / 3 | {summary} |"
        )

        # Detailed iteration breakdown
        details.append(f"\n### {impl.upper()} — iteration detail\n")
        for it in report["iterations"]:
            icon = "✅" if it["passed"] else "🔄" if it["iteration"] < iters else "❌"
            details.append(f"**Iteration {it['iteration']}** {icon}  ")
            details.append(f"```\n{it['summary']}\n```\n")

    lines.append("")
    lines.append(f"**Overall:** {'✅ Pipeline passed' if all_passed else '❌ Pipeline had failures'}")
    lines += details

    # Scaffold info
    scaffold_json = ROOT / "scaffold" / "scaffold.json"
    if scaffold_json.exists():
        scaffold = json.loads(scaffold_json.read_text())
        n_src   = sum(1 for f in scaffold["files"] if not f.get("is_test"))
        n_tests = sum(1 for f in scaffold["files"] if f.get("is_test"))
        lines += [
            "",
            "## Scaffold summary",
            f"- Scaffold version: `{scaffold.get('scaffold_version', '?')}`",
            f"- Source files generated: {n_src}",
            f"- Test files generated: {n_tests}",
        ]

    summary_md = "\n".join(lines) + "\n"
    out = REPORTS_DIR / "summary.md"
    out.write_text(summary_md)
    print(summary_md)
    print(f"[05] Report written → {out}")


if __name__ == "__main__":
    main()
