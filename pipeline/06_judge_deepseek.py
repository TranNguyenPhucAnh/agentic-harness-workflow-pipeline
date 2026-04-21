"""
pipeline/06_judge_deepseek.py
Step 6 — DeepSeek V3.2 (deepseek-reasoner) as Judge / Validator.

Runs ONLY after all vitest tests have passed (called by harness after step 4+5
exits 0).  Aggregates all pipeline artefacts into a single briefing, sends to
DeepSeek V3.2 for deep review, writes reports/judge_report.md.

Reads:
    spec.md
    scaffold/scaffold.json
    scaffold/glm_plan.json          (optional — present if GLM planner ran)
    scaffold/impl_qwen.json
    reports/qwen_iterations.json
    src/**/*.ts  src/**/*.tsx        (final implemented source)
    tests/**/*.ts  tests/**/*.tsx    (test files for reference)

Writes:
    reports/judge_report.md         ← human-readable final report
    reports/judge_raw.json          ← full model response + metadata
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
# deepseek-v3.2 with reasoning enabled via OpenRouter reasoning parameter
MODEL              = "deepseek/deepseek-v3.2"

ROOT        = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
PIPELINE_CTX = ROOT / "scaffold" / "pipeline_context.json"


# ── Prompt ────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a senior software engineer acting as a final code reviewer and sign-off authority.

You will receive a complete pipeline briefing containing:
1. The original product spec (spec.md)
2. The GLM 5.1 architectural plan (if available)
3. The Qwen implementation record and iteration history
4. All final implemented source files
5. All vitest test files

All unit tests have already passed.  Your job is NOT to re-run tests but to
perform a deeper qualitative review covering:

A. SPEC COMPLIANCE — Does the implementation match spec requirements?
   Check props, types, behaviour rules, acceptance criteria (§10).

B. CODE QUALITY — TypeScript strictness, no `any`, Tailwind-only styling,
   hook patterns, component structure, edge case handling.

C. TEST QUALITY — Are the tests meaningful? Do they actually validate the
   behaviour described in the spec, or are they trivially passing?

D. ARCHITECTURE — Is the file structure clean? Are dependencies correct?
   Any circular imports or coupling concerns?

E. GAPS / RISKS — Anything the tests don't cover that could break in production?

Return your review as a structured JSON object — raw JSON only, no markdown fences:
{
  "verdict": "APPROVED" | "APPROVED_WITH_NOTES" | "NEEDS_REVISION",
  "summary": "2-3 sentence executive summary",
  "sections": {
    "spec_compliance":  { "score": 1-5, "notes": "..." },
    "code_quality":     { "score": 1-5, "notes": "..." },
    "test_quality":     { "score": 1-5, "notes": "..." },
    "architecture":     { "score": 1-5, "notes": "..." },
    "gaps_risks":       { "notes": "..." }
  },
  "blocking_issues": [ "issue 1", "issue 2" ],
  "non_blocking_notes": [ "note 1", "note 2" ],
  "sign_off": "Your name as reviewer + timestamp placeholder"
}

Scoring guide: 5=excellent, 4=good, 3=acceptable, 2=needs work, 1=failing.
verdict APPROVED        → no blocking issues, score avg >= 3.5
verdict APPROVED_WITH_NOTES → no blocking issues but notable non-blocking notes
verdict NEEDS_REVISION  → one or more blocking issues found
"""


# ── Briefing builder ──────────────────────────────────────────────────────────

def _load_spec() -> str:
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()

def _read_safe(path: Path, label: str) -> str:
    if not path.exists():
        return f"[{label}: file not found at {path}]"
    return path.read_text()


def _collect_src_files(src_dir: Path) -> dict[str, str]:
    """Full collect — dùng cho test files."""
    files: dict[str, str] = {}
    for ext in ("*.ts", "*.tsx"):
        for p in sorted(src_dir.rglob(ext)):
            rel = str(p.relative_to(ROOT))
            files[rel] = p.read_text()
    return files

def _collect_changed_files(src_dir: Path) -> dict[str, str]:
    """
    Chỉ collect source files có nội dung khác so với stub gốc.
    Judge không cần xem file types/constants nếu chúng không thay đổi nhiều.
    Tiết kiệm ~40% input tokens của judge call.
    """
    # Load stub_map từ pipeline_context để so sánh
    stub_map: dict[str, str] = {}
    if PIPELINE_CTX.exists():
        ctx = json.loads(PIPELINE_CTX.read_text())
        stub_map = ctx.get("stub_map", {})
 
    changed: dict[str, str] = {}
    for ext in ("*.ts", "*.tsx"):
        for p in sorted(src_dir.rglob(ext)):
            rel     = str(p.relative_to(ROOT))
            current = p.read_text()
            stub    = stub_map.get(rel, "")
            # Coi là "changed" nếu không có stub (file mới) hoặc nội dung khác hẳn
            if not stub or current.strip() != stub.strip():
                changed[rel] = current
    return changed

def build_briefing() -> str:
    parts: list[str] = []

    # 1. Spec
    parts.append("## 1. spec.md\n\n" + _load_spec())

    # 2. GLM plan (optional)
    glm_plan_path = ROOT / "scaffold" / "glm_plan.json"
    if glm_plan_path.exists():
        plan = json.loads(glm_plan_path.read_text())
        parts.append(
            "## 2. GLM 5.1 Architectural Plan\n\n"
            f"```json\n{json.dumps(plan, indent=2)}\n```"
        )
    else:
        parts.append("## 2. GLM 5.1 Architectural Plan\n\n_Not available (--only-qwen mode)_")

    # 3. Qwen impl record
    impl_record_path = ROOT / "scaffold" / "impl_qwen.json"
    if impl_record_path.exists():
        rec = json.loads(impl_record_path.read_text())
        parts.append(
            "## 3. Qwen Implementation Record\n\n"
            f"- Mode: `{rec.get('mode', 'unknown')}`\n"
            f"- Files written: {len(rec.get('files', []))}\n"
            + "\n".join(f"  - `{f}`" for f in rec.get("files", []))
        )

    # 4. Iteration history (condensed)
    iter_path = REPORTS_DIR / "qwen_iterations.json"
    if iter_path.exists():
        report = json.loads(iter_path.read_text())
        iters  = report.get("iterations", [])
        lines  = [
            "## 4. Qwen Test Iteration History\n",
            f"- Final status: **{report.get('final_status', '?')}**",
            f"- Total iterations: {report.get('total_iterations', '?')} / "
            f"{report.get('max_iter', 3)}",
            "",
        ]
        for it in iters:
            icon = "✅" if it["passed"] else "🔄"
            lines.append(f"**Iteration {it['iteration']}** {icon} — {it.get('summary', '')}")
            for c in it.get("cluster_details", []):
                repaired = "✅" if c.get("repaired") else "❌"
                lines.append(f"  {repaired} `{c['cluster']}` ({c['failures']} failure(s))")
            lines.append("")
        parts.append("\n".join(lines))

    # 5. Final source files
    src_files = _collect_changed_files(ROOT / "src")   # <-- đổi function
    src_block = "\n\n".join(
        f"### {fp}\n```typescript\n{code}\n```"
        for fp, code in src_files.items()
    )
    parts.append(f"## 5. Implemented Source Files ({len(src_files)} changed files)\n\n{src_block}")
 
    test_files = _collect_src_files(ROOT / "tests")    # <-- giữ nguyên full
    test_block = "\n\n".join(
        f"### {fp}\n```typescript\n{code}\n```"
        for fp, code in test_files.items()
    )
    parts.append(f"## 6. Test Files ({len(test_files)} files)\n\n{test_block}")

    return "\n\n---\n\n".join(parts)


# ── API call ──────────────────────────────────────────────────────────────────

def call_deepseek_judge(briefing: str) -> tuple[str, list | None]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": briefing},
        ],
        "reasoning": {"enabled": True},
        "temperature": 0.1,
        "max_tokens": 16000,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    print("[06] Calling DeepSeek V3.2 (judge, reasoning ON) …")

    last_error = None

    with httpx.Client(timeout=300) as client:
        for attempt in range(2):  # 1 retry on empty response
            r = client.post(OPENROUTER_URL, headers=headers, json=payload)
            r.raise_for_status()

            data = r.json()

            usage = data.get("usage", {})
            prompt_t     = usage.get("prompt_tokens", "?")
            completion_t = usage.get("completion_tokens", "?")
            print(f"[06] Tokens: prompt={prompt_t}, completion={completion_t}")

            choice        = data["choices"][0]
            msg           = choice["message"]
            content       = msg.get("content")
            tool_calls    = msg.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            # Preserve reasoning_details for transparency (saved to judge_raw.json)
            reasoning_details = msg.get("reasoning_details")

            if tool_calls:
                raise RuntimeError(
                    f"DeepSeek judge returned tool_calls instead of text: {tool_calls}"
                )

            if content and content.strip():
                return content.strip(), reasoning_details

            last_error = (
                f"Empty content. finish_reason={finish_reason}, message={msg}"
            )
            print(f"[06] {last_error}", file=sys.stderr)

            if attempt == 0:
                print("[06] Retrying in 3s …", file=sys.stderr)
                time.sleep(3)

    raise RuntimeError(f"DeepSeek judge failed after retries: {last_error}")

# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[06] JSON parse failed: {e}", file=sys.stderr)
            print(f"[06] Raw (first 500):\n{raw[:500]}", file=sys.stderr)
            sys.exit(1)
    print("[06] No JSON object found in judge response.", file=sys.stderr)
    sys.exit(1)


# ── Report renderer ───────────────────────────────────────────────────────────

def render_report(review: dict) -> str:
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = review.get("verdict", "UNKNOWN")

    verdict_icon = {
        "APPROVED":             "✅",
        "APPROVED_WITH_NOTES":  "⚠️",
        "NEEDS_REVISION":       "❌",
    }.get(verdict, "❓")

    lines = [
        "# Judge Report — DeepSeek V3.2 Final Review",
        f"_Generated: {now}_",
        f"_Model: {MODEL}_",
        "",
        f"## Verdict: {verdict_icon} {verdict}",
        "",
        f"> {review.get('summary', '')}",
        "",
        "## Scores",
        "",
        "| Dimension | Score | Notes |",
        "|---|---|---|",
    ]

    sections = review.get("sections", {})
    dimension_labels = {
        "spec_compliance": "Spec Compliance",
        "code_quality":    "Code Quality",
        "test_quality":    "Test Quality",
        "architecture":    "Architecture",
        "gaps_risks":      "Gaps / Risks",
    }
    for key, label in dimension_labels.items():
        sec   = sections.get(key, {})
        score = sec.get("score", "—")
        notes = sec.get("notes", "—").replace("\n", " ")
        score_str = f"{score}/5" if isinstance(score, int) else str(score)
        lines.append(f"| {label} | {score_str} | {notes} |")

    # Blocking issues
    blocking = review.get("blocking_issues", [])
    lines += ["", "## Blocking Issues", ""]
    if blocking:
        for issue in blocking:
            lines.append(f"- ❌ {issue}")
    else:
        lines.append("_None — all checks passed._")

    # Non-blocking notes
    notes_list = review.get("non_blocking_notes", [])
    lines += ["", "## Non-blocking Notes", ""]
    if notes_list:
        for note in notes_list:
            lines.append(f"- ℹ️ {note}")
    else:
        lines.append("_None._")

    # Sign-off
    lines += [
        "",
        "---",
        f"**Sign-off:** {review.get('sign_off', 'DeepSeek V3.2')}",
    ]

    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[06] Building pipeline briefing …")
    briefing = build_briefing()
    print(f"[06] Briefing size: {len(briefing):,} chars")

    raw_response, reasoning_details = call_deepseek_judge(briefing)

    # Save raw response + reasoning chain
    raw_out = REPORTS_DIR / "judge_raw.json"
    raw_out.write_text(json.dumps({
        "model":            MODEL,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "response":         raw_response,
        "reasoning_details": reasoning_details,   # full chain-of-thought
    }, indent=2))
    print(f"[06] Raw response + reasoning saved → {raw_out}")

    review = _parse_json(raw_response)

    # Render and save markdown report
    report_md = render_report(review)
    report_out = REPORTS_DIR / "judge_report.md"
    report_out.write_text(report_md)

    print(f"\n[06] Judge report written → {report_out}")
    print(f"\n{'='*60}")
    print(report_md)
    print(f"{'='*60}")

    # Exit non-zero if judge found blocking issues
    verdict = review.get("verdict", "")
    if verdict == "NEEDS_REVISION":
        print("[06] Judge verdict: NEEDS_REVISION — blocking issues found.", file=sys.stderr)
        sys.exit(1)

    print(f"[06] Judge verdict: {verdict} ✅")


if __name__ == "__main__":
    main()
