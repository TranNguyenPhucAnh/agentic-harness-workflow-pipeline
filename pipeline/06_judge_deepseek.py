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

You will receive a complete pipeline briefing. Section 0 tells you whether this is a
FULL run (all files new) or a PARTIAL run (only some files changed).

For PARTIAL runs:
  - Focus your review on re-implemented files (marked in §5).
  - For reused files (§5b), only flag issues if they directly interact with changed files.
  - Do NOT block approval for pre-existing issues in reused files.

For FULL runs: review everything equally.

Review dimensions (apply to re-implemented files; secondary for reused):

A. SPEC COMPLIANCE — Props, types, behaviour rules, acceptance criteria (§10).
B. CODE QUALITY — TypeScript strict, no `any`, Tailwind-only, hook patterns, edge cases.
C. TEST QUALITY — Are tests meaningful or trivially passing?
D. ARCHITECTURE — Clean dependencies, no circular imports, correct file structure.
E. GAPS / RISKS — What tests don't cover that could break in production?

Return a structured JSON object — raw JSON only, no markdown fences:
{
  "verdict": "APPROVED" | "APPROVED_WITH_NOTES" | "NEEDS_REVISION",
  "run_type": "full" | "partial",
  "summary": "2-3 sentence executive summary",
  "sections": {
    "spec_compliance":  { "score": 1-5, "notes": "...", "scope": "re-implemented files" },
    "code_quality":     { "score": 1-5, "notes": "..." },
    "test_quality":     { "score": 1-5, "notes": "..." },
    "architecture":     { "score": 1-5, "notes": "..." },
    "gaps_risks":       { "notes": "..." }
  },
  "blocking_issues": [ "issue 1" ],
  "non_blocking_notes": [ "note 1" ],
  "partial_run_notes": "observations about reused files (partial runs only, else null)",
  "sign_off": "DeepSeek V3.2 + timestamp placeholder"
}

Scoring: 5=excellent, 4=good, 3=acceptable, 2=needs work, 1=failing.
APPROVED        → no blocking issues, avg score ≥ 3.5
APPROVED_WITH_NOTES → no blocking issues, notable non-blocking notes
NEEDS_REVISION  → one or more blocking issues found
"""


# ── Briefing builder ──────────────────────────────────────────────────────────

def _load_spec() -> str:
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()


def _load_delta() -> dict | None:
    """Load spec_delta.json if present — used to scope judge review."""
    p = ROOT / "scaffold" / "spec_delta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _affected_src_set(delta: dict | None) -> set[str]:
    """
    Return set of src/ files that were affected this run.
    Empty set means full run — judge reviews everything.
    """
    if delta is None or delta.get("is_first_run", True):
        return set()
    return {f for f in delta.get("affected_files", []) if f.startswith("src/")}

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

    delta        = _load_delta()
    affected_set = _affected_src_set(delta)
    is_partial   = bool(affected_set)   # True = only some files changed this run

    # 0. Run context header — tells judge whether this is full or partial
    if is_partial and delta:
        fv = delta.get("from_version") or "?"
        tv = delta.get("to_version", "?")
        changed_secs = delta.get("changed_sections", [])
        sums = delta.get("section_summaries", {})
        ctx_lines = [
            "## 0. Run context (IMPORTANT — read before reviewing)\n",
            f"**This is a PARTIAL run** — spec changed from `{fv}` to `{tv}`.",
            f"Only the following spec sections changed: {changed_secs}",
            "",
            "**Changed sections:**",
        ]
        for sec in changed_secs:
            note = sums.get(sec, "")
            ctx_lines.append(f"- §{sec}: {note}")
        ctx_lines += [
            "",
            "**Files re-implemented this run (your primary review focus):**",
        ]
        for fp in sorted(affected_set):
            ctx_lines.append(f"- `{fp}`")
        ctx_lines += [
            "",
            "**Files reused from previous run (unchanged — secondary review):**",
        ]
        impl_rec = ROOT / "scaffold" / "impl_qwen.json"
        skipped = []
        if impl_rec.exists():
            skipped = json.loads(impl_rec.read_text()).get("skipped_delta", [])
        for fp in skipped:
            ctx_lines.append(f"- `{fp}`")
        ctx_lines += [
            "",
            "**Review instructions for partial run:**",
            "- Focus spec-compliance and logic review on re-implemented files.",
            "- For reused files: only flag issues if they interact with changed files.",
            "- Do NOT block approval for issues in reused files that predate this run.",
        ]
        parts.append("\n".join(ctx_lines))
    else:
        parts.append(
            "## 0. Run context\n\n**Full run** — all files implemented from scratch."
        )

    # 1. Spec
    parts.append("## 1. spec.md\n\n" + _load_spec())
    
    addendum_path = ROOT / "scaffold" / "spec_addendum.md"
    if addendum_path.exists():
        parts.append(
            "## 1b. Spec Addendum (edge cases & invariants from previous runs)\n\n"
            + addendum_path.read_text()
        )

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
        skipped_d = rec.get("skipped_delta", [])
        rec_lines = [
            "## 3. Qwen Implementation Record\n",
            f"- Mode: `{rec.get('mode', 'unknown')}`",
            f"- Files implemented this run: {len(rec.get('files', []))}",
        ]
        for f in rec.get("files", []):
            rec_lines.append(f"  - `{f}`")
        if skipped_d:
            rec_lines.append(f"- Files reused (not re-implemented): {len(skipped_d)}")
            for f in skipped_d:
                rec_lines.append(f"  - `{f}` _(reused)_")
        parts.append("\n".join(rec_lines))

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

    # 5. Source files — scope to affected only on partial runs
    if is_partial and affected_set:
        # Primary: re-implemented files (full content)
        primary: dict[str, str] = {}
        for fp in sorted(affected_set):
            p = ROOT / fp
            if p.exists():
                primary[fp] = p.read_text()
        src_block = "\n\n".join(
            f"### {fp} _(re-implemented)_\n```typescript\n{code}\n```"
            for fp, code in primary.items()
        )
        parts.append(
            f"## 5. Re-implemented Source Files ({len(primary)} files)\n\n{src_block}"
        )

        # Secondary: reused files — signatures only to save tokens
        impl_rec = ROOT / "scaffold" / "impl_qwen.json"
        skipped_paths = []
        if impl_rec.exists():
            skipped_paths = json.loads(impl_rec.read_text()).get("skipped_delta", [])
        if skipped_paths:
            secondary_lines = ["## 5b. Reused Files — signatures only (not re-implemented)\n"]
            for fp in skipped_paths:
                p = ROOT / fp
                if p.exists():
                    # Emit only non-blank, non-body lines as a lightweight signature view
                    sig_lines = [
                        l for l in p.read_text().splitlines()
                        if l.strip() and not l.strip().startswith("//")
                        and "throw new Error" not in l
                    ][:30]   # cap at 30 lines per file
                    secondary_lines.append(f"### {fp}")
                    secondary_lines.append("```typescript")
                    secondary_lines.extend(sig_lines)
                    secondary_lines.append("```\n")
            parts.append("\n".join(secondary_lines))
    else:
        # Full run — collect all changed files as before
        src_files = _collect_changed_files(ROOT / "src")
        src_block = "\n\n".join(
            f"### {fp}\n```typescript\n{code}\n```"
            for fp, code in src_files.items()
        )
        parts.append(
            f"## 5. Implemented Source Files ({len(src_files)} changed files)\n\n{src_block}"
        )

    # 6. Test files — always full (judge needs to verify test quality)
    test_files = _collect_src_files(ROOT / "tests")
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
        f"_Run type: **{review.get('run_type', 'full')}**_",
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

    # Partial run notes
    partial_notes = review.get("partial_run_notes")
    if partial_notes:
        lines += ["", "## Partial Run Notes", "", f"_{partial_notes}_", ""]

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
