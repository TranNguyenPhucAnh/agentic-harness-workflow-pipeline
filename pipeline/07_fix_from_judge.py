"""
pipeline/07_fix_from_judge.py
Step 7 — Fix blocking issues identified by the DeepSeek judge.

Called automatically by harness.py when 06_judge_deepseek.py exits with
verdict NEEDS_REVISION.  Human approval is NOT required for blocking issues
because the judge already classified them; this script acts on that verdict.

What this script does
─────────────────────
1. Parse reports/judge_raw.json → extract blocking_issues + non_blocking_notes
2. Map each blocking issue to the source file(s) it affects
3. For each blocking issue → call Minimax once with:
     spec + affected src files + judge's exact description
4. Apply patches (scope-locked to src/ only — never tests/)
5. Run vitest to confirm fixes (exit 1 if still failing)
6. Write scaffold/judge_findings.md — injected into Minimax/Qwen prompts on
   future runs so the same mistakes are not repeated

Writes
──────
    scaffold/judge_findings.md      ← persistent cross-run memory
    reports/judge_fix_report.json   ← this run's fix log
    src/**                          ← patched files

Does NOT
────────
    - Modify test files
    - Modify spec.md  (spec changes require human + version bump)
    - Re-run the judge (harness.py does that after this script exits 0)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import httpx
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import time

ROOT         = Path(__file__).parent.parent
SPEC_PATH    = ROOT / "spec.md"
REPORTS_DIR  = ROOT / "reports"
SCAFFOLD_DIR = ROOT / "scaffold"
REPORTS_DIR.mkdir(exist_ok=True)

JUDGE_RAW_PATH   = REPORTS_DIR / "judge_raw.json"
FIX_REPORT_PATH  = REPORTS_DIR / "judge_fix_report.json"
FINDINGS_PATH    = SCAFFOLD_DIR / "judge_findings.md"


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class JudgeFinding:
    description: str
    severity:    str          # "blocking" | "non_blocking"
    files:       list[str]    # src/ paths this finding affects
    section:     str = ""     # judge section that flagged it (for context)


@dataclass
class FixRecord:
    finding:     str
    files:       list[str]
    patched:     bool
    files_written: list[str]
    note:        str


# ════════════════════════════════════════════════════════════════════════════
# API
# ════════════════════════════════════════════════════════════════════════════

def _openrouter_call(model_id: str, messages: list, max_tokens: int = 32768) -> str:
    import time
    api_key = os.environ["OPENROUTER_API_KEY"]
    for attempt in range(2):
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model_id, "messages": messages,
                  "temperature": 0.1, "max_tokens": max_tokens},
            timeout=300,
        )
        r.raise_for_status()
        data     = r.json()
        usage    = data.get("usage", {})
        print(f"    [tokens] {model_id}: "
              f"prompt={usage.get('prompt_tokens','?')}, "
              f"completion={usage.get('completion_tokens','?')}")
        content = data["choices"][0]["message"]["content"]
        if content and content.strip():
            return content
        if attempt == 0:
            print(f"    [warn] empty response from {model_id}, retrying in 3s …",
                  file=sys.stderr)
            time.sleep(3)
    return ""


def call_minimax(messages: list) -> str:
    return _openrouter_call("minimax/minimax-m2.7", messages)


def call_qwen(messages: list) -> str:
    return _openrouter_call("qwen/qwen3.6-plus", messages)


# ════════════════════════════════════════════════════════════════════════════
# Parse judge_raw.json
# ════════════════════════════════════════════════════════════════════════════

def load_judge_verdict() -> dict:
    if not JUDGE_RAW_PATH.exists():
        print(f"[07] ERROR: {JUDGE_RAW_PATH} not found.", file=sys.stderr)
        print("[07] Run 06_judge_deepseek.py first.", file=sys.stderr)
        sys.exit(1)
    raw_data = json.loads(JUDGE_RAW_PATH.read_text())
    raw_resp = raw_data.get("response", "")
    raw_resp = re.sub(r"^```[a-z]*\n?", "", raw_resp.strip())
    raw_resp = re.sub(r"\n?```$",        "", raw_resp.strip())
    try:
        return json.loads(raw_resp)
    except json.JSONDecodeError as e:
        print(f"[07] ERROR: could not parse judge response JSON: {e}", file=sys.stderr)
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# File mapping — which src files does each finding affect?
# ════════════════════════════════════════════════════════════════════════════

# Explicit path pattern in judge text (e.g. "src/hooks/useReplay.ts")
_RE_SRC_PATH = re.compile(r"src/[\w/]+\.(?:ts|tsx)")

# Keyword → likely files heuristic (fallback when no explicit path in text)
_KEYWORD_FILES: dict[str, list[str]] = {
    "useSensorData":  ["src/hooks/useSensorData.ts", "src/data/demoConstants.ts"],
    "useReplay":      ["src/hooks/useReplay.ts"],
    "demoConstants":  ["src/data/demoConstants.ts"],
    "useMemo":        ["src/hooks/useSensorData.ts"],
    "requestAnimationFrame": ["src/hooks/useReplay.ts"],
    "rAF":            ["src/hooks/useReplay.ts"],
    "setInterval":    ["src/hooks/useReplay.ts"],
    "anomaly":        ["src/hooks/useSensorData.ts"],
    "jumpToNext":     ["src/hooks/useReplay.ts"],
    "windowStart":    ["src/hooks/useReplay.ts"],
    "duplicate":      ["src/hooks/useSensorData.ts", "src/data/demoConstants.ts"],
}

# Component-wide scan for theme/Tailwind issues
_COMPONENT_SCAN_KEYWORDS = {"theme", "tailwind", "bg-white", "text-gray-800",
                             "dark theme", "light theme", "colour", "color"}


def _infer_files(text: str) -> list[str]:
    """Extract src/ file paths from finding text using explicit paths + keywords."""
    found: list[str] = []

    # 1. Explicit paths in text
    for m in _RE_SRC_PATH.findall(text):
        if m not in found:
            found.append(m)

    # 2. Keyword heuristics
    text_lower = text.lower()
    for kw, files in _KEYWORD_FILES.items():
        if kw.lower() in text_lower:
            for f in files:
                if f not in found and (ROOT / f).exists():
                    found.append(f)

    # 3. Component scan for theme issues
    if any(kw in text_lower for kw in _COMPONENT_SCAN_KEYWORDS):
        comp_dir = ROOT / "src" / "components"
        if comp_dir.exists():
            for p in sorted(comp_dir.rglob("*.tsx")):
                rel = str(p.relative_to(ROOT))
                if rel not in found:
                    found.append(rel)

    # Filter: only existing files
    return [f for f in found if (ROOT / f).exists()]


def extract_findings(verdict: dict) -> tuple[list[JudgeFinding], list[JudgeFinding]]:
    """Return (blocking, non_blocking) as JudgeFinding lists with file mappings."""
    # Extract section notes for richer context
    sections = verdict.get("sections", {})
    section_notes = {k: v.get("notes", "") for k, v in sections.items()}

    blocking: list[JudgeFinding] = []
    for desc in verdict.get("blocking_issues", []):
        # Find which section mentioned this issue
        section_hint = ""
        for sec_name, notes in section_notes.items():
            if any(word in notes.lower() for word in desc.lower().split()[:4]):
                section_hint = sec_name
                break
        blocking.append(JudgeFinding(
            description=desc,
            severity="blocking",
            files=_infer_files(desc + " " + section_notes.get(section_hint, "")),
            section=section_hint,
        ))

    non_blocking: list[JudgeFinding] = []
    for desc in verdict.get("non_blocking_notes", []):
        non_blocking.append(JudgeFinding(
            description=desc,
            severity="non_blocking",
            files=_infer_files(desc),
            section="",
        ))

    return blocking, non_blocking


# ════════════════════════════════════════════════════════════════════════════
# Fix prompts
# ════════════════════════════════════════════════════════════════════════════

JUDGE_FIX_SYSTEM_MINIMAX = """\
You are a senior TypeScript engineer fixing issues identified by a code reviewer (judge).

The judge has already diagnosed the problem. Your job is to implement the fix precisely.

Context you will receive:
1. spec.md — authoritative requirements
2. The exact judge finding (what is wrong and why)
3. All affected source files

Rules:
- Fix ONLY what the judge finding describes — do not refactor unrelated code
- If fixing a duplicate implementation: remove the hook from the wrong file,
  keep it only in src/hooks/; demoConstants.ts should export only constants
- TypeScript strict — no `any`
- Tailwind only — no inline styles (exception: dynamic width percentages)
- Output raw JSON only, no markdown fences:

{
  "files": [
    {
      "file_path": "src/hooks/useSensorData.ts",
      "code": "<full corrected file content>",
      "change_summary": "one sentence describing what was changed"
    }
  ],
  "root_cause": "one sentence: the precise bug",
  "fix_summary": "one sentence: what was done to fix it"
}

IMPORTANT: Return ALL affected files in the files array, even if only one changed.
"""

JUDGE_FIX_SYSTEM_QWEN = """\
You are a senior TypeScript/React developer fixing a surface issue identified by a judge.

The judge has diagnosed the problem. Fix it precisely — do not touch unrelated code.

Context: spec.md + judge finding + affected source files.

Rules:
- Tailwind only — use dark theme classes: bg-gray-800/900, text-gray-100/200/300
- TypeScript strict — no `any`
- Output raw JSON only, no markdown fences:

{
  "files": [
    {
      "file_path": "src/components/AnomalyFeed.tsx",
      "code": "<full corrected file content>",
      "change_summary": "one sentence"
    }
  ],
  "fix_summary": "one sentence"
}
"""


def _load_spec() -> str:
    compressed = SCAFFOLD_DIR / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else SPEC_PATH.read_text()


def _read_safe(path: Path) -> str:
    return path.read_text() if path.exists() else f"// FILE NOT FOUND: {path}\n"


def _parse_fix_response(raw: str, label: str) -> dict | None:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$",        "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"    [07] JSON parse failed for {label}: {e}", file=sys.stderr)
        print(f"    [07] Raw (first 300): {raw[:300]}", file=sys.stderr)
        return None


# ════════════════════════════════════════════════════════════════════════════
# Fix executor
# ════════════════════════════════════════════════════════════════════════════

# Blocking issues whose nature is architectural/logic → Minimax
# Non-blocking theme/Tailwind issues → Qwen
_SURFACE_KEYWORDS = {"theme", "tailwind", "colour", "color", "bg-", "text-",
                     "dark", "light", "class"}


def _choose_agent(finding: JudgeFinding) -> tuple[str, str]:
    """Return (agent_label, system_prompt) for this finding."""
    text_lower = finding.description.lower()
    if finding.severity == "non_blocking" and \
            any(kw in text_lower for kw in _SURFACE_KEYWORDS):
        return "qwen", JUDGE_FIX_SYSTEM_QWEN
    return "minimax", JUDGE_FIX_SYSTEM_MINIMAX


def fix_finding(
    finding: JudgeFinding,
    spec:    str,
    verdict: dict,
    verbose: bool,
) -> FixRecord:
    """Call the appropriate agent, apply patch, return FixRecord."""

    if not finding.files:
        print(f"  [07] No files mapped for: {finding.description[:60]}… — skipping")
        return FixRecord(
            finding=finding.description, files=[],
            patched=False, files_written=[],
            note="no files mapped — check heuristics or add explicit path in judge finding",
        )

    agent_label, system = _choose_agent(finding)
    call_fn = call_minimax if agent_label == "minimax" else call_qwen

    # Build context block: all affected files
    files_block = ""
    for fp in finding.files:
        code = _read_safe(ROOT / fp)
        files_block += f"\n### {fp}\n```typescript\n{code}\n```\n"

    # Include relevant section notes for richer context
    sections_block = ""
    if finding.section:
        sec = verdict.get("sections", {}).get(finding.section, {})
        if sec.get("notes"):
            sections_block = (
                f"\n### Judge section notes ({finding.section})\n"
                f"{sec['notes']}\n"
            )

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Judge finding (blocking issue to fix)\n"
        f"{finding.description}\n"
        f"{sections_block}\n"
        f"### Affected source files\n{files_block}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]

    print(f"  [07] → {agent_label.upper()} fixing: {finding.description[:70]}…")
    if verbose:
        print(f"  [07]   files: {finding.files}")

    raw = call_fn(messages).strip()
    if not raw:
        return FixRecord(
            finding=finding.description, files=finding.files,
            patched=False, files_written=[],
            note="model returned empty response",
        )

    patch = _parse_fix_response(raw, finding.description[:40])
    if not patch:
        return FixRecord(
            finding=finding.description, files=finding.files,
            patched=False, files_written=[],
            note="JSON parse failed",
        )

    # Apply patches — scope guard: never write to tests/
    written: list[str] = []
    for entry in patch.get("files", []):
        out_rel  = entry.get("file_path", "")
        out_path = ROOT / out_rel

        if not out_rel.startswith("src/"):
            print(f"  [07] ⚠ Scope violation: {out_rel} not under src/ — rejected",
                  file=sys.stderr)
            continue
        if "test" in out_rel.lower():
            print(f"  [07] ⚠ Tried to write test file {out_rel} — rejected",
                  file=sys.stderr)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(entry["code"])
        written.append(out_rel)
        change = entry.get("change_summary", "")
        print(f"  [07] ✓ Wrote {out_rel}" + (f" — {change}" if change else ""))

    fix_summary = patch.get("fix_summary", "")
    root_cause  = patch.get("root_cause", "")
    note = f"{root_cause} | {fix_summary}" if root_cause else fix_summary

    return FixRecord(
        finding=finding.description,
        files=finding.files,
        patched=bool(written),
        files_written=written,
        note=note,
    )


# ════════════════════════════════════════════════════════════════════════════
# vitest confirm
# ════════════════════════════════════════════════════════════════════════════

def run_vitest_confirm() -> tuple[bool, str]:
    print("\n[07] Running vitest to confirm fixes …")
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=ROOT, capture_output=True, text=True,
    )
    output = result.stdout + "\n" + result.stderr
    passed = result.returncode == 0
    summary = next(
        (l.strip() for l in output.splitlines()
         if ("passed" in l or "failed" in l) and "test" in l.lower()),
        "no summary line found",
    )
    icon = "✓" if passed else "✗"
    print(f"[07] vitest {icon} {summary}")
    return passed, output


# ════════════════════════════════════════════════════════════════════════════
# judge_findings.md — persistent cross-run memory
# ════════════════════════════════════════════════════════════════════════════

def write_judge_findings(
    blocking:     list[JudgeFinding],
    non_blocking: list[JudgeFinding],
    fix_records:  list[FixRecord],
    verdict:      dict,
) -> None:
    """
    Write scaffold/judge_findings.md.
    This file is injected into Minimax and Qwen system prompts on future runs
    so the same mistakes are not repeated.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scores = {k: v.get("score", "?")
              for k, v in verdict.get("sections", {}).items() if "score" in v}

    lines = [
        f"# Judge findings — {now}",
        f"_Verdict: {verdict.get('verdict')} | "
        f"Scores: {', '.join(f'{k}={v}/5' for k, v in scores.items())}_",
        "",
        "## Blocking issues (were fixed in this run — do not reintroduce)",
        "",
    ]

    fixed_set  = {r.finding for r in fix_records if r.patched}
    failed_set = {r.finding for r in fix_records if not r.patched}

    for f in blocking:
        status = "✓ fixed" if f.description in fixed_set else "✗ fix failed — needs human"
        lines.append(f"- [{status}] {f.description}")
        if f.files:
            lines.append(f"  → files: {', '.join(f.files)}")

    lines += [
        "",
        "## Non-blocking notes (watch out on future runs)",
        "",
    ]
    for f in non_blocking:
        lines.append(f"- {f.description}")

    lines += [
        "",
        "## Patterns to avoid (extracted from judge analysis)",
        "",
        "### Architecture",
    ]

    # Extract architecture notes from judge sections
    arch_notes = verdict.get("sections", {}).get("architecture", {}).get("notes", "")
    if arch_notes:
        lines.append(arch_notes)

    lines += [
        "",
        "### Code quality",
    ]
    quality_notes = verdict.get("sections", {}).get("code_quality", {}).get("notes", "")
    if quality_notes:
        lines.append(quality_notes)

    lines += [
        "",
        "---",
        "_This file is auto-generated. Do not edit manually._",
        "_To clear findings, delete this file before next run._",
    ]

    FINDINGS_PATH.write_text("\n".join(lines) + "\n")
    print(f"\n[07] Findings written → {FINDINGS_PATH}")
    print("[07] Injected into Minimax + Qwen prompts on next run.")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",        action="store_true")
    parser.add_argument("--fix-blocking",   action="store_true", default=True,
                        help="Fix blocking issues (default: True)")
    parser.add_argument("--fix-non-blocking", action="store_true", default=False,
                        help="Also attempt to fix non-blocking notes (default: False)")
    parser.add_argument("--skip-vitest",    action="store_true",
                        help="Skip vitest confirm after fixes (for debugging)")
    args = parser.parse_args()

    # ── Load verdict ──────────────────────────────────────────────────────────
    verdict = load_judge_verdict()

    if verdict.get("verdict") == "APPROVED":
        print("[07] Judge already APPROVED — nothing to fix.")
        sys.exit(0)

    print(f"[07] Judge verdict: {verdict['verdict']}")
    print(f"[07] Summary: {verdict.get('summary','')[:120]}")

    blocking, non_blocking = extract_findings(verdict)

    print(f"\n[07] Blocking issues ({len(blocking)}):")
    for f in blocking:
        mapped = f.files or ["(no files mapped)"]
        print(f"  • {f.description[:80]}")
        print(f"    files: {mapped}")

    print(f"\n[07] Non-blocking notes ({len(non_blocking)}):")
    for f in non_blocking:
        mapped = f.files or ["(no files mapped)"]
        print(f"  • {f.description[:80]}")
        print(f"    files: {mapped}, agent: {_choose_agent(f)[0]}")

    # ── Fix ───────────────────────────────────────────────────────────────────
    spec         = _load_spec()
    fix_records: list[FixRecord] = []

    to_fix: list[JudgeFinding] = []
    if args.fix_blocking:
        to_fix.extend(blocking)
    if args.fix_non_blocking:
        to_fix.extend(non_blocking)

    if not to_fix:
        print("\n[07] Nothing to fix (no target findings).")
    else:
        print(f"\n[07] Fixing {len(to_fix)} finding(s) …")
        for finding in to_fix:
            record = fix_finding(finding, spec, verdict, verbose=args.verbose)
            fix_records.append(record)

    # ── vitest confirm ────────────────────────────────────────────────────────
    vitest_passed = False
    vitest_summary = "skipped"

    if fix_records and not args.skip_vitest:
        vitest_passed, vitest_output = run_vitest_confirm()
        vitest_summary = vitest_output[-800:]
    elif args.skip_vitest:
        print("\n[07] Skipping vitest (--skip-vitest)")
        vitest_passed = True   # let harness decide

    # ── Write judge_findings.md ───────────────────────────────────────────────
    write_judge_findings(blocking, non_blocking, fix_records, verdict)

    # ── Fix report ────────────────────────────────────────────────────────────
    n_patched = sum(1 for r in fix_records if r.patched)
    report = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "judge_verdict":    verdict.get("verdict"),
        "blocking_count":   len(blocking),
        "non_blocking_count": len(non_blocking),
        "fix_attempts":     len(fix_records),
        "fixes_patched":    n_patched,
        "vitest_passed":    vitest_passed,
        "vitest_summary":   vitest_summary,
        "records":          [asdict(r) for r in fix_records],
    }
    FIX_REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n[07] Fix report → {FIX_REPORT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  STEP 7 SUMMARY")
    print(f"{'='*50}")
    print(f"  Blocking issues:    {len(blocking)}")
    print(f"  Fixes applied:      {n_patched}/{len(fix_records)}")
    print(f"  vitest after fix:   {'✓ PASS' if vitest_passed else '✗ FAIL'}")

    for r in fix_records:
        icon = "✅" if r.patched else "❌"
        print(f"  {icon} {r.finding[:65]}")
        if r.files_written:
            for fw in r.files_written:
                print(f"     → {fw}")
        if r.note:
            print(f"     note: {r.note[:80]}")

    failed_fixes = [r for r in fix_records if not r.patched]
    if failed_fixes:
        print(f"\n[07] ⚠ {len(failed_fixes)} fix(es) failed — needs human review:")
        for r in failed_fixes:
            print(f"     • {r.finding[:80]}")
            print(f"       {r.note}")

    # Exit code: 0 only if vitest passed (or skipped)
    # harness.py uses this to decide whether to re-run judge
    if not vitest_passed and not args.skip_vitest:
        print("\n[07] Tests still failing after judge fixes — human review needed.")
        sys.exit(1)

    print("\n[07] Done. harness.py will re-run judge to confirm improvement.")
    sys.exit(0)


if __name__ == "__main__":
    main()
