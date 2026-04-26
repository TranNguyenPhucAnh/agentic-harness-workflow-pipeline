"""
pipeline/04_test_and_iterate.py
Step 4+5 — Clustered repair: Qwen (surface bugs) + Minimax (logic bugs).

Role split:
    Qwen 3.6+    — Layer 1 (surface): DOM selector, import path, Tailwind class,
                   float precision, text contract. ONE targeted pass per cluster.
                   If Qwen detects a logic bug it signals LOGIC_BUG and steps aside.
    Minimax 2.7  — Layer 2 (logic): hook state machines, data generation, range
                   constraints, AC-* acceptance criteria. Scope-locked to
                   src/hooks/, src/data/, src/types/, src/utils/ only.

Cluster ownership:
    - Hook/data files (MINIMAX_SCOPE) → skip Qwen, go straight to Minimax L2.
    - Component files → Qwen L1 first; if LOGIC_BUG and in scope → Minimax L2;
      if out of scope and unfixable → ESCALATED→human.
    - Once a cluster is owned by Minimax, Qwen will NOT touch it again.
    - Minimax patches outside its scope are rejected silently.

Phase flow:
    Phase B : run vitest → parse output → list[FailureCluster]
    Phase C : per-cluster dispatch:
        P0 — Consistency: cross-check test vs code vs spec (no file writes)
             Verdicts: CODE_BUG | TEST_FRAGILE | SPEC_AMBIG | THRESHOLD_OK
             TEST_FRAGILE / THRESHOLD_OK → allowed to patch test file (query/threshold only)
             SPEC_AMBIG                  → escalate to human immediately, skip repair
             CODE_BUG                   → normal repair flow below
        L0 — Static     : esbuild/float fixes (no LLM)
        L1 — Qwen       : surface bugs (components); skipped for hook/data files
        L2 — Minimax    : logic bugs (hooks/data); activated on stale/scope/LOGIC_BUG
        L3 — Give-up    : ESCALATED after max_cluster_attempts
    Phase D : rerun vitest → repeat

Writes:
    reports/qwen_iterations.json
    reports/escalated_clusters.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import httpx
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable
import time

ROOT        = Path(__file__).parent.parent
SPEC_PATH   = ROOT / "spec.md"
GLM_PLAN    = ROOT / "scaffold" / "glm_plan.json"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SRC_DIR = "src"

# Files Minimax is allowed to write — hooks and pure logic only
MINIMAX_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^src/hooks/"),
    re.compile(r"^src/data/"),
    re.compile(r"^src/types/"),
    re.compile(r"^src/utils/"),
]


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFailure:
    test_file:     str
    test_name:     str
    error_snippet: str


@dataclass
class FailureCluster:
    """One cluster = one test file + its corresponding src file."""
    test_file:  str
    src_file:   str
    failures:   list[TestFailure] = field(default_factory=list)

    attempt_count:      int  = field(default=0)
    last_fingerprint:   str  = field(default="")
    escalated:          bool = field(default=False)
    is_transform_error: bool = field(default=False)
    # "qwen" until Minimax takes over; then "minimax" permanently
    owner:              str  = field(default="qwen")

    @property
    def key(self) -> str:
        return self.test_file

    def error_block(self) -> str:
        return "\n\n".join(
            f"  x {f.test_name}\n{f.error_snippet}" for f in self.failures
        )

    def fingerprint(self) -> str:
        return re.sub(r"\s+", " ", self.error_block()).strip()[:400]

    def is_minimax_scope(self) -> bool:
        return any(p.match(self.src_file) for p in MINIMAX_SCOPE_PATTERNS)


@dataclass
class ClusterRepairRecord:
    cluster:              str
    src_file:             str
    failures:             int
    repaired:             bool
    layer_used:           str   # "static"|"qwen_targeted"|"minimax_logic"|"test_rewrite"|"skipped"
    escalated:            bool
    escalated_to:         str   # ""|"minimax"|"human"
    owner:                str   # "qwen"|"minimax"
    note:                 str = ""
    consistency_verdict:  str = ""   # P0 verdict: CODE_BUG|TEST_FRAGILE|SPEC_AMBIG|THRESHOLD_OK|""


@dataclass
class IterationRecord:
    iteration:         int
    passed:            bool
    summary:           str
    clusters_found:    int
    clusters_repaired: int
    cluster_details:   list[dict]
    log_snippet:       str


# ════════════════════════════════════════════════════════════════════════════
# API helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_spec() -> str:
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else SPEC_PATH.read_text()

def _openrouter_call(model_id: str, messages: list, max_tokens: int = 32768) -> str:
    import time
    api_key = os.environ["OPENROUTER_API_KEY"]

    for attempt in range(2):   # 1 retry on empty response
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_id, "messages": messages,
                  "temperature": 0.1, "max_tokens": max_tokens},
            timeout=300,
        )
        r.raise_for_status()
        data         = r.json()
        usage        = data.get("usage", {})
        prompt_t     = usage.get("prompt_tokens", "?")
        completion_t = usage.get("completion_tokens", "?")
        print(f"    [tokens] {model_id}: prompt={prompt_t}, completion={completion_t}")
        content = data["choices"][0]["message"]["content"]
        if content and content.strip():
            return content
        # Empty response — wait and retry once
        if attempt == 0:
            print(f"    [warn] {model_id} returned empty response, retrying in 3s …",
                  file=sys.stderr)
            time.sleep(3)

    return ""   # caller handles empty string as parse error


def call_qwen(messages: list) -> str:
    return _openrouter_call("qwen/qwen3.6-plus", messages)


def call_minimax(messages: list) -> str:
    return _openrouter_call("minimax/minimax-m2.7", messages)


# ════════════════════════════════════════════════════════════════════════════
# GLM plan — global_notes loader
# ════════════════════════════════════════════════════════════════════════════

def _load_glm_global_notes() -> str:
    if not GLM_PLAN.exists():
        return ""
    try:
        return json.loads(GLM_PLAN.read_text()).get("global_notes", "")
    except Exception:
        return ""
        
# ════════════════════════════════════════════════════════════════════════════
# Judge findings loader — cross-run regression prevention
# ════════════════════════════════════════════════════════════════════════════

FINDINGS_PATH = ROOT / "scaffold" / "judge_findings.md"


def _load_judge_findings() -> str:
    """
    Load scaffold/judge_findings.md written by 07_fix_from_judge.py.
    Returns empty string if file doesn't exist (first run, no judge yet).
    Injected into both Minimax and Qwen prompts so the same mistakes
    from previous runs are not repeated.
    """
    if not FINDINGS_PATH.exists():
        return ""
    try:
        content = FINDINGS_PATH.read_text().strip()
        return content if content else ""
    except Exception:
        return ""

# ════════════════════════════════════════════════════════════════════════════
# Phase B — run vitest + parse failures
# ════════════════════════════════════════════════════════════════════════════

def run_vitest() -> tuple[bool, str]:
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return result.returncode == 0, result.stdout + "\n" + result.stderr


_RE_TEST_FILE   = re.compile(r"^\s*(FAIL|PASS)\s+(tests/\S+\.test\.[tj]sx?)", re.MULTILINE)
_RE_FAIL_TEST   = re.compile(r"^\s+[x\u00d7\u2717\u274c]\s+(.+)$", re.MULTILINE)
_RE_ERROR_BLOCK = re.compile(
    r"(AssertionError|Error|TypeError|ReferenceError)[^\n]*\n(?:[ \t]+[^\n]*\n)*",
    re.MULTILINE,
)
_RE_TRANSFORM_ERR = re.compile(
    r"(Transform failed|ERROR: Expected|SyntaxError.*esbuild|error TS\d+)",
    re.IGNORECASE,
)


def _infer_src_file(test_file: str) -> str:
    rel = test_file.replace("tests/", "", 1)
    rel = re.sub(r"\.test\.(tsx?)$", r".\1", rel)
    rel = re.sub(r"\.test\.(ts)$",   r".\1", rel)
    return f"{SRC_DIR}/{rel}"


def parse_failures(output: str) -> list[FailureCluster]:
    clusters: dict[str, FailureCluster] = {}
    file_matches = list(_RE_TEST_FILE.finditer(output))
    sections: list[tuple[str, str, str]] = []
    for i, m in enumerate(file_matches):
        start = m.end()
        end   = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(output)
        sections.append((m.group(1), m.group(2), output[start:end]))

    for status, test_file, section in sections:
        if status != "FAIL":
            continue
        src_file = _infer_src_file(test_file)
        cluster  = clusters.setdefault(
            test_file, FailureCluster(test_file=test_file, src_file=src_file),
        )
        if _RE_TRANSFORM_ERR.search(section):
            cluster.is_transform_error = True
        fail_names = _RE_FAIL_TEST.findall(section)
        errors     = _RE_ERROR_BLOCK.findall(section)
        for j, name in enumerate(fail_names):
            snippet = errors[j] if j < len(errors) else section[:500]
            cluster.failures.append(TestFailure(
                test_file=test_file, test_name=name.strip(),
                error_snippet=snippet.strip()[:600],   # <-- thêm [:600]
            ))
        if not cluster.failures:
            cluster.failures.append(TestFailure(
                test_file=test_file, test_name="(parse fallback)",
                error_snippet=section[:800].strip(),   # <-- giảm từ 1500 → 800
            ))
    return list(clusters.values())


def merge_cluster_state(
    new_clusters: list[FailureCluster],
    prev_state:   dict[str, FailureCluster],
) -> list[FailureCluster]:
    for c in new_clusters:
        if c.key in prev_state:
            prev = prev_state[c.key]
            c.attempt_count    = prev.attempt_count
            c.last_fingerprint = prev.last_fingerprint
            c.escalated        = prev.escalated
            c.owner            = prev.owner   # preserve ownership
    return new_clusters


# ════════════════════════════════════════════════════════════════════════════
# Layer 0 — Static pre-pass (no LLM)
# ════════════════════════════════════════════════════════════════════════════

_RE_JSX_GENERIC    = re.compile(
    r"(<\w[\w.]*)<(\w[\w,\s]*)>(\s*(?:events|data|items|props|value)\s*=)",
)
_RE_TEMPLATE_WIDTH = re.compile(r"(`\$\{)([^}]*\*\s*100)(\}%`)")
_RE_FLOAT_WIDTH    = re.compile(r"(width:\s*)(\d+\.\d+)(%)")


def _static_fix_transform(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file not found"
    orig    = path.read_text()
    patched = _RE_JSX_GENERIC.sub(r"\1\3", orig)
    if patched != orig:
        path.write_text(patched)
        return True, "removed JSX generic type param causing esbuild parse error"
    return False, "no static transform pattern matched"


def _static_fix_src(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file not found"
    orig    = path.read_text()
    patched = _RE_TEMPLATE_WIDTH.sub(r"`${Math.round(\2)}\3", orig)
    patched = _RE_FLOAT_WIDTH.sub(
        lambda m: f"{m.group(1)}{round(float(m.group(2)))}{m.group(3)}", patched,
    )
    if patched != orig:
        path.write_text(patched)
        return True, "rounded floating-point percentage widths"
    return False, "no static src pattern matched"


def layer0_static_prepass(cluster: FailureCluster, verbose: bool) -> tuple[bool, str]:
    if verbose:
        print(f"    [L0] Static pre-pass for {cluster.test_file} …")
    if cluster.is_transform_error:
        fixed, desc = _static_fix_transform(ROOT / cluster.test_file)
        if fixed:
            print(f"    [L0] ✓ {desc}")
            return True, desc
    fixed, desc = _static_fix_src(ROOT / cluster.src_file)
    if fixed:
        print(f"    [L0] ✓ {desc}")
        return True, desc
    if verbose:
        print("    [L0] No static pattern matched.")
    return False, "no static fix applicable"


# ════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════

FIX_SYSTEM_QWEN = """\
You are a senior TypeScript/React developer doing a SURFACE-LEVEL fix for ONE failing cluster.

Your scope is LIMITED to surface bugs:
  - Wrong DOM selector or missing test-id attribute
  - Incorrect import path or missing export
  - Tailwind class name typo or wrong colour/state class
  - Floating-point precision in style values
  - Text content or badge label mismatch
  - Missing or wrong aria attribute

DO NOT:
  - Rewrite hook logic, state machines, or data generation
  - Change public interfaces (types, props, function signatures)
  - Touch any file other than the one src file listed

If the bug is a LOGIC or ALGORITHM issue (state transition, math, data shape),
return the source file UNCHANGED and set:
  "explanation": "LOGIC_BUG — needs Minimax debugger"

Return JSON (no other keys):
{
  "file_path": "src/components/SummaryStickyBar.tsx",
  "code": "<full file content — UNCHANGED if logic bug>",
  "explanation": "what was fixed, or LOGIC_BUG"
}
TypeScript strict — no `any`. Tailwind only. Raw JSON. No markdown fences.
"""

def _build_qwen_system_with_findings(findings: str) -> str:
    """
    Wrap FIX_SYSTEM_QWEN with judge findings block when available.
    Findings are placed BEFORE the task instructions so they act as
    a negative-example primer — model reads 'do not do X' before the task.
    """
    if not findings:
        return FIX_SYSTEM_QWEN

    # Extract only the non-blocking + patterns sections for Qwen
    # (Qwen handles surface/Tailwind issues — filter to relevant lines)
    relevant_lines: list[str] = []
    capture = False
    for line in findings.splitlines():
        if "## Non-blocking" in line or "## Patterns to avoid" in line:
            capture = True
        elif line.startswith("## ") and capture:
            # Stop at next major section that isn't relevant
            if "Blocking" in line:
                capture = False
        if capture:
            relevant_lines.append(line)

    findings_block = "\n".join(relevant_lines).strip()
    if not findings_block:
        return FIX_SYSTEM_QWEN

    return (
        f"## Previous run — do NOT repeat these mistakes\n"
        f"{findings_block}\n\n"
        f"---\n\n"
        + FIX_SYSTEM_QWEN
    )

def _load_knowledge_base() -> str:
    """Load knowledge_base.md if it exists — accumulated human fix patterns."""
    kb = ROOT / "scaffold" / "knowledge_base.md"
    if not kb.exists():
        return ""
    content = kb.read_text().strip()
    # Strip the file header (first 3 lines) — only pass the entries
    lines = content.splitlines()
    body_lines = [l for l in lines if not l.startswith("# ") and not l.startswith("_")]
    return "\n".join(body_lines).strip()


def _build_minimax_system(global_notes: str, judge_findings: str = "") -> str:
    notes_block = (
        f"\n## GLM Architect's Global Notes (MUST follow)\n{global_notes}\n"
        if global_notes else ""
    )

    # Judge findings: blocking issues as "do not reintroduce" checklist
    findings_block = ""
    if judge_findings:
        findings_block = (
            f"\n## Judge findings from previous run — do NOT reintroduce\n"
            f"{judge_findings}\n"
        )

    # Knowledge base: accumulated patterns from human fixes
    # Each entry documents a bug pattern that AI failed to fix autonomously.
    # Injected here so Minimax learns from human interventions over time.
    kb_content = _load_knowledge_base()
    kb_block = (
        f"\n## Accumulated knowledge from human fixes — study these patterns\n"
        f"These are bugs that the AI repair loop could NOT fix — a human had to\n"
        f"intervene. Pay close attention: if the current cluster resembles any of\n"
        f"these patterns, apply the same fix strategy.\n\n"
        f"{kb_content}\n"
        if kb_content else ""
    )

    return f"""\
You are a senior TypeScript logic debugger specialising in hooks and data generation.
{notes_block}{findings_block}{kb_block}
You receive a failing cluster for a hook or data file.
Fix the LOGIC — not the UI, not the styling.

SCOPE (STRICTLY ENFORCED):
  - You may ONLY write to: src/hooks/, src/data/, src/types/, src/utils/
  - Never touch src/components/ — those files are owned by another agent
  - Never change public interfaces unless they directly contradict spec + test

WHAT TO FIX:
  - Hook state machine transitions (play/pause/reset/scrub semantics)
  - Data generation (anomaly injection, rates, cluster timing)
  - Range/constraint violations (AC-4 point count, anomaly rate 0.2–1.5%)
  - NaN / undefined from uninitialised constants or wrong array access
  - Off-by-one in window/index calculations

HOW:
  1. Read each test assertion as a hard requirement.
  2. Trace the state timeline step by step.
  3. If this cluster resembles a pattern in the knowledge base above, apply that fix.
  4. Identify the root cause — one specific line or function.
  5. Rewrite ONLY the broken function(s). Leave everything else intact.

Return JSON:
{{
  "file_path": "src/hooks/useReplay.ts",
  "code": "<full corrected file>",
  "root_cause": "one sentence: what was logically wrong",
  "explanation": "what you changed and why"
}}
TypeScript strict — no `any`. Raw JSON. No markdown fences.
"""


def _build_state_timeline(test_code: str, max_entries: int = 12) -> str:
    lines    = test_code.splitlines()
    timeline: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("import"):
            continue
        if s.startswith("describe(") or s.startswith("it(") or s.startswith("test("):
            timeline.append(f"TEST: {s[:100]}")
        elif "render(" in s or "fireEvent" in s or "userEvent" in s or "act(" in s:
            timeline.append(f"  ACTION: {s[:100]}")
        elif s.startswith("expect("):
            timeline.append(f"  ASSERT: {s[:100]}")
        else:
            continue
        if len(timeline) >= max_entries * 3:    # <-- NEW: hard stop
            timeline.append("  … (truncated)")
            break
    return "\n".join(timeline) if timeline else "(could not extract timeline)"


def _read_file_safe(path: Path) -> str:
    return path.read_text() if path.exists() else f"// FILE NOT FOUND: {path}\n"


# ════════════════════════════════════════════════════════════════════════════
# Phase 0 — Consistency checker
# ════════════════════════════════════════════════════════════════════════════

CONSISTENCY_SYSTEM = """\
You are a test-vs-code consistency auditor. You do NOT fix code or tests.
Your only job: read the spec, the failing test, the source file, and the assertion,
then classify who is wrong.

Return raw JSON only — no markdown fences, no preamble:
{
  "verdict": "CODE_BUG" | "TEST_FRAGILE" | "SPEC_AMBIG" | "THRESHOLD_OK",
  "confidence": "high" | "medium" | "low",
  "reasoning": "one paragraph — cite the specific spec section, test line, and code line",
  "test_patch_allowed": true | false,
  "test_patch_rationale": "only if test_patch_allowed=true — exactly what to change and why"
}

Verdict definitions:
  CODE_BUG      — code does not implement the spec correctly; test expectation is valid.
                  test_patch_allowed MUST be false.
  TEST_FRAGILE  — test query or assertion is brittle (wrong selector, DOM structure mismatch,
                  text split across nodes, timing issue) but the INTENT is correct.
                  The fix is to rewrite the query/assertion, NOT to change what is being tested.
                  test_patch_allowed = true.
  SPEC_AMBIG    — spec is genuinely ambiguous about this behaviour; test and code both have
                  valid interpretations. Cannot be resolved automatically.
                  test_patch_allowed MUST be false.
  THRESHOLD_OK  — code behaviour is correct per spec, but test uses a threshold too strict
                  for the implementation (e.g. spec says "~0.4% ± variance" but test asserts
                  exact ≤ 1.5% when actual variance can exceed that).
                  test_patch_allowed = true ONLY if relaxing the threshold does NOT mask a real bug.

CRITICAL: never set test_patch_allowed=true for CODE_BUG. That would hide the real defect.
"""

TEST_REPAIR_SYSTEM = """\
You are fixing a FRAGILE TEST or THRESHOLD. The test intent is correct — only the
implementation of the assertion is brittle or too strict.

Allowed changes ONLY:
  TEST_FRAGILE  — fix DOM query selectors, async patterns, or text-content matchers:
                  • Replace getByText(regex) that matches multiple elements with
                    within(container).getByText(), getByRole(), or getByTestId()
                  • Replace text split across nodes with:
                    getByText((_, el) => el?.textContent?.includes('...'))
                  • Fix missing act() wrappers for state updates
  THRESHOLD_OK  — relax a numerical threshold WITH explicit spec citation:
                  • e.g. change toBeLessThanOrEqual(0.015) to toBeLessThanOrEqual(0.025)
                    citing "spec says ~0.4% with natural variance, no hard upper bound"

NOT allowed — ANY of these voids the patch:
  • Changing what behaviour is being tested
  • Removing assertions or reducing assertion count
  • Changing expected values to match wrong code behaviour
  • Adding trivial pass-through assertions like expect(true).toBe(true)
  • Touching src/ files

Return raw JSON only:
{
  "file_path": "tests/components/AnomalyFeed.test.tsx",
  "code": "<full corrected test file>",
  "changes_made": ["one item per change, quoted verbatim before → after"],
  "explanation": "one sentence"
}
"""


def check_consistency(
    cluster:  FailureCluster,
    spec:     str,
    verbose:  bool = False,
) -> dict:
    """
    Phase 0: cross-check test vs code vs spec. Does NOT modify any files.
    Returns verdict dict. Falls back to CODE_BUG on any error.
    """
    src_code  = _read_file_safe(ROOT / cluster.src_file)
    test_code = _read_file_safe(ROOT / cluster.test_file)
    error_log = cluster.error_block()

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Test file: {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Source file: {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Failing assertions\n```\n{error_log}\n```"
    )

    messages = [
        {"role": "system", "content": CONSISTENCY_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if verbose:
        print(f"    [P0] Consistency check → Qwen ({cluster.test_file}) …")

    try:
        raw = call_qwen(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        result = json.loads(raw)
        verdict = result.get("verdict", "CODE_BUG")
        conf    = result.get("confidence", "low")
        print(f"    [P0] verdict={verdict} confidence={conf}")
        if verbose:
            print(f"    [P0] reasoning: {result.get('reasoning','')[:200]}")
        return result
    except Exception as e:
        print(f"    [P0] Check failed ({e}), defaulting to CODE_BUG.", file=sys.stderr)
        return {
            "verdict": "CODE_BUG", "confidence": "low",
            "test_patch_allowed": False,
            "reasoning": f"consistency check error: {e}",
            "test_patch_rationale": "",
        }


def repair_test_file(
    cluster:   FailureCluster,
    verdict:   dict,
    verbose:   bool = False,
) -> bool:
    """
    P0 branch: rewrite fragile test query or relax threshold.
    Returns True if patch was applied and file written.
    """
    spec      = _load_spec()
    test_code = _read_file_safe(ROOT / cluster.test_file)
    src_code  = _read_file_safe(ROOT / cluster.src_file)
    error_log = cluster.error_block()

    rationale_block = (
        f"### Auditor rationale\n{verdict.get('test_patch_rationale', '')}\n\n"
    )

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Source file (DO NOT MODIFY): {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Test file to fix: {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Failing assertions\n```\n{error_log}\n```\n\n"
        f"{rationale_block}"
        f"Verdict: {verdict.get('verdict')} — fix ONLY what the rationale describes."
    )

    messages = [
        {"role": "system", "content": TEST_REPAIR_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if verbose:
        print(f"    [P0-fix] Rewriting test: {cluster.test_file} …")

    try:
        raw = call_qwen(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$",        "", raw)
        patch = json.loads(raw)
    except Exception as e:
        print(f"    [P0-fix] Parse error: {e}", file=sys.stderr)
        return False

    out_path = ROOT / patch.get("file_path", cluster.test_file)

    # Safety: only allow writes to tests/
    if not str(out_path).startswith(str(ROOT / "tests")):
        print(f"    [P0-fix] ⚠ Scope violation: tried to write {out_path}. Rejected.",
              file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch["code"])

    changes = patch.get("changes_made", [])
    print(f"    [P0-fix] ✓ Test updated — {patch.get('explanation', '(no explanation)')}")
    for ch in changes:
        print(f"      • {ch}")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Shared repair executor
# ════════════════════════════════════════════════════════════════════════════

def _call_repair(
    cluster:     FailureCluster,
    call_api:    Callable[[list], str],
    system:      str,
    extra_ctx:   str  = "",
    verbose:     bool = False,
    layer_name:  str  = "L1",
    scope_check: bool = False,
) -> tuple[bool, str]:
    """
    Call model, apply patch.
    Returns (patched: bool, explanation: str).
    Special: if layer_name=="L1" and explanation contains "LOGIC_BUG", returns (False, "LOGIC_BUG").
    """
    spec      = _load_spec()
    src_code  = _read_file_safe(ROOT / cluster.src_file)
    test_code = _read_file_safe(ROOT / cluster.test_file)
    error_log = cluster.error_block()

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Test file (read-only): {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Source file to fix: {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Failing tests\n```\n{error_log}\n```"
        + (f"\n\n### Expected state timeline\n```\n{extra_ctx}\n```" if extra_ctx else "")
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]

    model_label = "Qwen" if layer_name == "L1" else "Minimax"
    if verbose:
        print(f"    [{layer_name}] → {model_label} "
              f"(attempt #{cluster.attempt_count + 1}, "
              f"{len(cluster.failures)} failure(s)) …")

    try:
        raw = call_api(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        patch = json.loads(raw)
    except Exception as e:
        print(f"    [{layer_name}] Parse error: {e}", file=sys.stderr)
        return False, f"parse error: {e}"

    explanation = patch.get("explanation", "")

    # Qwen LOGIC_BUG signal — do not write file, route to Minimax
    if layer_name == "L1" and "LOGIC_BUG" in explanation.upper():
        print(f"    [L1] Qwen signalled LOGIC_BUG — deferring to Minimax.")
        return False, "LOGIC_BUG"

    out_rel = patch.get("file_path", cluster.src_file)

    # Scope guard for Minimax
    if scope_check and not any(p.match(out_rel) for p in MINIMAX_SCOPE_PATTERNS):
        print(f"    [{layer_name}] ⚠ Scope violation: tried to write {out_rel}. "
              f"Allowed: hooks/, data/, types/, utils/. Patch rejected.")
        return False, f"scope violation: {out_rel}"

    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch["code"])

    root_cause = patch.get("root_cause", "")
    summary    = f"{root_cause} — {explanation}" if root_cause else explanation
    print(f"    [{layer_name}] ✓ Patched {out_rel} — {summary or '(no explanation)'}")
    return True, explanation


# ════════════════════════════════════════════════════════════════════════════
# Phase C — cluster dispatch
# ════════════════════════════════════════════════════════════════════════════

def repair_cluster(
    cluster:              FailureCluster,
    global_notes:         str,
    max_cluster_attempts: int,
    judge_findings:       str = "",
    verbose:              bool = False,
) -> ClusterRepairRecord:
    """
    Dispatch cluster through P0 → L0 → L1/L2 → L3.

    Full decision tree (top to bottom, first match wins):

      P0  Consistency check — FIRST ATTEMPT ONLY (skipped on retries):
            SPEC_AMBIG                        → ESCALATED→human immediately
            TEST_FRAGILE / THRESHOLD_OK
              + test_patch_allowed=true       → repair_test_file() → done
            CODE_BUG (or low confidence)      → fall through to L0 below

      L3  Give-up guard — checked before every LLM call:
            cluster.escalated=True            → SKIP (already given up)
            attempt_count >= max_attempts     → ESCALATED→human

      L0  Static pre-pass (no LLM):
            esbuild transform error           → patch test file (JSX generic fix)
            float precision in src            → Math.round() patch
            matched                           → done
            no match                          → fall through

      L1  Qwen surface fix — COMPONENTS ONLY:
            cluster.owner == "minimax"        → skip L1 (Qwen locked out)
            cluster.is_minimax_scope()        → skip L1 (hook/data → straight to L2)
            is_stale (fingerprint unchanged)  → skip L1 (→ L2)
            otherwise → call Qwen:
              fix applied                     → done, owner stays "qwen"
              LOGIC_BUG signal + in scope     → transfer owner→"minimax", → L2
              LOGIC_BUG + out of scope        → ESCALATED→human

      L2  Minimax logic fix — HOOKS / DATA ONLY (scope-locked):
            scope_check: rejects patches outside src/hooks/, src/data/, src/types/, src/utils/
            fix applied                       → done, owner stays "minimax"
            patch rejected / parse error      → repaired=False (will retry next iteration)
    """
    # ── L3 guard ─────────────────────────────────────────────────────────────
    if cluster.escalated:
        print(f"    [SKIP] {cluster.test_file} — ESCALATED→{cluster.owner}, skipping.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True,
            escalated_to=cluster.owner, owner=cluster.owner,
            note="previously escalated",
        )

    # ── Phase 0: consistency check (only on first attempt per cluster) ────────
    spec = _load_spec()
    if cluster.attempt_count == 0:
        cv = check_consistency(cluster, spec, verbose=verbose)
        verdict = cv.get("verdict", "CODE_BUG")

        if verdict == "SPEC_AMBIG":
            cluster.escalated = True
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=False,
                layer_used="skipped", escalated=True,
                escalated_to="human", owner=cluster.owner,
                note=f"spec ambiguous: {cv.get('reasoning','')[:150]}",
                consistency_verdict=verdict,
            )

        if verdict in ("TEST_FRAGILE", "THRESHOLD_OK") and cv.get("test_patch_allowed"):
            ok = repair_test_file(cluster, cv, verbose=verbose)
            cluster.attempt_count += 1
            cluster.last_fingerprint = cluster.fingerprint()
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=ok,
                layer_used="test_rewrite", escalated=not ok,
                escalated_to="human" if not ok else "",
                owner=cluster.owner,
                note=(
                    cv.get("test_patch_rationale", "")[:150]
                    if ok else "test rewrite failed"
                ),
                consistency_verdict=verdict,
            )

        # CODE_BUG or low-confidence → fall through to normal repair
        consistency_verdict_label = verdict
    else:
        consistency_verdict_label = ""   # skip P0 on retries

    # ── L0: static pre-pass ───────────────────────────────────────────────────
    l0_fixed, l0_desc = layer0_static_prepass(cluster, verbose)
    if l0_fixed:
        cluster.attempt_count   += 1
        cluster.last_fingerprint = cluster.fingerprint()
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=True,
            layer_used="static", escalated=False,
            escalated_to="", owner=cluster.owner, note=l0_desc,
            consistency_verdict=consistency_verdict_label,
        )

    # ── Give-up check ─────────────────────────────────────────────────────────
    if cluster.attempt_count >= max_cluster_attempts:
        cluster.escalated = True
        esc_to = "human"
        print(f"    [L3] ⚠ Gave up on {cluster.test_file} after "
              f"{cluster.attempt_count} attempt(s). ESCALATED→{esc_to}.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True, escalated_to=esc_to,
            owner=cluster.owner,
            note=f"gave up after {cluster.attempt_count} LLM attempts",
            consistency_verdict=consistency_verdict_label,
        )

    current_fp = cluster.fingerprint()
    is_stale   = cluster.attempt_count > 0 and cluster.last_fingerprint == current_fp

    # Decide whether to skip L1 and go straight to Minimax
    skip_qwen = (
        cluster.owner == "minimax"         # already transferred
        or cluster.is_minimax_scope()      # hook/data → Minimax territory
        or is_stale                        # no progress → escalate to Minimax
    )

    if not skip_qwen:
        # ── L1: Qwen surface fix ─────────────────────────────────────────────
        ok, note = _call_repair(
            cluster, call_qwen,
            system=_build_qwen_system_with_findings(judge_findings),
            verbose=verbose, layer_name="L1",
        )
        cluster.attempt_count   += 1
        cluster.last_fingerprint = current_fp

        if ok:
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=True,
                layer_used="qwen_targeted", escalated=False,
                escalated_to="", owner="qwen",
                consistency_verdict=consistency_verdict_label,
            )

        # Qwen failed: escalate to Minimax if file is in scope
        if cluster.is_minimax_scope():
            print(f"    [L1→L2] Transferring {cluster.test_file} to Minimax.")
            cluster.owner = "minimax"
        else:
            # Component that Qwen couldn't fix and Minimax can't touch → human
            cluster.escalated = True
            return ClusterRepairRecord(
                cluster=cluster.key, src_file=cluster.src_file,
                failures=len(cluster.failures), repaired=False,
                layer_used="qwen_targeted", escalated=True,
                escalated_to="human", owner="qwen",
                note=f"Qwen failed on component outside Minimax scope: {note}",
                consistency_verdict=consistency_verdict_label,
            )

    # ── L2: Minimax logic debugger ────────────────────────────────────────────
    cluster.owner  = "minimax"
    test_code      = _read_file_safe(ROOT / cluster.test_file)
    timeline       = _build_state_timeline(test_code)
    minimax_system = _build_minimax_system(global_notes, judge_findings)

    ok, note = _call_repair(
        cluster, call_minimax,
        system=minimax_system,
        extra_ctx=timeline,
        verbose=verbose, layer_name="L2",
        scope_check=True,
    )
    cluster.attempt_count   += 1
    cluster.last_fingerprint = current_fp

    return ClusterRepairRecord(
        cluster=cluster.key, src_file=cluster.src_file,
        failures=len(cluster.failures), repaired=ok,
        layer_used="minimax_logic", escalated=False,
        escalated_to="", owner="minimax", note=note,
        consistency_verdict=consistency_verdict_label,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main — B / C / D loop
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--max-cluster-attempts", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--impl", default="qwen", choices=["qwen"],
                        help="Kept for harness.py compat — always qwen+minimax internally")
    args = parser.parse_args()

    max_iter             = args.max_iter
    max_cluster_attempts = args.max_cluster_attempts
    verbose              = args.verbose

    global_notes = _load_glm_global_notes()
    if global_notes:
        print(f"[04] GLM global_notes loaded ({len(global_notes)} chars) "
              f"— will be injected into Minimax prompts")

    judge_findings = _load_judge_findings()
    if judge_findings:
        print(f"[04] Judge findings loaded ({len(judge_findings)} chars) "
              f"— injected into Minimax + Qwen prompts (regression prevention)")

    iteration_records: list[IterationRecord]     = []
    cluster_state:     dict[str, FailureCluster] = {}
    escalated_log:     list[dict]                = []

    for iteration in range(1, max_iter + 1):
        tag = f"[04][{iteration}/{max_iter}]"

        print(f"\n{tag} Phase B — running vitest …")
        passed, output = run_vitest()

        summary_line = next(
            (l.strip() for l in output.splitlines()
             if ("passed" in l or "failed" in l) and "test" in l.lower()),
            output.strip().splitlines()[-1] if output.strip() else "no output",
        )
        print(f"{tag} {summary_line}")

        if passed:
            print(f"{tag} ✓ All tests passed.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=True, summary=summary_line,
                clusters_found=0, clusters_repaired=0, cluster_details=[],
                log_snippet="",
            ))
            break

        clusters = parse_failures(output)
        clusters = merge_cluster_state(clusters, cluster_state)

        print(f"{tag} {len(clusters)} failing cluster(s):")
        for c in clusters:
            markers = []
            if c.attempt_count > 0 and c.last_fingerprint == c.fingerprint():
                markers.append("STALE")
            if c.escalated:
                markers.append("ESCALATED")
            if c.owner == "minimax":
                markers.append("MINIMAX")
            scope_label  = "[hook/data]" if c.is_minimax_scope() else "[component]"
            marker_str   = f"  [{', '.join(markers)}]" if markers else ""
            print(f"  * {scope_label} {c.test_file} "
                  f"({len(c.failures)} failure(s)){marker_str}")

        if not clusters:
            print(f"{tag} Could not parse clusters. Stopping.", file=sys.stderr)
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=0, clusters_repaired=0, cluster_details=[],
                log_snippet=output[-1200:],
            ))
            break

        if iteration == max_iter:
            print(f"{tag} Reached max_iter — {len(clusters)} cluster(s) remaining.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=len(clusters), clusters_repaired=0,
                cluster_details=[{
                    "cluster":  c.key, "failures": len(c.failures),
                    "escalated": c.escalated, "owner": c.owner,
                    "attempts": c.attempt_count,
                } for c in clusters],
                log_snippet=output[-1200:],
            ))
            break

        print(f"{tag} Phase C — dispatching {len(clusters)} cluster(s) …")
        repaired = 0
        cluster_details = []

        for cluster in clusters:
            print(f"  -> {cluster.test_file} "
                  f"(owner={cluster.owner}, attempt #{cluster.attempt_count + 1})")
            record = repair_cluster(
                cluster, global_notes, max_cluster_attempts,
                judge_findings=judge_findings, verbose=verbose,
            )
            cluster_state[cluster.key] = cluster
            repaired += int(record.repaired)
            detail = {
                "cluster":              record.cluster,
                "src_file":             record.src_file,
                "failures":             record.failures,
                "repaired":             record.repaired,
                "layer_used":           record.layer_used,
                "escalated":            record.escalated,
                "escalated_to":         record.escalated_to,
                "owner":                record.owner,
                "note":                 record.note,
                "consistency_verdict":  record.consistency_verdict,
            }
            cluster_details.append(detail)
            if record.escalated:
                escalated_log.append({"iteration": iteration, **detail})

        print(f"{tag} Phase C done — {repaired}/{len(clusters)} patched.")
        iteration_records.append(IterationRecord(
            iteration=iteration, passed=False, summary=summary_line,
            clusters_found=len(clusters), clusters_repaired=repaired,
            cluster_details=cluster_details,
            log_snippet=output[-1200:],
        ))

    # ── Reports ───────────────────────────────────────────────────────────────
    final_passed = bool(iteration_records and iteration_records[-1].passed)
    report = {
        "impl":                 "qwen+minimax",
        "max_iter":             max_iter,
        "max_cluster_attempts": max_cluster_attempts,
        "total_iterations":     len(iteration_records),
        "final_status":         "PASS" if final_passed else "FAIL",
        "iterations":           [asdict(r) for r in iteration_records],
    }
    (REPORTS_DIR / "qwen_iterations.json").write_text(json.dumps(report, indent=2))
    print(f"\n[04] Report → reports/qwen_iterations.json")

    if escalated_log:
        esc_path = REPORTS_DIR / "escalated_clusters.json"
        esc_path.write_text(json.dumps({
            "total_escalated": len(escalated_log),
            "clusters":        escalated_log,
        }, indent=2))
        print(f"[04] ⚠ Escalated → {esc_path} ({len(escalated_log)} cluster(s))")

    if not final_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
