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
    cluster:       str
    src_file:      str
    failures:      int
    repaired:      bool
    layer_used:    str   # "static"|"qwen_targeted"|"minimax_logic"|"skipped"
    escalated:     bool
    escalated_to:  str   # ""|"minimax"|"human"
    owner:         str   # "qwen"|"minimax"
    note:          str = ""


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
    api_key = os.environ["OPENROUTER_API_KEY"]
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model_id, "messages": messages,
              "temperature": 0.1, "max_tokens": max_tokens},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


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


def _build_minimax_system(global_notes: str) -> str:
    notes_block = (
        f"\n## GLM Architect's Global Notes (MUST follow)\n{global_notes}\n"
        if global_notes else ""
    )
    return f"""\
You are a senior TypeScript logic debugger specialising in hooks and data generation.
{notes_block}
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
  3. Identify the root cause — one specific line or function.
  4. Rewrite ONLY the broken function(s). Leave everything else intact.

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
    verbose:              bool = False,
) -> ClusterRepairRecord:
    """
    Dispatch cluster through L0 → L1/L2 → L3.

    Routing rules:
      - Hook/data files (MINIMAX_SCOPE) → skip L1 Qwen, go straight to L2 Minimax
      - Component files → L1 Qwen first
          * Qwen fixes it                 → done, owner stays "qwen"
          * Qwen signals LOGIC_BUG        → transfer to L2 Minimax (if in scope)
          * stale fingerprint             → L2 Minimax (if in scope)
          * unfixable + out of scope      → ESCALATED→human
      - Once cluster.owner == "minimax"   → always go to L2, Qwen locked out
    """
    # L3 guard
    if cluster.escalated:
        print(f"    [SKIP] {cluster.test_file} — ESCALATED→{cluster.owner}, skipping.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True,
            escalated_to=cluster.owner, owner=cluster.owner,
            note="previously escalated",
        )

    # L0: static pre-pass
    l0_fixed, l0_desc = layer0_static_prepass(cluster, verbose)
    if l0_fixed:
        cluster.attempt_count   += 1
        cluster.last_fingerprint = cluster.fingerprint()
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=True,
            layer_used="static", escalated=False,
            escalated_to="", owner=cluster.owner, note=l0_desc,
        )

    # Give-up check
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
            system=FIX_SYSTEM_QWEN,
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
            )

    # ── L2: Minimax logic debugger ───────────────────────────────────────────
    cluster.owner = "minimax"
    test_code      = _read_file_safe(ROOT / cluster.test_file)
    timeline       = _build_state_timeline(test_code)
    minimax_system = _build_minimax_system(global_notes)

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
                cluster, global_notes, max_cluster_attempts, verbose=verbose,
            )
            cluster_state[cluster.key] = cluster
            repaired += int(record.repaired)
            detail = {
                "cluster":      record.cluster,
                "src_file":     record.src_file,
                "failures":     record.failures,
                "repaired":     record.repaired,
                "layer_used":   record.layer_used,
                "escalated":    record.escalated,
                "escalated_to": record.escalated_to,
                "owner":        record.owner,
                "note":         record.note,
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
