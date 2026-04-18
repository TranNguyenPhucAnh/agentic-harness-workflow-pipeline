"""
pipeline/04_test_and_iterate.py
Step 4+5 — Clustered repair pipeline with 3-layer escalation.

Phase A  (upstream) : 03a wrote src/ (Qwen executor).
Phase B  (this file): run vitest → parse output → list[FailureCluster]
Phase C  (this file): per-cluster repair with 3 escalation layers:
    Layer 0 — Static pre-pass: fix deterministic esbuild/transform errors
              without calling any model (fast, free, no-LLM)
    Layer 1 — Targeted repair: standard Qwen call with focused prompt
    Layer 2 — Escalated repair: richer prompt (state timeline, full rewrite)
              triggered when cluster error fingerprint didn't change after L1
    Layer 3 — Give-up: mark cluster ESCALATED after max_cluster_attempts,
              log for human review, skip further LLM calls on that cluster
Phase D  (this file): rerun full vitest after all clusters processed → repeat

Usage:
    python pipeline/04_test_and_iterate.py --max-iter 3
    python pipeline/04_test_and_iterate.py --max-iter 5 --verbose
    python pipeline/04_test_and_iterate.py --max-iter 5 --max-cluster-attempts 3

Writes:
    reports/qwen_iterations.json
    reports/escalated_clusters.json   ← clusters that hit give-up threshold
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
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SRC_DIR = "src"   # Qwen is the sole executor; all output goes to src/


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFailure:
    test_file:     str   # "tests/components/SummaryStickyBar.test.tsx"
    test_name:     str   # individual test description
    error_snippet: str   # raw error block


@dataclass
class FailureCluster:
    """One cluster = one test file + its corresponding src file."""
    test_file:  str
    src_file:   str
    failures:   list[TestFailure] = field(default_factory=list)

    # Escalation state — mutated across iterations
    attempt_count:      int  = field(default=0)
    last_fingerprint:   str  = field(default="")
    escalated:          bool = field(default=False)
    is_transform_error: bool = field(default=False)

    @property
    def key(self) -> str:
        return self.test_file

    def error_block(self) -> str:
        return "\n\n".join(
            f"  x {f.test_name}\n{f.error_snippet}" for f in self.failures
        )

    def fingerprint(self) -> str:
        """Stable digest of error content — used to detect stale repairs."""
        return re.sub(r"\s+", " ", self.error_block()).strip()[:400]


@dataclass
class ClusterRepairRecord:
    cluster:    str
    src_file:   str
    failures:   int
    repaired:   bool
    layer_used: str   # "static" | "targeted" | "escalated" | "skipped"
    escalated:  bool
    note:       str = ""


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


# ════════════════════════════════════════════════════════════════════════════
# Phase B — run vitest + parse failures
# ════════════════════════════════════════════════════════════════════════════

def run_vitest() -> tuple[bool, str]:
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0, output


_RE_TEST_FILE   = re.compile(r"^\s*(FAIL|PASS)\s+(tests/\S+\.test\.[tj]sx?)", re.MULTILINE)
_RE_FAIL_TEST   = re.compile(r"^\s+[x\u00d7\u2717\u274c]\s+(.+)$", re.MULTILINE)
_RE_ERROR_BLOCK = re.compile(
    r"(AssertionError|Error|TypeError|ReferenceError)[^\n]*\n(?:[ \t]+[^\n]*\n)*",
    re.MULTILINE,
)
# esbuild / tsc transform errors — deterministic, no-LLM fixable
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
            test_file,
            FailureCluster(test_file=test_file, src_file=src_file),
        )

        # Flag transform errors for Layer 0
        if _RE_TRANSFORM_ERR.search(section):
            cluster.is_transform_error = True

        fail_names = _RE_FAIL_TEST.findall(section)
        errors     = _RE_ERROR_BLOCK.findall(section)

        for j, name in enumerate(fail_names):
            snippet = errors[j] if j < len(errors) else section[:500]
            cluster.failures.append(TestFailure(
                test_file=test_file,
                test_name=name.strip(),
                error_snippet=snippet.strip(),
            ))

        if not cluster.failures:
            cluster.failures.append(TestFailure(
                test_file=test_file,
                test_name="(parse fallback)",
                error_snippet=section[:1500].strip(),
            ))

    return list(clusters.values())


def merge_cluster_state(
    new_clusters: list[FailureCluster],
    prev_state:   dict[str, FailureCluster],
) -> list[FailureCluster]:
    """
    Carry over attempt_count, last_fingerprint, escalated from previous
    iteration so staleness detection and give-up logic work across outer loops.
    """
    for c in new_clusters:
        if c.key in prev_state:
            prev = prev_state[c.key]
            c.attempt_count    = prev.attempt_count
            c.last_fingerprint = prev.last_fingerprint
            c.escalated        = prev.escalated
    return new_clusters


# ════════════════════════════════════════════════════════════════════════════
# Layer 0 — Static pre-pass (no LLM)
# ════════════════════════════════════════════════════════════════════════════

# JSX generic syntax that confuses esbuild: <Component<T> prop= → <Component prop=
_RE_JSX_GENERIC = re.compile(
    r"(<\w[\w.]*)<(\w[\w,\s]*)>(\s*(?:events|data|items|props|value)\s*=)",
)
# Floating-point width in template literals: ${val * 100}% → ${Math.round(val*100)}%
_RE_TEMPLATE_WIDTH = re.compile(r"(`\$\{)([^}]*\*\s*100)(\}%`)")
# Literal float percentage in style strings: 55.000001% → 55%
_RE_FLOAT_WIDTH = re.compile(r"(width:\s*)(\d+\.\d+)(%)")


def _static_fix_transform(test_path: Path) -> tuple[bool, str]:
    """
    Auto-fix known esbuild transform errors in test files.
    Only touches syntax ambiguities — never touches assertions or logic.
    """
    if not test_path.exists():
        return False, "test file not found"

    original = test_path.read_text()
    # Remove JSX generic type params that confuse esbuild parser
    patched = _RE_JSX_GENERIC.sub(r"\1\3", original)

    if patched != original:
        test_path.write_text(patched)
        return True, "removed JSX generic type param causing esbuild parse error"

    return False, "no static transform pattern matched"


def _static_fix_src(src_path: Path) -> tuple[bool, str]:
    """
    Auto-fix known source-side issues without LLM.
    Currently: floating-point percentage widths → Math.round().
    """
    if not src_path.exists():
        return False, "src file not found"

    original = src_path.read_text()
    patched  = original

    # Fix template literal widths: ${value * 100}% → ${Math.round(value * 100)}%
    patched = _RE_TEMPLATE_WIDTH.sub(r"`${Math.round(\2)}\3", patched)
    # Fix literal float percentages in style strings
    patched = _RE_FLOAT_WIDTH.sub(
        lambda m: f"{m.group(1)}{round(float(m.group(2)))}{m.group(3)}",
        patched,
    )

    if patched != original:
        src_path.write_text(patched)
        return True, "rounded floating-point percentage widths to avoid precision mismatch"

    return False, "no static src pattern matched"


def layer0_static_prepass(
    cluster: FailureCluster,
    verbose: bool,
) -> tuple[bool, str]:
    """
    Layer 0: deterministic fixes — no model call.
    Transform errors → try test file fix first, then src.
    Non-transform errors → try src file fix only.
    """
    if verbose:
        print(f"    [L0] Static pre-pass for {cluster.test_file} …")

    if cluster.is_transform_error:
        fixed, desc = _static_fix_transform(ROOT / cluster.test_file)
        if fixed:
            print(f"    [L0] ✓ Test file patched: {desc}")
            return True, desc

    fixed, desc = _static_fix_src(ROOT / cluster.src_file)
    if fixed:
        print(f"    [L0] ✓ Src file patched: {desc}")
        return True, desc

    if verbose:
        print("    [L0] No static pattern matched — escalating to LLM.")
    return False, "no static fix applicable"


# ════════════════════════════════════════════════════════════════════════════
# Layer 1 — Targeted repair (standard Qwen prompt)
# ════════════════════════════════════════════════════════════════════════════

FIX_SYSTEM_TARGETED = """\
You are a senior TypeScript/React developer doing a targeted fix for ONE failing test cluster.
You receive:
1. spec.md (for context)
2. The test file that is failing (read-only — do NOT modify it)
3. The source file to fix
4. The specific test errors for this cluster only

Return a JSON object with this EXACT schema (no other keys):
{
  "file_path": "src/components/SummaryStickyBar.tsx",
  "code": "<full fixed file content>",
  "explanation": "one line describing what was fixed"
}

Rules:
- Return EXACTLY ONE file object (the source file, not the test file).
- Fix ONLY what the error log indicates. Do not rewrite unrelated logic.
- TypeScript strict mode — no `any`.
- Tailwind only — no inline styles, no CSS modules.
- Output raw JSON only. No markdown fences.
"""


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — Escalated repair (richer context + rewrite permission)
# ════════════════════════════════════════════════════════════════════════════

FIX_SYSTEM_ESCALATED = """\
You are a senior TypeScript/React developer doing a DEEP FIX for a stubborn failing cluster.

IMPORTANT: Previous repair attempts have NOT fixed this cluster — the error fingerprint
is UNCHANGED. This means the bug is a semantic/logic issue, not a surface syntax error.
A targeted patch will NOT work. You must reason about the full state machine.

You receive:
1. spec.md — read the behaviour description for this file very carefully
2. The test file (read-only — do NOT modify it)
3. The source file in its current (broken) state
4. The exact failing test errors
5. A step-by-step expected state timeline extracted from the test — follow it exactly

Your task:
- REWRITE the relevant function/hook/component from scratch if needed.
- Ensure EVERY assertion in the failing tests passes.
- Keep all exported interfaces intact so other files are not broken.

Return a JSON object:
{
  "file_path": "src/hooks/useReplay.ts",
  "code": "<full rewritten file content>",
  "explanation": "what was fundamentally wrong and how you fixed it"
}

TypeScript strict mode — no `any`. Tailwind only. Raw JSON only. No markdown fences.
"""


def _build_state_timeline(test_code: str) -> str:
    """
    Extract a bullet-point state timeline from act() / expect() blocks
    to give the escalated model a clear before/after sequence.
    """
    lines    = test_code.splitlines()
    timeline: list[str] = []
    in_test  = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"(it|test)\s*\(", stripped):
            in_test = True
            timeline.append(f"TEST: {stripped[:100]}")
        elif in_test and stripped.startswith("act("):
            timeline.append(f"  ACTION: {stripped[:100]}")
        elif in_test and stripped.startswith("expect("):
            timeline.append(f"  ASSERT: {stripped[:100]}")
        elif in_test and stripped in ("});", "})"):
            in_test = False
            timeline.append("")

    return "\n".join(timeline) if timeline else "(could not extract timeline)"


def _read_file_safe(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return f"// FILE NOT FOUND: {path}\n// This file needs to be created.\n"


def _call_repair(
    cluster:    FailureCluster,
    call_api:   Callable[[list], str],
    system:     str,
    extra_ctx:  str = "",
    verbose:    bool = False,
    layer_name: str = "L1",
) -> bool:
    """Shared LLM repair executor for Layer 1 and Layer 2."""
    spec      = SPEC_PATH.read_text()
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

    if verbose:
        print(f"    [{layer_name}] → Qwen call "
              f"(attempt #{cluster.attempt_count + 1}, "
              f"{len(cluster.failures)} failure(s)) …")

    try:
        raw = call_api(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        patch = json.loads(raw)
    except Exception as e:
        print(f"    [{layer_name}] Parse error: {e}", file=sys.stderr)
        return False

    out_rel  = patch.get("file_path", cluster.src_file)
    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch["code"])
    print(f"    [{layer_name}] ✓ Patched {out_rel} — "
          f"{patch.get('explanation', '(no explanation)')}")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Phase C — dispatch per cluster through escalation layers
# ════════════════════════════════════════════════════════════════════════════

def repair_cluster(
    cluster:              FailureCluster,
    call_api:             Callable[[list], str],
    max_cluster_attempts: int,
    verbose:              bool = False,
) -> ClusterRepairRecord:
    """
    Dispatch a single cluster through Layer 0 → 1 → 2 → give-up.

    Decision tree:
        1. Already escalated?           → skip (Layer 3 guard)
        2. Layer 0 matches pattern?     → static fix, return
        3. attempt_count >= max?        → mark ESCALATED, return
        4. fingerprint unchanged?       → Layer 2 (escalated prompt + timeline)
        5. otherwise                    → Layer 1 (standard targeted prompt)
    """
    # Layer 3 guard: already given up in a previous iteration
    if cluster.escalated:
        print(f"    [SKIP] {cluster.test_file} — ESCALATED, requires human review.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True,
            note="previously escalated — requires human review",
        )

    # Layer 0: static pre-pass
    l0_fixed, l0_desc = layer0_static_prepass(cluster, verbose)
    if l0_fixed:
        cluster.attempt_count   += 1
        cluster.last_fingerprint = cluster.fingerprint()
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=True,
            layer_used="static", escalated=False, note=l0_desc,
        )

    # Layer 3 check: exceeded max LLM attempts → give up
    if cluster.attempt_count >= max_cluster_attempts:
        cluster.escalated = True
        print(f"    [L3] ⚠ Gave up on {cluster.test_file} after "
              f"{cluster.attempt_count} LLM attempt(s). Marking ESCALATED.")
        return ClusterRepairRecord(
            cluster=cluster.key, src_file=cluster.src_file,
            failures=len(cluster.failures), repaired=False,
            layer_used="skipped", escalated=True,
            note=f"gave up after {cluster.attempt_count} LLM attempts — "
                 "error fingerprint unchanged or repair ineffective",
        )

    # Staleness check: same error as last attempt?
    current_fp = cluster.fingerprint()
    is_stale   = (
        cluster.attempt_count > 0
        and cluster.last_fingerprint == current_fp
    )

    if is_stale:
        # Layer 2: escalated prompt with state timeline
        print(f"    [L2] Stale fingerprint — switching to escalated prompt.")
        test_code = _read_file_safe(ROOT / cluster.test_file)
        timeline  = _build_state_timeline(test_code)
        ok = _call_repair(
            cluster, call_api,
            system=FIX_SYSTEM_ESCALATED,
            extra_ctx=timeline,
            verbose=verbose,
            layer_name="L2",
        )
        layer_used = "escalated"
    else:
        # Layer 1: standard targeted prompt
        ok = _call_repair(
            cluster, call_api,
            system=FIX_SYSTEM_TARGETED,
            verbose=verbose,
            layer_name="L1",
        )
        layer_used = "targeted"

    # Update state AFTER the repair attempt
    cluster.attempt_count   += 1
    cluster.last_fingerprint = current_fp

    return ClusterRepairRecord(
        cluster=cluster.key, src_file=cluster.src_file,
        failures=len(cluster.failures), repaired=ok,
        layer_used=layer_used, escalated=False,
    )


# ════════════════════════════════════════════════════════════════════════════
# Main — B / C / D loop
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Max outer B→C→D iterations (default: 3)")
    parser.add_argument("--max-cluster-attempts", type=int, default=2,
                        help="Max LLM repair attempts per cluster before give-up (default: 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra debug info per cluster and layer")
    # Kept for harness.py backward compat — value always qwen
    parser.add_argument("--impl", default="qwen", choices=["qwen"],
                        help="Executor model (always qwen — GLM is planner only)")
    args = parser.parse_args()

    max_iter             = args.max_iter
    max_cluster_attempts = args.max_cluster_attempts
    verbose              = args.verbose

    iteration_records: list[IterationRecord]     = []
    cluster_state:     dict[str, FailureCluster] = {}   # persisted across iterations
    escalated_log:     list[dict]                = []

    for iteration in range(1, max_iter + 1):
        tag = f"[04][QWEN][{iteration}/{max_iter}]"

        # ── Phase B: full vitest run ──────────────────────────────────────────
        print(f"\n{tag} Phase B — running full vitest suite …")
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

        # ── Parse failures + merge persisted escalation state ─────────────────
        clusters = parse_failures(output)
        clusters = merge_cluster_state(clusters, cluster_state)

        print(f"{tag} {len(clusters)} failing cluster(s):")
        for c in clusters:
            tags = []
            if c.attempt_count > 0 and c.last_fingerprint == c.fingerprint():
                tags.append("STALE")
            if c.escalated:
                tags.append("ESCALATED")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"  * {c.test_file}  ({len(c.failures)} failure(s)){tag_str}")

        if not clusters:
            print(f"{tag} Could not parse clusters (compile error?). Stopping.",
                  file=sys.stderr)
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=0, clusters_repaired=0, cluster_details=[],
                log_snippet=output[-2000:],
            ))
            break

        if iteration == max_iter:
            print(f"{tag} Reached max_iter — {len(clusters)} cluster(s) still failing.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=len(clusters), clusters_repaired=0,
                cluster_details=[{
                    "cluster":   c.key,
                    "failures":  len(c.failures),
                    "escalated": c.escalated,
                    "attempts":  c.attempt_count,
                } for c in clusters],
                log_snippet=output[-2000:],
            ))
            break

        # ── Phase C: repair loop ──────────────────────────────────────────────
        print(f"{tag} Phase C — dispatching {len(clusters)} cluster(s) …")
        repaired        = 0
        cluster_details = []

        for cluster in clusters:
            attempt_label = f"attempt #{cluster.attempt_count + 1}"
            print(f"  -> {cluster.test_file} ({attempt_label})")

            record = repair_cluster(
                cluster, call_qwen, max_cluster_attempts, verbose=verbose,
            )
            # Persist updated escalation state across outer iterations
            cluster_state[cluster.key] = cluster

            repaired += int(record.repaired)
            detail = {
                "cluster":    record.cluster,
                "src_file":   record.src_file,
                "failures":   record.failures,
                "repaired":   record.repaired,
                "layer_used": record.layer_used,
                "escalated":  record.escalated,
                "note":       record.note,
            }
            cluster_details.append(detail)

            if record.escalated:
                escalated_log.append({"iteration": iteration, **detail})

        print(f"{tag} Phase C done — {repaired}/{len(clusters)} patched.")

        iteration_records.append(IterationRecord(
            iteration=iteration, passed=False, summary=summary_line,
            clusters_found=len(clusters), clusters_repaired=repaired,
            cluster_details=cluster_details,
            log_snippet=output[-2000:],
        ))

        # Phase D: top of loop reruns vitest automatically

    # ── Reports ───────────────────────────────────────────────────────────────
    final_passed = bool(iteration_records and iteration_records[-1].passed)

    report = {
        "impl":                 "qwen",
        "max_iter":             max_iter,
        "max_cluster_attempts": max_cluster_attempts,
        "total_iterations":     len(iteration_records),
        "final_status":         "PASS" if final_passed else "FAIL",
        "iterations":           [asdict(r) for r in iteration_records],
    }
    (REPORTS_DIR / "qwen_iterations.json").write_text(json.dumps(report, indent=2))
    print(f"\n[04] Iteration report → reports/qwen_iterations.json")

    if escalated_log:
        esc_path = REPORTS_DIR / "escalated_clusters.json"
        esc_path.write_text(json.dumps({
            "total_escalated": len(escalated_log),
            "clusters":        escalated_log,
        }, indent=2))
        print(f"[04] ⚠ Escalated clusters → {esc_path} "
              f"({len(escalated_log)} require human review)")

    if not final_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
