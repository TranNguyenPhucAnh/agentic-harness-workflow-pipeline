"""
pipeline/04_test_and_iterate.py
Step 4+5 — Clustered repair pipeline (Phase B/C/D).

Phase A  (upstream) : 03a wrote src/ (Qwen executor).
Phase B  (this file): run vitest, parse output → list[FailureCluster]
Phase C  (this file): for each cluster → 1 targeted Qwen call → patch src/
Phase D  (this file): rerun full vitest after all clusters patched → repeat up to max_iter

Usage:
    python pipeline/04_test_and_iterate.py --max-iter 3
    python pipeline/04_test_and_iterate.py --max-iter 5 --verbose

Writes: reports/qwen_iterations.json
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

ROOT          = Path(__file__).parent.parent
SPEC_PATH     = ROOT / "spec.md"
REPORTS_DIR   = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

SRC_DIR = "src"   # Qwen is the sole executor; all output goes to src/


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFailure:
    test_file: str        # e.g. "tests/components/SummaryStickyBar.test.tsx"
    test_name: str        # individual test description
    error_snippet: str    # raw error block for this test


@dataclass
class FailureCluster:
    """One cluster = one test file + its corresponding src file."""
    test_file: str                      # tests/… path
    src_file: str                       # src/… path (may not exist yet)
    failures: list[TestFailure] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.test_file

    def error_block(self) -> str:
        return "\n\n".join(
            f"  x {f.test_name}\n{f.error_snippet}" for f in self.failures
        )


@dataclass
class IterationRecord:
    iteration: int
    passed: bool
    summary: str
    clusters_found: int
    clusters_repaired: int
    cluster_details: list[dict]
    log_snippet: str


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
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_qwen(messages: list) -> str:
    return _openrouter_call("qwen/qwen3.6-plus", messages)


# Repair always uses Qwen — GLM is a planner (03b), not a code fixer
CALLERS: dict[str, Callable[[list], str]] = {
    "qwen": call_qwen,
}


# ════════════════════════════════════════════════════════════════════════════
# Phase B — run vitest + parse failures into clusters
# ════════════════════════════════════════════════════════════════════════════

def run_vitest() -> tuple[bool, str]:
    """Run `npx vitest run --reporter=verbose`. Returns (passed, raw_output)."""
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0, output


# Patterns for vitest verbose output -----------------------------------------
# Test file header:  " FAIL  tests/components/SummaryStickyBar.test.tsx"
_RE_TEST_FILE   = re.compile(r"^\s*(FAIL|PASS)\s+(tests/\S+\.test\.[tj]sx?)", re.MULTILINE)
# Individual failing test: " x test name" or unicode variants
_RE_FAIL_TEST   = re.compile(r"^\s+[x\u00d7\u2717\u274c]\s+(.+)$", re.MULTILINE)
# Error block: AssertionError / TypeError / Error + following indented lines
_RE_ERROR_BLOCK = re.compile(
    r"(AssertionError|Error|TypeError|ReferenceError)[^\n]*\n(?:[ \t]+[^\n]*\n)*",
    re.MULTILINE,
)


def _infer_src_file(test_file: str) -> str:
    """
    Derive the src file path from a test file path.
      tests/components/Foo.test.tsx  ->  src/components/Foo.tsx
      tests/hooks/useBar.test.ts     ->  src/hooks/useBar.ts
    """
    rel = test_file.replace("tests/", "", 1)          # components/Foo.test.tsx
    rel = re.sub(r"\.test\.(tsx?)$", r".\1", rel)     # components/Foo.tsx
    rel = re.sub(r"\.test\.(ts)$",   r".\1", rel)     # hooks/useBar.ts
    return f"{SRC_DIR}/{rel}"


def parse_failures(output: str) -> list[FailureCluster]:
    """
    Phase B: parse vitest verbose output -> list[FailureCluster].

    1. Find all FAIL test file sections.
    2. For each section extract individual test names + error snippets.
    3. Group into FailureCluster(test_file, src_file, failures).
    """
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

        fail_names = _RE_FAIL_TEST.findall(section)
        errors     = _RE_ERROR_BLOCK.findall(section)

        for j, name in enumerate(fail_names):
            snippet = errors[j] if j < len(errors) else section[:500]
            cluster.failures.append(TestFailure(
                test_file=test_file,
                test_name=name.strip(),
                error_snippet=snippet.strip(),
            ))

        # Fallback when regex finds no individual failures (e.g. compile errors)
        if not cluster.failures:
            cluster.failures.append(TestFailure(
                test_file=test_file,
                test_name="(parse fallback)",
                error_snippet=section[:1500].strip(),
            ))

    return list(clusters.values())


# ════════════════════════════════════════════════════════════════════════════
# Phase C — targeted repair per cluster
# ════════════════════════════════════════════════════════════════════════════

FIX_SYSTEM = """\
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


def _read_file_safe(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return f"// FILE NOT FOUND: {path}\n// This file needs to be created.\n"


def repair_cluster(
    cluster: FailureCluster,
    call_api: Callable[[list], str],
    verbose: bool = False,
) -> bool:
    """
    Phase C: call Qwen once for this cluster, apply patch to src/.
    Returns True if patch was applied successfully.
    """
    spec      = SPEC_PATH.read_text()
    src_path  = ROOT / cluster.src_file
    test_path = ROOT / cluster.test_file
    src_code  = _read_file_safe(src_path)
    test_code = _read_file_safe(test_path)
    error_log = cluster.error_block()

    user_content = (
        f"### spec.md\n\n{spec}\n\n"
        f"### Test file (read-only): {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Source file to fix: {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Failing tests in this cluster\n```\n{error_log}\n```"
    )

    messages = [
        {"role": "system", "content": FIX_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    if verbose:
        print(f"    [C] Sending cluster '{cluster.test_file}' ({len(cluster.failures)} failures) to model …")

    try:
        raw = call_api(messages).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        patch = json.loads(raw)
    except Exception as e:
        print(f"    [C] Parse error for cluster '{cluster.test_file}': {e}", file=sys.stderr)
        return False

    out_rel  = patch.get("file_path", cluster.src_file)
    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(patch["code"])
    print(f"    [C] Patched {out_rel} — {patch.get('explanation', '(no explanation)')}")
    return True


# ════════════════════════════════════════════════════════════════════════════
# Main — B / C / D loop
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Max outer B->C->D iterations")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print extra debug info per cluster")
    # --impl kept for harness.py backward compat but value is ignored (always qwen)
    parser.add_argument("--impl", default="qwen", choices=["qwen"],
                        help="Executor model (always qwen — GLM is planner only)")
    args = parser.parse_args()

    max_iter = args.max_iter
    verbose  = args.verbose
    call_api = CALLERS["qwen"]

    iteration_records: list[IterationRecord] = []

    for iteration in range(1, max_iter + 1):
        tag = f"[04][QWEN][{iteration}/{max_iter}]"

        # ── Phase B / D: full suite run ───────────────────────────────────
        print(f"\n{tag} Phase B — running full vitest suite …")
        passed, output = run_vitest()

        summary_line = next(
            (l.strip() for l in output.splitlines()
             if ("passed" in l or "failed" in l) and "test" in l.lower()),
            output.strip().splitlines()[-1] if output.strip() else "no output",
        )
        print(f"{tag} {summary_line}")

        if passed:
            print(f"{tag} All tests passed.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=True, summary=summary_line,
                clusters_found=0, clusters_repaired=0, cluster_details=[],
                log_snippet="",
            ))
            break

        # ── Phase B: cluster the failures ─────────────────────────────────
        clusters = parse_failures(output)
        print(f"{tag} {len(clusters)} failing cluster(s):")
        for c in clusters:
            print(f"  * {c.test_file}  ({len(c.failures)} failure(s))")

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
            print(f"{tag} Reached max_iter with {len(clusters)} cluster(s) still failing.")
            iteration_records.append(IterationRecord(
                iteration=iteration, passed=False, summary=summary_line,
                clusters_found=len(clusters), clusters_repaired=0,
                cluster_details=[{"cluster": c.key, "failures": len(c.failures)}
                                 for c in clusters],
                log_snippet=output[-2000:],
            ))
            break

        # ── Phase C: repair each cluster ──────────────────────────────────
        print(f"{tag} Phase C — targeted repair ({len(clusters)} cluster(s)) …")
        repaired = 0
        cluster_details = []

        for cluster in clusters:
            print(f"  -> {cluster.test_file}")
            ok = repair_cluster(cluster, call_api, verbose=verbose)
            repaired += int(ok)
            cluster_details.append({
                "cluster":  cluster.key,
                "src_file": cluster.src_file,
                "failures": len(cluster.failures),
                "repaired": ok,
            })

        print(f"{tag} Phase C done — {repaired}/{len(clusters)} patched.")

        iteration_records.append(IterationRecord(
            iteration=iteration, passed=False, summary=summary_line,
            clusters_found=len(clusters), clusters_repaired=repaired,
            cluster_details=cluster_details,
            log_snippet=output[-2000:],
        ))

        # Phase D: top of loop reruns full suite automatically

    # ── Report ────────────────────────────────────────────────────────────────
    final_passed = bool(iteration_records and iteration_records[-1].passed)
    report = {
        "impl":             "qwen",
        "max_iter":         max_iter,
        "total_iterations": len(iteration_records),
        "final_status":     "PASS" if final_passed else "FAIL",
        "iterations":       [asdict(r) for r in iteration_records],
    }
    (REPORTS_DIR / "qwen_iterations.json").write_text(json.dumps(report, indent=2))
    print(f"\n[04] Report -> reports/qwen_iterations.json")

    if not final_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
