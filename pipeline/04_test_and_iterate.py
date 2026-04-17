"""
pipeline/04_test_and_iterate.py
Step 4+5 — Run vitest against an implementation, iterate on failure.

Usage:
    python pipeline/04_test_and_iterate.py --impl qwen      --max-iter 3
    python pipeline/04_test_and_iterate.py --impl glm       --max-iter 3

For GLM: temporarily copies src_glm/ → src/ before testing, restores after.
Writes:  reports/{impl}_iterations.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import httpx
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SPEC_PATH    = ROOT / "spec.md"
SCAFFOLD_JSON = ROOT / "scaffold" / "scaffold.json"
REPORTS_DIR  = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ── API helpers ──────────────────────────────────────────────────────────────

def call_qwen(messages: list) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "qwen/qwen3.6-plus", "messages": messages, "temperature": 0.1, "max_tokens": 32768},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_glm(messages: list) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "z-ai/glm-5.1", "messages": messages, "temperature": 0.1, "max_tokens": 32768},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


CALLERS = {"qwen": call_qwen, "glm": call_glm}


# ── Test runner ───────────────────────────────────────────────────────────────

def run_vitest(src_dir: Path) -> tuple[bool, str]:
    """
    Run vitest. src_dir is symlinked to ROOT/src before running.
    Returns (passed: bool, output: str).
    """
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "VITEST_SRC": str(src_dir)},
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0, output


def swap_src(impl: str) -> Path:
    """For GLM, copy src_glm/ → src/ (backup original first). Returns backup path."""
    src = ROOT / "src"
    if impl == "glm":
        backup = ROOT / "_src_backup"
        if src.exists():
            shutil.copytree(src, backup, dirs_exist_ok=True)
        shutil.copytree(ROOT / "src_glm", src, dirs_exist_ok=True)
        return backup
    return src   # qwen writes directly to src/, nothing to swap


def restore_src(impl: str, backup: Path) -> None:
    if impl == "glm" and backup != ROOT / "src":
        shutil.rmtree(ROOT / "src", ignore_errors=True)
        if backup.exists():
            shutil.copytree(backup, ROOT / "src", dirs_exist_ok=True)
            shutil.rmtree(backup, ignore_errors=True)


# ── Fix loop ──────────────────────────────────────────────────────────────────

FIX_SYSTEM = """You are a TypeScript/React developer fixing failing tests.
You will receive:
1. The original spec.md
2. The current source files that are failing
3. The test error log

Return a JSON object:
{
  "files": [
    { "file_path": "src/hooks/useSensorData.ts", "code": "<fixed full content>" }
  ],
  "explanation": "brief description of what was fixed"
}

Rules:
- Fix ONLY what is failing. Do not rewrite working code.
- Do NOT modify test files.
- TypeScript strict, no `any`, Tailwind only.
- Output raw JSON only. No markdown fences.
"""


def collect_src_files(impl: str) -> dict:
    """Collect all non-test source files for the given impl."""
    src_dir = ROOT / ("src_glm" if impl == "glm" else "src")
    files = {}
    for p in src_dir.rglob("*.ts"):
        files[str(p.relative_to(ROOT))] = p.read_text()
    for p in src_dir.rglob("*.tsx"):
        files[str(p.relative_to(ROOT))] = p.read_text()
    return files


def apply_fix(impl: str, fix_result: dict) -> None:
    for entry in fix_result.get("files", []):
        rel = entry["file_path"]
        # For GLM, remap src/ → src_glm/
        if impl == "glm" and rel.startswith("src/"):
            rel = "src_glm/" + rel[len("src/"):]
        path = ROOT / rel
        if path.exists():   # only overwrite existing files
            path.write_text(entry["code"])
            print(f"  [fix] {rel}")


def request_fix(impl: str, error_log: str, call_api) -> dict:
    spec = SPEC_PATH.read_text()
    src_files = collect_src_files(impl)

    src_block = "\n\n".join(
        f"### {fp}\n```typescript\n{code}\n```"
        for fp, code in src_files.items()
    )

    messages = [
        {"role": "system", "content": FIX_SYSTEM},
        {"role": "user", "content": (
            f"### spec.md\n\n{spec}\n\n"
            f"### Current source files\n\n{src_block}\n\n"
            f"### Test error log\n\n```\n{error_log}\n```"
        )},
    ]

    raw = call_api(messages).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    return json.loads(raw)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl",     required=True, choices=["qwen", "glm"])
    parser.add_argument("--max-iter", type=int, default=3)
    args = parser.parse_args()

    impl     = args.impl
    max_iter = args.max_iter
    call_api = CALLERS[impl]

    iterations = []
    backup = swap_src(impl)

    try:
        for iteration in range(1, max_iter + 1):
            print(f"\n[04] [{impl.upper()}] Iteration {iteration}/{max_iter} — running vitest …")
            passed, output = run_vitest(ROOT / "src")

            # Parse test summary from vitest output
            summary_line = next(
                (l for l in output.splitlines() if "passed" in l or "failed" in l), ""
            )
            print(f"[04] [{impl.upper()}] {summary_line.strip()}")

            iterations.append({
                "iteration": iteration,
                "passed": passed,
                "summary": summary_line.strip(),
                "log_snippet": output[-3000:],   # last 3k chars
            })

            if passed:
                print(f"[04] [{impl.upper()}] ✓ All tests passed on iteration {iteration}.")
                break

            if iteration == max_iter:
                print(f"[04] [{impl.upper()}] ✗ Still failing after {max_iter} iterations.")
                break

            # Attempt fix
            print(f"[04] [{impl.upper()}] Requesting fix from model …")
            try:
                fix = request_fix(impl, output, call_api)
                apply_fix(impl, fix)
                print(f"[04] [{impl.upper()}] Fix applied: {fix.get('explanation', '(no explanation)')}")
            except Exception as e:
                print(f"[04] [{impl.upper()}] Fix request failed: {e}", file=sys.stderr)
                break

    finally:
        restore_src(impl, backup)

    # Write per-impl report
    report = {
        "impl": impl,
        "total_iterations": len(iterations),
        "final_status": "PASS" if iterations and iterations[-1]["passed"] else "FAIL",
        "iterations": iterations,
    }
    report_path = REPORTS_DIR / f"{impl}_iterations.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[04] Report written → {report_path}")

    # Exit non-zero if final iteration failed (so GH Actions marks step red)
    if report["final_status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
