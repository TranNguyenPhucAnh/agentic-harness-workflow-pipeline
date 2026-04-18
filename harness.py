#!/usr/bin/env python3
"""
harness.py — Local dev runner for the LLM pipeline.
Mirrors the GitHub Actions workflow, runs on your machine.

Architecture:
    Gemini         → scaffold JSON (stubs + signatures)
    GLM 5.1        → planner: decomposes scaffold into glm_plan.json
    Qwen 3.6+      → executor: implements src/ per-file (with plan) or single-call
    vitest         → test + targeted repair loop (max N iterations)
    DeepSeek V3.2  → judge: qualitative review + sign-off (runs only on green)

Usage:
    # Full pipeline (scaffold → plan → implement → test → report → judge)
    python harness.py

    # Skip GLM planning — Qwen single-call mode, judge still runs
    python harness.py --only-qwen

    # Reuse existing scaffold.json
    python harness.py --skip-scaffold

    # Reuse existing glm_plan.json
    python harness.py --skip-scaffold --skip-plan

    # Skip everything up to test
    python harness.py --test-only

    # Skip judge (faster, no DeepSeek call)
    python harness.py --skip-judge
    python harness.py --test-only --skip-judge

    # Override iteration cap
    python harness.py --max-iter 5

    # Verbose cluster debug output
    python harness.py --test-only --verbose

    # Override per-cluster LLM attempt cap
    python harness.py --test-only --max-cluster-attempts 3

Typical debug loops:
    python harness.py --test-only --skip-judge --max-iter 3
    python harness.py --skip-scaffold --skip-plan --skip-judge

Requirements:
    pip install httpx
    GEMINI_API_KEY, OPENROUTER_API_KEY must be set as env vars
    (or in a .env file — loaded automatically)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def load_dotenv():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def run_step(label: str, script: str, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(ROOT / "pipeline" / script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
    print(f"  {status}  ({elapsed:.1f}s)")
    return result.returncode == 0


def skip_step(label: str, reason: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  [SKIPPED — {reason}]")
    print(f"{'='*60}")


def check_env(keys: list[str]) -> bool:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        print(f"[harness] Missing env vars: {', '.join(missing)}")
        print("          Set them in .env or export them before running.")
        return False
    return True


def check_file_exists(path: Path, flag: str) -> bool:
    if not path.exists():
        print(f"[harness] {flag} set but {path} not found.")
        return False
    return True


def check_src_exists() -> bool:
    src_dir = ROOT / "src"
    if not src_dir.exists() or not any(src_dir.rglob("*.ts")):
        print("[harness] --test-only set but src/ is empty or missing.")
        print("          Run without --test-only first to generate implementation.")
        return False
    return True


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Local LLM pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-scaffold", action="store_true",
                        help="Reuse existing scaffold/scaffold.json — skip Gemini call")
    parser.add_argument("--skip-plan", action="store_true",
                        help="Reuse existing scaffold/glm_plan.json — skip GLM planning call")
    parser.add_argument("--only-qwen", action="store_true",
                        help="Skip GLM planning entirely; Qwen runs single-call mode")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip scaffold + plan + implement; jump straight to vitest")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip DeepSeek V3.2 judge step (useful during debug loops)")
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Max fix iterations for test loop (default: 3)")
    parser.add_argument("--max-cluster-attempts", type=int, default=2,
                        help="Max LLM repair attempts per cluster before give-up (default: 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Pass --verbose to 04_test_and_iterate.py for cluster debug output")
    args = parser.parse_args()

    # --test-only implies skipping all generation steps
    if args.test_only:
        args.skip_scaffold = True
        args.skip_plan = True

    results: dict[str, bool] = {}

    # ── Step 2: Gemini scaffold ──────────────────────────────────────────────
    if not args.skip_scaffold:
        if not check_env(["GEMINI_API_KEY"]):
            sys.exit(1)
        ok = run_step("Step 2 — Gemini scaffold", "02_scaffold_gemini.py")
        results["scaffold"] = ok
        if not ok:
            print("\n[harness] Scaffold failed. Stopping.")
            sys.exit(1)
    else:
        if not check_file_exists(ROOT / "scaffold" / "scaffold.json", "--skip-scaffold"):
            sys.exit(1)
        skip_step("Step 2 — Gemini scaffold", "reusing scaffold/scaffold.json")

    # ── Step 3b: GLM plan ────────────────────────────────────────────────────
    glm_plan_path = ROOT / "scaffold" / "glm_plan.json"

    if args.only_qwen or args.test_only:
        skip_step("Step 3b — GLM 5.1 plan",
                  "--only-qwen" if args.only_qwen else "--test-only")
        plan_available = False

    elif args.skip_plan:
        if not check_file_exists(glm_plan_path, "--skip-plan"):
            print("          Tip: run without --skip-plan to regenerate, "
                  "or use --only-qwen to skip planning entirely.")
            sys.exit(1)
        skip_step("Step 3b — GLM 5.1 plan", "reusing scaffold/glm_plan.json")
        plan_available = True

    else:
        if not check_env(["OPENROUTER_API_KEY"]):
            sys.exit(1)
        ok = run_step("Step 3b — GLM 5.1 plan", "03b_implement_glm.py")
        results["glm_plan"] = ok
        if not ok:
            print("\n[harness] GLM planning failed.")
            print("          Tip: --skip-plan to reuse last plan, "
                  "or --only-qwen to skip planning entirely.")
            sys.exit(1)
        plan_available = True

    # ── Step 3a: Qwen implement ──────────────────────────────────────────────
    if not args.test_only:
        if not check_env(["OPENROUTER_API_KEY"]):
            sys.exit(1)

        qwen_args: list[str] = []
        if plan_available:
            qwen_args.append("--use-glm-plan")

        mode_label = "per-file + GLM plan" if plan_available else "single-call"
        ok = run_step(
            f"Step 3a — Qwen implement ({mode_label})",
            "03a_implement_qwen.py",
            qwen_args,
        )
        results["impl_qwen"] = ok
    else:
        if not check_src_exists():
            sys.exit(1)
        skip_step("Step 3a — Qwen implement", "reusing existing src/")
        results["impl_qwen"] = True

    # ── Step 4+5: Test + iterate ─────────────────────────────────────────────
    test_args = ["--impl", "qwen", "--max-iter", str(args.max_iter),
                 "--max-cluster-attempts", str(args.max_cluster_attempts)]
    if args.verbose:
        test_args.append("--verbose")

    ok = run_step("Step 4+5 — Qwen test + iterate", "04_test_and_iterate.py", test_args)
    results["test_qwen"] = ok
    tests_passed = ok

    # ── Step 5b: Aggregate report ────────────────────────────────────────────
    run_step("Step 5b — Aggregate report", "05_report.py")

    # ── Step 6: DeepSeek V3.2 judge ────────────────────────────────────────────
    # Judge runs ONLY when tests passed. Skipped entirely if tests failed or
    # --skip-judge is set (avoids burning API budget on broken code).
    if args.skip_judge:
        skip_step("Step 6 — DeepSeek V3.2 judge", "--skip-judge")

    elif not tests_passed:
        skip_step(
            "Step 6 — DeepSeek V3.2 judge",
            "tests failed — fix tests first before requesting judge sign-off",
        )

    else:
        if not check_env(["OPENROUTER_API_KEY"]):
            print("[harness] WARNING: cannot run judge without OPENROUTER_API_KEY.")
        else:
            ok = run_step("Step 6 — DeepSeek V3.2 judge", "06_judge_deepseek.py")
            results["judge"] = ok

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for key, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {key}")

    all_ok = all(results.values())
    print(f"\n  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")
    print(f"\n  Reports:")
    print(f"    Pipeline  → reports/summary.md")
    if not args.skip_judge and tests_passed:
        print(f"    Judge     → reports/judge_report.md")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
