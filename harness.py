#!/usr/bin/env python3
"""
harness.py — Local dev runner for the LLM pipeline.
Mirrors the GitHub Actions workflow, runs on your machine.

Architecture:
    Gemini         → scaffold JSON (stubs + signatures)
    GLM 5.1        → planner: decomposes scaffold into glm_plan.json
    Qwen 3.6+      → executor: implements src/ per-file (with plan) or single-call
    vitest         → test loop: Qwen fixes surface bugs, Minimax 2.7 fixes logic bugs
    DeepSeek V3.2  → judge: qualitative review + sign-off (runs only on green)
    07_fix         → auto-fix blocking issues from judge (runs on NEEDS_REVISION)
    07_knowledge   → long-term knowledge distillation (run manually after human review)

Usage:
    # Full pipeline (scaffold → plan → implement → test → report → judge → fix → re-judge)
    python harness.py

    # Skip GLM planning — Qwen single-call mode, judge still runs
    python harness.py --only-qwen

    # Reuse existing scaffold.json
    python harness.py --skip-scaffold

    # Reuse existing glm_plan.json
    python harness.py --skip-scaffold --skip-plan

    # Skip everything up to test
    python harness.py --test-only

    # Skip judge entirely (faster, no DeepSeek call)
    python harness.py --skip-judge
    python harness.py --test-only --skip-judge

    # Skip auto-fix step 7 (run judge but stop there)
    python harness.py --skip-fix

    # Cap how many times judge→fix can loop (default: 2)
    python harness.py --max-judge-rounds 3

    # Override iteration cap
    python harness.py --max-iter 5

    # Verbose cluster debug output
    python harness.py --test-only --verbose

    # Override per-cluster LLM attempt cap
    python harness.py --test-only --max-cluster-attempts 3

Typical debug loops:
    python harness.py --test-only --skip-judge --max-iter 3
    python harness.py --skip-scaffold --skip-plan --skip-judge

After pipeline completes with judge APPROVED_WITH_NOTES or NEEDS_REVISION:
    python pipeline/07_update_knowledge.py          # interactive knowledge distillation
    python pipeline/07_update_knowledge.py --dry-run  # preview only

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


def _read_judge_verdict() -> str:
    """Read current judge verdict from reports/judge_raw.json. Returns '' if not found."""
    import json, re as _re
    raw_path = ROOT / "reports" / "judge_raw.json"
    if not raw_path.exists():
        return ""
    try:
        raw_data = json.loads(raw_path.read_text())
        resp = raw_data.get("response", "")
        resp = _re.sub(r"^```[a-z]*\n?", "", resp.strip())
        resp = _re.sub(r"\n?```$",        "", resp.strip())
        return json.loads(resp).get("verdict", "")
    except Exception:
        return ""


def _run_judge_fix_loop(args, results: dict) -> None:
    """
    Runs the judge → fix → re-judge loop up to args.max_judge_rounds times.

    Round structure:
        1. Run 06_judge_deepseek.py
        2. Read verdict from judge_raw.json
        3. If APPROVED or APPROVED_WITH_NOTES → done
        4. If NEEDS_REVISION and not --skip-fix:
             a. Run 07_fix_from_judge.py
             b. If fix script exits 0 (vitest still green) → re-run judge
             c. If fix script exits 1 (vitest now failing) → stop, mark failed
        5. If --skip-fix → stop after first judge regardless of verdict
        6. After max_judge_rounds → stop, report final verdict
    """
    max_rounds   = args.max_judge_rounds
    skip_fix     = args.skip_fix

    for round_num in range(1, max_rounds + 1):
        round_label = f"round {round_num}/{max_rounds}" if max_rounds > 1 else ""
        label_suffix = f" ({round_label})" if round_label else ""

        # ── Judge ──────────────────────────────────────────────────────────
        ok = run_step(
            f"Step 6 — DeepSeek V3.2 judge{label_suffix}",
            "06_judge_deepseek.py",
        )
        results[f"judge_r{round_num}"] = ok

        verdict = _read_judge_verdict()
        print(f"\n[harness] Judge verdict: {verdict or '(unknown)'}")

        # APPROVED → done
        if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
            print(f"[harness] ✅ Judge {verdict} — pipeline complete.")
            break

        # NEEDS_REVISION but last round → stop
        if round_num == max_rounds:
            print(f"[harness] ⚠ Reached max_judge_rounds ({max_rounds}) "
                  f"with verdict {verdict}.")
            print(f"[harness] Run manually: python pipeline/07_update_knowledge.py")
            break

        # NEEDS_REVISION and --skip-fix → stop
        if skip_fix:
            skip_step(
                f"Step 7 — Fix from judge{label_suffix}",
                "--skip-fix set — review judge_report.md manually",
            )
            break

        # NEEDS_REVISION → trigger 07_fix_from_judge
        if verdict == "NEEDS_REVISION":
            fix_ok = run_step(
                f"Step 7 — Fix from judge{label_suffix}",
                "07_fix_from_judge.py",
            )
            results[f"judge_fix_r{round_num}"] = fix_ok

            if not fix_ok:
                # 07_fix exits 1 when vitest still failing after patches
                print(f"\n[harness] ⚠ Judge fix step failed "
                      f"(vitest still failing after patches).")
                print(f"[harness] Human review required — "
                      f"see reports/judge_fix_report.json")
                break

            # Fix applied and vitest green → loop back for re-judge
            print(f"\n[harness] Fix applied successfully — re-running judge …")
            # Re-run aggregate report before re-judge so judge sees fresh state
            run_step("Step 5b — Aggregate report (post-fix)", "05_report.py")
            continue

        # Judge exited non-zero for a reason other than NEEDS_REVISION
        # (e.g. API error, parse failure) — don't loop
        print(f"[harness] Judge step failed (non-verdict error) — stopping.")
        break


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
    parser.add_argument("--skip-fix", action="store_true",
                        help="Run judge but skip 07_fix_from_judge even on NEEDS_REVISION")
    parser.add_argument("--max-judge-rounds", type=int, default=2,
                        help="Max judge→fix→re-judge rounds before giving up (default: 2)")
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

    if args.only_qwen:
        skip_step("Step 3b — GLM 5.1 plan", "--only-qwen")
        plan_available = False

    elif args.test_only:
        # --test-only skips all generation, but if a plan file exists we still
        # pass it to 03a so Qwen benefits from it (no extra API call needed).
        plan_available = glm_plan_path.exists()
        reason = "reusing existing glm_plan.json" if plan_available else "no glm_plan.json found"
        skip_step("Step 3b — GLM 5.1 plan", f"--test-only ({reason})")

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

    # ── Step 6 + 7: Judge → fix → re-judge loop ──────────────────────────────
    # Flow:
    #   tests green → judge runs
    #     APPROVED / APPROVED_WITH_NOTES → done ✓
    #     NEEDS_REVISION → 07_fix_from_judge → re-run vitest → re-judge
    #     repeat up to --max-judge-rounds
    #
    # Guard rails:
    #   - judge never runs on failed tests
    #   - 07_fix can only write to src/ (scope-locked in the script)
    #   - loop exits after max_judge_rounds regardless of verdict
    #   - --skip-fix: run judge but never trigger auto-fix

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
            _run_judge_fix_loop(args, results)

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
        judge_verdict = _read_judge_verdict()
        if judge_verdict in ("APPROVED_WITH_NOTES", "NEEDS_REVISION"):
            print(f"\n  Judge verdict: {judge_verdict}")
            print(f"  Run knowledge update when ready:")
            print(f"    python pipeline/07_update_knowledge.py")
            print(f"    python pipeline/07_update_knowledge.py --dry-run  # preview")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
