#!/usr/bin/env python3
"""
harness.py — Local dev runner for the LLM pipeline.
 
Pipeline stages (run in order):
    Step 1  spec_diff        — detect spec changes, write spec_delta.json (no LLM)
    Step 2  Gemini           — generate scaffold stubs + test files
    Step 3b GLM 5.1          — plan: decompose each stub into ordered sub-tasks
    Step 3a Qwen 3.6+        — implement src/ files guided by plan
    Step 4  vitest loop      — run tests; Qwen fixes surface bugs, Minimax fixes logic
    Step 5b report           — aggregate summary.md
    Step 6  DeepSeek V3.2    — judge: qualitative review + sign-off (green only)
    Step 7  07_fix_from_judge — auto-patch blocking issues from judge (NEEDS_REVISION only)
            └─ re-runs Step 5b + Step 6 after each fix, up to --max-judge-rounds
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Generation flags (control Steps 1–3):
 
  --force                Re-run ALL steps even if spec_delta says nothing changed.
                         Use when you want a clean slate regardless of history.
 
  --dry-run              Print what WOULD run without executing anything.
                         Useful to verify delta decisions before committing.
 
  --skip-scaffold        Skip Step 2 (Gemini). Reuse existing scaffold/scaffold.json.
                         Use when spec §7/§8 (file tree + schema) did NOT change.
 
  --skip-plan            Skip Step 3b (GLM). Reuse existing scaffold/glm_plan.json.
                         Use when you want to re-implement but keep the same plan.
 
  --only-qwen            Skip Step 3b entirely (no GLM plan at all).
                         Qwen runs in single-call mode instead of per-file mode.
                         Faster and cheaper; lower quality for complex specs.
 
  --test-only            Skip Steps 1–3a entirely. Jump straight to vitest (Step 4).
                         Reuses whatever is currently in src/.
                         Use during debug loops when src/ is already populated.
 
Test loop flags (control Step 4):
 
  --max-iter N           Max number of full vitest→repair→vitest outer loops.
                         Default: 3. Raise to 5+ for stubborn clusters.
                         Each iteration = run vitest + repair all failing clusters.
 
  --max-cluster-attempts N
                         Max LLM repair calls per individual failing cluster before
                         giving up and marking it ESCALATED.
                         Default: 2. First attempt uses Qwen (surface), second
                         uses Minimax (logic). Raise to 3 if Minimax needs more tries.
 
  --verbose              Print per-cluster debug output: which layer ran, token counts,
                         state timeline extracted, scope violations, etc.
 
Judge flags (control Steps 6–7):
 
  --skip-judge           Skip Step 6 (DeepSeek) and Step 7 entirely.
                         Use during active debug loops to save API cost.
                         Run without this flag for final sign-off.
 
  --skip-fix             Run Step 6 (judge) but skip Step 7 (auto-fix).
                         Judge report is written; you review it manually.
                         Use when you want judge feedback without automated patches.

  --from-judge           Skip Steps 1–6 entirely. Assumes tests are already green
                         and judge_raw.json already exists from a previous run.
                         Feeds that existing review into Step 7 (07_fix_from_judge)
                         without calling the judge API again, then re-judges once
                         after the fix to confirm. No API cost for the first judge call.
                         Use after: tests passed + judge already ran + you want to
                         act on the review without paying for another judge call.
 
  --max-judge-rounds N   How many times the judge→fix→re-judge loop can repeat.
                         Default: 2 (judge once, fix once, re-judge once).
                         Each round: judge runs → if NEEDS_REVISION → fix → re-judge.
                         Stops early on APPROVED or APPROVED_WITH_NOTES.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON WORKFLOWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
First run / spec changed:
    python harness.py
    # spec_diff detects changes automatically; full pipeline runs
 
Spec changed, scaffold still valid (only component props changed):
    python harness.py --skip-scaffold
    # Reuses scaffold.json; re-plans + re-implements + tests + judges
 
Debug loop (tests failing, iterate quickly without spending on judge):
    python harness.py --test-only --skip-judge --max-iter 5
    # Runs vitest loop only; increase --max-iter if clusters keep failing
 
Debug loop with more attempts per stubborn cluster:
    python harness.py --test-only --skip-judge --max-iter 5 --max-cluster-attempts 3
 
Final sign-off after debug loop passes:
    python harness.py --test-only
    # Runs vitest (should pass) → report → judge → auto-fix if needed
 
Force clean re-run (ignore all cached state):
    python harness.py --force
 
Preview what would run without executing:
    python harness.py --dry-run
    python harness.py --test-only --dry-run
 
After judge reports APPROVED_WITH_NOTES or NEEDS_REVISION:
    python pipeline/07_update_knowledge.py           # distill findings to knowledge base
    python pipeline/07_update_knowledge.py --dry-run  # preview only

Tests green + judge already ran, act on existing review without re-calling judge API:
    python harness.py --from-judge
    # Skip Steps 1–6, feed judge_raw.json into fix loop, re-judge once after fix

    python harness.py --from-judge --skip-fix
    # Same but only print existing verdict — no auto-fix
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requirements:
    pip install httpx
    GEMINI_API_KEY, OPENROUTER_API_KEY in .env or exported as env vars
"""

import argparse
import json
import os
import re as _re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

DELTA_PATH   = ROOT / "scaffold" / "spec_delta.json"
PREV_SRC_DIR = ROOT / "scaffold" / "prev_src"


# ════════════════════════════════════════════════════════════════════════════
# Core helpers
# ════════════════════════════════════════════════════════════════════════════

def load_dotenv() -> None:
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
        print("[harness] src/ is empty or missing.")
        print("          Run without --test-only first to generate implementation.")
        return False
    return True


# ════════════════════════════════════════════════════════════════════════════
# Delta helpers
# ════════════════════════════════════════════════════════════════════════════

def load_delta() -> dict | None:
    if not DELTA_PATH.exists():
        return None
    try:
        return json.loads(DELTA_PATH.read_text())
    except Exception:
        return None


def delta_requires(delta: dict | None, step: str) -> bool:
    """True if delta says step must re-run, or delta unavailable."""
    if delta is None:
        return True
    return delta.get("rerun_steps", {}).get(step, True)


def print_delta_summary(delta: dict) -> None:
    fv  = delta.get("from_version") or "(none)"
    tv  = delta.get("to_version", "?")
    print(f"\n[harness] Spec: {fv} → {tv}")
    if delta.get("is_first_run"):
        print("[harness] First run — full pipeline.")
        return
    changed  = delta.get("changed_sections", [])
    affected = delta.get("affected_files", [])
    rerun    = [k for k, v in delta.get("rerun_steps", {}).items() if v]
    skip     = [k for k, v in delta.get("rerun_steps", {}).items() if not v]
    sums     = delta.get("section_summaries", {})
    if changed:
        print(f"[harness] Changed §: {changed}")
        for sec in changed:
            if sec in sums:
                print(f"    §{sec}: {sums[sec]}")
    print(f"[harness] Affected files  : {len(affected)}")
    print(f"[harness] Steps to re-run : {rerun or '(none)'}")
    print(f"[harness] Steps to skip   : {skip or '(none)'}")


# ════════════════════════════════════════════════════════════════════════════
# src/ snapshot + restore
# ════════════════════════════════════════════════════════════════════════════

def snapshot_src() -> None:
    """Save current src/ as prev_src/ for future delta partial restores."""
    src = ROOT / "src"
    if not src.exists():
        return
    if PREV_SRC_DIR.exists():
        shutil.rmtree(PREV_SRC_DIR)
    shutil.copytree(src, PREV_SRC_DIR)
    print(f"[harness] src/ snapshot → {PREV_SRC_DIR.relative_to(ROOT)}")


def restore_unaffected_files(delta: dict) -> int:
    """
    Copy unaffected src/ files from prev_src/ so Qwen only implements
    the files that changed. Returns number of files restored.
    """
    unaffected = [f for f in delta.get("unaffected_files", []) if f.startswith("src/")]
    if not unaffected or not PREV_SRC_DIR.exists():
        return 0
    restored = 0
    for rel in unaffected:
        prev = PREV_SRC_DIR / rel[len("src/"):]
        dest = ROOT / rel
        if prev.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prev, dest)
            restored += 1
    if restored:
        print(f"[harness] Restored {restored} unaffected file(s) from prev_src/")
    return restored


# ════════════════════════════════════════════════════════════════════════════
# Judge helpers
# ════════════════════════════════════════════════════════════════════════════

def _read_judge_verdict() -> str:
    """Read current judge verdict from reports/judge_raw.json. Returns '' if not found."""
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
    Judge → fix → re-judge loop, up to args.max_judge_rounds times.

    Round structure:
        1. Run 06_judge_deepseek.py
        2. Read verdict from judge_raw.json
        3. APPROVED / APPROVED_WITH_NOTES → done ✓
        4. NEEDS_REVISION + not --skip-fix:
             a. Run 07_fix_from_judge.py
             b. Exit 0 (vitest still green) → re-run aggregate report → re-judge
             c. Exit 1 (vitest now failing) → stop, mark failed
        5. --skip-fix → stop after first judge
        6. max_judge_rounds exceeded → stop, report final verdict
    """
    max_rounds = args.max_judge_rounds
    skip_fix   = args.skip_fix

    for round_num in range(1, max_rounds + 1):
        round_sfx = f" (round {round_num}/{max_rounds})" if max_rounds > 1 else ""

        # ── Judge ──────────────────────────────────────────────────────────
        ok = run_step(
            f"Step 6 — DeepSeek V3.2 judge{round_sfx}",
            "06_judge_deepseek.py",
        )
        results[f"judge_r{round_num}"] = ok

        verdict = _read_judge_verdict()
        print(f"\n[harness] Judge verdict: {verdict or '(unknown)'}")

        if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
            print(f"[harness] ✅ Judge {verdict} — pipeline complete.")
            break

        if round_num == max_rounds:
            print(f"[harness] ⚠ Reached max_judge_rounds ({max_rounds}) "
                  f"with verdict {verdict}.")
            print(f"[harness] Run manually: python pipeline/07_update_knowledge.py")
            break

        if skip_fix:
            skip_step(
                f"Step 7 — Fix from judge{round_sfx}",
                "--skip-fix set — review judge_report.md manually",
            )
            break

        if verdict == "NEEDS_REVISION":
            fix_args = []
            if args.verbose:
                fix_args.append("--verbose")
            if getattr(args, "fix_non_blocking", False):
                fix_args.append("--fix-non-blocking")

            fix_ok = run_step(
                f"Step 7 — Fix from judge{round_sfx}",
                "07_fix_from_judge.py",
                fix_args or None,
            )
            results[f"judge_fix_r{round_num}"] = fix_ok

            if not fix_ok:
                print(f"\n[harness] ⚠ Judge fix failed (vitest still failing after patches).")
                print(f"[harness] Human review required — see reports/judge_fix_report.json")
                break

            # Fix applied and vitest green → refresh report + re-judge
            print(f"\n[harness] Fix applied successfully — re-running judge …")
            run_step("Step 5b — Aggregate report (post-fix)", "05_report.py")
            continue

        # Judge exited non-zero for non-verdict reason (API error, parse failure)
        print(f"[harness] Judge step failed (non-verdict error) — stopping.")
        break


def _run_fix_from_existing_judge(args, results: dict) -> None:
    """
    Feed an existing judge_raw.json into the fix loop without calling the API.
    Used when --from-judge is set.

    Flow:
        1. Verify reports/judge_raw.json exists
        2. Read verdict
        3. APPROVED / APPROVED_WITH_NOTES → nothing to fix, done
        4. NEEDS_REVISION + not --skip-fix:
             a. Run 07_fix_from_judge.py (applies patches, re-runs vitest internally)
             b. If fix OK → refresh report → re-judge ONCE for final sign-off
             c. If fix fails → stop, report for human review
        5. NEEDS_REVISION + --skip-fix → print verdict, done
    """
    raw_path = ROOT / "reports" / "judge_raw.json"
    if not raw_path.exists():
        print("[harness] --from-judge: reports/judge_raw.json not found.")
        print("          Run the full pipeline first to generate a judge report.")
        results["judge_from_existing"] = False
        return

    verdict = _read_judge_verdict()
    print(f"\n[harness] Existing judge verdict: {verdict or '(unknown)'}")
    results["judge_from_existing"] = True

    if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
        print(f"[harness] ✅ Already {verdict} — nothing to fix.")
        return

    if verdict != "NEEDS_REVISION":
        print(f"[harness] ⚠ Unrecognised verdict '{verdict}' — stopping.")
        results["judge_from_existing"] = False
        return

    if args.skip_fix:
        skip_step(
            "Step 7 — Fix from judge (existing review)",
            "--skip-fix set — review judge_report.md manually",
        )
        return

    fix_args = []
    if args.verbose:
        fix_args.append("--verbose")
    if getattr(args, "fix_non_blocking", False):
        fix_args.append("--fix-non-blocking")
        
    # Apply fix
    fix_ok = run_step(
        "Step 7 — Fix from judge (existing review)",
        "07_fix_from_judge.py",
        fix_args or None,
    )
    results["judge_fix"] = fix_ok

    if not fix_ok:
        print(f"\n[harness] ⚠ Fix failed (vitest still failing after patches).")
        print(f"[harness] Human review required — see reports/judge_fix_report.json")
        return

    # Fix applied → refresh report → re-judge once for final sign-off
    print(f"\n[harness] Fix applied — refreshing report + re-judging …")
    run_step("Step 5b — Aggregate report (post-fix)", "05_report.py")

    if not check_env(["OPENROUTER_API_KEY"]):
        print("[harness] WARNING: cannot re-judge without OPENROUTER_API_KEY.")
        return

    ok = run_step("Step 6 — DeepSeek V3.2 judge (post-fix)", "06_judge_deepseek.py")
    results["judge_post_fix"] = ok
    final = _read_judge_verdict()
    if final:
        print(f"\n[harness] Post-fix verdict: {final}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Local LLM pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Delta-aware flags
    parser.add_argument("--force", action="store_true",
                        help="Ignore spec_delta.json — re-run all steps unconditionally")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing anything")
    # Generation override flags
    parser.add_argument("--skip-scaffold", action="store_true",
                        help="Skip Gemini scaffold (overrides delta)")
    parser.add_argument("--skip-plan", action="store_true",
                        help="Skip GLM planning (overrides delta)")
    parser.add_argument("--retry-impl", action="store_true",
                        help="Retry only failed files from last impl run "
                         "(reads impl_qwen.json failed_files). "
                         "Implies --skip-scaffold --skip-plan.")
    parser.add_argument("--only-qwen", action="store_true",
                        help="Skip GLM planning entirely; Qwen single-call mode")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip scaffold + plan + implement; jump straight to vitest")
    # Judge flags
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judge entirely (overrides delta)")
    parser.add_argument("--skip-fix", action="store_true",
                        help="Run judge but skip 07_fix_from_judge even on NEEDS_REVISION")
    parser.add_argument("--from-judge", action="store_true",
                        help="Skip Steps 1–6; feed existing reports/judge_raw.json "
                             "into fix loop directly, then re-judge once after fix")
    parser.add_argument("--max-judge-rounds", type=int, default=2,
                        help="Max judge→fix→re-judge rounds before giving up (default: 2)")
    # Test loop flags
    parser.add_argument("--max-iter", type=int, default=3,
                        help="Max fix iterations for test loop (default: 3)")
    parser.add_argument("--max-cluster-attempts", type=int, default=2,
                        help="Max LLM repair attempts per cluster before give-up (default: 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Pass --verbose to 04_test_and_iterate.py")
    args = parser.parse_args()

    if args.test_only:
        args.skip_scaffold = True
        args.skip_plan     = True

    if args.retry_impl:
       args.skip_scaffold = True
       args.skip_plan     = True
       # --only-files sẽ được build từ impl_qwen.json bên dưới
 
    # --from-judge: skip everything up to judge, feed existing review
    if args.from_judge:
        args.skip_scaffold = True
        args.skip_plan     = True
        args.test_only     = True   # skip Steps 1–5a
        args.skip_judge    = True   # skip Step 6 (judge API call)

    results: dict[str, bool] = {}

    # ── Step 1: Spec diff ────────────────────────────────────────────────────
    # Always fast (no LLM). Skipped only with --test-only (no spec change expected).
    delta: dict | None = None

    if args.test_only:
        skip_step("Step 1 — Spec diff", "--test-only")
    else:
        run_step("Step 1 — Spec diff", "spec_diff.py")
        delta = load_delta()
        if delta:
            print_delta_summary(delta)
        if args.force:
            print("[harness] --force: delta loaded for info only — all steps will re-run.")
            delta = None   # treat as no delta → everything re-runs

    # ── Auto-skip from delta (manual flags take priority) ────────────────────
    if delta and not delta.get("is_first_run"):
        if not delta_requires(delta, "scaffold") and not args.skip_scaffold:
            args.skip_scaffold = True
            print("[harness] delta: §7/§8 unchanged → scaffold skipped")
        if (not delta_requires(delta, "plan")
                and not args.skip_plan and not args.only_qwen):
            args.skip_plan = True
            print("[harness] delta: no affected files → plan skipped")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[harness] DRY RUN — nothing executed.")
        plan_str = ("skip (only-qwen)" if args.only_qwen
                    else "skip" if args.skip_plan else "run")
        steps = [
            ("scaffold",       "skip" if args.skip_scaffold else "run"),
            ("plan",           plan_str),
            ("implement",      "skip" if args.test_only else "run"),
            ("test",           "run"),
            ("report",         "run"),
            ("judge+fix loop", "skip" if args.skip_judge else f"run (max {args.max_judge_rounds} rounds)"),
        ]
        for name, action in steps:
            icon = "▶" if "run" in action else "⏭"
            print(f"  {icon}  {name:20}  {action}")
        return

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
        plan_available = glm_plan_path.exists()
        reason = ("reusing existing glm_plan.json" if plan_available
                  else "no glm_plan.json found")
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
    
        # Delta: restore unaffected files before Qwen runs
        # Skip restore on --retry-impl — prev_src/ already applied from last run
        if delta and not delta.get("is_first_run") and not args.retry_impl:
            restore_unaffected_files(delta)
    
        qwen_args: list[str] = []
        if plan_available:
            qwen_args.append("--use-glm-plan")
    
        # --retry-impl overrides delta --only-files
        if args.retry_impl:
            impl_rec_path = ROOT / "scaffold" / "impl_qwen.json"
            if impl_rec_path.exists():
                try:
                    rec    = json.loads(impl_rec_path.read_text())
                    failed = rec.get("failed_files", [])
                    if failed:
                        qwen_args += ["--only-files", ",".join(failed)]
                        print(f"[harness] --retry-impl: retrying {len(failed)} failed file(s).")
                    else:
                        print("[harness] --retry-impl: no failed_files in impl_qwen.json — nothing to retry.")
                        sys.exit(0)
                except Exception:
                    print("[harness] --retry-impl: could not read impl_qwen.json.")
                    sys.exit(1)
            else:
                print("[harness] --retry-impl: impl_qwen.json not found — run full impl first.")
                sys.exit(1)
    
        # Delta --only-files (skipped when --retry-impl already set --only-files)
        elif delta and not delta.get("is_first_run"):
            src_affected = [f for f in delta.get("affected_files", [])
                            if f.startswith("src/")]
            if src_affected:
                qwen_args += ["--only-files", ",".join(src_affected)]
                print(f"[harness] Qwen will implement {len(src_affected)} "
                      f"affected file(s) only.")
    
        mode_label = "per-file + GLM plan" if plan_available else "single-call"
        if args.retry_impl:
            mode_label += " | retry-impl"
        elif delta and not delta.get("is_first_run"):
            n = len([f for f in delta.get("affected_files", []) if f.startswith("src/")])
            mode_label += f" | {n} affected"
    
        ok = run_step(
            f"Step 3a — Qwen implement ({mode_label})",
            "03a_implement_qwen.py",
            qwen_args,
        )
        results["impl_qwen"] = ok
    
        if not ok:
            impl_rec_path = ROOT / "scaffold" / "impl_qwen.json"
            if impl_rec_path.exists():
                try:
                    rec    = json.loads(impl_rec_path.read_text())
                    failed = rec.get("failed_files", [])
                    if failed:
                        print(f"\n[harness] {len(failed)} file(s) failed to implement:")
                        for fp in failed:
                            print(f"    {fp}")
                        print(f"\n[harness] Retry: python harness.py --retry-impl")
                except Exception:
                    pass
             
        # Snapshot src/ after successful implement for future delta runs
        if ok:
            snapshot_src()

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

    # ── Step 6 + 7: Judge → fix → re-judge loop ─────────────────────────────
    if args.skip_judge and not args.from_judge:
        skip_step("Step 6 — DeepSeek V3.2 judge", "--skip-judge")

    elif not tests_passed:
        skip_step(
            "Step 6 — DeepSeek V3.2 judge",
            "tests failed — fix tests first before requesting judge sign-off",
        )

    elif args.from_judge:
        # --from-judge: skip_judge was set to suppress Step 6 above;
        # now feed the existing review into the fix loop instead.
        skip_step("Step 6 — DeepSeek V3.2 judge", "--from-judge (reusing judge_raw.json)")
        _run_fix_from_existing_judge(args, results)

    else:
        if not check_env(["OPENROUTER_API_KEY"]):
            print("[harness] WARNING: cannot run judge without OPENROUTER_API_KEY.")
        else:
            _run_judge_fix_loop(args, results)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    if delta and not delta.get("is_first_run"):
        fv = delta.get("from_version") or "?"
        tv = delta.get("to_version", "?")
        n  = len(delta.get("affected_files", []))
        print(f"  Spec: {fv} → {tv}  ({n} file(s) affected)")
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
            print(f"    python pipeline/07_update_knowledge.py --dry-run")
    
    # ── Persist apply record (cross-run state tracking) ──────────────────────
    # Write spec_applied.json so next spec_diff run knows what was last applied.
    # Only written on overall PASS — failed runs don't count as "applied".
    if all_ok and delta:
        try:
            from pipeline.spec_diff import write_applied
        except ImportError:
            # spec_diff.py lives in pipeline/ subdirectory
            import importlib.util, sys as _sys
            _spec = importlib.util.spec_from_file_location(
                "spec_diff", ROOT / "pipeline" / "spec_diff.py"
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            write_applied = _mod.write_applied

        applied_steps = [k for k, v in results.items() if v]
        write_applied(
            version=delta.get("to_version", "unknown"),
            steps=applied_steps,
            status="PASS",
        )
        print(f"\n  Apply record → scaffold/spec_applied.json  "
              f"(v{delta.get('to_version', '?')} marked as applied)")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
