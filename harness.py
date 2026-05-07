#!/usr/bin/env python3
"""
harness.py — Orchestrator for the LLM pipeline.

harness.py is the ONLY entrypoint for end-to-end pipeline runs.
Each script in pipeline/ is a pure step runner: executes one step,
writes owned artifacts, exits. No step script navigates the pipeline.

Pipeline steps (canonical order):
    absorb      01_absorber.py         — scan codebase, build knowledge maps
    clarify     00_clarificator.py     — clarify requirements interactively
    scaffold    02_scaffold_gemini.py  — generate stub files + test files
    plan        03b_implement_glm.py   — decompose stubs into ordered task plan
    implement   03a_implement_qwen.py  — implement src/ files guided by plan
    test        04_test_and_iterate.py — vitest loop with Qwen+Minimax repair
    report      05_report.py           — aggregate summary.md
    judge       06_judge_deepseek.py   — qualitative review + sign-off
    fix         07_fix_from_judge.py   — auto-patch NEEDS_REVISION findings
                └─ re-runs report + judge after each fix, up to --max-judge-rounds
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MINI MODE — daily driver for small / targeted tasks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Bypasses the full pipeline. Implemented in pipeline/mini_mode.py.
  Rule of thumb: no spec.md change needed → --mini.

  --mini "PROMPT"          Patch local files or rewrite a context file.
                           Loads knowledge context, calls Qwen, verifies,
                           retries ×2, logs to run/mini_log.json.

  --files f1 f2 ...        (code mode) Files to patch. If omitted, LLM
                           suggests and you confirm interactively.

  --context-file FILE      (DE/MLOps mode) Read FILE as task input.
                           LLM rewrites its content. Task type is
                           auto-detected from extension (.sql → sql,
                           .py → python, .yaml → config, etc.).

  --output-file FILE       Where to write the result. Defaults to
                           overwriting --context-file in place.

  --task-type TYPE         Override auto-detection.
                           Values: code | sql | python | config | text | auto
                           Verifiers: vitest | sqlfluff | py_compile+ruff |
                                      yaml/json parse | LLM self-review

  --dry-run                Print plan without writing anything.

  Examples:
    python harness.py --mini "fix button color" --files src/Header.tsx
    python harness.py --mini "optimize for partition pruning" \\
        --context-file queries/daily_agg.sql
    python harness.py --mini "split DAG into two tasks" \\
        --context-file dags/ingest_orders.py --output-file dags/ingest_orders.py
    python harness.py --mini "..." --context-file queries/q.sql --dry-run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Generation flags (control Steps 1–3):
 
  --force                Re-run ALL steps even if spec_delta says nothing changed.
                         Use when you want a clean slate regardless of history.
 
  --dry-run              Print what WOULD run without executing anything.
                         Useful to verify delta decisions before committing.
 
  --skip-scaffold        Skip Step 2 (Gemini). Reuse existing artifacts/state/scaffold.json.
                         Use when spec §7/§8 (file tree + schema) did NOT change.
 
  --skip-plan            Skip Step 3b (GLM). Reuse existing artifacts/state/plan.json.
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
 
Clarify requirement before building (async client Q&A):
    python harness.py --clarify-only --clarify-input requirement.md
    python harness.py --clarify-only                    # interactive mode

Build after clarification is done:
    python harness.py --skip-clarify

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
import datetime
import json
import os
import re as _re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# harness.py lives at ROOT level — same level as artifacts/
ROOT = Path(__file__).parent

# === WRITE AUTHORITY: harness ===
# OWNS  : (none — orchestrator only; mini_mode.py owns run/mini_log.json)
# READS : all artifacts (coordinates pipeline steps)
# NOTE  : mini mode is delegated entirely to pipeline/mini_mode.py

import sys as _sys
_sys.path.insert(0, str(ROOT))
# LazyPath constants — resolve to the correct project artifact root at use time.
# ensure_dirs() is called inside main() after --project is parsed and
# PIPELINE_PROJECT env var is set.  Do NOT call ensure_dirs() here.
from artifacts.paths import (
    SPEC_DELTA as DELTA_PATH,
    SCAFFOLD_JSON,
    PLAN_JSON as GLM_PLAN_PATH,
    IMPL_RECORD as IMPL_RECORD_PATH,
    UPDATE_LOG as UPDATE_LOG_PATH,
    JUDGE_RAW as JUDGE_RAW_PATH,
    CLARIFICATION_REPORT,
    CLARIFIED_REQ,
)
# NOTE: prev_src is a transient staging dir, not a pipeline artifact.
# Resolved lazily so it uses the correct project artifact root.
def _prev_src_dir() -> Path:
    from artifacts.paths import artifact_root
    return artifact_root() / "state" / "prev_src"

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
    prev_src = _prev_src_dir()
    if prev_src.exists():
        shutil.rmtree(prev_src)
    shutil.copytree(src, prev_src)
    print(f"[harness] src/ snapshot → {prev_src.relative_to(ROOT)}")


def restore_unaffected_files(delta: dict) -> int:
    """
    Copy unaffected src/ files from prev_src/ so Qwen only implements
    the files that changed. Returns number of files restored.
    """
    unaffected = [f for f in delta.get("unaffected_files", []) if f.startswith("src/")]
    prev_src = _prev_src_dir()
    if not unaffected or not prev_src.exists():
        return 0
    restored = 0
    for rel in unaffected:
        prev = prev_src / rel[len("src/"):]
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
    """Read current judge verdict from JUDGE_RAW_PATH. Returns '' if not found."""
    raw_path = JUDGE_RAW_PATH
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
                print(f"[harness] Human review required — see {UPDATE_LOG_PATH}")
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
        print(f"[harness] Human review required — see {UPDATE_LOG_PATH}")
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

# ════════════════════════════════════════════════════════════════════════════
# Step registry — single source of truth for pipeline order
# ════════════════════════════════════════════════════════════════════════════

STEPS = ["absorb", "clarify", "scaffold", "plan",
         "implement", "test", "report", "judge", "fix"]

STEP_SCRIPTS: dict[str, str] = {
    "absorb":    "01_absorber.py",
    "clarify":   "00_clarificator.py",
    "scaffold":  "02_scaffold_gemini.py",
    "plan":      "03b_implement_glm.py",
    "implement": "03a_implement_qwen.py",
    "test":      "04_test_and_iterate.py",
    "report":    "05_report.py",
    "judge":     "06_judge_deepseek.py",
    "fix":       "07_fix_from_judge.py",
}

STEP_ENV_KEYS: dict[str, list[str]] = {
    "scaffold":  ["GEMINI_API_KEY"],
    "plan":      ["OPENROUTER_API_KEY"],
    "implement": ["OPENROUTER_API_KEY"],
    "test":      ["OPENROUTER_API_KEY"],
    "judge":     ["OPENROUTER_API_KEY"],
    "fix":       ["OPENROUTER_API_KEY"],
}


def _resolve_run_range(args: argparse.Namespace) -> tuple[str, str]:
    """
    Return (from_step, until_step) after validating all flag combinations.
    Raises SystemExit on invalid input.

    Rules:
        --from-X                    → X .. report  (or last step)
        --from-X --until-Y          → X .. Y
        --X  (shorthand)            → X .. X
        no flags                    → absorb .. fix  (full run)

    Error cases:
        --until-Y without --from-X  → error
        --from-X where X > Y        → error
        multiple shorthand flags    → error
        shorthand mixed with from/until → error
    """
    from_step  = getattr(args, "from_step",  None)
    until_step = getattr(args, "until_step", None)

    # Collect shorthand flags (--absorb, --clarify, ...)
    shorthands = [s for s in STEPS if getattr(args, s, False)]

    if shorthands and (from_step or until_step):
        _die(f"Cannot mix --{shorthands[0]} shorthand with --from/--until flags.")

    if len(shorthands) > 1:
        _die(f"Only one step shorthand allowed at a time, got: "
             f"{' '.join('--'+s for s in shorthands)}")

    if shorthands:
        return shorthands[0], shorthands[0]

    if until_step and not from_step:
        _die(f"--until-{until_step} requires --from-<step>.")

    if not from_step and not until_step:
        return STEPS[0], STEPS[-1]   # full run

    from_idx  = STEPS.index(from_step)
    until_idx = STEPS.index(until_step) if until_step else len(STEPS) - 1
    until_step = until_step or STEPS[-1]

    if from_idx > until_idx:
        order = " → ".join(STEPS)
        _die(f"--from-{from_step} comes after --until-{until_step}.\n  Order: {order}")
    return from_step, until_step


def _die(msg: str) -> None:
    print(f"[harness] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _print_dry_run(from_step: str, until_step: str, args: argparse.Namespace) -> None:
    """Print what would run without executing anything."""
    from_idx  = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)
    print("\n[harness] DRY RUN — nothing will be executed.")
    print(f"  Project : {os.environ.get('PIPELINE_PROJECT', '?')}")
    print(f"  Range   : {from_step} → {until_step}")
    print()
    for i, step in enumerate(STEPS):
        if from_idx <= i <= until_idx:
            script = STEP_SCRIPTS[step]
            print(f"  ▶  {step:<12}  {script}")
        else:
            print(f"  ⏭  {step:<12}  (skipped)")
    print()
    if args.force:
        print("  --force: delta checks will be bypassed — all steps re-run.")
    print()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="Pipeline orchestrator. Run full end-to-end or a sub-range of steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py                              # full run
  python harness.py --from-clarify               # clarify → fix
  python harness.py --from-implement --until-test  # implement + test only
  python harness.py --scaffold                   # scaffold step only
  python harness.py --dry-run                    # preview full run
  python harness.py --from-test --dry-run        # preview test → fix
""",
    )

    # ── Project ───────────────────────────────────────────────────────────────
    parser.add_argument("--project", type=str, default=None,
                        metavar="NAME",
                        help="Project name → artifacts_<slug>/. "
                             "If omitted, interactive prompt lists existing projects.")

    # ── Flow control: range ───────────────────────────────────────────────────
    for step in STEPS:
        parser.add_argument(f"--from-{step}", dest="from_step",
                            action="store_const", const=step,
                            help=f"Start pipeline from step '{step}'.")
        parser.add_argument(f"--until-{step}", dest="until_step",
                            action="store_const", const=step,
                            help=f"Stop pipeline after step '{step}'.")

    # ── Flow control: shorthand (single step) ────────────────────────────────
    for step in STEPS:
        parser.add_argument(f"--{step}", dest=step,
                            action="store_true",
                            help=f"Run only step '{step}' (shorthand for "
                                 f"--from-{step} --until-{step}).")

    # ── Behaviour modifiers (not flow control) ───────────────────────────────
    parser.add_argument("--dry-run", action="store_true",
                        help="Print steps that would run without executing anything.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if spec_delta says nothing changed.")
    parser.add_argument("--only-qwen", action="store_true",
                        help="Skip GLM plan step; Qwen runs in single-call mode.")
    parser.add_argument("--retry-impl", action="store_true",
                        help="Re-implement only files listed as failed in impl_record.json.")
    parser.add_argument("--max-judge-rounds", type=int, default=2,
                        metavar="N",
                        help="Max judge→fix→re-judge iterations (default: 2).")
    parser.add_argument("--max-iter", type=int, default=3,
                        metavar="N",
                        help="Max vitest→repair outer loops (default: 3).")
    parser.add_argument("--max-cluster-attempts", type=int, default=2,
                        metavar="N",
                        help="Max LLM repair calls per failing cluster (default: 2).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-cluster debug output during test step.")
    parser.add_argument("--skip-fix", action="store_true",
                        help="Run judge but skip auto-fix step.")
    parser.add_argument("--clarify-input", type=str, default=None,
                        metavar="FILE",
                        help="Pass file path as input to clarify step.")

    # ── Mini mode (unchanged) ─────────────────────────────────────────────────
    parser.add_argument("--mini", type=str, default=None, metavar="PROMPT",
                        help="Mini mode: targeted task without full pipeline.")
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--context-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None)

    args = parser.parse_args()

    # ── Project resolution ────────────────────────────────────────────────────
    if not args.project:
        args.project = _interactive_project_select(ROOT)
    os.environ["PIPELINE_PROJECT"] = args.project

    from artifacts.paths import ensure_dirs, artifact_root
    ensure_dirs()

    _project_info = {
        "name": args.project,
        "artifact_root": str(artifact_root()),
    }
    print(f"[harness] Project  : {_project_info['name']}")
    print(f"[harness] Workspace: {_project_info['artifact_root']}")

    # ── Mini mode: delegate entirely, no flow control ─────────────────────────
    if args.mini is not None:
        from pipeline.mini_mode import run_mini  # type: ignore
        run_mini(
            prompt           = args.mini,
            files            = args.files,
            context_file     = Path(args.context_file) if args.context_file else None,
            output_file      = Path(args.output_file)  if args.output_file  else None,
            task_type_override = args.task_type,
            dry_run          = args.dry_run,
        )
        return

    # ── Resolve step range ────────────────────────────────────────────────────
    from_step, until_step = _resolve_run_range(args)

    if args.dry_run:
        _print_dry_run(from_step, until_step, args)
        return

    from_idx  = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)
    steps_to_run = STEPS[from_idx : until_idx + 1]

    print(f"[harness] Steps    : {from_step} → {until_step}")
    print()

    results: dict[str, bool] = {}

    # ── spec_diff: always runs before scaffold/plan/implement if in range ─────
    # spec_diff is an internal pre-step, not exposed as a named pipeline step,
    # because users never need to skip it independently.
    delta: dict | None = None
    needs_diff = any(s in steps_to_run for s in ("scaffold", "plan", "implement"))
    if needs_diff:
        run_step("spec diff", "spec_diff.py")
        delta = load_delta()
        if delta:
            print_delta_summary(delta)
        if args.force:
            print("[harness] --force: delta ignored — all steps will re-run.")
            delta = None

    # ── Execute steps ─────────────────────────────────────────────────────────
    tests_passed = True
    plan_available = GLM_PLAN_PATH.exists()

    for step in steps_to_run:
        ok = _run_step(step, args, delta, plan_available, results, tests_passed)
        results[step] = ok
        if step == "plan" and ok:
            plan_available = True
        if step == "test":
            tests_passed = ok
        if step == "implement" and ok:
            snapshot_src()
        if not ok and step in ("scaffold", "implement"):
            print(f"\n[harness] {step} failed — stopping pipeline.")
            sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results, delta, args, tests_passed)

    all_ok = all(results.values())

    # Persist apply record on full PASS
    if all_ok and delta and "implement" in results:
        _write_apply_record(delta, results)

    sys.exit(0 if all_ok else 1)


def _run_step(
    step: str,
    args: argparse.Namespace,
    delta: dict | None,
    plan_available: bool,
    results: dict,
    tests_passed: bool,
) -> bool:
    """Dispatch a single step with its extra args. Returns success bool."""

    extra: list[str] = []

    if step == "absorb":
        ok = run_step("absorb", STEP_SCRIPTS["absorb"])
        return ok

    if step == "clarify":
        clarify_args: list[str] = []
        if getattr(args, "clarify_input", None):
            clarify_args += ["--input", args.clarify_input]
        ok = run_step("clarify", STEP_SCRIPTS["clarify"], clarify_args or None)
        if not ok:
            print("\n[harness] Clarify failed.")
        return ok

    if step == "scaffold":
        # Auto-skip when delta says §7/§8 unchanged (unless --force already cleared delta)
        if delta and not delta.get("is_first_run") and not delta_requires(delta, "scaffold"):
            skip_step("scaffold", "delta: §7/§8 unchanged — reusing scaffold.json")
            return True
        if not check_env(STEP_ENV_KEYS.get("scaffold", [])):
            sys.exit(1)
        ok = run_step("scaffold", STEP_SCRIPTS["scaffold"])
        if not ok:
            print("\n[harness] Scaffold failed — stopping.")
        return ok

    if step == "plan":
        if args.only_qwen:
            skip_step("plan", "--only-qwen")
            return True
        if delta and not delta.get("is_first_run") and not delta_requires(delta, "plan"):
            skip_step("plan", "delta: no affected files — reusing plan.json")
            return True
        if not check_env(STEP_ENV_KEYS.get("plan", [])):
            sys.exit(1)
        ok = run_step("plan", STEP_SCRIPTS["plan"])
        if not ok:
            print("\n[harness] Plan failed.\n"
                  "  Tip: --only-qwen to skip planning entirely.")
        return ok

    if step == "implement":
        if not check_env(STEP_ENV_KEYS.get("implement", [])):
            sys.exit(1)
        qwen_args: list[str] = []
        if plan_available:
            qwen_args.append("--use-glm-plan")
        if args.retry_impl:
            qwen_args = _retry_impl_args(qwen_args)
            if qwen_args is None:
                return False
        elif delta and not delta.get("is_first_run"):
            restore_unaffected_files(delta)
            src_affected = [f for f in delta.get("affected_files", [])
                            if f.startswith("src/")]
            if src_affected:
                qwen_args += ["--only-files", ",".join(src_affected)]
                print(f"[harness] implement: {len(src_affected)} affected file(s) only.")
        mode = "per-file+plan" if plan_available else "single-call"
        ok = run_step(f"implement ({mode})", STEP_SCRIPTS["implement"], qwen_args)
        if not ok:
            _print_impl_failed()
        return ok

    if step == "test":
        test_args = ["--impl", "qwen",
                     "--max-iter", str(args.max_iter),
                     "--max-cluster-attempts", str(args.max_cluster_attempts)]
        if args.verbose:
            test_args.append("--verbose")
        return run_step("test", STEP_SCRIPTS["test"], test_args)

    if step == "report":
        return run_step("report", STEP_SCRIPTS["report"])

    if step == "judge":
        if not tests_passed:
            skip_step("judge", "tests failed — fix tests before judge sign-off")
            return True
        if not check_env(STEP_ENV_KEYS.get("judge", [])):
            print("[harness] WARNING: cannot run judge without OPENROUTER_API_KEY.")
            return False
        _run_judge_fix_loop(args, results)
        return results.get("judge", False)

    if step == "fix":
        # fix is handled inside _run_judge_fix_loop; if judge ran, skip standalone
        if "judge" in results:
            skip_step("fix", "already handled by judge loop")
            return True
        _run_fix_from_existing_judge(args, results)
        return results.get("fix_from_judge", True)

    return True


def _retry_impl_args(qwen_args: list[str]) -> list[str] | None:
    """Build --only-files args from impl_record failed_files. Returns None on error."""
    if not IMPL_RECORD_PATH.exists():
        print("[harness] --retry-impl: impl_record.json not found — run full impl first.")
        return None
    try:
        rec    = json.loads(IMPL_RECORD_PATH.read_text())
        failed = rec.get("failed_files", [])
        if not failed:
            print("[harness] --retry-impl: no failed_files in impl_record.json — nothing to retry.")
            return None
        qwen_args += ["--only-files", ",".join(failed)]
        print(f"[harness] --retry-impl: retrying {len(failed)} failed file(s).")
        return qwen_args
    except Exception:
        print("[harness] --retry-impl: could not read impl_record.json.")
        return None


def _print_impl_failed() -> None:
    if not IMPL_RECORD_PATH.exists():
        return
    try:
        rec    = json.loads(IMPL_RECORD_PATH.read_text())
        failed = rec.get("failed_files", [])
        if failed:
            print(f"\n[harness] {len(failed)} file(s) failed to implement:")
            for fp in failed:
                print(f"    {fp}")
            print("\n[harness] Retry: python harness.py --retry-impl")
    except Exception:
        pass


def _print_summary(
    results: dict,
    delta: dict | None,
    args: argparse.Namespace,
    tests_passed: bool,
) -> None:
    from artifacts.paths import CLARIFICATION_REPORT, artifact_root
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
    if CLARIFICATION_REPORT.exists() and results.get("clarify"):
        print(f"    Clarify  → {CLARIFICATION_REPORT}")
    print(f"    Pipeline → {artifact_root() / 'reports' / 'summary.md'}")
    if results.get("judge") and tests_passed:
        print(f"    Judge    → {artifact_root() / 'reports' / 'judge_report.md'}")
        judge_verdict = _read_judge_verdict()
        if judge_verdict in ("APPROVED_WITH_NOTES", "NEEDS_REVISION"):
            print(f"\n  Judge verdict: {judge_verdict}")
            print("  Run knowledge update when ready:")
            print("    python pipeline/07_update_knowledge.py")


def _write_apply_record(delta: dict, results: dict) -> None:
    try:
        from pipeline.spec_diff import write_applied  # type: ignore
    except ImportError:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "spec_diff", ROOT / "pipeline" / "spec_diff.py"
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        write_applied = _mod.write_applied

    applied_steps = [k for k, v in results.items() if v]
    write_applied(version=delta.get("to_version", "unknown"),
                  steps=applied_steps, status="PASS")
    from artifacts.paths import SPEC_APPLIED
    print(f"\n  Apply record → {SPEC_APPLIED}  "
          f"(v{delta.get('to_version', '?')} marked as applied)")


if __name__ == "__main__":
    main()
