#!/usr/bin/env python3
"""
harness.py — Generic LLM pipeline runner.

The pipeline is organised around AGENT ROLES, not specific models.
You choose which model fills each role via pipeline.config.json (or env vars).
The techstack is also configurable — not baked in.

Agent roles:
    scaffold_agent   — reads spec.md, generates scaffold stubs + test files
    planner_agent    — reasons over stubs, produces an ordered implementation plan
    executor_agent   — implements src/ files guided by the plan
    tester_agent     — repairs failing test clusters (surface layer)
    logic_agent      — repairs deep logic failures (escalation layer)
    judge_agent      — qualitative review + sign-off

Pipeline stages (run in order):
    Step 1  spec_diff        — detect spec changes, write spec_delta.json (no LLM)
    Step 2  scaffold_agent   — generate scaffold stubs + test files
    Step 3b planner_agent    — decompose each stub into ordered sub-tasks
    Step 3a executor_agent   — implement src/ files guided by plan
    Step 4  test loop        — run tests; tester_agent fixes surface, logic_agent fixes logic
    Step 5b report           — aggregate summary.md
    Step 6  judge_agent      — qualitative review + sign-off (green only)
    Step 7  fix_from_judge   — auto-patch blocking issues (NEEDS_REVISION only)
             └─ re-runs Step 5b + Step 6 after each fix, up to --max-judge-rounds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIGURATION (pipeline/pipeline_config.json)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    {
      "techstack": "react-vite-ts",          // or "nextjs", "vue", "svelte", …
      "agents": {
        "scaffold":  { "model": "gemini-2.5-flash",  "provider": "gemini" },
        "planner":   { "model": "glm-5.1",           "provider": "openrouter" },
        "executor":  { "model": "qwen3-30b",         "provider": "openrouter" },
        "tester":    { "model": "qwen3-30b",         "provider": "openrouter" },
        "logic":     { "model": "minimax-m1",        "provider": "openrouter" },
        "judge":     { "model": "deepseek-v3",       "provider": "openrouter" }
      }
    }

    Each pipeline step reads its agent config from this file at runtime.
    Swap any model by editing the config — no code changes needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generation flags (control Steps 1–3):

  --force                Re-run ALL steps even if spec_delta says nothing changed.
                         Use when you want a clean slate regardless of history.

  --dry-run              Print what WOULD run without executing anything.
                         Useful to verify delta decisions before committing.

  --skip-scaffold        Skip Step 2 (scaffold_agent). Reuse existing scaffold/scaffold.json.
                         Use when spec §7/§8 (file tree + schema) did NOT change.

  --skip-plan            Skip Step 3b (planner_agent). Reuse existing scaffold/plan.json.
                         Use when you want to re-implement but keep the same plan.

  --only-executor        Skip Step 3b entirely (no plan at all).
                         executor_agent runs in single-call mode.
                         Faster and cheaper; lower quality for complex specs.

  --test-only            Skip Steps 1–3a entirely. Jump straight to tests (Step 4).
                         Reuses whatever is currently in src/.

Test loop flags (control Step 4):

  --max-iter N           Max number of full test→repair→test outer loops.
                         Default: 3. Raise to 5+ for stubborn clusters.

  --max-cluster-attempts N
                         Max LLM repair calls per individual failing cluster before
                         giving up and marking it ESCALATED.
                         Default: 2. First attempt = tester_agent, second = logic_agent.

  --verbose              Print per-cluster debug output.

Judge flags (control Steps 6–7):

  --skip-judge           Skip Step 6 (judge_agent) and Step 7 entirely.

  --skip-fix             Run Step 6 (judge) but skip Step 7 (auto-fix).

  --from-judge           Skip Steps 1–6. Feed existing reports/judge_raw.json into
                         the fix loop, then re-judge once after the fix.

  --max-judge-rounds N   How many times the judge→fix→re-judge loop can repeat.
                         Default: 2.

Retry flags:

  --retry-impl           Retry only failed files from the last executor run.
                         Implies --skip-scaffold --skip-plan.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON WORKFLOWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

First run / spec changed:
    python harness.py

Spec changed, scaffold still valid:
    python harness.py --skip-scaffold

Debug loop (iterate quickly without judge cost):
    python harness.py --test-only --skip-judge --max-iter 5

Debug with more repair attempts per cluster:
    python harness.py --test-only --skip-judge --max-iter 5 --max-cluster-attempts 3

Final sign-off after debug loop passes:
    python harness.py --test-only

Force clean re-run:
    python harness.py --force

Preview what would run:
    python harness.py --dry-run

Act on existing judge review without re-calling API:
    python harness.py --from-judge
    python harness.py --from-judge --skip-fix

After judge reports APPROVED_WITH_NOTES or NEEDS_REVISION:
    python pipeline/07_update_knowledge.py
    python pipeline/07_update_knowledge.py --dry-run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requirements:
    pip install httpx
    API keys configured in .env or exported as env vars.
    See pipeline/pipeline_config.json for agent → model mapping.
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

DELTA_PATH      = ROOT / "scaffold" / "spec_delta.json"
PREV_SRC_DIR    = ROOT / "scaffold" / "prev_src"
PIPELINE_CONFIG = ROOT / "pipeline" / "pipeline_config.json"


# ════════════════════════════════════════════════════════════════════════════
# Pipeline config loader
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG: dict = {
    "techstack": "react-vite-ts",
    "agents": {
        "scaffold": {"model": "gemini-2.5-flash",  "provider": "gemini"},
        "planner":  {"model": "glm-5.1",           "provider": "openrouter"},
        "executor": {"model": "qwen3-30b-a3b",     "provider": "openrouter"},
        "tester":   {"model": "qwen3-30b-a3b",     "provider": "openrouter"},
        "logic":    {"model": "minimax-m1",         "provider": "openrouter"},
        "judge":    {"model": "deepseek-v3",        "provider": "openrouter"},
    },
}

def load_pipeline_config() -> dict:
    """Load pipeline_config.json, falling back to defaults."""
    if PIPELINE_CONFIG.exists():
        try:
            cfg = json.loads(PIPELINE_CONFIG.read_text())
            # Merge with defaults so missing keys don't crash downstream
            merged = _DEFAULT_CONFIG.copy()
            merged["techstack"] = cfg.get("techstack", merged["techstack"])
            agents = merged["agents"].copy()
            agents.update(cfg.get("agents", {}))
            merged["agents"] = agents
            return merged
        except Exception as e:
            print(f"[harness] WARNING: could not parse pipeline_config.json: {e}")
            print("[harness]          Using built-in defaults.")
    return _DEFAULT_CONFIG.copy()


def agent_label(cfg: dict, role: str) -> str:
    """Return a display label like 'gemini-2.5-flash (scaffold)'."""
    agent = cfg["agents"].get(role, {})
    model = agent.get("model", role)
    return f"{model} ({role})"


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


def _required_env_for_provider(provider: str) -> list[str]:
    """Return the env var(s) needed for a given provider."""
    _MAP = {
        "gemini":      ["GEMINI_API_KEY"],
        "openrouter":  ["OPENROUTER_API_KEY"],
        "anthropic":   ["ANTHROPIC_API_KEY"],
        "openai":      ["OPENAI_API_KEY"],
        "mistral":     ["MISTRAL_API_KEY"],
    }
    return _MAP.get(provider, [])


def check_env_for_role(cfg: dict, role: str) -> bool:
    agent    = cfg["agents"].get(role, {})
    provider = agent.get("provider", "openrouter")
    keys     = _required_env_for_provider(provider)
    missing  = [k for k in keys if not os.environ.get(k)]
    if missing:
        print(f"[harness] Missing env vars for {role} agent ({provider}): "
              f"{', '.join(missing)}")
        print("          Set them in .env or export before running.")
        return False
    return True


def check_file_exists(path: Path, flag: str) -> bool:
    if not path.exists():
        print(f"[harness] {flag} set but {path} not found.")
        return False
    return True


def check_src_exists() -> bool:
    src_dir = ROOT / "src"
    # Check for any source-like file (ts, tsx, js, jsx, py, …)
    source_exts = {".ts", ".tsx", ".js", ".jsx", ".py", ".vue", ".svelte"}
    if not src_dir.exists() or not any(
        f.suffix in source_exts for f in src_dir.rglob("*") if f.is_file()
    ):
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
    fv = delta.get("from_version") or "(none)"
    tv = delta.get("to_version", "?")
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
    Copy unaffected src/ files from prev_src/ so the executor only implements
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


def _run_judge_fix_loop(args, cfg: dict, results: dict) -> None:
    """
    Judge → fix → re-judge loop, up to args.max_judge_rounds times.

    Round structure:
        1. Run 06_judge.py  (reads judge agent config internally)
        2. Read verdict from judge_raw.json
        3. APPROVED / APPROVED_WITH_NOTES → done ✓
        4. NEEDS_REVISION + not --skip-fix:
             a. Run 07_fix_from_judge.py
             b. Exit 0 (tests still green) → refresh report → re-judge
             c. Exit 1 (tests now failing) → stop, mark failed
        5. --skip-fix → stop after first judge
        6. max_judge_rounds exceeded → stop, report final verdict
    """
    max_rounds  = args.max_judge_rounds
    skip_fix    = args.skip_fix
    judge_label = agent_label(cfg, "judge")

    for round_num in range(1, max_rounds + 1):
        round_sfx = f" (round {round_num}/{max_rounds})" if max_rounds > 1 else ""

        ok = run_step(
            f"Step 6 — Judge [{judge_label}]{round_sfx}",
            "06_judge.py",
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
            print("[harness] Run manually: python pipeline/07_update_knowledge.py")
            break

        if skip_fix:
            skip_step(
                f"Step 7 — Fix from judge{round_sfx}",
                "--skip-fix set — review judge_report.md manually",
            )
            break

        if verdict == "NEEDS_REVISION":
            fix_ok = run_step(
                f"Step 7 — Fix from judge{round_sfx}",
                "07_fix_from_judge.py",
            )
            results[f"judge_fix_r{round_num}"] = fix_ok

            if not fix_ok:
                print("\n[harness] ⚠ Judge fix failed (tests still failing after patches).")
                print("[harness] Human review required — see reports/judge_fix_report.json")
                break

            print("\n[harness] Fix applied — refreshing report + re-judging …")
            run_step("Step 5b — Aggregate report (post-fix)", "05_report.py")
            continue

        print("[harness] Judge step failed (non-verdict error) — stopping.")
        break


def _run_fix_from_existing_judge(args, cfg: dict, results: dict) -> None:
    """
    Feed an existing judge_raw.json into the fix loop without calling the API.
    Used when --from-judge is set.
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

    fix_ok = run_step(
        "Step 7 — Fix from judge (existing review)",
        "07_fix_from_judge.py",
    )
    results["judge_fix"] = fix_ok

    if not fix_ok:
        print("\n[harness] ⚠ Fix failed (tests still failing after patches).")
        print("[harness] Human review required — see reports/judge_fix_report.json")
        return

    print("\n[harness] Fix applied — refreshing report + re-judging …")
    run_step("Step 5b — Aggregate report (post-fix)", "05_report.py")

    if not check_env_for_role(cfg, "judge"):
        print("[harness] WARNING: cannot re-judge — missing API key for judge agent.")
        return

    judge_label = agent_label(cfg, "judge")
    ok = run_step(f"Step 6 — Judge [{judge_label}] (post-fix)", "06_judge.py")
    results["judge_post_fix"] = ok
    final = _read_judge_verdict()
    if final:
        print(f"\n[harness] Post-fix verdict: {final}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()
    cfg = load_pipeline_config()

    parser = argparse.ArgumentParser(
        description="Generic LLM pipeline runner (role-based agents, configurable techstack)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Delta-aware flags
    parser.add_argument("--force", action="store_true",
                        help="Ignore spec_delta.json — re-run all steps unconditionally")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing anything")
    # Generation override flags
    parser.add_argument("--skip-scaffold", action="store_true",
                        help="Skip scaffold_agent step (overrides delta)")
    parser.add_argument("--skip-plan", action="store_true",
                        help="Skip planner_agent step (overrides delta)")
    parser.add_argument("--retry-impl", action="store_true",
                        help="Retry only failed files from last executor run "
                             "(reads scaffold/impl_executor.json failed_files). "
                             "Implies --skip-scaffold --skip-plan.")
    parser.add_argument("--only-executor", action="store_true",
                        help="Skip planner_agent entirely; executor runs in single-call mode")
    parser.add_argument("--test-only", action="store_true",
                        help="Skip scaffold + plan + implement; jump straight to tests")
    # Judge flags
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judge_agent entirely (overrides delta)")
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

    # Flag normalisation
    if args.test_only:
        args.skip_scaffold = True
        args.skip_plan     = True

    if args.retry_impl:
        args.skip_scaffold = True
        args.skip_plan     = True

    if args.from_judge:
        args.skip_scaffold = True
        args.skip_plan     = True
        args.test_only     = True
        args.skip_judge    = True  # suppress direct judge call; handled by fix flow

    results: dict[str, bool] = {}

    # Print active config summary
    print(f"\n[harness] Techstack : {cfg['techstack']}")
    for role, agent in cfg["agents"].items():
        print(f"[harness] {role:<12}: {agent.get('model', '?')}  "
              f"({agent.get('provider', '?')})")

    # ── Step 1: Spec diff ────────────────────────────────────────────────────
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
            delta = None

    # ── Auto-skip from delta (manual flags take priority) ────────────────────
    if delta and not delta.get("is_first_run"):
        if not delta_requires(delta, "scaffold") and not args.skip_scaffold:
            args.skip_scaffold = True
            print("[harness] delta: §7/§8 unchanged → scaffold skipped")
        if (not delta_requires(delta, "plan")
                and not args.skip_plan and not args.only_executor):
            args.skip_plan = True
            print("[harness] delta: no affected files → plan skipped")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[harness] DRY RUN — nothing executed.")
        plan_str = ("skip (only-executor)" if args.only_executor
                    else "skip" if args.skip_plan else "run")
        steps = [
            ("spec diff",      "skip" if args.test_only else "run"),
            ("scaffold",       f"skip" if args.skip_scaffold else f"run [{agent_label(cfg, 'scaffold')}]"),
            ("plan",           f"skip" if args.skip_plan else f"run [{agent_label(cfg, 'planner')}]  {plan_str}"),
            ("implement",      f"skip" if args.test_only else f"run [{agent_label(cfg, 'executor')}]"),
            ("test",           f"run [{agent_label(cfg, 'tester')}] / [{agent_label(cfg, 'logic')}]"),
            ("report",         "run"),
            ("judge+fix loop", f"skip" if args.skip_judge
                               else f"run [{agent_label(cfg, 'judge')}] (max {args.max_judge_rounds} rounds)"),
        ]
        for name, action in steps:
            icon = "▶" if "run" in action else "⏭"
            print(f"  {icon}  {name:20}  {action}")
        return

    # ── Step 2: Scaffold ─────────────────────────────────────────────────────
    if not args.skip_scaffold:
        if not check_env_for_role(cfg, "scaffold"):
            sys.exit(1)
        scaffold_label = agent_label(cfg, "scaffold")
        ok = run_step(f"Step 2 — Scaffold [{scaffold_label}]", "02_scaffold.py")
        results["scaffold"] = ok
        if not ok:
            print("\n[harness] Scaffold failed. Stopping.")
            sys.exit(1)
    else:
        if not check_file_exists(ROOT / "scaffold" / "scaffold.json", "--skip-scaffold"):
            sys.exit(1)
        skip_step("Step 2 — Scaffold", "reusing scaffold/scaffold.json")

    # ── Step 3b: Planner ─────────────────────────────────────────────────────
    plan_json_path = ROOT / "scaffold" / "plan.json"
    plan_available = False

    if args.only_executor:
        skip_step("Step 3b — Planner", "--only-executor")
        plan_available = False

    elif args.test_only:
        plan_available = plan_json_path.exists()
        reason = ("reusing existing plan.json" if plan_available
                  else "no plan.json found")
        skip_step("Step 3b — Planner", f"--test-only ({reason})")

    elif args.skip_plan:
        if not check_file_exists(plan_json_path, "--skip-plan"):
            print("          Tip: run without --skip-plan to regenerate, "
                  "or use --only-executor to skip planning entirely.")
            sys.exit(1)
        skip_step("Step 3b — Planner", "reusing scaffold/plan.json")
        plan_available = True

    else:
        if not check_env_for_role(cfg, "planner"):
            sys.exit(1)
        planner_label = agent_label(cfg, "planner")
        ok = run_step(f"Step 3b — Planner [{planner_label}]", "03b_plan.py")
        results["plan"] = ok
        if not ok:
            print("\n[harness] Planning failed.")
            print("          Tip: --skip-plan to reuse last plan, "
                  "or --only-executor to skip planning entirely.")
            sys.exit(1)
        plan_available = True

    # ── Step 3a: Executor implement ──────────────────────────────────────────
    if not args.test_only:
        if not check_env_for_role(cfg, "executor"):
            sys.exit(1)

        # Delta: restore unaffected files before executor runs
        if delta and not delta.get("is_first_run") and not args.retry_impl:
            restore_unaffected_files(delta)

        executor_args: list[str] = []
        if plan_available:
            executor_args.append("--use-plan")

        # --retry-impl: re-run only previously failed files
        impl_rec_path = ROOT / "scaffold" / "impl_executor.json"
        if args.retry_impl:
            if impl_rec_path.exists():
                try:
                    rec    = json.loads(impl_rec_path.read_text())
                    failed = rec.get("failed_files", [])
                    if failed:
                        executor_args += ["--only-files", ",".join(failed)]
                        print(f"[harness] --retry-impl: retrying {len(failed)} failed file(s).")
                    else:
                        print("[harness] --retry-impl: no failed_files in impl_executor.json — nothing to retry.")
                        sys.exit(0)
                except Exception:
                    print("[harness] --retry-impl: could not read impl_executor.json.")
                    sys.exit(1)
            else:
                print("[harness] --retry-impl: impl_executor.json not found — run full impl first.")
                sys.exit(1)

        # Delta --only-files (skipped when --retry-impl already set --only-files)
        elif delta and not delta.get("is_first_run"):
            src_affected = [f for f in delta.get("affected_files", [])
                            if f.startswith("src/")]
            if src_affected:
                executor_args += ["--only-files", ",".join(src_affected)]
                print(f"[harness] Executor will implement {len(src_affected)} "
                      f"affected file(s) only.")

        executor_label = agent_label(cfg, "executor")
        mode_label = "per-file + plan" if plan_available else "single-call"
        if args.retry_impl:
            mode_label += " | retry-impl"
        elif delta and not delta.get("is_first_run"):
            n = len([f for f in delta.get("affected_files", []) if f.startswith("src/")])
            mode_label += f" | {n} affected"

        ok = run_step(
            f"Step 3a — Executor [{executor_label}] ({mode_label})",
            "03a_implement.py",
            executor_args,
        )
        results["impl"] = ok

        if not ok and impl_rec_path.exists():
            try:
                rec    = json.loads(impl_rec_path.read_text())
                failed = rec.get("failed_files", [])
                if failed:
                    print(f"\n[harness] {len(failed)} file(s) failed to implement:")
                    for fp in failed:
                        print(f"    {fp}")
                    print("\n[harness] Retry: python harness.py --retry-impl")
            except Exception:
                pass

        if ok:
            snapshot_src()

    else:
        if not check_src_exists():
            sys.exit(1)
        skip_step("Step 3a — Executor implement", "reusing existing src/")
        results["impl"] = True

    # ── Step 4+5: Test + iterate ─────────────────────────────────────────────
    test_args = [
        "--max-iter",             str(args.max_iter),
        "--max-cluster-attempts", str(args.max_cluster_attempts),
    ]
    if args.verbose:
        test_args.append("--verbose")

    tester_label = agent_label(cfg, "tester")
    logic_label  = agent_label(cfg, "logic")
    ok = run_step(
        f"Step 4+5 — Test loop [{tester_label}] / [{logic_label}]",
        "04_test_and_iterate.py",
        test_args,
    )
    results["test"] = ok
    tests_passed = ok

    # ── Step 5b: Aggregate report ────────────────────────────────────────────
    run_step("Step 5b — Aggregate report", "05_report.py")

    # ── Step 6 + 7: Judge → fix → re-judge loop ─────────────────────────────
    judge_label = agent_label(cfg, "judge")

    if args.skip_judge and not args.from_judge:
        skip_step(f"Step 6 — Judge [{judge_label}]", "--skip-judge")

    elif not tests_passed:
        skip_step(
            f"Step 6 — Judge [{judge_label}]",
            "tests failed — fix tests first before requesting judge sign-off",
        )

    elif args.from_judge:
        skip_step(f"Step 6 — Judge [{judge_label}]",
                  "--from-judge (reusing reports/judge_raw.json)")
        _run_fix_from_existing_judge(args, cfg, results)

    else:
        if not check_env_for_role(cfg, "judge"):
            print("[harness] WARNING: cannot run judge without API key.")
        else:
            _run_judge_fix_loop(args, cfg, results)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Techstack: {cfg['techstack']}")
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

    # ── Persist apply record ─────────────────────────────────────────────────
    if all_ok and delta:
        try:
            from pipeline.spec_diff import write_applied
        except ImportError:
            import importlib.util
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
