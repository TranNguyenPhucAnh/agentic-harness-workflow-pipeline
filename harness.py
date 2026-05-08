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
    plan        03b_implement_glm.py   — decompose work into ordered task plan
    implement   03a_implement_qwen.py  — implement src/ files guided by plan
    test        04_test_and_iterate.py — vitest loop with Qwen+Minimax repair
    report      05_report.py           — aggregate summary.md
    judge       06_judge_deepseek.py   — qualitative review + sign-off
    fix         07_fix_from_judge.py   — auto-patch NEEDS_REVISION findings
                └─ re-runs report + judge after each fix, up to --max-judge-rounds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE MODES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --scope full           Spec-driven full pipeline.
                         Uses spec_diff, scaffold.json, plan.json.

  --scope mini           Targeted daily-driver pipeline.
                         Skips scaffold/spec_diff and routes planner/implementer
                         through mini contracts:
                             state/plan_mini.json
                             run/analysis_mini.json
                         Intended for small changes where spec.md does not need
                         to change.

  Legacy --mini PROMPT   Backward-compatible entrypoint delegated to
                         pipeline/mini_mode.py. Prefer:
                             python harness.py --scope mini ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generation flags:

  --project NAME         Project name → artifacts_<slug>/.
                         If omitted, interactive prompt lists existing projects.

  --scope full|mini      Pipeline scope. Default: full.

  --auto-continue        Non-interactive mode. Reserved for interactive state
                         machine checkpoints; currently means no extra prompts
                         from harness itself.

  --force                Re-run ALL full-scope generation steps even if
                         spec_delta says nothing changed.

  --dry-run              Print what WOULD run without executing anything.

  --only-qwen            Skip GLM plan step; Qwen runs in single-call mode.

  --retry-impl           Re-implement only files listed as failed in
                         run/impl_record.json.

Range flags:

  --from-STEP            Start from STEP.
  --until-STEP           Stop after STEP.
  --STEP                 Run only STEP.

Steps:
  absorb, clarify, scaffold, plan, implement, test, report, judge, fix

Test loop flags:

  --max-iter N
  --max-cluster-attempts N
  --verbose

Judge flags:

  --skip-fix
  --from-judge
  --max-judge-rounds N

Clarification:

  --clarify-input FILE

Legacy mini mode:

  --mini PROMPT
  --files f1 f2 ...
  --context-file FILE
  --output-file FILE
  --task-type TYPE

Requirements:
    pip install httpx
    GEMINI_API_KEY, OPENROUTER_API_KEY in .env or exported as env vars
"""

from __future__ import annotations

import argparse
import json
import os
import re as _re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# harness.py lives at ROOT level — same level as artifacts/
ROOT = Path(__file__).parent

# === WRITE AUTHORITY: harness ===
# OWNS  : orchestration only; optional legacy mini delegated to mini_mode.py
# READS : all artifacts
# NOTE  : step scripts own their respective artifact writes.

sys.path.insert(0, str(ROOT))

# LazyPath constants — resolve to the correct project artifact root at use time.
# ensure_dirs() is called inside main() after --project is parsed and
# PIPELINE_PROJECT env var is set. Do NOT call ensure_dirs() here.
from artifacts.paths import (  # noqa: E402
    SPEC_DELTA as DELTA_PATH,
    PLAN_JSON as GLM_PLAN_PATH,
    PLAN_MINI as MINI_PLAN_PATH,
    IMPL_RECORD as IMPL_RECORD_PATH,
    UPDATE_LOG as UPDATE_LOG_PATH,
    JUDGE_RAW as JUDGE_RAW_PATH,
    CLARIFICATION_REPORT,
)


# ════════════════════════════════════════════════════════════════════════════
# Step registry — single source of truth for pipeline order
# ════════════════════════════════════════════════════════════════════════════

STEPS = [
    "absorb",
    "clarify",
    "scaffold",
    "plan",
    "implement",
    "test",
    "report",
    "judge",
    "fix",
]

STEP_SCRIPTS: dict[str, str] = {
    "absorb": "01_absorber.py",
    "clarify": "00_clarificator.py",
    "scaffold": "02_scaffold_gemini.py",
    "plan": "03b_implement_glm.py",
    "implement": "03a_implement_qwen.py",
    "test": "04_test_and_iterate.py",
    "report": "05_report.py",
    "judge": "06_judge_deepseek.py",
    "fix": "07_fix_from_judge.py",
}

STEP_ENV_KEYS: dict[str, list[str]] = {
    "scaffold": ["GEMINI_API_KEY"],
    "plan": ["OPENROUTER_API_KEY"],
    "implement": ["OPENROUTER_API_KEY"],
    "test": ["OPENROUTER_API_KEY"],
    "judge": ["OPENROUTER_API_KEY"],
    "fix": ["OPENROUTER_API_KEY"],
}

SCOPE_CHOICES = ("full", "mini")


# NOTE: prev_src is a transient staging dir, not a pipeline artifact.
# Resolved lazily so it uses the correct project artifact root.
def _prev_src_dir() -> Path:
    from artifacts.paths import artifact_root

    return artifact_root() / "state" / "prev_src"


# ════════════════════════════════════════════════════════════════════════════
# Core helpers
# ════════════════════════════════════════════════════════════════════════════

def _die(msg: str) -> None:
    print(f"[harness] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def run_step(label: str, script: str, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(ROOT / "pipeline" / script)] + (extra_args or [])

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
    print(f"  {status}  ({elapsed:.1f}s)")
    return result.returncode == 0


def skip_step(label: str, reason: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}  [SKIPPED — {reason}]")
    print(f"{'=' * 60}")


def check_env(keys: list[str]) -> bool:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        print(f"[harness] Missing env vars: {', '.join(missing)}")
        print("          Set them in .env or export them before running.")
        return False
    return True


def _artifact_rel(path: Any) -> str:
    try:
        from artifacts.paths import artifact_root

        return str(Path(path).relative_to(artifact_root()))
    except Exception:
        return str(path)


# ════════════════════════════════════════════════════════════════════════════
# Project selection
# ════════════════════════════════════════════════════════════════════════════

def _interactive_project_select(root: Path) -> str:
    """
    List existing artifacts_* projects and let the user pick one or create new.
    Falls back to a plain name prompt if no projects exist yet.
    """
    projects = sorted(
        p.name.removeprefix("artifacts_")
        for p in root.glob("artifacts_*")
        if p.is_dir()
    )

    if projects:
        print("\nExisting projects:")
        for i, project in enumerate(projects, 1):
            print(f"  {i}. {project}")
        print(f"  {len(projects) + 1}. New project")
        print()

        choice = input("Select project (number or name): ").strip()

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(projects):
                return projects[idx - 1]
            if idx == len(projects) + 1:
                name = input("New project name: ").strip()
                if name:
                    return name

        if choice and not choice.isdigit():
            return choice

    name = input("Project name: ").strip()
    if not name:
        _die("Project name is required.")
    return name


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
    from_version = delta.get("from_version") or "(none)"
    to_version = delta.get("to_version", "?")

    print(f"\n[harness] Spec: {from_version} → {to_version}")

    if delta.get("is_first_run"):
        print("[harness] First run — full pipeline.")
        return

    changed = delta.get("changed_sections", [])
    affected = delta.get("affected_files", [])
    rerun = [key for key, value in delta.get("rerun_steps", {}).items() if value]
    skip = [key for key, value in delta.get("rerun_steps", {}).items() if not value]
    summaries = delta.get("section_summaries", {})

    if changed:
        print(f"[harness] Changed §: {changed}")
        for section in changed:
            if section in summaries:
                print(f"    §{section}: {summaries[section]}")

    print(f"[harness] Affected files  : {len(affected)}")
    print(f"[harness] Steps to re-run : {rerun or '(none)'}")
    print(f"[harness] Steps to skip   : {skip or '(none)'}")


# ════════════════════════════════════════════════════════════════════════════
# src/ snapshot + restore
# ════════════════════════════════════════════════════════════════════════════

def snapshot_src() -> None:
    """Save current repo src/ as project-scoped prev_src/ for delta restores."""
    src = ROOT / "src"
    if not src.exists():
        return

    prev_src = _prev_src_dir()
    if prev_src.exists():
        shutil.rmtree(prev_src)

    shutil.copytree(src, prev_src)
    print(f"[harness] src/ snapshot → {_artifact_rel(prev_src)}")


def restore_unaffected_files(delta: dict) -> int:
    """
    Copy unaffected src/ files from prev_src/ so Qwen only implements
    the files that changed. Returns number of files restored.
    """
    unaffected = [
        file
        for file in delta.get("unaffected_files", [])
        if isinstance(file, str) and file.startswith("src/")
    ]

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
    """Read current judge verdict from run/judge_raw.json. Returns '' if unavailable."""
    raw_path = JUDGE_RAW_PATH
    if not raw_path.exists():
        return ""

    try:
        raw_data = json.loads(raw_path.read_text())
        response = raw_data.get("response", "")
        response = _re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", response.strip())
        response = _re.sub(r"\n?```$", "", response.strip())
        return json.loads(response).get("verdict", "")
    except Exception:
        return ""


def _scope_args_for_script(script: str, scope: str) -> list[str]:
    """
    Return scope args for scripts that support scope.

    Current contract:
      - 03b_implement_glm.py supports --scope full|mini
      - 03a_implement_qwen.py supports --scope full|mini

    Other steps remain backward-compatible and receive no --scope unless they
    are refactored explicitly.
    """
    if script in {"03b_implement_glm.py", "03a_implement_qwen.py"}:
        return ["--scope", scope]
    return []


def _run_judge_fix_loop(args: argparse.Namespace, results: dict[str, bool]) -> None:
    """
    Judge → fix → re-judge loop, up to args.max_judge_rounds times.
    """
    max_rounds = args.max_judge_rounds
    skip_fix = args.skip_fix

    for round_num in range(1, max_rounds + 1):
        round_sfx = f" (round {round_num}/{max_rounds})" if max_rounds > 1 else ""

        ok = run_step(
            f"Step 6 — DeepSeek V3.2 judge{round_sfx}",
            STEP_SCRIPTS["judge"],
        )
        results[f"judge_r{round_num}"] = ok
        results["judge"] = ok

        verdict = _read_judge_verdict()
        print(f"\n[harness] Judge verdict: {verdict or '(unknown)'}")

        if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
            print(f"[harness] ✅ Judge {verdict} — pipeline complete.")
            results["judge"] = True
            break

        if round_num == max_rounds:
            print(
                f"[harness] ⚠ Reached max_judge_rounds ({max_rounds}) "
                f"with verdict {verdict or '(unknown)'}."
            )
            print("[harness] Run manually when ready: python pipeline/07_update_knowledge.py")
            break

        if skip_fix:
            skip_step(
                f"Step 7 — Fix from judge{round_sfx}",
                "--skip-fix set — review judge_report.md manually",
            )
            break

        if verdict == "NEEDS_REVISION":
            fix_args: list[str] = []
            if args.verbose:
                fix_args.append("--verbose")
            if getattr(args, "fix_non_blocking", False):
                fix_args.append("--fix-non-blocking")

            fix_ok = run_step(
                f"Step 7 — Fix from judge{round_sfx}",
                STEP_SCRIPTS["fix"],
                fix_args or None,
            )
            results[f"judge_fix_r{round_num}"] = fix_ok

            if not fix_ok:
                print("\n[harness] ⚠ Judge fix failed; human review required.")
                print(f"[harness] See {UPDATE_LOG_PATH}")
                break

            print("\n[harness] Fix applied successfully — re-running report + judge …")
            report_ok = run_step("Step 5b — Aggregate report (post-fix)", STEP_SCRIPTS["report"])
            results[f"report_post_fix_r{round_num}"] = report_ok
            if not report_ok:
                results["judge"] = False
                break
            continue

        print("[harness] Judge step failed or returned non-actionable verdict — stopping.")
        break


def _run_fix_from_existing_judge(args: argparse.Namespace, results: dict[str, bool]) -> None:
    """
    Feed an existing run/judge_raw.json into the fix loop without calling the
    judge API first. Used when --from-judge is set.
    """
    raw_path = JUDGE_RAW_PATH
    if not raw_path.exists():
        print("[harness] --from-judge: run/judge_raw.json not found.")
        print("          Run the full pipeline first to generate a judge report.")
        results["fix_from_judge"] = False
        return

    verdict = _read_judge_verdict()
    print(f"\n[harness] Existing judge verdict: {verdict or '(unknown)'}")

    if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
        print(f"[harness] ✅ Already {verdict} — nothing to fix.")
        results["fix_from_judge"] = True
        return

    if verdict != "NEEDS_REVISION":
        print(f"[harness] ⚠ Unrecognised verdict '{verdict}' — stopping.")
        results["fix_from_judge"] = False
        return

    if args.skip_fix:
        skip_step(
            "Step 7 — Fix from judge (existing review)",
            "--skip-fix set — review judge_report.md manually",
        )
        results["fix_from_judge"] = True
        return

    fix_args: list[str] = []
    if args.verbose:
        fix_args.append("--verbose")
    if getattr(args, "fix_non_blocking", False):
        fix_args.append("--fix-non-blocking")

    fix_ok = run_step(
        "Step 7 — Fix from judge (existing review)",
        STEP_SCRIPTS["fix"],
        fix_args or None,
    )
    results["fix_from_judge"] = fix_ok

    if not fix_ok:
        print("\n[harness] ⚠ Fix failed; human review required.")
        print(f"[harness] See {UPDATE_LOG_PATH}")
        return

    print("\n[harness] Fix applied — refreshing report + re-judging …")
    report_ok = run_step("Step 5b — Aggregate report (post-fix)", STEP_SCRIPTS["report"])
    results["report_post_fix"] = report_ok

    if not report_ok:
        results["fix_from_judge"] = False
        return

    if not check_env(["OPENROUTER_API_KEY"]):
        print("[harness] WARNING: cannot re-judge without OPENROUTER_API_KEY.")
        return

    judge_ok = run_step("Step 6 — DeepSeek V3.2 judge (post-fix)", STEP_SCRIPTS["judge"])
    results["judge_post_fix"] = judge_ok

    final = _read_judge_verdict()
    if final:
        print(f"\n[harness] Post-fix verdict: {final}")


# ════════════════════════════════════════════════════════════════════════════
# Range resolution + dry-run
# ════════════════════════════════════════════════════════════════════════════

def _validate_scope(scope: str) -> None:
    if scope not in SCOPE_CHOICES:
        _die(f"Invalid --scope {scope!r}. Expected one of: {', '.join(SCOPE_CHOICES)}")


def _resolve_run_range(args: argparse.Namespace) -> tuple[str, str]:
    """
    Return (from_step, until_step) after validating all flag combinations.

    Rules:
        --from-X                    → X .. fix
        --from-X --until-Y          → X .. Y
        --X shorthand               → X .. X
        no flags                    → absorb .. fix

    Mini scope keeps the same canonical order but auto-skips scaffold at
    execution time.
    """
    from_step = getattr(args, "from_step", None)
    until_step = getattr(args, "until_step", None)

    shorthands = [step for step in STEPS if getattr(args, step, False)]

    if shorthands and (from_step or until_step):
        _die(f"Cannot mix --{shorthands[0]} shorthand with --from/--until flags.")

    if len(shorthands) > 1:
        _die(
            "Only one step shorthand allowed at a time, got: "
            + " ".join(f"--{step}" for step in shorthands)
        )

    if shorthands:
        return shorthands[0], shorthands[0]

    if until_step and not from_step:
        _die(f"--until-{until_step} requires --from-<step>.")

    if not from_step and not until_step:
        return STEPS[0], STEPS[-1]

    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step) if until_step else len(STEPS) - 1
    until_step = until_step or STEPS[-1]

    if from_idx > until_idx:
        order = " → ".join(STEPS)
        _die(f"--from-{from_step} comes after --until-{until_step}.\n  Order: {order}")

    return from_step, until_step


def _print_dry_run(from_step: str, until_step: str, args: argparse.Namespace) -> None:
    """Print what would run without executing anything."""
    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)

    print("\n[harness] DRY RUN — nothing will be executed.")
    print(f"  Project : {os.environ.get('PIPELINE_PROJECT', '?')}")
    print(f"  Scope   : {args.scope}")
    print(f"  Range   : {from_step} → {until_step}")
    print()

    for i, step in enumerate(STEPS):
        in_range = from_idx <= i <= until_idx
        if not in_range:
            print(f"  ⏭  {step:<12}  (skipped)")
            continue

        if args.scope == "mini" and step == "scaffold":
            print(f"  ⏭  {step:<12}  (skipped: mini scope)")
            continue

        script = STEP_SCRIPTS[step]
        extra = _scope_args_for_script(script, args.scope)
        suffix = f" {' '.join(extra)}" if extra else ""
        print(f"  ▶  {step:<12}  {script}{suffix}")

    print()

    if args.scope == "full" and args.force:
        print("  --force: delta checks will be bypassed — all full-scope steps re-run.")

    if args.scope == "mini":
        print("  mini scope: spec_diff/scaffold are skipped; planner writes plan_mini.json.")

    print()


# ════════════════════════════════════════════════════════════════════════════
# Step dispatch
# ════════════════════════════════════════════════════════════════════════════

def _run_step(
    step: str,
    args: argparse.Namespace,
    delta: dict | None,
    plan_available: bool,
    results: dict[str, bool],
    tests_passed: bool,
) -> bool:
    """Dispatch a single step with its extra args. Returns success bool."""

    if step == "absorb":
        return run_step("absorb", STEP_SCRIPTS["absorb"])

    if step == "clarify":
        clarify_args: list[str] = []
        if getattr(args, "clarify_input", None):
            clarify_args += ["--input", args.clarify_input]

        ok = run_step("clarify", STEP_SCRIPTS["clarify"], clarify_args or None)
        if not ok:
            print("\n[harness] Clarify failed.")
        return ok

    if step == "scaffold":
        if args.scope == "mini":
            skip_step("scaffold", "mini scope uses targeted planner; no scaffold needed")
            return True

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

        if args.scope == "full":
            if delta and not delta.get("is_first_run") and not delta_requires(delta, "plan"):
                skip_step("plan", "delta: no affected files — reusing plan.json")
                return True

        if not check_env(STEP_ENV_KEYS.get("plan", [])):
            sys.exit(1)

        plan_args = _scope_args_for_script(STEP_SCRIPTS["plan"], args.scope)
        ok = run_step("plan", STEP_SCRIPTS["plan"], plan_args)

        if not ok:
            print(
                "\n[harness] Plan failed.\n"
                "  Tip: --only-qwen to skip planning entirely."
            )

        return ok

    if step == "implement":
        if not check_env(STEP_ENV_KEYS.get("implement", [])):
            sys.exit(1)

        qwen_args = _scope_args_for_script(STEP_SCRIPTS["implement"], args.scope)

        if plan_available:
            qwen_args.append("--use-glm-plan")

        if args.retry_impl:
            retry_args = _retry_impl_args(qwen_args)
            if retry_args is None:
                return False
            qwen_args = retry_args

        elif args.scope == "full" and delta and not delta.get("is_first_run"):
            restore_unaffected_files(delta)
            src_affected = [
                file
                for file in delta.get("affected_files", [])
                if isinstance(file, str) and file.startswith("src/")
            ]
            if src_affected:
                qwen_args += ["--only-files", ",".join(src_affected)]
                print(f"[harness] implement: {len(src_affected)} affected file(s) only.")

        if args.scope == "mini":
            mode = "mini-targeted+plan" if plan_available else "mini-targeted"
        else:
            mode = "per-file+plan" if plan_available else "single-call"

        ok = run_step(f"implement ({mode})", STEP_SCRIPTS["implement"], qwen_args)

        if not ok:
            _print_impl_failed()

        return ok

    if step == "test":
        test_args = [
            "--impl",
            "qwen",
            "--max-iter",
            str(args.max_iter),
            "--max-cluster-attempts",
            str(args.max_cluster_attempts),
        ]
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
        if "judge" in results:
            skip_step("fix", "already handled by judge loop")
            return True

        _run_fix_from_existing_judge(args, results)
        return results.get("fix_from_judge", False)

    return True


def _retry_impl_args(qwen_args: list[str]) -> list[str] | None:
    """Build --only-files args from impl_record failed_files. Returns None on error."""
    if not IMPL_RECORD_PATH.exists():
        print("[harness] --retry-impl: impl_record.json not found — run full impl first.")
        return None

    try:
        record = json.loads(IMPL_RECORD_PATH.read_text())
        failed = record.get("failed_files", [])

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
        record = json.loads(IMPL_RECORD_PATH.read_text())
        failed = record.get("failed_files", [])

        if failed:
            print(f"\n[harness] {len(failed)} file(s) failed to implement:")
            for file in failed:
                print(f"    {file}")
            print("\n[harness] Retry: python harness.py --retry-impl")

    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Summary + apply record
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(
    results: dict[str, bool],
    delta: dict | None,
    args: argparse.Namespace,
    tests_passed: bool,
) -> None:
    from artifacts.paths import artifact_root

    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")

    print(f"  Scope: {args.scope}")

    if delta and not delta.get("is_first_run"):
        from_version = delta.get("from_version") or "?"
        to_version = delta.get("to_version", "?")
        affected_count = len(delta.get("affected_files", []))
        print(f"  Spec : {from_version} → {to_version}  ({affected_count} file(s) affected)")

    for key, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {key}")

    all_ok = all(results.values())
    print(f"\n  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")

    print("\n  Reports:")
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


def _write_apply_record(delta: dict, results: dict[str, bool]) -> None:
    try:
        from pipeline.spec_diff import write_applied  # type: ignore
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "spec_diff",
            ROOT / "pipeline" / "spec_diff.py",
        )
        if spec is None or spec.loader is None:
            print("[harness] WARNING: could not import spec_diff.write_applied")
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        write_applied = module.write_applied

    applied_steps = [key for key, value in results.items() if value]
    write_applied(
        version=delta.get("to_version", "unknown"),
        steps=applied_steps,
        status="PASS",
    )

    from artifacts.paths import SPEC_APPLIED

    print(
        f"\n  Apply record → {SPEC_APPLIED}  "
        f"(v{delta.get('to_version', '?')} marked as applied)"
    )


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="Pipeline orchestrator. Run full end-to-end or a sub-range of steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py
  python harness.py --scope full --from-clarify
  python harness.py --scope full --from-implement --until-test
  python harness.py --scope mini --from-clarify --until-implement
  python harness.py --scope mini --auto-continue
  python harness.py --scaffold
  python harness.py --dry-run
  python harness.py --from-test --dry-run
""",
    )

    # ── Project ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Project name → artifacts_<slug>/. "
            "If omitted, interactive prompt lists existing projects."
        ),
    )

    # ── Scope ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--scope",
        choices=list(SCOPE_CHOICES),
        default="full",
        help=(
            "Pipeline scope: 'full' runs spec-driven flow; "
            "'mini' runs targeted planner/implementer flow."
        ),
    )

    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Non-interactive: skip confirmation prompts controlled by harness.",
    )

    # ── Flow control: range ───────────────────────────────────────────────────
    for step in STEPS:
        parser.add_argument(
            f"--from-{step}",
            dest="from_step",
            action="store_const",
            const=step,
            help=f"Start pipeline from step '{step}'.",
        )
        parser.add_argument(
            f"--until-{step}",
            dest="until_step",
            action="store_const",
            const=step,
            help=f"Stop pipeline after step '{step}'.",
        )

    # ── Flow control: shorthand ───────────────────────────────────────────────
    for step in STEPS:
        parser.add_argument(
            f"--{step}",
            dest=step,
            action="store_true",
            help=(
                f"Run only step '{step}' "
                f"(shorthand for --from-{step} --until-{step})."
            ),
        )

    # ── Behaviour modifiers ──────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps that would run without executing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all full-scope steps even if spec_delta says nothing changed.",
    )
    parser.add_argument(
        "--only-qwen",
        action="store_true",
        help="Skip GLM plan step; Qwen runs in single-call mode.",
    )
    parser.add_argument(
        "--retry-impl",
        action="store_true",
        help="Re-implement only files listed as failed in impl_record.json.",
    )
    parser.add_argument(
        "--max-judge-rounds",
        type=int,
        default=2,
        metavar="N",
        help="Max judge→fix→re-judge iterations (default: 2).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        metavar="N",
        help="Max vitest→repair outer loops (default: 3).",
    )
    parser.add_argument(
        "--max-cluster-attempts",
        type=int,
        default=2,
        metavar="N",
        help="Max LLM repair calls per failing cluster (default: 2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cluster debug output during test step.",
    )
    parser.add_argument(
        "--skip-fix",
        action="store_true",
        help="Run judge but skip auto-fix step.",
    )
    parser.add_argument(
        "--from-judge",
        action="store_true",
        help=(
            "Skip normal flow and consume existing run/judge_raw.json, "
            "then run fix and one post-fix judge."
        ),
    )
    parser.add_argument(
        "--clarify-input",
        type=str,
        default=None,
        metavar="FILE",
        help="Pass file path as input to clarify step.",
    )

    # ── Legacy mini mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--mini",
        type=str,
        default=None,
        metavar="PROMPT",
        help=(
            "Legacy mini mode: targeted task delegated to pipeline/mini_mode.py. "
            "Prefer --scope mini."
        ),
    )
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--context-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None)

    return parser


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    _validate_scope(args.scope)

    # ── Project resolution ────────────────────────────────────────────────────
    if not args.project:
        args.project = _interactive_project_select(ROOT)

    os.environ["PIPELINE_PROJECT"] = args.project

    from artifacts.paths import ensure_dirs, artifact_root

    ensure_dirs()

    print(f"[harness] Project  : {args.project}")
    print(f"[harness] Workspace: {artifact_root()}")
    print(f"[harness] Scope    : {args.scope}")

    # ── Legacy mini mode ──────────────────────────────────────────────────────
    if args.mini is not None:
        print(
            "[harness] WARNING: --mini is legacy and delegates to pipeline/mini_mode.py.\n"
            "          Prefer: python harness.py --scope mini ..."
        )

        from pipeline.mini_mode import run_mini  # type: ignore

        run_mini(
            prompt=args.mini,
            files=args.files,
            context_file=Path(args.context_file) if args.context_file else None,
            output_file=Path(args.output_file) if args.output_file else None,
            task_type_override=args.task_type,
            dry_run=args.dry_run,
        )
        return

    # ── --from-judge special flow ─────────────────────────────────────────────
    if args.from_judge:
        results: dict[str, bool] = {}
        if args.dry_run:
            print("\n[harness] DRY RUN — would consume existing run/judge_raw.json and run fix.")
            return

        _run_fix_from_existing_judge(args, results)
        _print_summary(results, delta=None, args=args, tests_passed=True)
        sys.exit(0 if all(results.values()) else 1)

    # ── Resolve step range ────────────────────────────────────────────────────
    from_step, until_step = _resolve_run_range(args)

    if args.dry_run:
        _print_dry_run(from_step, until_step, args)
        return

    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)
    steps_to_run = STEPS[from_idx:until_idx + 1]

    print(f"[harness] Steps    : {from_step} → {until_step}")
    print()

    results: dict[str, bool] = {}

    # ── spec_diff: full scope only ────────────────────────────────────────────
    delta: dict | None = None
    needs_diff = (
        args.scope == "full"
        and any(step in steps_to_run for step in ("scaffold", "plan", "implement"))
    )

    if needs_diff:
        run_step("spec diff", "spec_diff.py")
        delta = load_delta()

        if delta:
            print_delta_summary(delta)

        if args.force:
            print("[harness] --force: delta ignored — all steps will re-run.")
            delta = None

    elif args.scope == "mini":
        print("[harness] mini scope: skipping spec_diff and scaffold contracts.")

    # ── Execute steps ─────────────────────────────────────────────────────────
    tests_passed = True
    plan_available = MINI_PLAN_PATH.exists() if args.scope == "mini" else GLM_PLAN_PATH.exists()

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

    # Persist apply record on full-scope PASS only.
    if args.scope == "full" and all_ok and delta and "implement" in results:
        _write_apply_record(delta, results)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
