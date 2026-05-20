"""
modules/post_interactive.py — Post-run interactive prompt for pipeline scripts.

Runs AFTER a script completes (in the finally block). Two sequential prompts:

  1. COMMIT PROMPT (if script owns a long-term *_log.json artifact):
     "Append to <module>/<log>.json? [y/n]"
     → y: calls the write_fn provided by the script to perform the append.
     → n: skips append, prints notice. Does NOT affect the next prompt.

  2. NEXT STEP PROMPT:
     Shows what the next step consumes/produces, asks proceed or stop.

The two prompts are independent — skipping the log commit does not skip the
next step, and vice versa.

────────────────────────────────────────────────────────────────────────────────
Usage per script
────────────────────────────────────────────────────────────────────────────────
    from modules.post_interactive import commit_log, prompt_next_step

    # In finally block of main(), after print_summary and print_artifact_summary:
    finally:
        print_summary("[01]")
        print_artifact_summary("[01]")
        commit_log(
            role     = "absorber",
            write_fn = lambda: _append_codebase_log(log_entry),
            prefix   = "[01]",
        )
        prompt_next_step("absorber", prefix="[01]")

    # Scripts without a long-term log (archivist handles its own approval flow):
    finally:
        print_summary("[13]")
        print_artifact_summary("[13]")
        prompt_next_step("archivist", prefix="[13]")

────────────────────────────────────────────────────────────────────────────────
Long-term log ownership per role
────────────────────────────────────────────────────────────────────────────────
  absorber     → absorber/codebase_log.json
  clarificator → clarificator/decision_log.json
  enricher     → enricher/prompt_log.json
  spectracker  → spectracker/version_log.json
  scaffolder   → scaffolder/skeleton_log.json
  planner      → planner/plan_log.json
  executor     → executor/manifest_log.json
  debugger     → debugger/test_log.json
  reporter     → reporter/execution_log.json
  judge        → judge/verdict_log.json
  patcher      → patcher/attempt_log.json
  archivist    → (handled separately — no commit_log call)
  specwright   → (no log — versioning delegated to spectracker)
  harness      → (harness_run_log.json written by harness, not via this module)

────────────────────────────────────────────────────────────────────────────────
Spectracker exception
────────────────────────────────────────────────────────────────────────────────
spectracker requires human judgment to confirm a spec version is stable before
marking it applied. It cannot be auto-suggested in the normal linear chain.

Instead, archivist (last step) suggests spectracker via suggest_after_run.
This reflects the natural workflow: after archivist curates knowledge at the
end of a pipeline run, the user decides if the spec is ready to mark applied.

specwright also suggests spectracker as its linear next_step, but the
requires_manual_confirm flag ensures the prompt clearly explains this is a
manual confirmation step, not an automatic processing step.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── StepInfo ─────────────────────────────────────────────────────────────────

@dataclass
class StepInfo:
    script:                  str
    consumes:                list[Any]  = field(default_factory=list)
    produces:                list[Any]  = field(default_factory=list)
    next_step:               str | None = None
    long_term_log:           Any | None = None   # *_log.json constant, or None
    # Override: suggest this role instead of next_step after run.
    suggest_after_run:       str | None = None
    # Flag: this step requires explicit human confirmation before running.
    requires_manual_confirm: bool       = False


# ─── Pipeline chain ───────────────────────────────────────────────────────────
#
# SOURCE OF TRUTH for step → artifact relationships, mirroring OWNERSHIP.md.
# Uses path constants from artifacts/paths.py — never raw strings.
#
# long_term_log: the single *_log.json artifact each script owns.
#   None  → script has no log (specwright), or handles its own flow (archivist).
#
# To reorder steps: update next_step pointers only.
# To add a step: add StepInfo entry + update next_step of preceding step.

def _build_chain() -> dict[str, StepInfo]:
    """Build lazily — paths.py requires PIPELINE_PROJECT env var at resolve time."""
    from artifacts.paths import (
        # absorber
        ABSORBER_CODEBASE_MAP,
        ABSORBER_CODEBASE_SNAPSHOT,
        ABSORBER_CODEBASE_LOG,
        # clarificator
        CLARIFICATOR_SESSION,
        CLARIFICATOR_DECISION_LOG,
        # enricher
        ENRICHER_OVERWRITE_PROMPT,
        ENRICHER_PROMPT_LOG,
        # spectracker
        SPECTRACKER_VERSION_DELTA,
        SPECTRACKER_VERSION_LOG,
        # scaffolder
        SCAFFOLD_JSON,
        SCAFFOLDER_SKELETON_LOG,
        # planner
        PLANNER_FULL_PLAN,
        PLANNER_MINI_PLAN,
        PLANNER_PLAN_LOG,
        # executor
        EXECUTOR_OVERWRITE_MANIFEST,
        EXECUTOR_MANIFEST_LOG,
        # debugger
        DEBUGGER_OVERWRITE_TEST_SUMMARY,
        DEBUGGER_TEST_LOG,
        # reporter
        REPORTER_EXECUTION_SUMMARY,
        REPORTER_EXECUTION_LOG,
        # judge
        JUDGE_OVERWRITE_VERDICT_RAW,
        JUDGE_VERDICT_SUMMARY,
        JUDGE_VERDICT_LOG,
        # patcher
        PATCHER_OVERWRITE_FIX_SUMMARY,
        PATCHER_ATTEMPT_LOG,
        # archivist
        ARCHIVIST_KNOWLEDGE_LOG,
        ARCHIVIST_SPEC_GAPS,
        ARCHIVIST_CURATION_LOG,
        # spec
        get_spec_path,
    )

    class _SpecPath:
        """Lazy wrapper — resolves at display time, not import time."""
        def __str__(self):  return str(get_spec_path())
        def __repr__(self): return f"spec:{get_spec_path().name}"

    SPEC = _SpecPath()

    return {
        "absorber": StepInfo(
            script        = "01_absorber.py",
            consumes      = [],
            produces      = [
                ABSORBER_CODEBASE_MAP,
                ABSORBER_CODEBASE_SNAPSHOT,
                ABSORBER_CODEBASE_LOG,
            ],
            long_term_log = ABSORBER_CODEBASE_LOG,
            next_step     = "clarificator",
        ),
        "clarificator": StepInfo(
            script        = "02_clarificator.py",
            consumes      = [
                ABSORBER_CODEBASE_SNAPSHOT,
                ABSORBER_CODEBASE_MAP,
                CLARIFICATOR_DECISION_LOG,
            ],
            produces      = [
                CLARIFICATOR_SESSION,
                CLARIFICATOR_DECISION_LOG,
            ],
            long_term_log = CLARIFICATOR_DECISION_LOG,
            next_step     = "enricher",
        ),
        "enricher": StepInfo(
            script        = "03_enricher.py",
            consumes      = [
                CLARIFICATOR_SESSION,
                ABSORBER_CODEBASE_MAP,
                CLARIFICATOR_DECISION_LOG,
                ARCHIVIST_KNOWLEDGE_LOG,
            ],
            produces      = [
                ENRICHER_OVERWRITE_PROMPT,
                ENRICHER_PROMPT_LOG,
            ],
            long_term_log = ENRICHER_PROMPT_LOG,
            next_step     = "specwright",
        ),
        "specwright": StepInfo(
            script        = "04_specwright.py",
            consumes      = [ENRICHER_OVERWRITE_PROMPT],
            produces      = [SPEC],
            long_term_log = None,   # versioning delegated to spectracker
            next_step     = "spectracker",
        ),
        "spectracker": StepInfo(
            script                  = "05_spectracker.py",
            consumes                = [SPEC, SPECTRACKER_VERSION_LOG],
            produces                = [
                SPECTRACKER_VERSION_DELTA,
                SPECTRACKER_VERSION_LOG,
            ],
            long_term_log           = SPECTRACKER_VERSION_LOG,
            next_step               = "scaffolder",
            requires_manual_confirm = True,
        ),
        "scaffolder": StepInfo(
            script        = "06_scaffolder.py",
            consumes      = [SPEC],
            produces      = [
                SCAFFOLD_JSON,
                SCAFFOLDER_SKELETON_LOG,
            ],
            long_term_log = SCAFFOLDER_SKELETON_LOG,
            next_step     = "planner",
        ),
        "planner": StepInfo(
            script        = "07_planner.py",
            consumes      = [
                SPEC,
                SCAFFOLD_JSON,
                ABSORBER_CODEBASE_MAP,
                ARCHIVIST_KNOWLEDGE_LOG,
            ],
            produces      = [
                PLANNER_FULL_PLAN,
                PLANNER_MINI_PLAN,
                PLANNER_PLAN_LOG,
            ],
            long_term_log = PLANNER_PLAN_LOG,
            next_step     = "executor",
        ),
        "executor": StepInfo(
            script        = "08_executor.py",
            consumes      = [
                SPEC,
                SCAFFOLD_JSON,
                PLANNER_FULL_PLAN,
                ABSORBER_CODEBASE_MAP,
                ARCHIVIST_KNOWLEDGE_LOG,
            ],
            produces      = [
                EXECUTOR_OVERWRITE_MANIFEST,
                EXECUTOR_MANIFEST_LOG,
            ],
            long_term_log = EXECUTOR_MANIFEST_LOG,
            next_step     = "debugger",
        ),
        "debugger": StepInfo(
            script        = "09_debugger.py",
            consumes      = [
                EXECUTOR_OVERWRITE_MANIFEST,
                PLANNER_FULL_PLAN,
                PATCHER_ATTEMPT_LOG,
            ],
            produces      = [
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                DEBUGGER_TEST_LOG,
            ],
            long_term_log = DEBUGGER_TEST_LOG,
            next_step     = "reporter",
        ),
        "reporter": StepInfo(
            script        = "10_reporter.py",
            consumes      = [
                EXECUTOR_OVERWRITE_MANIFEST,
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                SCAFFOLD_JSON,
                PLANNER_FULL_PLAN,
            ],
            produces      = [
                REPORTER_EXECUTION_SUMMARY,
                REPORTER_EXECUTION_LOG,
            ],
            long_term_log = REPORTER_EXECUTION_LOG,
            next_step     = "judge",
        ),
        "judge": StepInfo(
            script        = "11_judge.py",
            consumes      = [
                SPEC,
                EXECUTOR_OVERWRITE_MANIFEST,
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                ARCHIVIST_KNOWLEDGE_LOG,
                ARCHIVIST_SPEC_GAPS,
            ],
            produces      = [
                JUDGE_OVERWRITE_VERDICT_RAW,
                JUDGE_VERDICT_SUMMARY,
                JUDGE_VERDICT_LOG,
            ],
            long_term_log = JUDGE_VERDICT_LOG,
            next_step     = "patcher",
        ),
        "patcher": StepInfo(
            script        = "12_patcher.py",
            consumes      = [
                JUDGE_OVERWRITE_VERDICT_RAW,
                PATCHER_ATTEMPT_LOG,
            ],
            produces      = [
                PATCHER_OVERWRITE_FIX_SUMMARY,
                PATCHER_ATTEMPT_LOG,
            ],
            long_term_log = PATCHER_ATTEMPT_LOG,
            next_step     = "archivist",
        ),
        "archivist": StepInfo(
            script             = "13_archivist.py",
            consumes           = [
                JUDGE_OVERWRITE_VERDICT_RAW,
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                ARCHIVIST_KNOWLEDGE_LOG,
                ARCHIVIST_SPEC_GAPS,
            ],
            produces           = [
                ARCHIVIST_KNOWLEDGE_LOG,
                ARCHIVIST_SPEC_GAPS,
                ARCHIVIST_CURATION_LOG,
            ],
            long_term_log      = None,   # archivist handles its own approval flow
            next_step          = None,
            suggest_after_run  = "spectracker",
        ),
    }


_PIPELINE_CHAIN: dict[str, StepInfo] | None = None


def get_chain() -> dict[str, StepInfo]:
    """Return the pipeline chain dict, building it lazily on first call."""
    global _PIPELINE_CHAIN
    if _PIPELINE_CHAIN is None:
        _PIPELINE_CHAIN = _build_chain()
    return _PIPELINE_CHAIN


# ─── Display helpers ──────────────────────────────────────────────────────────

_W = 72


def _resolve_next(role: str) -> tuple[str, StepInfo] | None:
    """
    Return (next_role, next_info) for a given role.
    Respects suggest_after_run override over next_step.
    Returns None if no next step.
    """
    chain     = get_chain()
    info      = chain.get(role)
    if info is None:
        return None
    next_key  = info.suggest_after_run or info.next_step
    if next_key is None:
        return None
    next_info = chain.get(next_key)
    if next_info is None:
        return None
    return next_key, next_info


def _print_next_step_box(next_role: str, next_info: StepInfo) -> None:
    """Print the consumes/produces box for the upcoming step."""
    pad = max(0, _W - 18 - len(next_role) - len(next_info.script))
    print(f"\n  ┌─ Next step: {next_role}  ({next_info.script}) {'─' * pad}┐")

    if next_info.consumes:
        print("  │  Consumes:")
        for p in next_info.consumes:
            print(f"  │    · {Path(str(p)).name}")
    else:
        print("  │  Consumes: (none)")

    if next_info.produces:
        print("  │  Produces:")
        for p in next_info.produces:
            print(f"  │    · {Path(str(p)).name}")
    else:
        print("  │  Produces: (none)")

    print(f"  └{'─' * (_W - 2)}┘\n")


def _run_command(next_info: StepInfo) -> str:
    """Build the suggested run command for the next step."""
    project = os.environ.get("PIPELINE_PROJECT", "<project>")
    step    = next_info.script.split("_", 1)[1].replace(".py", "")
    return f"python3 harness.py --{step} --project {project}"


# ─── Public API ───────────────────────────────────────────────────────────────

def commit_log(
    role:     str,
    write_fn: Callable[[], None],
    prefix:   str = "[pipeline]",
) -> bool:
    """
    Prompt the user to approve appending to this role's long-term *_log.json.

    Looks up the long_term_log artifact for the given role. If None (specwright,
    archivist) prints nothing and returns True immediately.

    Parameters
    ----------
    role     : current script's role key (e.g. "absorber")
    write_fn : zero-argument callable that performs the actual append when called.
               Only called if the user approves.
    prefix   : log prefix for terminal output (e.g. "[01]")

    Returns
    -------
    bool — True if committed, False if skipped.
    """
    chain = get_chain()
    info  = chain.get(role)

    # Role not in chain, or no long-term log defined → skip silently
    if info is None or info.long_term_log is None:
        return True

    log_path    = Path(str(info.long_term_log))
    # Display as <module>/<filename> — drop the artifact root prefix
    # log_path is an absolute path; show the last 2 parts for clarity
    display     = "/".join(log_path.parts[-2:])

    print(f"\n{'─' * _W}")
    print(f"  {prefix} Commit to long-term log")
    print(f"{'─' * _W}")
    print(f"  Append this run's entry to: {display}")
    print()

    while True:
        try:
            answer = input("  → Append? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{prefix} Interrupted — skipping log commit.")
            return False

        if answer in ("y", "yes"):
            try:
                write_fn()
                print(f"  ✓ Appended to {display}\n")
            except Exception as exc:
                print(f"  ✗ Failed to append: {exc}\n", file=sys.stderr)
                return False
            return True

        if answer in ("n", "no"):
            print(f"  Skipped — {display} not updated.\n")
            return False

        print("  Please enter y or n.")


def prompt_next_step(role: str, prefix: str = "[pipeline]") -> None:
    """
    Post-run interactive prompt. Call in the finally block of main(),
    after commit_log().

    Shows the next step's ownership info, then asks the user to proceed or stop.
    Prints the suggested run command if the user chooses to proceed.

    Parameters
    ----------
    role   : current script's role key (e.g. "absorber")
    prefix : log prefix for consistent terminal output (e.g. "[01]")
    """
    resolved = _resolve_next(role)
    if resolved is None:
        if role in get_chain():
            print(f"\n{prefix} Pipeline complete — no next step.")
        return

    next_role, next_info = resolved
    chain        = get_chain()
    current_info = chain.get(role)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * _W}")
    if current_info and current_info.suggest_after_run == next_role:
        print(f"  {prefix} Pipeline run complete.")
        print(f"  Suggested: run spectracker to mark spec version as applied.")
    else:
        print(f"  {prefix} Script complete — suggested next step: {next_role}")
    print(f"{'─' * _W}")

    # ── Ownership box ─────────────────────────────────────────────────────────
    _print_next_step_box(next_role, next_info)

    # ── Manual confirm notice ─────────────────────────────────────────────────
    if next_info.requires_manual_confirm:
        print(f"  ⚠  {next_role} requires manual confirmation.")
        print( "     Run it only when you are satisfied the spec is stable")
        print( "     and all pipeline iterations for this version are complete.\n")

    # ── Proceed / stop ────────────────────────────────────────────────────────
    print(f"  [1] proceed — run {next_role} now")
    print( "  [2] stop    — exit, run manually later\n")

    while True:
        try:
            choice = input("  → Choose 1 / 2: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{prefix} Interrupted.")
            return

        if choice in ("1", "proceed"):
            cmd = _run_command(next_info)
            print(f"\n{prefix} Run:\n\n    {cmd}\n")
            return
        if choice in ("2", "stop"):
            print(f"\n{prefix} Stopped. To run {next_role} later:\n")
            print(f"    {_run_command(next_info)}\n")
            return
        print("  Please enter 1 or 2.")


# ─── Standalone: inspect chain ────────────────────────────────────────────────

def print_chain() -> None:
    """Print the full pipeline chain. Usage: python3 -m modules.post_interactive"""
    chain = get_chain()
    print(f"\n{'═' * _W}")
    print("  PIPELINE CHAIN")
    print(f"{'═' * _W}")
    for role, info in chain.items():
        flags = []
        if info.requires_manual_confirm: flags.append("manual-confirm")
        if info.suggest_after_run:       flags.append(f"suggest→{info.suggest_after_run}")
        if info.long_term_log is None:   flags.append("no-log")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        log_name = Path(str(info.long_term_log)).name if info.long_term_log else "(none)"
        print(f"\n  [{role}]  {info.script}{flag_str}")
        print(f"    consumes  : {[Path(str(p)).name for p in info.consumes] or '(none)'}")
        print(f"    produces  : {[Path(str(p)).name for p in info.produces] or '(none)'}")
        print(f"    log       : {log_name}")
        print(f"    next      : {info.next_step or '(end)'}")
    print(f"\n{'═' * _W}\n")


if __name__ == "__main__":
    print_chain()
