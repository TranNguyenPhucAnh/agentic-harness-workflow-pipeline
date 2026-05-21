"""
modules/post_interactive.py — Post-run interactive prompt for pipeline scripts.

Runs AFTER a script completes (in the finally block), shows what the next step
will consume/produce, then asks the user whether to proceed or stop.

────────────────────────────────────────────────────────────────────────────────
Usage per script
────────────────────────────────────────────────────────────────────────────────
    from modules.post_interactive import prompt_next_step

    # In finally block of main(), after print_summary and print_artifact_summary:
    finally:
        print_summary("[01]")
        print_artifact_summary("[01]")
        prompt_next_step("absorber", prefix="[01]")

Flow
────
  1. Look up current role in PIPELINE_CHAIN
  2. Resolve next step (respects suggest_after_run override)
  3. If next step has requires_manual_confirm=True → show special notice
  4. Print next step's consumes/produces
  5. Ask: [1] proceed  [2] stop
  6. If proceed → print suggested run command

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
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ─── StepInfo ─────────────────────────────────────────────────────────────────

@dataclass
class StepInfo:
    script:                str
    consumes:              list[Any]   = field(default_factory=list)
    produces:              list[Any]   = field(default_factory=list)
    next_step:             str | None  = None
    # Override: suggest this role instead of next_step after run.
    # Used by archivist to suggest spectracker at end of pipeline.
    suggest_after_run:     str | None  = None
    # Flag: this step requires explicit human confirmation before running.
    # prompt_next_step will show a notice instead of a run command.
    requires_manual_confirm: bool      = False


# ─── Pipeline chain ───────────────────────────────────────────────────────────
#
# SOURCE OF TRUTH for step → artifact relationships, mirroring OWNERSHIP.md.
# Uses path constants from artifacts/paths.py — never raw strings.
#
# To reorder steps: update next_step pointers only.
# To add a step: add StepInfo entry + update next_step of preceding step.

def _build_chain() -> dict[str, StepInfo]:
    """Build lazily — paths.py requires PIPELINE_PROJECT env var at resolve time."""
    from artifacts.paths import (
        ABSORBER_CODEBASE_MAP,
        ABSORBER_CODEBASE_LOG,
        ABSORBER_CODEBASE_SNAPSHOT,
        CLARIFICATOR_SESSION,
        CLARIFICATOR_DECISION_LOG,
        ENRICHER_OVERWRITE_PROMPT,
        SCAFFOLD_JSON,
        PLANNER_FULL_PLAN,
        PLANNER_MINI_PLAN,
        PLANNER_MINI_IMPACT,
        EXECUTOR_OVERWRITE_MANIFEST,
        DEBUGGER_OVERWRITE_TEST_SUMMARY,
        REPORTER_EXECUTION_SUMMARY,
        JUDGE_OVERWRITE_VERDICT_RAW,
        JUDGE_VERDICT_SUMMARY,
        PATCHER_OVERWRITE_FIX_SUMMARY,
        PATCHER_FINDINGS_SNAPSHOT,
        PATCHER_ATTEMPT_LOG,
        ARCHIVIST_KNOWLEDGE_LOG,
        ARCHIVIST_SPEC_GAPS,
        ARCHIVIST_CURATION_LOG,
        SPECTRACKER_VERSION_DELTA,
        SPECTRACKER_APPLIED,
        SPECTRACKER_VERSION_LOG,
        get_spec_path,
    )

    class _SpecPath:
        """Lazy wrapper — resolves at display time, not import time."""
        def __str__(self):  return str(get_spec_path())
        def __repr__(self): return f"spec:{get_spec_path().name}"

    SPEC = _SpecPath()

    return {
        "absorber": StepInfo(
            script    = "01_absorber.py",
            consumes  = [],
            produces  = [
                ABSORBER_CODEBASE_MAP,
                ABSORBER_CODEBASE_LOG,
                ABSORBER_CODEBASE_SNAPSHOT,
            ],
            next_step = "clarificator",
        ),
        "clarificator": StepInfo(
            script    = "02_clarificator.py",
            consumes  = [
                ABSORBER_CODEBASE_MAP,
                ABSORBER_CODEBASE_SNAPSHOT,
                CLARIFICATOR_DECISION_LOG,
            ],
            produces  = [
                CLARIFICATOR_SESSION,
                CLARIFICATOR_DECISION_LOG,
            ],
            next_step = "enricher",
        ),
        "enricher": StepInfo(
            script    = "03_enricher.py",
            consumes  = [
                CLARIFICATOR_SESSION,
                ABSORBER_CODEBASE_MAP,
                CLARIFICATOR_DECISION_LOG,
                ARCHIVIST_KNOWLEDGE_LOG,
            ],
            produces  = [ENRICHER_OVERWRITE_PROMPT],
            next_step = "specwright",
        ),
        "specwright": StepInfo(
            script    = "04_specwright.py",
            consumes  = [ENRICHER_OVERWRITE_PROMPT],
            produces  = [SPEC],
            next_step = "spectracker",
        ),
        "spectracker": StepInfo(
            script             = "05_spectracker.py",
            consumes           = [SPEC, SPECTRACKER_APPLIED],
            produces           = [
                SPECTRACKER_VERSION_DELTA,
                SPECTRACKER_APPLIED,
                SPECTRACKER_VERSION_LOG,
            ],
            next_step              = "scaffolder",
            requires_manual_confirm = True,
        ),
        "scaffolder": StepInfo(
            script    = "06_scaffolder.py",
            consumes  = [SPEC],
            produces  = [SCAFFOLD_JSON],
            next_step = "planner",
        ),
        "planner": StepInfo(
            script    = "07_planner.py",
            consumes  = [
                SPEC,
                SCAFFOLD_JSON,
                ABSORBER_CODEBASE_MAP,
                ARCHIVIST_KNOWLEDGE_LOG,
            ],
            produces  = [PLANNER_FULL_PLAN, PLANNER_MINI_PLAN, PLANNER_MINI_IMPACT],
            next_step = "executor",
        ),
        "executor": StepInfo(
            script    = "08_executor.py",
            consumes  = [
                SPEC,
                SCAFFOLD_JSON,
                PLANNER_FULL_PLAN,
                ABSORBER_CODEBASE_MAP,
                ARCHIVIST_KNOWLEDGE_LOG,
            ],
            produces  = [EXECUTOR_OVERWRITE_MANIFEST],
            next_step = "debugger",
        ),
        "debugger": StepInfo(
            script    = "09_debugger.py",
            consumes  = [
                EXECUTOR_OVERWRITE_MANIFEST,
                PLANNER_FULL_PLAN,
                PATCHER_FINDINGS_SNAPSHOT,
            ],
            produces  = [DEBUGGER_OVERWRITE_TEST_SUMMARY],
            next_step = "reporter",
        ),
        "reporter": StepInfo(
            script    = "10_reporter.py",
            consumes  = [
                EXECUTOR_OVERWRITE_MANIFEST,
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                SCAFFOLD_JSON,
                PLANNER_FULL_PLAN,
            ],
            produces  = [REPORTER_EXECUTION_SUMMARY],
            next_step = "judge",
        ),
        "judge": StepInfo(
            script    = "11_judge.py",
            consumes  = [
                SPEC,
                EXECUTOR_OVERWRITE_MANIFEST,
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                ARCHIVIST_KNOWLEDGE_LOG,
                ARCHIVIST_SPEC_GAPS,
            ],
            produces  = [JUDGE_OVERWRITE_VERDICT_RAW, JUDGE_VERDICT_SUMMARY],
            next_step = "patcher",
        ),
        "patcher": StepInfo(
            script    = "12_patcher.py",
            consumes  = [
                JUDGE_OVERWRITE_VERDICT_RAW,
                PATCHER_FINDINGS_SNAPSHOT,
            ],
            produces  = [
                PATCHER_OVERWRITE_FIX_SUMMARY,
                PATCHER_FINDINGS_SNAPSHOT,
                PATCHER_ATTEMPT_LOG,
            ],
            next_step = "archivist",
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
            next_step          = None,       # end of linear chain
            suggest_after_run  = "spectracker",  # natural post-pipeline confirmation
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
    chain    = get_chain()
    info     = chain.get(role)
    if info is None:
        return None
    next_key = info.suggest_after_run or info.next_step
    if next_key is None:
        return None
    next_info = chain.get(next_key)
    if next_info is None:
        return None
    return next_key, next_info


def _print_next_step_box(next_role: str, next_info: StepInfo) -> None:
    """Print the consumes/produces box for the upcoming step."""
    pad   = max(0, _W - 18 - len(next_role) - len(next_info.script))
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
    return f"python3 harness.py --{next_info.script.split('_', 1)[1].replace('.py', '')} --project {project}"


# ─── Public API ───────────────────────────────────────────────────────────────

def prompt_next_step(role: str, prefix: str = "[pipeline]") -> None:
    """
    Post-run interactive prompt. Call in the finally block of main().

    Shows the next step's ownership info, then asks the user to proceed or stop.
    Prints the suggested run command if the user chooses to proceed.

    Parameters
    ----------
    role   : current script's role key (e.g. "absorber")
    prefix : log prefix for consistent terminal output (e.g. "[01]")
    """
    resolved = _resolve_next(role)
    if resolved is None:
        # End of pipeline or unknown role — nothing to suggest
        if role in get_chain():
            print(f"\n{prefix} Pipeline complete — no next step.")
        return

    next_role, next_info = resolved
    chain = get_chain()
    current_info = chain.get(role)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * _W}")
    if current_info and current_info.suggest_after_run == next_role:
        # suggest_after_run path (e.g. archivist → spectracker)
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

    # ── Commit run to harness_run_log.json ───────────────────────────────────
    _maybe_commit_run_log(role, prefix)

    # ── Proceed / stop ────────────────────────────────────────────────────────
    print(f"  [1] proceed — run {next_role} now")
    print( "  [2] stop    — exit, run manually later\n")

    while True:
        choice = input("  → Choose 1 / 2: ").strip()
        if choice in ("1", "proceed"):
            cmd = _run_command(next_info)
            print(f"\n{prefix} Run:\n\n    {cmd}\n")
            return
        if choice in ("2", "stop"):
            print(f"\n{prefix} Stopped. To run {next_role} later:\n")
            print(f"    {_run_command(next_info)}\n")
            return
        print("  Please enter 1 or 2.")



# ─── Run log commit ───────────────────────────────────────────────────────────

def _maybe_commit_run_log(role: str, prefix: str = "[pipeline]") -> None:
    """
    Prompt user to commit this run to harness_run_log.json.
    Appends a terse entry: step, timestamp, status=completed.
    Skips silently on EOFError (non-interactive / harness subprocess).
    """
    import json
    from datetime import datetime, timezone

    try:
        ans = input("  Commit this run to harness_run_log.json? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return  # non-interactive — skip silently

    if ans in ("n", "no"):
        print(f"  {prefix} Run not committed to log.")
        return

    # Resolve log path
    try:
        from artifacts.paths import artifact_root
        log_path = Path(str(artifact_root())) / "harness_run_log.json"
    except Exception:
        print(f"  {prefix}[warn] Could not resolve artifact root — run log skipped.")
        return

    entry = {
        "step":       role,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "status":     "completed",
        "project":    os.environ.get("PIPELINE_PROJECT", "unknown"),
    }

    existing: list[dict] = []
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            if isinstance(data, dict) and "entries" in data:
                existing = data["entries"]
            elif isinstance(data, list):
                existing = data
        except Exception:
            pass

    existing.append(entry)
    try:
        log_path.write_text(json.dumps({"entries": existing}, indent=2))
        print(f"  {prefix} Run committed to {log_path.name} (total entries: {len(existing)})")
    except Exception as e:
        print(f"  {prefix}[warn] Could not write run log: {e}")


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
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"\n  [{role}]  {info.script}{flag_str}")
        print(f"    consumes : {[Path(str(p)).name for p in info.consumes] or '(none)'}")
        print(f"    produces : {[Path(str(p)).name for p in info.produces] or '(none)'}")
        print(f"    next     : {info.next_step or '(end)'}")
    print(f"\n{'═' * _W}\n")


if __name__ == "__main__":
    print_chain()
