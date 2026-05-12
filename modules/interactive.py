"""
modules/interactive.py — Interactive review + pipeline ownership display.

Provides two things:
  1. PIPELINE_CHAIN  — source of truth for each step's consumes/produces/next_step.
                       Mirrors OWNERSHIP.md in importable form, using path constants
                       from artifacts/paths.py so paths never diverge.
  2. review_artifact() — interactive review loop: show content preview, display
                         next-step ownership, ask user to proceed or abort.

────────────────────────────────────────────────────────────────────────────────
Usage per script
────────────────────────────────────────────────────────────────────────────────
    from modules.interactive import review_artifact, Action
    from modules.interactive import PIPELINE_CHAIN   # optional, for inspection

    # After LLM produces an artifact:
    action, final_text = review_artifact(
        content     = enriched_prompt_text,
        role        = "enricher",           # key into PIPELINE_CHAIN
        title       = "Enriched prompt — review before sending to spec agent",
        preview     = "full",               # "full" | int (line count)
        prefix      = "[03]",               # for consistent log prefix
        choices     = [                     # omit "edit" if not needed
            Action.CONFIRM,
            Action.ABORT,
        ],
    )
    if action == Action.ABORT:
        return   # caller decides what to do

    # final_text == content (or edited content if Action.EDIT was chosen + completed)

────────────────────────────────────────────────────────────────────────────────
Review flow (Approach 2 — confirm-then-show)
────────────────────────────────────────────────────────────────────────────────
  Step 1: Show content preview
  Step 2: Present choice menu (confirm / edit / abort)
  Step 3: If user picks CONFIRM → show next-step ownership (consumes/produces)
           → ask: [1] proceed  [2] abort
  Step 4: Return (action, final_text) to caller

The second confirmation is lightweight (just two options) and gives the user
a chance to see exactly what the next step will consume/produce before committing.
If PIPELINE_CHAIN has no entry for the role, ownership display is skipped.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


# ─── Action enum ─────────────────────────────────────────────────────────────

class Action(Enum):
    CONFIRM = auto()   # proceed to next step
    EDIT    = auto()   # open $EDITOR, then re-review
    ABORT   = auto()   # stop here, artifact already saved


# ─── Pipeline ownership chain ─────────────────────────────────────────────────
#
# SOURCE OF TRUTH for step → artifact relationships, mirroring OWNERSHIP.md.
# Uses path constants from artifacts/paths.py — never raw strings.
#
# Each StepInfo encodes:
#   script     : filename of the owning script
#   consumes   : artifacts this step reads (LazyPath / Path / str)
#   produces   : artifacts this step writes
#   next_step  : role key of the step that runs after this one (or None)
#
# To add a new step: add an entry here + update next_step of the preceding step.
# To reorder: update next_step pointers only — no script changes needed.

@dataclass
class StepInfo:
    script:    str
    consumes:  list[Any] = field(default_factory=list)
    produces:  list[Any] = field(default_factory=list)
    next_step: str | None = None


def _build_chain() -> dict[str, StepInfo]:
    """
    Build PIPELINE_CHAIN lazily to avoid importing paths at module load time
    (paths.py requires PIPELINE_PROJECT env var when resolving).
    Called once on first access via the module-level property.
    """
    from artifacts.paths import (
        # absorber outputs
        ABSORBER_CODEBASE_MAP,
        ABSORBER_CONFIG_MAP,
        ABSORBER_BLAME_MAP,
        ABSORBER_CODEBASE_SNAPSHOT,
        ABSORBER_GIT_SNAPSHOT,
        # clarificator
        CLARIFIED_REQ,
        CLARIFICATOR_OVERWRITE_RAW,
        CLARIFICATOR_OVERWRITE_QUESTIONS,
        CLARIFICATOR_DECISION_LOG,
        # enricher
        ENRICHER_OVERWRITE_PROMPT,
        # scaffolder
        SCAFFOLD_JSON,
        # planner
        PLANNER_FULL_PLAN,
        PLANNER_MINI_PLAN,
        PLANNER_MINI_IMPACT,
        # executor
        EXECUTOR_OVERWRITE_MANIFEST,
        # debugger
        DEBUGGER_OVERWRITE_TEST_SUMMARY,
        # reporter
        REPORTER_EXECUTION_SUMMARY,
        # judge
        JUDGE_OVERWRITE_VERDICT_RAW,
        JUDGE_VERDICT_SUMMARY,
        # patcher
        PATCHER_OVERWRITE_FIX_SUMMARY,
        PATCHER_FINDINGS_SNAPSHOT,
        PATCHER_ATTEMPT_LOG,
        # archivist
        ARCHIVIST_KNOWLEDGE_LOG,
        ARCHIVIST_SPEC_GAPS,
        ARCHIVIST_CURATION_LOG,
        # spectracker
        SPECTRACKER_VERSION_DELTA,
        SPECTRACKER_APPLIED,
        SPECTRACKER_VERSION_LOG,
    )
    # get_spec_path() is a function, not a constant — call it lazily
    from artifacts.paths import get_spec_path

    class _SpecPath:
        """Thin lazy wrapper so spec path resolves at display time, not import time."""
        def __str__(self): return str(get_spec_path())
        def __repr__(self): return f"spec_path({get_spec_path().name})"

    SPEC = _SpecPath()

    return {
        "absorber": StepInfo(
            script    = "01_absorber.py",
            consumes  = [],
            produces  = [
                ABSORBER_CODEBASE_MAP,
                ABSORBER_CONFIG_MAP,
                ABSORBER_BLAME_MAP,
                ABSORBER_CODEBASE_SNAPSHOT,
                ABSORBER_GIT_SNAPSHOT,
            ],
            next_step = "clarificator",
        ),
        "clarificator": StepInfo(
            script    = "02_clarificator.py",
            consumes  = [
                ABSORBER_CODEBASE_SNAPSHOT,
                ABSORBER_GIT_SNAPSHOT,
                ABSORBER_CODEBASE_MAP,
                CLARIFICATOR_DECISION_LOG,
            ],
            produces  = [
                CLARIFIED_REQ,
                CLARIFICATOR_OVERWRITE_RAW,
                CLARIFICATOR_OVERWRITE_QUESTIONS,
                CLARIFICATOR_DECISION_LOG,
            ],
            next_step = "enricher",
        ),
        "enricher": StepInfo(
            script    = "03_enricher.py",
            consumes  = [
                CLARIFIED_REQ,
                ABSORBER_CODEBASE_MAP,
                ABSORBER_CONFIG_MAP,
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
            script    = "05_spectracker.py",
            consumes  = [SPEC, SPECTRACKER_APPLIED],
            produces  = [
                SPECTRACKER_VERSION_DELTA,
                SPECTRACKER_APPLIED,
                SPECTRACKER_VERSION_LOG,
            ],
            next_step = "scaffolder",
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
            script    = "13_archivist.py",
            consumes  = [
                JUDGE_OVERWRITE_VERDICT_RAW,
                DEBUGGER_OVERWRITE_TEST_SUMMARY,
                ARCHIVIST_KNOWLEDGE_LOG,
                ARCHIVIST_SPEC_GAPS,
            ],
            produces  = [
                ARCHIVIST_KNOWLEDGE_LOG,
                ARCHIVIST_SPEC_GAPS,
                ARCHIVIST_CURATION_LOG,
            ],
            next_step = None,
        ),
    }


# Lazy singleton — built on first access
_PIPELINE_CHAIN: dict[str, StepInfo] | None = None


def get_chain() -> dict[str, StepInfo]:
    """Return the pipeline chain dict, building it on first call."""
    global _PIPELINE_CHAIN
    if _PIPELINE_CHAIN is None:
        _PIPELINE_CHAIN = _build_chain()
    return _PIPELINE_CHAIN


# Convenience alias for direct import
PIPELINE_CHAIN = get_chain  # callable, not the dict — call get_chain() for dict


# ─── Display helpers ──────────────────────────────────────────────────────────

_W = 72  # banner width


def _banner(title: str) -> None:
    print(f"\n{'─' * _W}")
    print(f"  {title}")
    print(f"{'─' * _W}")


def _print_ownership(role: str, prefix: str) -> None:
    """Print next-step consumes/produces table for the given role."""
    chain = get_chain()
    info  = chain.get(role)
    if info is None or info.next_step is None:
        return

    next_info = chain.get(info.next_step)
    if next_info is None:
        return

    print(f"\n  ┌─ Next step: {info.next_step}  ({next_info.script}) {'─' * max(0, _W - 30 - len(info.next_step) - len(next_info.script))}┐")

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

    print(f"  └{'─' * (_W - 2)}┘")


def _open_in_editor(content: str, hint_suffix: str = ".md") -> str | None:
    """
    Write content to a temp file, open $EDITOR, return modified content.
    Returns None if editor not found or returned empty content.
    """
    editor = os.environ.get("EDITOR", "nano")
    fd, tmp_path = tempfile.mkstemp(suffix=hint_suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        subprocess.run([editor, tmp_path], check=True)
        edited = Path(tmp_path).read_text()
        return edited if edited.strip() else None
    except FileNotFoundError:
        print(f"  [interactive][warn] Editor '{editor}' not found. Set $EDITOR env var.")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"  [interactive][warn] Editor exited with error: {exc}")
        return None
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ─── Main public API ──────────────────────────────────────────────────────────

def review_artifact(
    content:  str,
    role:     str,
    title:    str,
    *,
    preview:  int | str = 60,
    prefix:   str       = "[pipeline]",
    choices:  list[Action] | None = None,
) -> tuple[Action, str]:
    """
    Interactive review loop for a pipeline artifact.

    Parameters
    ----------
    content  : artifact text to review
    role     : pipeline role key (e.g. "enricher") — used to look up next-step info
    title    : banner title shown above content preview
    preview  : "full" to show entire content, or int for max line count
    prefix   : log prefix for consistent terminal output (e.g. "[03]")
    choices  : list of Action values to offer; defaults to [CONFIRM, ABORT]
               include Action.EDIT to enable editor option

    Returns
    -------
    (Action, str) — the chosen action + final content (may differ if EDIT was used)
    """
    if choices is None:
        choices = [Action.CONFIRM, Action.ABORT]

    current_content = content

    while True:
        # ── Step 1: show content preview ─────────────────────────────────────
        _banner(title)
        lines = current_content.strip().splitlines()

        if preview == "full":
            print("\n" + current_content.strip())
        else:
            limit = int(preview)
            print("\n" + "\n".join(lines[:limit]))
            if len(lines) > limit:
                print(f"\n  ... [{len(lines) - limit} more lines not shown]")

        print(f"\n{'─' * _W}\n")

        # ── Step 2: present choice menu ───────────────────────────────────────
        menu: list[tuple[str, str, Action]] = []
        n = 1
        for action in choices:
            if action == Action.CONFIRM:
                menu.append((str(n), "confirm", Action.CONFIRM))
            elif action == Action.EDIT:
                menu.append((str(n), "edit    — open $EDITOR to modify", Action.EDIT))
            elif action == Action.ABORT:
                menu.append((str(n), "abort   — stop here, artifact already saved", Action.ABORT))
            n += 1

        for key, label, _ in menu:
            print(f"  [{key}] {label}")

        keys = [k for k, _, _ in menu]
        while True:
            choice = input(f"\n  → Choose {' / '.join(keys)}: ").strip()
            matched = next((act for k, _, act in menu if choice in (k, act.name.lower())), None)
            if matched is not None:
                break
            print(f"  Please enter {' or '.join(keys)}.")

        # ── Step 3: handle choice ─────────────────────────────────────────────
        if matched == Action.EDIT:
            edited = _open_in_editor(current_content)
            if edited:
                print(f"{prefix} Updated content loaded.")
                current_content = edited
            else:
                print(f"{prefix} Editor returned empty — keeping original.")
            continue   # loop back to show updated content

        if matched == Action.ABORT:
            return Action.ABORT, current_content

        # matched == Action.CONFIRM — show ownership then ask proceed/abort
        _print_ownership(role, prefix)

        chain = get_chain()
        if role in chain and chain[role].next_step is not None:
            print(f"\n  [1] proceed — continue to {chain[role].next_step}")
            print( "  [2] abort   — stop here\n")
            while True:
                proceed = input("  → Choose 1 / 2: ").strip()
                if proceed in ("1", "proceed"):
                    return Action.CONFIRM, current_content
                if proceed in ("2", "abort"):
                    return Action.ABORT, current_content
                print("  Please enter 1 or 2.")
        else:
            # No next step defined — confirm proceeds immediately
            return Action.CONFIRM, current_content


# ─── Standalone: print full pipeline chain ───────────────────────────────────

def print_chain() -> None:
    """Print the full pipeline ownership chain. Useful for debugging."""
    chain = get_chain()
    print(f"\n{'═' * 72}")
    print("  PIPELINE CHAIN")
    print(f"{'═' * 72}")
    for role, info in chain.items():
        print(f"\n  [{role}]  {info.script}")
        print(f"    consumes : {[Path(str(p)).name for p in info.consumes] or '(none)'}")
        print(f"    produces : {[Path(str(p)).name for p in info.produces] or '(none)'}")
        print(f"    next     : {info.next_step or '(end)'}")
    print(f"\n{'═' * 72}\n")


if __name__ == "__main__":
    print_chain()
