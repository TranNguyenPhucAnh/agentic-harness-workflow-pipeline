"""
modules/artifact_tracking.py — Artifact read/write tracking for pipeline scripts.

Replaces the boilerplate block duplicated in every pipeline script:

    _ARTIFACTS_READ: set[str] = set()
    _ARTIFACTS_WRITTEN: set[str] = set()
    def _track_read(path): ...
    def _track_write(path): ...
    def _print_artifact_access_summary(): ...

Usage per script
────────────────
    from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary

    # when reading a pipeline artifact:
    track_read(SPEC_PATH)

    # when writing a pipeline artifact:
    track_write(MANIFEST_PATH)

    # in finally of main():
    print_artifact_summary("[08]")

Design note
───────────
Module-level sets are safe here because harness launches each script in a
separate subprocess (subprocess.run), so state never leaks across scripts.
Call reset() only in tests or multi-run single-process scenarios.
"""

from __future__ import annotations

from typing import Any


# ─── Module-level state ───────────────────────────────────────────────────────

_read:    set[str] = set()
_written: set[str] = set()


# ─── Public API ───────────────────────────────────────────────────────────────

def track_read(path: Any) -> None:
    """Record a pipeline artifact as read by this script."""
    _read.add(str(path))


def track_write(path: Any) -> None:
    """Record a pipeline artifact as written/created/updated by this script."""
    _written.add(str(path))


def print_summary(prefix: str = "[pipeline]") -> None:
    """
    Print the read/write summary for this script run.

    Call this in the finally block of main() so it always prints even on error.

    Example output:
        [08] Artifacts read:
        [08]   READ  /project/artifacts_foo/specwright_spec_foo.md
        [08] Artifacts created/updated/overwritten/appended:
        [08]   WRITE /project/artifacts_foo/execution/executor_overwrite_manifest.json
    """
    print(f"{prefix} Artifacts read:")
    if _read:
        for item in sorted(_read):
            print(f"{prefix}   READ  {item}")
    else:
        print(f"{prefix}   READ  (none)")

    print(f"{prefix} Artifacts created/updated/overwritten/appended:")
    if _written:
        for item in sorted(_written):
            print(f"{prefix}   WRITE {item}")
    else:
        print(f"{prefix}   WRITE (none)")


def get_read() -> frozenset[str]:
    """Return a snapshot of all paths read this run."""
    return frozenset(_read)


def get_written() -> frozenset[str]:
    """Return a snapshot of all paths written this run."""
    return frozenset(_written)


def reset() -> None:
    """
    Reset all tracking state.

    Call at the start of a script when running multiple logical "runs"
    in one process (tests, REPL). Not needed in normal harness usage
    because each script runs in its own subprocess.
    """
    _read.clear()
    _written.clear()
