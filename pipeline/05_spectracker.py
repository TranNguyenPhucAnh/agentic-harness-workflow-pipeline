"""
pipeline/05_spectracker.py
==========================
Spec diff engine — detect what changed between spec versions.

Single responsibility: version tracking and section-level diff.
Does NOT know about file structure, pipeline topology, or project conventions.

Canonical position in full flow:
    01_absorber
    02_clarificator
    03_enricher
    04_specwright
    05_spectracker   ← runs after canonical spec exists
    06_scaffolder
    07_planner
    08_executor
    09_debugger
    10_reporter
    11_judge
    12_patcher
    13_archivist

Reads:
    artifacts_<slug>/spec/specwright_spec_<slug>.md
    artifacts_<slug>/spectracker/version_log.json

Writes:
    artifacts_<slug>/spectracker/version_delta.json   ← short-term, overwrite per run
    artifacts_<slug>/spectracker/version_log.json     ← long-term, append-only
                                                        (snapshot + applied state merged)

Finalization write (called by harness only, after successful full-scope run):
    Updates applied=true on the relevant entry in version_log.json.

Important lifecycle note:
    Normal spectracker runs compute a proposed delta but DO NOT mark the version
    as applied. The harness calls write_applied() only after the downstream
    full-scope pipeline succeeds. This prevents a spec version from being marked
    applied before implementation/test/judge completion.

Direct execution:
    python 05_spectracker.py --project my-app
    PIPELINE_PROJECT=my-app python 05_spectracker.py

Show only, no writes:
    python 05_spectracker.py --project my-app --show

History:
    python 05_spectracker.py --project my-app --history
    python 05_spectracker.py --project my-app --history --last 3

Manual applied-state recovery/fallback:
    python 05_spectracker.py --project my-app --mark-applied --status PASS

Default missing-spec behavior:
    If specwright_spec_<slug>.md does not exist, spectracker skips cleanly and
    exits 0. Use --strict to turn missing spec into exit 1.

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# === WRITE AUTHORITY: spectracker ===
# OWNS  : artifacts_<slug>/spectracker/version_delta.json    (short-term, overwrite)
#          artifacts_<slug>/spectracker/version_log.json     (long-term, append + update applied)
# READS : artifacts_<slug>/spec/specwright_spec_<slug>.md
#          artifacts_<slug>/spectracker/version_log.json     (self-read for delta baseline)

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    SPECTRACKER_VERSION_DELTA,
    SPECTRACKER_VERSION_LOG,
    ensure_dirs,
    get_spec_path,
)
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

ROLE = "spectracker"
DELTA_OUT = SPECTRACKER_VERSION_DELTA
VERSION_LOG = SPECTRACKER_VERSION_LOG


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

def _configure_project(
    project: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 05_spectracker.py directly."
    )


# ════════════════════════════════════════════════════════════════════════════
# Section parser
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecSection:
    key: str      # "4.3", "10", "0" etc.
    title: str    # "4.3 Some Title"
    content: str  # full text of the section: header + body
    hash: str     # sha256 of content for change detection


def _section_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode()).hexdigest()[:16]


def parse_spec_version(text: str) -> str:
    """Extract version from spec header comment."""
    m = re.search(r"^#\s*Version:\s*(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def _section_sort_key(key: str) -> list[int]:
    """Sort section keys like 4, 4.3, 10 numerically."""
    try:
        return [int(x) for x in key.split(".")]
    except Exception:
        return [999999]


def parse_sections(text: str) -> dict[str, SpecSection]:
    """
    Parse spec.md into sections keyed by number, e.g. "4", "4.3", "10".

    Handles:
      ## 4. Title
      ### 4.3 Title
    """
    header_re = re.compile(
        r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.\s+(.+)$",
        re.MULTILINE,
    )

    matches = list(header_re.finditer(text))
    sections: dict[str, SpecSection] = {}

    for i, m in enumerate(matches):
        key = m.group(2)
        title = f"{m.group(2)}. {m.group(3).strip()}"
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        sections[key] = SpecSection(
            key=key,
            title=title,
            content=content,
            hash=_section_hash(content),
        )

    return sections


# ════════════════════════════════════════════════════════════════════════════
# Core diff logic
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecDelta:
    from_version: str | None
    to_version: str
    is_first_run: bool
    changed_sections: list[str]
    unchanged_sections: list[str]
    new_sections: list[str]
    removed_sections: list[str]
    section_summaries: dict[str, str]
    baseline_source: str | None = None


def _summarise_change(key: str, old_content: str, new_content: str) -> str:
    _ = key

    old_lines = set(old_content.splitlines())
    new_lines = set(new_content.splitlines())

    added = [line.strip() for line in (new_lines - old_lines) if line.strip()]
    removed = [line.strip() for line in (old_lines - new_lines) if line.strip()]

    prop_added = [
        line for line in added
        if line.startswith(("export ", "interface ", "type ", "  ")) and ":" in line
    ]
    prop_removed = [
        line for line in removed
        if line.startswith(("export ", "interface ", "type ", "  ")) and ":" in line
    ]

    parts: list[str] = []

    if prop_added:
        parts.append(f"added: {prop_added[0][:80]}")
    if prop_removed:
        parts.append(f"removed: {prop_removed[0][:80]}")
    if not parts and added:
        parts.append(f"+{len(added)} line(s)")
    if not parts and removed:
        parts.append(f"-{len(removed)} line(s)")
    if not parts:
        parts.append("content changed")

    return "; ".join(parts)


def compute_delta(
    current_text: str,
    previous_text: str | None,
    baseline_source: str | None = None,
) -> SpecDelta:
    current_ver = parse_spec_version(current_text)
    current_secs = parse_sections(current_text)

    is_first_run = previous_text is None
    prev_ver = parse_spec_version(previous_text) if previous_text else None
    prev_secs = parse_sections(previous_text) if previous_text else {}

    changed: list[str] = []
    unchanged: list[str] = []
    new_secs: list[str] = []
    removed: list[str] = []
    summaries: dict[str, str] = {}

    if is_first_run:
        changed = sorted(current_secs.keys(), key=_section_sort_key)
        for key in changed:
            summaries[key] = "initial section"
    else:
        all_keys = set(current_secs) | set(prev_secs)

        for key in sorted(all_keys, key=_section_sort_key):
            if key not in prev_secs:
                new_secs.append(key)
                summaries[key] = "new section"
            elif key not in current_secs:
                removed.append(key)
                summaries[key] = "section removed"
            elif current_secs[key].hash != prev_secs[key].hash:
                changed.append(key)
                summaries[key] = _summarise_change(
                    key,
                    prev_secs[key].content,
                    current_secs[key].content,
                )
            else:
                unchanged.append(key)

    return SpecDelta(
        from_version=prev_ver,
        to_version=current_ver,
        is_first_run=is_first_run,
        changed_sections=changed,
        unchanged_sections=unchanged,
        new_sections=new_secs,
        removed_sections=removed,
        section_summaries=summaries,
        baseline_source=baseline_source,
    )


def _print_delta_summary(delta: SpecDelta) -> None:
    print(f"[spectracker] {delta.from_version or '(none)'} → {delta.to_version}")

    if delta.baseline_source:
        print(f"[spectracker] Baseline: {delta.baseline_source}")

    if delta.is_first_run:
        print("[spectracker] First run — full downstream pipeline required.")
    else:
        print(f"[spectracker] Changed  §: {delta.changed_sections or '(none)'}")
        print(f"[spectracker] New      §: {delta.new_sections or '(none)'}")
        print(f"[spectracker] Removed  §: {delta.removed_sections or '(none)'}")

        for sec in delta.changed_sections:
            if sec in delta.section_summaries:
                print(f"    §{sec}: {delta.section_summaries[sec]}")


# ════════════════════════════════════════════════════════════════════════════
# version_log.json management
# ════════════════════════════════════════════════════════════════════════════
#
# Schema: { "entries": [ <VersionLogEntry>, ... ] }
#
# Each entry:
# {
#   "version": "v1.3",
#   "generated_at": "2026-05-16T08:00:00Z",
#   "applied": false,
#   "applied_at": null,
#   "changed_sections": ["2", "4"],
#   "affected_files": ["src/auth/service.py"],
#   "spec_content": "..."   ← full spec text at this version (replaces per-file snapshots)
# }
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class VersionLogEntry:
    version: str
    generated_at: str
    applied: bool
    applied_at: str | None
    changed_sections: list[str]
    affected_files: list[str]
    spec_content: str


def _load_version_log() -> list[dict[str, Any]]:
    """Load version_log.json entries. Returns [] if file missing or corrupt."""
    if not VERSION_LOG.exists():
        return []

    try:
        track_read(VERSION_LOG)
        data = json.loads(Path(VERSION_LOG).read_text())
        if isinstance(data, dict):
            return data.get("entries", [])
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _save_version_log(entries: list[dict[str, Any]]) -> None:
    """Write version_log.json atomically."""
    VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    Path(VERSION_LOG).write_text(json.dumps({"entries": entries}, indent=2))
    track_write(VERSION_LOG)


def _find_entry_by_version(
    entries: list[dict[str, Any]],
    version: str,
) -> dict[str, Any] | None:
    """Find the latest entry matching a version string."""
    for entry in reversed(entries):
        if entry.get("version") == version:
            return entry
    return None


def _append_version_log_entry(
    delta: SpecDelta,
    spec_content: str,
    affected_files: list[str] | None = None,
) -> bool:
    """
    Append a new entry to version_log.json for the current version.

    Returns True if appended, False if this version already has an entry
    (write-once per version semantics for the snapshot).
    """
    entries = _load_version_log()

    # Write-once: don't duplicate if version already logged
    if _find_entry_by_version(entries, delta.to_version) is not None:
        return False

    now = datetime.now(timezone.utc).isoformat()

    # Merge changed + new sections as the "changed" set for downstream consumers
    all_changed = sorted(
        set(delta.changed_sections + delta.new_sections),
        key=_section_sort_key,
    )

    entry: dict[str, Any] = {
        "version": delta.to_version,
        "generated_at": now,
        "applied": False,
        "applied_at": None,
        "changed_sections": all_changed,
        "affected_files": affected_files or [],
        "spec_content": spec_content,
    }

    entries.append(entry)
    _save_version_log(entries)
    return True


# ════════════════════════════════════════════════════════════════════════════
# Applied state (merged into version_log.json)
# ════════════════════════════════════════════════════════════════════════════

def get_last_applied_version() -> str | None:
    """Return the version string of the most recently applied entry, or None."""
    entries = _load_version_log()

    for entry in reversed(entries):
        if entry.get("applied"):
            return entry.get("version")

    return None


def load_applied() -> dict[str, Any] | None:
    """
    Backward-compatible: return a dict resembling the old applied state.
    Returns None if no version has been applied yet.
    """
    entries = _load_version_log()

    for entry in reversed(entries):
        if entry.get("applied"):
            return {
                "last_applied_version": entry["version"],
                "applied_at": entry.get("applied_at"),
                "final_status": "PASS",
            }

    return None


def write_applied(
    version: str,
    status: str = "PASS",
) -> None:
    """
    Mark a spec version as applied by updating its entry in version_log.json.

    Intended caller:
        harness.py finalization, after downstream full-scope pipeline success.

    If the version entry doesn't exist yet (edge case: manual --mark-applied
    before a normal run), creates a minimal entry.
    """
    status = status.upper().strip()
    now = datetime.now(timezone.utc).isoformat()

    entries = _load_version_log()
    entry = _find_entry_by_version(entries, version)

    if entry is not None:
        entry["applied"] = True
        entry["applied_at"] = now
    else:
        # Create minimal entry for manual mark-applied
        entries.append({
            "version": version,
            "generated_at": now,
            "applied": True,
            "applied_at": now,
            "changed_sections": [],
            "affected_files": [],
            "spec_content": "",
        })

    _save_version_log(entries)
    print(f"[spectracker] Applied: {version} ({status}) at {now}")


# ════════════════════════════════════════════════════════════════════════════
# Baseline resolution (replaces old snapshot file approach)
# ════════════════════════════════════════════════════════════════════════════

def _determine_baseline(
    current_ver: str,
    from_version: str | None,
) -> tuple[str | None, str]:
    """
    Return (previous_text, baseline_source).

    Priority:
      1. --from <version>  (explicit override) — lookup in version_log.json
      2. last applied version's spec_content from version_log.json
      3. latest entry in version_log.json != current version
      4. None (first run)
    """
    entries = _load_version_log()

    if from_version:
        entry = _find_entry_by_version(entries, from_version)
        if entry is None or not entry.get("spec_content"):
            raise FileNotFoundError(
                f"Version '{from_version}' not found in version_log.json "
                f"or has empty spec_content."
            )
        return entry["spec_content"], f"--from {from_version}"

    # Try last applied version
    last_applied = get_last_applied_version()
    if last_applied and last_applied != current_ver:
        entry = _find_entry_by_version(entries, last_applied)
        if entry and entry.get("spec_content"):
            return entry["spec_content"], f"last applied ({last_applied})"

        # Fallback: latest entry != current
        print(
            f"[spectracker] WARN: last applied version {last_applied} has no "
            "spec_content; falling back to latest entry.",
            file=sys.stderr,
        )

    # Fallback: latest entry with spec_content != current version
    for entry in reversed(entries):
        if entry.get("version") != current_ver and entry.get("spec_content"):
            return entry["spec_content"], f"latest logged ({entry['version']})"

    # No baseline found — first run
    return None, "first run (no baseline)"


# ════════════════════════════════════════════════════════════════════════════
# History display
# ════════════════════════════════════════════════════════════════════════════

def print_history(n: int = 0) -> None:
    """Print version_log.json entries in human-readable format."""
    entries = _load_version_log()

    if not entries:
        print("[spectracker] No version history yet.")
        return

    display = entries[-n:] if n > 0 else entries

    print(f"\n[spectracker] Version history ({len(entries)} total, showing {len(display)}):")
    print("-" * 60)

    for entry in display:
        version = entry.get("version", "?")
        generated = entry.get("generated_at", "?")[:19]
        applied = entry.get("applied", False)
        applied_at = (entry.get("applied_at") or "")[:19]
        changed = entry.get("changed_sections", [])

        icon = "✅" if applied else "⏳"
        applied_str = f"  applied {applied_at}" if applied else "  (pending)"

        print(f"  {icon} {version}  generated {generated}{applied_str}")
        if changed:
            print(f"      changed §: {changed}")

    print()


# ════════════════════════════════════════════════════════════════════════════
# CLI / Main
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="05_spectracker.py",
        description="Spec diff engine — section-level version tracking.",
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Run as a silent preflight check (called by harness before mid-pipeline steps). "
            "Skips post-run prompts and long-term artifact commit."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "Project name for direct execution. Sets PIPELINE_PROJECT before "
            "resolving artifact paths."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print delta summary to stdout and exit without writing delta/history.",
    )
    parser.add_argument(
        "--from",
        dest="from_version",
        help="Force compare against a specific version snapshot.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Print version history and applied state, then exit.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        metavar="N",
        help="With --history: show only last N entries. 0 = all.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing canonical spec as an error instead of a clean skip.",
    )
    parser.add_argument(
        "--mark-applied",
        action="store_true",
        help=(
            "Mark the current or supplied spec version as applied. Intended as "
            "manual fallback; harness calls write_applied() automatically after "
            "successful full-scope completion."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help=(
            "Version to use with --mark-applied. If omitted, spectracker reads "
            "the canonical spec and extracts # Version."
        ),
    )
    parser.add_argument(
        "--status",
        default="PASS",
        choices=["PASS", "FAIL"],
        help="Status to record with --mark-applied.",
    )

    return parser


def _handle_mark_applied(args: argparse.Namespace) -> None:
    version = args.version

    if not version:
        spec_path = get_spec_path()
        if not spec_path.exists():
            print(
                f"[spectracker] ERROR: cannot infer version because "
                f"{spec_path} does not exist. Pass --version explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)

        track_read(spec_path)
        version = parse_spec_version(spec_path.read_text())

    write_applied(version=version, status=args.status)


def main() -> None:
    exit_code = 0

    try:
        parser = _build_parser()
        args = parser.parse_args()

        _configure_project(args.project, parser)

        # Important: do not call ensure_dirs() at import-time.
        # PIPELINE_PROJECT must be available before artifact paths are resolved.
        ensure_dirs()

        if args.history:
            print_history(n=args.last)
            return

        if args.mark_applied:
            _handle_mark_applied(args)
            return

        spec_path = get_spec_path()
        if not spec_path.exists():
            msg = (
                f"[spectracker] SKIP: canonical spec not found: {spec_path}. "
                "Run specwright first."
            )

            if args.strict:
                print(msg.replace("SKIP", "ERROR"), file=sys.stderr)
                sys.exit(1)

            print(msg)
            return

        track_read(spec_path)
        current_text = spec_path.read_text()
        current_ver = parse_spec_version(current_text)

        try:
            prev_text, baseline_source = _determine_baseline(
                current_ver=current_ver,
                from_version=args.from_version,
            )
        except FileNotFoundError as exc:
            print(f"[spectracker] ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        delta = compute_delta(
            current_text=current_text,
            previous_text=prev_text,
            baseline_source=baseline_source,
        )

        _print_delta_summary(delta)

        if args.show:
            return

        # ── Write short-term: version_delta.json (overwrite per run) ──
        DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
        Path(DELTA_OUT).write_text(json.dumps(asdict(delta), indent=2))
        track_write(DELTA_OUT)
        print(f"[spectracker] Delta       → {DELTA_OUT}")

        # ── Append long-term: version_log.json (write-once per version) ──
        #
        # Preflight mode: auto-append only if there is a real version bump
        # (changed/new/removed sections). No prompt — silent.
        # Normal mode: always append (post_interactive handles keep/discard prompt).
        is_preflight    = getattr(args, "preflight", False)
        has_version_bump = bool(
            delta.changed_sections
            or delta.new_sections
            or delta.removed_sections
        )

        if is_preflight and not has_version_bump:
            print(f"[spectracker] Version log → skipped (preflight, no changes)")
        else:
            appended = _append_version_log_entry(
                delta=delta,
                spec_content=current_text,
                affected_files=[],  # populated by downstream consumers if needed
            )
            if appended:
                label = " (auto-appended, preflight)" if is_preflight else f" (new entry: {delta.to_version})"
                print(f"[spectracker] Version log → {VERSION_LOG}{label}")
            else:
                print(f"[spectracker] Version log → {VERSION_LOG} (entry for {delta.to_version} already exists)")

        # Summary
        has_section_delta = bool(
            delta.is_first_run
            or delta.changed_sections
            or delta.new_sections
            or delta.removed_sections
        )

        if has_section_delta:
            print(f"[spectracker] Section changes detected — downstream rerun required.")
        else:
            print("[spectracker] No section changes detected; downstream can skip.")

    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[spectracker][error] {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        print_artifact_summary("[05]")
        if not getattr(args, "preflight", False):
            prompt_next_step(ROLE, prefix="[05]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()