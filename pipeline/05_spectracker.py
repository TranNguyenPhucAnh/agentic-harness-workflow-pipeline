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
    artifacts_<slug>/specwright_spec_<slug>.md
    artifacts_<slug>/state/spectracker_applied_version.json
    artifacts_<slug>/knowledge/history/<version>.md
    artifacts_<slug>/knowledge/history/spectracker_version_log.md

Writes:
    artifacts_<slug>/sessions/<NNN>/cache/spectracker_overwrite_version_delta.json
    artifacts_<slug>/knowledge/history/<version>.md                    ← write-once
    artifacts_<slug>/knowledge/history/spectracker_version_log.md      ← append-only

Finalization write (called by harness only, after successful full-scope run):
    artifacts_<slug>/state/spectracker_applied_version.json

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
from pathlib import Path
from typing import Any

# === WRITE AUTHORITY: spectracker ===
# OWNS  : artifacts_<slug>/sessions/<NNN>/cache/spectracker_overwrite_version_delta.json
#         artifacts_<slug>/state/spectracker_applied_version.json
#         artifacts_<slug>/knowledge/history/spectracker_version_log.md
#         artifacts_<slug>/knowledge/history/<version>.md   (dynamic, write-once)
# READS : artifacts_<slug>/specwright_spec_<slug>.md
#         artifacts_<slug>/state/spectracker_applied_version.json
#         artifacts_<slug>/knowledge/history/<version>.md   (dynamic snapshots)
#         artifacts_<slug>/knowledge/history/spectracker_version_log.md

import sys as _sys

_sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    HISTORY_DIR,
    SPECTRACKER_APPLIED,
    SPECTRACKER_VERSION_DELTA,
    SPECTRACKER_VERSION_LOG,
    ensure_dirs,
    get_spec_path,
)
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402

KNOWLEDGE_HISTORY_DIR = HISTORY_DIR
CHANGELOG            = SPECTRACKER_VERSION_LOG
DELTA_OUT            = SPECTRACKER_VERSION_DELTA
APPLIED_PATH         = SPECTRACKER_APPLIED


# ════════════════════════════════════════════════════════════════════════════
# Artifact access tracking
# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

ROLE = "spectracker"

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


def _safe_version_filename(version: str) -> str:
    """
    Keep version-based dynamic filenames safe.

    Allows common semver-ish/version characters while replacing path separators
    and other unsafe chars with '-'.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "-", version.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


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
        key   = m.group(2)
        title = f"{m.group(2)}. {m.group(3).strip()}"
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
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
    from_version:      str | None
    to_version:        str
    is_first_run:      bool
    changed_sections:  list[str]
    unchanged_sections: list[str]
    new_sections:      list[str]
    removed_sections:  list[str]
    section_summaries: dict[str, str]
    baseline_source:   str | None = None


def _summarise_change(key: str, old_content: str, new_content: str) -> str:
    _ = key

    old_lines = set(old_content.splitlines())
    new_lines  = set(new_content.splitlines())

    added   = [line.strip() for line in (new_lines - old_lines) if line.strip()]
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
    current_ver  = parse_spec_version(current_text)
    current_secs = parse_sections(current_text)

    is_first_run = previous_text is None
    prev_ver     = parse_spec_version(previous_text) if previous_text else None
    prev_secs    = parse_sections(previous_text) if previous_text else {}

    changed:   list[str] = []
    unchanged: list[str] = []
    new_secs:  list[str] = []
    removed:   list[str] = []
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
# History management
# ════════════════════════════════════════════════════════════════════════════

def _snapshot_path(version: str) -> Path:
    # Naming: spectracker_spec_snapshot_<version>.md
    #   owner prefix : spectracker (naming_rules Rule 1+2)
    #   semantic     : spec_snapshot — write-once raw spec content at this version
    #   lifecycle    : write-once exception — NOT _overwrite_ (not per-run), NOT _log (not append)
    #                  documented in TAXONOMY.md + OWNERSHIP.md Special Notes
    #   extension    : .md — human-readable spec content (Rule 3)
    return KNOWLEDGE_HISTORY_DIR / f"spectracker_spec_snapshot_{_safe_version_filename(version)}.md"


def _save_snapshot_write_once(version: str, text: str) -> tuple[Path, bool]:
    """
    Save raw spec snapshot to knowledge/history/<ver>.md.

    Write-once semantics: if absent, create; if present, leave untouched.
    Returns (path, created).
    """
    KNOWLEDGE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(version)

    if path.exists():
        track_read(path)
        return path, False

    path.write_text(apply_md_header(text, path, owner="05_spectracker.py"))
    track_write(path)
    return path, True


def _load_latest_snapshot(exclude_version: str) -> tuple[str | None, str | None]:
    """Return (version, text) of the most recent snapshot != exclude_version."""
    if not KNOWLEDGE_HISTORY_DIR.exists():
        return None, None

    exclude_stem = f"spectracker_spec_snapshot_{_safe_version_filename(exclude_version)}"

    # Only match files following the new naming: spectracker_spec_snapshot_<version>.md
    # This avoids accidentally picking up spectracker_version_log.md or other .md files.
    snapshots = sorted(
        KNOWLEDGE_HISTORY_DIR.glob("spectracker_spec_snapshot_*.md"),
        key=lambda p: p.stem,
    )

    for snap in reversed(snapshots):
        if snap.stem != exclude_stem:
            track_read(snap)
            return snap.stem.removeprefix("spectracker_spec_snapshot_"), snap.read_text()

    return None, None


def _load_snapshot(version: str) -> str | None:
    path = _snapshot_path(version)

    if not path.exists():
        return None

    track_read(path)
    return path.read_text()


def _determine_baseline(
    current_ver: str,
    from_version: str | None,
) -> tuple[str | None, str]:
    """
    Return (previous_text, baseline_source).

    Priority:
      1. --from <version>  (explicit override)
      2. last applied version snapshot
      3. latest snapshot (any version != current)
      4. None (first run)
    """
    if from_version:
        prev_text = _load_snapshot(from_version)

        if prev_text is None:
            raise FileNotFoundError(f"snapshot {from_version} not found")

        return prev_text, f"--from {from_version}"

    last_applied = get_last_applied_version()

    if last_applied and last_applied != current_ver:
        prev_text = _load_snapshot(last_applied)

        if prev_text is not None:
            return prev_text, f"last applied ({last_applied})"

        print(
            f"[spectracker] WARN: last applied snapshot {last_applied} not found; "
            "falling back to latest snapshot.",
            file=sys.stderr,
        )

        _, fallback_text = _load_latest_snapshot(exclude_version=current_ver)
        return fallback_text, "latest snapshot"

    _, prev_text = _load_latest_snapshot(exclude_version=current_ver)
    return prev_text, "latest snapshot"


# ════════════════════════════════════════════════════════════════════════════
# Changelog (aggregate log only)
# ════════════════════════════════════════════════════════════════════════════

def _changelog_has_version(version: str) -> bool:
    if not CHANGELOG.exists():
        return False

    track_read(CHANGELOG)
    content = CHANGELOG.read_text()
    pattern = rf"^## \[{re.escape(version)}\]\s+—"
    return bool(re.search(pattern, content, flags=re.MULTILINE))


def _build_changelog_entry(delta: SpecDelta) -> str:
    from datetime import datetime, timezone

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: list[str] = [
        f"## [{delta.to_version}] — {date_str}",
        f"_(from {delta.from_version or 'initial'})_",
        "",
    ]

    if delta.changed_sections or delta.new_sections or delta.removed_sections:
        if delta.new_sections:
            lines.append("### Added")
            for sec in delta.new_sections:
                note = delta.section_summaries.get(sec, "new section")
                lines.append(f"- §{sec}: {note}")
            lines.append("")

        if delta.changed_sections:
            lines.append("### Changed")
            for sec in delta.changed_sections:
                note = delta.section_summaries.get(sec, "")
                lines.append(f"- §{sec}{': ' + note if note else ''}")
            lines.append("")

        if delta.removed_sections:
            lines.append("### Removed")
            for sec in delta.removed_sections:
                note = delta.section_summaries.get(sec, "section removed")
                lines.append(f"- §{sec}: {note}")
            lines.append("")
    else:
        lines += ["### No section changes detected", ""]

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def _append_changelog(delta: SpecDelta) -> bool:
    """
    Append entry to aggregate changelog if this version is not already present.

    Returns True if appended, False if already exists.
    """
    if _changelog_has_version(delta.to_version):
        return False

    entry = _build_changelog_entry(delta)
    existing = ""

    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)

    if CHANGELOG.exists():
        track_read(CHANGELOG)
        existing = CHANGELOG.read_text()

    CHANGELOG.write_text(existing + entry)
    track_write(CHANGELOG)
    return True


def print_changelog(n: int = 0) -> None:
    if not CHANGELOG.exists():
        print("[spectracker] No changelog yet.")
        return

    track_read(CHANGELOG)
    content = CHANGELOG.read_text()

    if not n:
        print(content)
        return

    entries = re.split(r"(?=^## \[)", content, flags=re.MULTILINE)
    entries = [e for e in entries if e.strip()]

    for entry in entries[-n:]:
        print(entry)


# ════════════════════════════════════════════════════════════════════════════
# Applied state
# ════════════════════════════════════════════════════════════════════════════

def load_applied() -> dict | None:
    if not APPLIED_PATH.exists():
        return None

    try:
        track_read(APPLIED_PATH)
        return json.loads(APPLIED_PATH.read_text())
    except Exception:
        return None


def get_last_applied_version() -> str | None:
    applied = load_applied()
    return applied.get("last_applied_version") if applied else None


def write_applied(
    version: str,
    status: str = "PASS",
) -> None:
    """
    Mark a spec version as applied.

    Intended caller:
        harness.py finalization, after downstream full-scope pipeline success.

    Schema (hybrid — top-level fields overwrite; run_history[] appends):
        last_applied_version: str
        applied_at: ISO timestamp
        final_status: "PASS" | "FAIL"
        run_history: [{version, applied_at, status}]
    """
    from datetime import datetime, timezone

    status = status.upper().strip()
    now    = datetime.now(timezone.utc).isoformat()

    applied = load_applied() or {"run_history": []}

    applied["last_applied_version"] = version
    applied["applied_at"]           = now
    applied["final_status"]         = status

    run_history: list[dict[str, Any]] = applied.get("run_history", [])
    run_history.append(
        {
            "version":    version,
            "applied_at": now,
            "status":     status,
        }
    )
    applied["run_history"] = run_history

    APPLIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPLIED_PATH.write_text(json.dumps(applied, indent=2))
    track_write(APPLIED_PATH)


def print_run_history() -> None:
    applied = load_applied()

    if not applied:
        print("[spectracker] No run history yet.")
        return

    history = applied.get("run_history", [])
    print(f"\n[spectracker] Run history ({len(history)} run(s)):")

    for entry in history:
        icon       = "✅" if entry.get("status") == "PASS" else "❌"
        version    = entry.get("version", "?")
        applied_at = entry.get("applied_at", "?")[:19]
        print(f"  {icon} {version}  {applied_at}")


# ════════════════════════════════════════════════════════════════════════════
# CLI / Main
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="05_spectracker.py",
        description="Spec diff engine — section-level version tracking.",
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
        help="Print spectracker_version_log.md and run history, then exit.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        metavar="N",
        help="With --history: show only last N changelog entries. 0 = all.",
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

    print(
        f"[spectracker] Applied version updated → {APPLIED_PATH} "
        f"({version}, {args.status})"
    )


def main() -> None:
    exit_code = 0

    try:
        parser = _build_parser()
        args   = parser.parse_args()

        _configure_project(args.project, parser)

        # Important: do not call ensure_dirs() at import-time.
        # PIPELINE_PROJECT must be available before artifact paths are resolved.
        ensure_dirs()

        if args.history:
            print_changelog(n=args.last)
            print_run_history()
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
        current_ver  = parse_spec_version(current_text)

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

        # Write session delta.
        DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
        DELTA_OUT.write_text(json.dumps(asdict(delta), indent=2))
        track_write(DELTA_OUT)
        print(f"[spectracker] Delta     → {DELTA_OUT}")

        # Write-once raw spec snapshot.
        snapshot_path, snapshot_created = _save_snapshot_write_once(
            current_ver,
            current_text,
        )
        if snapshot_created:
            print(f"[spectracker] Snapshot  → {snapshot_path}")
        else:
            print(f"[spectracker] Snapshot  → {snapshot_path} (exists; write-once)")

        # Append to aggregate changelog only when there are section changes.
        has_section_delta = bool(
            delta.is_first_run
            or delta.changed_sections
            or delta.new_sections
            or delta.removed_sections
        )

        if has_section_delta:
            appended = _append_changelog(delta)
            if appended:
                print(f"[spectracker] Changelog → {CHANGELOG} (entry for {delta.to_version})")
            else:
                print(f"[spectracker] Changelog → {CHANGELOG} (entry for {delta.to_version} already exists)")
        else:
            print("[spectracker] No section changes detected; changelog unchanged.")

    except SystemExit as exc:
        code      = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[spectracker][error] {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        print_artifact_summary("[05]")
        prompt_next_step(ROLE, prefix="[05]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
