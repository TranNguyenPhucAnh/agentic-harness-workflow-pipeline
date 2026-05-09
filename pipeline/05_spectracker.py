"""
pipeline/05_spectracker.py
==========================
Spec diff engine — detect what changed between spec versions and map to affected files.

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
    artifacts_<slug>/cache/spectracker_session_version_delta.json
    artifacts_<slug>/knowledge/history/<version>.md                    ← write-once
    artifacts_<slug>/knowledge/history/<version>.changelog.md          ← write-once
    artifacts_<slug>/knowledge/history/spectracker_version_log.md      ← append-only

Finalization write:
    artifacts_<slug>/state/spectracker_applied_version.json

Important lifecycle note:
    Normal spectracker runs compute a proposed delta but DO NOT mark the version
    as applied. The harness should call write_applied() only after the downstream
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
    python 05_spectracker.py --project my-app --mark-applied --status PASS --steps scaffolder,planner,executor,debugger,reporter,judge

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
# OWNS  : artifacts_<slug>/cache/spectracker_session_version_delta.json
#         artifacts_<slug>/state/spectracker_applied_version.json
#         artifacts_<slug>/knowledge/history/spectracker_version_log.md
#         artifacts_<slug>/knowledge/history/<version>.md            (dynamic, write-once)
#         artifacts_<slug>/knowledge/history/<version>.changelog.md  (dynamic, write-once)
# READS : artifacts_<slug>/specwright_spec_<slug>.md
#         artifacts_<slug>/state/spectracker_applied_version.json
#         artifacts_<slug>/knowledge/history/<version>.md            (dynamic snapshots)
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


# Local aliases — map canonical constants to short internal names.
KNOWLEDGE_HISTORY_DIR = HISTORY_DIR
CHANGELOG = SPECTRACKER_VERSION_LOG
DELTA_OUT = SPECTRACKER_VERSION_DELTA
APPLIED_PATH = SPECTRACKER_APPLIED


# ════════════════════════════════════════════════════════════════════════════
# Artifact access tracking
# ════════════════════════════════════════════════════════════════════════════

_ARTIFACTS_READ: set[str] = set()
_ARTIFACTS_WRITTEN: set[str] = set()


def _track_read(path: Any) -> None:
    _ARTIFACTS_READ.add(str(path))


def _track_write(path: Any) -> None:
    _ARTIFACTS_WRITTEN.add(str(path))


def _print_artifact_access_summary() -> None:
    print("[spectracker] Artifacts read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[spectracker]   READ  {item}")
    else:
        print("[spectracker]   READ  (none)")

    print("[spectracker] Artifacts created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[spectracker]   WRITE {item}")
    else:
        print("[spectracker]   WRITE (none)")


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

def _configure_project(
    project: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    """
    Configure project context for direct execution.

    Harness normally sets PIPELINE_PROJECT before invoking this script.
    Direct usage can pass --project.
    """
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
    title: str    # "4.3 `AnomalyFeed`"
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
# File → section mapping
# ════════════════════════════════════════════════════════════════════════════

_STATIC_SECTION_FILE_MAP: dict[str, list[str]] = {
    "3": [
        "src/App.tsx",
        "src/main.tsx",
    ],
    "4.1": [
        "src/components/SummaryStickyBar.tsx",
        "tests/components/SummaryStickyBar.test.tsx",
    ],
    "4.2": [
        "src/components/ReplayControls.tsx",
        "tests/components/ReplayControls.test.tsx",
    ],
    "4.3": [
        "src/components/AnomalyFeed.tsx",
        "tests/components/AnomalyFeed.test.tsx",
    ],
    "4.4": [
        "src/components/ModelGates.tsx",
        "tests/components/ModelGates.test.tsx",
    ],
    "4.5": [
        "src/hooks/useSensorData.ts",
        "tests/hooks/useSensorData.test.ts",
    ],
    "4.6": [
        "src/hooks/useReplay.ts",
        "tests/hooks/useReplay.test.ts",
    ],
    "5": [
        "src/types/sensor.ts",
    ],
    "6": [
        "src/data/demoConstants.ts",
    ],
    "7": [],
    "10": [],
}

_SCAFFOLDER_TRIGGER_SECTIONS = {"7", "8"}
_TEST_ONLY_TRIGGER_SECTIONS = {"10"}
_IGNORED_SECTIONS = {"0", "1", "2", "9", "11"}


def _test_file_for_src(fp: str) -> str:
    """
    Derive a test file path from a src/ file path.

    Examples:
        src/hooks/useReplay.ts       → tests/hooks/useReplay.test.ts
        src/components/Card.tsx      → tests/components/Card.test.tsx
    """
    test_fp = fp.replace("src/", "tests/", 1)
    test_fp = re.sub(r"\.(tsx?)$", r".test.\1", test_fp)
    return test_fp


def _dedupe_sorted(items: list[str]) -> list[str]:
    return sorted(dict.fromkeys(items))


def _extract_file_map_from_spec(
    sections: dict[str, SpecSection],
) -> dict[str, list[str]]:
    """
    Build section → files map.

    Starts with the static known mapping, then augments it with explicit spec
    declarations of the form:

        **File:** `src/foo/bar.ts`
    """
    file_map = {k: list(v) for k, v in _STATIC_SECTION_FILE_MAP.items()}

    file_re = re.compile(r"\*\*File:\*\*\s+`(src/[^`]+)`")

    for key, section in sections.items():
        found = file_re.findall(section.content)
        for fp in found:
            existing = file_map.setdefault(key, [])

            if fp not in existing:
                existing.append(fp)

            test_fp = _test_file_for_src(fp)
            if test_fp not in existing:
                existing.append(test_fp)

    return {key: _dedupe_sorted(files) for key, files in file_map.items()}


def _merge_file_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}

    for fmap in maps:
        for key, files in fmap.items():
            bucket = merged.setdefault(key, [])
            for fp in files:
                if fp not in bucket:
                    bucket.append(fp)

    return {key: _dedupe_sorted(files) for key, files in merged.items()}


def _all_files_from_map(file_map: dict[str, list[str]]) -> list[str]:
    files: list[str] = []

    for mapped in file_map.values():
        for fp in mapped:
            if fp not in files:
                files.append(fp)

    return sorted(files)


def _files_for_changed_sections(
    impacted_sections: list[str],
    file_map: dict[str, list[str]],
    all_known_files: list[str],
) -> tuple[list[str], list[str]]:
    affected: set[str] = set()

    for key in impacted_sections:
        if key in _IGNORED_SECTIONS or key in _SCAFFOLDER_TRIGGER_SECTIONS:
            continue

        for fp in file_map.get(key, []):
            affected.add(fp)

    # Type contract changes affect hooks/components and their tests.
    if "5" in impacted_sections:
        for fp in all_known_files:
            if fp.startswith("src/hooks/") or fp.startswith("src/components/"):
                affected.add(fp)
                affected.add(_test_file_for_src(fp))

    # Demo constants/data changes affect hooks and their tests.
    if "6" in impacted_sections:
        for fp in all_known_files:
            if fp.startswith("src/hooks/"):
                affected.add(fp)
                affected.add(_test_file_for_src(fp))

    unaffected = [f for f in all_known_files if f not in affected]
    return sorted(affected), sorted(unaffected)


def _decide_rerun_steps(
    impacted_sections: list[str],
    affected_files: list[str],
    is_first_run: bool,
) -> dict[str, bool]:
    """
    Decide downstream harness steps.

    Canonical keys intentionally match harness step names:
      - scaffolder
      - planner
      - executor
      - debugger
      - reporter
      - judge
    """
    if is_first_run:
        return {
            "scaffolder": True,
            "planner": True,
            "executor": True,
            "debugger": True,
            "reporter": True,
            "judge": True,
        }

    scaffolder = bool(affected_files) or any(
        key in _SCAFFOLDER_TRIGGER_SECTIONS for key in impacted_sections
    )

    planner = bool(affected_files)
    executor = bool(affected_files)

    debugger = executor or any(
        key in _TEST_ONLY_TRIGGER_SECTIONS for key in impacted_sections
    )

    reporter = any((scaffolder, planner, executor, debugger))
    judge = debugger

    return {
        "scaffolder": scaffolder,
        "planner": planner,
        "executor": executor,
        "debugger": debugger,
        "reporter": reporter,
        "judge": judge,
    }


# ════════════════════════════════════════════════════════════════════════════
# History management
# ════════════════════════════════════════════════════════════════════════════

def _snapshot_path(version: str) -> Path:
    return KNOWLEDGE_HISTORY_DIR / f"{_safe_version_filename(version)}.md"


def _per_version_changelog_path(version: str) -> Path:
    return KNOWLEDGE_HISTORY_DIR / f"{_safe_version_filename(version)}.changelog.md"


def _save_snapshot_write_once(version: str, text: str) -> tuple[Path, bool]:
    """
    Save raw spec snapshot to knowledge/history/<ver>.md.

    Write-once semantics:
      - If absent, create it.
      - If present, leave untouched.
    Returns:
      (path, created)
    """
    KNOWLEDGE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(version)

    if path.exists():
        _track_read(path)
        return path, False

    path.write_text(text)
    _track_write(path)
    return path, True


def _load_latest_snapshot(exclude_version: str) -> tuple[str | None, str | None]:
    """Return (version, text) of the most recent snapshot != exclude_version."""
    if not KNOWLEDGE_HISTORY_DIR.exists():
        return None, None

    exclude_stem = _safe_version_filename(exclude_version)

    snapshots = sorted(
        [
            p
            for p in KNOWLEDGE_HISTORY_DIR.glob("*.md")
            if not p.name.endswith(".changelog.md")
            and p.name != "spectracker_version_log.md"
        ],
        key=lambda p: p.stem,
    )

    for snap in reversed(snapshots):
        ver = snap.stem
        if ver != exclude_stem:
            _track_read(snap)
            return ver, snap.read_text()

    return None, None


def _load_snapshot(version: str) -> str | None:
    path = _snapshot_path(version)

    if not path.exists():
        return None

    _track_read(path)
    return path.read_text()


# ════════════════════════════════════════════════════════════════════════════
# Applied state
# ════════════════════════════════════════════════════════════════════════════

def load_applied() -> dict | None:
    if not APPLIED_PATH.exists():
        return None

    try:
        _track_read(APPLIED_PATH)
        return json.loads(APPLIED_PATH.read_text())
    except Exception:
        return None


def get_last_applied_version() -> str | None:
    applied = load_applied()
    return applied.get("last_applied_version") if applied else None


def write_applied(
    version: str,
    steps: list[str] | None = None,
    status: str = "PASS",
) -> None:
    """
    Mark a spec version as applied.

    Intended caller:
        harness.py finalization, after downstream full-scope pipeline success.

    This function owns writes to:
        state/spectracker_applied_version.json

    Lifecycle:
        hybrid — top-level fields overwrite; run_history[] appends.
    """
    from datetime import datetime, timezone

    status = status.upper().strip()
    steps = steps or []

    now = datetime.now(timezone.utc).isoformat()

    applied = load_applied() or {"run_history": []}

    applied["last_applied_version"] = version
    applied["applied_at"] = now
    applied["applied_steps"] = steps
    applied["final_status"] = status

    run_history: list[dict[str, Any]] = applied.get("run_history", [])
    run_history.append(
        {
            "version": version,
            "applied_at": now,
            "status": status,
            "steps": steps,
        }
    )
    applied["run_history"] = run_history

    APPLIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPLIED_PATH.write_text(json.dumps(applied, indent=2))
    _track_write(APPLIED_PATH)


def print_run_history() -> None:
    applied = load_applied()

    if not applied:
        print("[spectracker] No run history yet.")
        return

    history = applied.get("run_history", [])
    print(f"\n[spectracker] Run history ({len(history)} run(s)):")

    for entry in history:
        icon = "✅" if entry.get("status") == "PASS" else "❌"
        steps = ", ".join(entry.get("steps", []))
        version = entry.get("version", "?")
        applied_at = entry.get("applied_at", "?")[:19]
        print(f"  {icon} {version}  {applied_at}  [{steps}]")


# ════════════════════════════════════════════════════════════════════════════
# Changelog management
# ════════════════════════════════════════════════════════════════════════════

def _changelog_has_version(version: str) -> bool:
    if not CHANGELOG.exists():
        return False

    _track_read(CHANGELOG)
    content = CHANGELOG.read_text()
    pattern = rf"^## \[{re.escape(version)}\]\s+—"
    return bool(re.search(pattern, content, flags=re.MULTILINE))


def _build_changelog_entry(delta: "SpecDelta") -> str:
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

    if delta.affected_files:
        lines.append("### Affected files")
        for fp in delta.affected_files:
            lines.append(f"- `{fp}`")
        lines.append("")

    rerun = [k for k, v in delta.rerun_steps.items() if v]
    if rerun:
        lines.append(f"### Steps re-run: {', '.join(rerun)}")
        lines.append("")
    else:
        lines.append("### Steps re-run: none")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def _append_changelog_write_once(delta: "SpecDelta") -> tuple[bool, bool]:
    """
    Append aggregate changelog entry and create per-version changelog.

    Write-once semantics:
      - Aggregate log appends entry only if this version is not already present.
      - Per-version changelog is created only if absent.

    Returns:
      (aggregate_appended, per_version_created)
    """
    entry = _build_changelog_entry(delta)

    aggregate_appended = False
    per_version_created = False

    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    if not _changelog_has_version(delta.to_version):
        existing = ""
        if CHANGELOG.exists():
            _track_read(CHANGELOG)
            existing = CHANGELOG.read_text()

        CHANGELOG.write_text(existing + entry)
        _track_write(CHANGELOG)
        aggregate_appended = True

    KNOWLEDGE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    cl_path = _per_version_changelog_path(delta.to_version)

    if cl_path.exists():
        _track_read(cl_path)
    else:
        cl_path.write_text(entry)
        _track_write(cl_path)
        per_version_created = True

    return aggregate_appended, per_version_created


def print_changelog(n: int = 0) -> None:
    if not CHANGELOG.exists():
        print("[spectracker] No changelog yet.")
        return

    _track_read(CHANGELOG)
    content = CHANGELOG.read_text()

    if not n:
        print(content)
        return

    entries = re.split(r"(?=^## \[)", content, flags=re.MULTILINE)
    entries = [e for e in entries if e.strip()]

    for entry in entries[-n:]:
        print(entry)


# ════════════════════════════════════════════════════════════════════════════
# Section summary generator
# ════════════════════════════════════════════════════════════════════════════

def _summarise_change(key: str, old_content: str, new_content: str) -> str:
    _ = key

    old_lines = set(old_content.splitlines())
    new_lines = set(new_content.splitlines())

    added = [line.strip() for line in (new_lines - old_lines) if line.strip()]
    removed = [line.strip() for line in (old_lines - new_lines) if line.strip()]

    prop_added = [
        line
        for line in added
        if line.startswith(("export ", "interface ", "type ", "  ")) and ":" in line
    ]
    prop_removed = [
        line
        for line in removed
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
    affected_files: list[str]
    unaffected_files: list[str]
    rerun_steps: dict[str, bool]
    section_summaries: dict[str, str]
    baseline_source: str | None = None


def compute_delta(
    current_text: str,
    previous_text: str | None,
    all_known_files: list[str] | None = None,
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
        unchanged = []
        new_secs = []
        removed = []

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

    current_file_map = _extract_file_map_from_spec(current_secs)
    previous_file_map = _extract_file_map_from_spec(prev_secs) if prev_secs else {}
    file_map = _merge_file_maps(previous_file_map, current_file_map)

    if all_known_files is None:
        all_known_files = _all_files_from_map(file_map)

    impacted_sections = sorted(
        set(changed + new_secs + removed),
        key=_section_sort_key,
    )

    affected, unaffected = _files_for_changed_sections(
        impacted_sections,
        file_map,
        all_known_files,
    )

    rerun = _decide_rerun_steps(
        impacted_sections,
        affected,
        is_first_run,
    )

    return SpecDelta(
        from_version=prev_ver,
        to_version=current_ver,
        is_first_run=is_first_run,
        changed_sections=changed,
        unchanged_sections=unchanged,
        new_sections=new_secs,
        removed_sections=removed,
        affected_files=affected,
        unaffected_files=unaffected,
        rerun_steps=rerun,
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

    print(f"[spectracker] Affected files   : {len(delta.affected_files)}")
    for fp in delta.affected_files:
        print(f"    {fp}")

    print(f"[spectracker] Unaffected files : {len(delta.unaffected_files)}")
    print(
        "[spectracker] Re-run steps     : "
        f"{[k for k, v in delta.rerun_steps.items() if v] or '(none)'}"
    )


def _determine_baseline(
    current_ver: str,
    from_version: str | None,
) -> tuple[str | None, str]:
    """
    Return (previous_text, baseline_source).
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
# CLI / Main
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="05_spectracker.py",
        description="Spec diff engine",
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

    # Applied-state finalization / recovery.
    parser.add_argument(
        "--mark-applied",
        action="store_true",
        help=(
            "Mark the current or supplied spec version as applied. Intended as "
            "manual fallback; harness should normally call write_applied() "
            "after successful full-scope completion."
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
    parser.add_argument(
        "--steps",
        default="",
        help=(
            "Comma-separated applied steps for --mark-applied, e.g. "
            "scaffolder,planner,executor,debugger,reporter,judge"
        ),
    )

    return parser


def _handle_mark_applied(args: argparse.Namespace) -> None:
    version = args.version

    if not version:
        spec_path = get_spec_path()
        if not spec_path.exists():
            msg = (
                f"[spectracker] ERROR: cannot infer version because "
                f"{spec_path} does not exist. Pass --version explicitly."
            )
            print(msg, file=sys.stderr)
            sys.exit(1)

        _track_read(spec_path)
        version = parse_spec_version(spec_path.read_text())

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    write_applied(
        version=version,
        steps=steps,
        status=args.status,
    )

    print(
        f"[spectracker] Applied version updated → {APPLIED_PATH} "
        f"({version}, {args.status})"
    )


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

        _track_read(spec_path)
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

        # Write session delta.
        DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
        DELTA_OUT.write_text(json.dumps(asdict(delta), indent=2))
        _track_write(DELTA_OUT)
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

        # Changelog only if this is an initial run, a new version, or there are changes.
        has_section_delta = bool(
            delta.is_first_run
            or delta.changed_sections
            or delta.new_sections
            or delta.removed_sections
        )

        if has_section_delta:
            aggregate_appended, per_version_created = _append_changelog_write_once(delta)

            if aggregate_appended:
                print(
                    f"[spectracker] Changelog → {CHANGELOG} "
                    f"(entry for {delta.to_version})"
                )
            else:
                print(
                    f"[spectracker] Changelog → {CHANGELOG} "
                    f"(entry for {delta.to_version} already exists)"
                )

            cl_path = _per_version_changelog_path(delta.to_version)
            if per_version_created:
                print(f"[spectracker] Per-version changelog → {cl_path}")
            else:
                print(f"[spectracker] Per-version changelog → {cl_path} (exists; write-once)")
        else:
            print("[spectracker] No section changes detected; changelog unchanged.")

    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[spectracker][error] {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
