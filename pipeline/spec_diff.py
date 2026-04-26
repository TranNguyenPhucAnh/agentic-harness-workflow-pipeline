"""
pipeline/spec_diff.py
Spec diff engine — detect what changed between spec versions and map to affected files.

Reads:
    spec.md                              ← current spec (single source of truth)
    artifacts/state/spec_applied.json    ← last successfully applied version
    artifacts/knowledge/history/         ← raw spec snapshots per version

Writes:
    artifacts/cache/spec_delta.json      ← delta for this run
    artifacts/knowledge/history/<ver>.md ← raw snapshot of current spec
    artifacts/knowledge/history/<ver>.changelog.md ← changelog entry
    artifacts/knowledge/history/spec.changelog    ← aggregated changelog

For taxonomy details see docs/artifacts.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).parent.parent

# New artifact paths
SPEC_PATH = ROOT / "spec.md"
STATE_DIR = ROOT / "artifacts" / "state"
CACHE_DIR = ROOT / "artifacts" / "cache"
KNOWLEDGE_HISTORY_DIR = ROOT / "artifacts" / "knowledge" / "history"

DELTA_OUT = CACHE_DIR / "spec_delta.json"
APPLIED_PATH = STATE_DIR / "spec_applied.json"

# Ensure directories exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Section parser (unchanged)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecSection:
    key:     str    # "4.3", "10", "0" etc.
    title:   str    # "4.3 `AnomalyFeed`"
    content: str    # full text of the section (header + body)
    hash:    str    # sha256 of content for change detection


def _section_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode()).hexdigest()[:16]


def parse_spec_version(text: str) -> str:
    """Extract version from spec header comment."""
    m = re.search(r"^#\s*Version:\s*(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def parse_sections(text: str) -> dict[str, SpecSection]:
    """
    Parse spec.md into sections keyed by number (e.g. "4", "4.3", "10").
    Handles both ## N. Title and ### N.M Title formats.
    """
    header_re = re.compile(
        r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.\s+(.+)$", re.MULTILINE
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
            key=key, title=title, content=content,
            hash=_section_hash(content),
        )

    return sections


# ════════════════════════════════════════════════════════════════════════════
# File→section mapping (unchanged)
# ════════════════════════════════════════════════════════════════════════════

_STATIC_SECTION_FILE_MAP: dict[str, list[str]] = {
    "4.1": ["src/components/SummaryStickyBar.tsx",
            "tests/components/SummaryStickyBar.test.tsx"],
    "4.2": ["src/components/ReplayControls.tsx",
            "tests/components/ReplayControls.test.tsx"],
    "4.3": ["src/components/AnomalyFeed.tsx",
            "tests/components/AnomalyFeed.test.tsx"],
    "4.4": ["src/components/ModelGates.tsx",
            "tests/components/ModelGates.test.tsx"],
    "4.5": ["src/hooks/useSensorData.ts",
            "tests/hooks/useSensorData.test.ts"],
    "4.6": ["src/hooks/useReplay.ts",
            "tests/hooks/useReplay.test.ts"],
    "5":   ["src/types/sensor.ts"],
    "6":   ["src/data/demoConstants.ts"],
    "7":   [],
    "10":  [],
    "3":   ["src/App.tsx", "src/main.tsx"],
}

_SCAFFOLD_TRIGGER_SECTIONS = {"7", "8"}
_TEST_ONLY_TRIGGER_SECTIONS = {"10"}
_IGNORED_SECTIONS = {"0", "1", "2", "9", "11"}


def _extract_file_map_from_spec(sections: dict[str, SpecSection]) -> dict[str, list[str]]:
    file_map = {k: list(v) for k, v in _STATIC_SECTION_FILE_MAP.items()}
    file_re  = re.compile(r"\*\*File:\*\*\s+`(src/[^`]+)`")

    for key, section in sections.items():
        found = file_re.findall(section.content)
        for fp in found:
            existing = file_map.setdefault(key, [])
            if fp not in existing:
                existing.append(fp)
                test_fp = fp.replace("src/", "tests/", 1)
                test_fp = re.sub(r"\.(tsx?)$", r".test.\1", test_fp)
                test_fp = re.sub(r"\.(ts)$",   r".test.\1", test_fp)
                if test_fp not in existing:
                    existing.append(test_fp)

    return file_map


def _files_for_changed_sections(
    changed: list[str],
    file_map: dict[str, list[str]],
    all_known_files: list[str],
) -> tuple[list[str], list[str]]:
    affected: set[str] = set()

    for key in changed:
        if key in _IGNORED_SECTIONS or key in _SCAFFOLD_TRIGGER_SECTIONS:
            continue
        for fp in file_map.get(key, []):
            affected.add(fp)

    if "5" in changed:
        for fp in all_known_files:
            if fp.startswith("src/hooks/") or fp.startswith("src/components/"):
                affected.add(fp)
            test = fp.replace("src/", "tests/", 1)
            test = re.sub(r"\.(tsx?)$", r".test.\1", test)
            test = re.sub(r"\.(ts)$",   r".test.\1", test)
            affected.add(test)

    if "6" in changed:
        for fp in all_known_files:
            if fp.startswith("src/hooks/"):
                affected.add(fp)
                test = fp.replace("src/", "tests/", 1)
                test = re.sub(r"\.(ts)$", r".test.\1", test)
                affected.add(test)

    unaffected = [f for f in all_known_files if f not in affected]
    return sorted(affected), sorted(unaffected)


def _decide_rerun_steps(
    changed: list[str],
    affected_files: list[str],
    is_first_run: bool,
) -> dict[str, bool]:
    if is_first_run:
        return {"scaffold": True, "plan": True, "implement": True,
                "test": True, "judge": True}

    scaffold = bool(affected_files) or any(k in _SCAFFOLD_TRIGGER_SECTIONS for k in changed)
    plan      = bool(affected_files)
    implement = bool(affected_files)
    test      = implement or any(k in _TEST_ONLY_TRIGGER_SECTIONS for k in changed)
    judge     = test

    return {
        "scaffold": scaffold,
        "plan":     plan,
        "implement": implement,
        "test":     test,
        "judge":    judge,
    }


# ════════════════════════════════════════════════════════════════════════════
# History management (writing to knowledge/history/)
# ════════════════════════════════════════════════════════════════════════════

def _save_snapshot(version: str, text: str) -> None:
    """Save raw spec snapshot to knowledge/history/<ver>.md"""
    path = KNOWLEDGE_HISTORY_DIR / f"{version}.md"
    path.write_text(text)


def _load_latest_snapshot(exclude_version: str) -> tuple[str | None, str | None]:
    """Return (version, text) of the most recent snapshot != exclude_version."""
    if not KNOWLEDGE_HISTORY_DIR.exists():
        return None, None
    snapshots = sorted(KNOWLEDGE_HISTORY_DIR.glob("*.md"), key=lambda p: p.stem)
    for snap in reversed(snapshots):
        ver = snap.stem
        if ver != exclude_version:
            return ver, snap.read_text()
    return None, None


def _load_snapshot(version: str) -> str | None:
    path = KNOWLEDGE_HISTORY_DIR / f"{version}.md"
    return path.read_text() if path.exists() else None


# ════════════════════════════════════════════════════════════════════════════
# Applied state (state/)
# ════════════════════════════════════════════════════════════════════════════

def load_applied() -> dict | None:
    if not APPLIED_PATH.exists():
        return None
    try:
        return json.loads(APPLIED_PATH.read_text())
    except Exception:
        return None


def get_last_applied_version() -> str | None:
    applied = load_applied()
    return applied.get("last_applied_version") if applied else None


def write_applied(version: str, steps: list[str], status: str) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    applied = load_applied() or {"run_history": []}
    applied["last_applied_version"] = version
    applied["applied_at"]           = now
    applied["applied_steps"]        = steps
    applied["final_status"]         = status

    run_history: list[dict] = applied.get("run_history", [])
    run_history.append({
        "version":    version,
        "applied_at": now,
        "status":     status,
        "steps":      steps,
    })
    applied["run_history"] = run_history

    APPLIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    APPLIED_PATH.write_text(json.dumps(applied, indent=2))


def print_run_history() -> None:
    applied = load_applied()
    if not applied:
        print("[spec_diff] No run history yet.")
        return
    history = applied.get("run_history", [])
    print(f"\n[spec_diff] Run history ({len(history)} run(s)):")
    for entry in history:
        icon = "✅" if entry.get("status") == "PASS" else "❌"
        steps = ", ".join(entry.get("steps", []))
        print(f"  {icon} {entry['version']}  {entry['applied_at'][:19]}  [{steps}]")


# ════════════════════════════════════════════════════════════════════════════
# Changelog management (knowledge/history/)
# ════════════════════════════════════════════════════════════════════════════

def _append_changelog(delta: "SpecDelta") -> None:
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
                lines.append(f"- §{sec}")
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

    lines.append("---")
    lines.append("")
    entry = "\n".join(lines)

    # Append to aggregated spec.changelog
    CHANGELOG = KNOWLEDGE_HISTORY_DIR / "spec.changelog"
    existing = changelog_path.read_text() if changelog_path.exists() else ""
    changelog_path.write_text(existing + entry)

    # Also save per-version changelog
    cl_path = KNOWLEDGE_HISTORY_DIR / f"{delta.to_version}.changelog.md"
    cl_path.write_text(entry)


def print_changelog(n: int = 0) -> None:
    CHANGELOG = KNOWLEDGE_HISTORY_DIR / "spec.changelog"
    if not changelog_path.exists():
        print("[spec_diff] No changelog yet.")
        return
    content = changelog_path.read_text()
    if not n:
        print(content)
        return
    entries = re.split(r"(?=^## \[)", content, flags=re.MULTILINE)
    entries = [e for e in entries if e.strip()]
    for entry in entries[-n:]:
        print(entry)


# ════════════════════════════════════════════════════════════════════════════
# Section summary generator (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def _summarise_change(key: str, old_content: str, new_content: str) -> str:
    old_lines = set(old_content.splitlines())
    new_lines = set(new_content.splitlines())
    added   = [l.strip() for l in (new_lines - old_lines) if l.strip()]
    removed = [l.strip() for l in (old_lines - new_lines) if l.strip()]

    prop_added   = [l for l in added   if l.startswith(("export ", "interface ", "type ", "  ")) and ":" in l]
    prop_removed = [l for l in removed if l.startswith(("export ", "interface ", "type ", "  ")) and ":" in l]

    parts: list[str] = []
    if prop_added:
        parts.append(f"added: {prop_added[0][:60]}")
    if prop_removed:
        parts.append(f"removed: {prop_removed[0][:60]}")
    if not parts and added:
        parts.append(f"+{len(added)} line(s)")
    if not parts and removed:
        parts.append(f"-{len(removed)} line(s)")
    if not parts:
        parts.append("content changed")

    return "; ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# Core diff logic (unchanged)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecDelta:
    from_version:       str | None
    to_version:         str
    is_first_run:       bool
    changed_sections:   list[str]
    unchanged_sections: list[str]
    new_sections:       list[str]
    removed_sections:   list[str]
    affected_files:     list[str]
    unaffected_files:   list[str]
    rerun_steps:        dict[str, bool]
    section_summaries:  dict[str, str]


def compute_delta(
    current_text: str,
    previous_text: str | None,
    all_known_files: list[str] | None = None,
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

    all_keys = set(current_secs) | set(prev_secs)
    for key in sorted(all_keys, key=lambda k: [int(x) for x in k.split(".")]):
        if key not in prev_secs:
            new_secs.append(key)
            summaries[key] = "new section"
        elif key not in current_secs:
            removed.append(key)
            summaries[key] = "section removed"
        elif current_secs[key].hash != prev_secs[key].hash:
            changed.append(key)
            summaries[key] = _summarise_change(
                key, prev_secs[key].content, current_secs[key].content
            )
        else:
            unchanged.append(key)

    if is_first_run:
        changed   = sorted(current_secs.keys(), key=lambda k: [int(x) for x in k.split(".")])
        unchanged = []

    file_map = _extract_file_map_from_spec(current_secs)
    if all_known_files is None:
        all_known_files = []
        for files in file_map.values():
            for fp in files:
                if fp not in all_known_files:
                    all_known_files.append(fp)

    affected, unaffected = _files_for_changed_sections(
        changed + new_secs, file_map, all_known_files
    )
    rerun = _decide_rerun_steps(changed + new_secs, affected, is_first_run)

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
    )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Spec diff engine")
    parser.add_argument("--show", action="store_true",
                        help="Print delta summary to stdout and exit (no writes)")
    parser.add_argument("--from", dest="from_version",
                        help="Force compare against specific version snapshot")
    parser.add_argument("--history", action="store_true",
                        help="Print aggregated spec.changelog and run history, then exit")
    parser.add_argument("--last", type=int, default=0, metavar="N",
                        help="With --history: show only last N changelog entries (0=all)")
    args = parser.parse_args()

    if args.history:
        print_changelog(n=args.last)
        print_run_history()
        return

    if not SPEC_PATH.exists():
        print(f"[spec_diff] ERROR: {SPEC_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    current_text = SPEC_PATH.read_text()
    current_ver  = parse_spec_version(current_text)

    # Determine baseline
    if args.from_version:
        prev_text = _load_snapshot(args.from_version)
        if prev_text is None:
            print(f"[spec_diff] ERROR: snapshot {args.from_version} not found.",
                  file=sys.stderr)
            sys.exit(1)
        baseline_source = f"--from {args.from_version}"
    else:
        last_applied = get_last_applied_version()
        if last_applied and last_applied != current_ver:
            prev_text = _load_snapshot(last_applied)
            baseline_source = f"last applied ({last_applied})"
        else:
            _, prev_text = _load_latest_snapshot(exclude_version=current_ver)
            baseline_source = "latest snapshot"

    print(f"[spec_diff] Baseline: {baseline_source}")

    delta = compute_delta(current_text, prev_text)

    print(f"[spec_diff] {delta.from_version or '(none)'} → {delta.to_version}")
    if delta.is_first_run:
        print("[spec_diff] First run — full pipeline required.")
    else:
        print(f"[spec_diff] Changed  §: {delta.changed_sections or '(none)'}")
        print(f"[spec_diff] New      §: {delta.new_sections or '(none)'}")
        print(f"[spec_diff] Removed  §: {delta.removed_sections or '(none)'}")
        print(f"[spec_diff] Affected files   : {len(delta.affected_files)}")
        for fp in delta.affected_files:
            note = delta.section_summaries.get(
                next((k for k, files in _STATIC_SECTION_FILE_MAP.items()
                      if fp in files), ""), ""
            )
            print(f"    {fp}" + (f"  ← {note}" if note else ""))
        print(f"[spec_diff] Unaffected files : {len(delta.unaffected_files)}")
        print(f"[spec_diff] Re-run steps     : "
              f"{[k for k, v in delta.rerun_steps.items() if v]}")

    if args.show:
        return

    # Write outputs
    DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DELTA_OUT.write_text(json.dumps(asdict(delta), indent=2))
    print(f"[spec_diff] Delta     → {DELTA_OUT}")

    _save_snapshot(current_ver, current_text)
    print(f"[spec_diff] Snapshot  → {KNOWLEDGE_HISTORY_DIR}/{current_ver}.md")

    if delta.from_version != delta.to_version or delta.is_first_run:
        _append_changelog(delta)
        print(f"[spec_diff] Changelog → {KNOWLEDGE_HISTORY_DIR}/spec.changelog  "
              f"(entry for {delta.to_version})")


if __name__ == "__main__":
    main()
