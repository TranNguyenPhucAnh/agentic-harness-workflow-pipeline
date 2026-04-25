"""
pipeline/spec_diff.py
Spec diff engine — detect what changed between spec versions and map to affected files.

Reads:
    spec.md                          ← current spec (single source of truth)
    scaffold/spec_applied.json       ← last successfully applied version (harness writes this)
    scaffold/spec_history/           ← raw spec snapshots per version

Writes:
    scaffold/spec_delta.json         ← delta for this run (consumed by harness + downstream)
    scaffold/spec_history/<ver>.md   ← raw snapshot of current spec
    scaffold/spec_history/<ver>.changelog.md  ← human-readable changelog entry for this version
    spec.changelog                   ← aggregated changelog across all versions (git-style)

Key design:
    spec_diff compares current spec against the LAST SUCCESSFULLY APPLIED version
    (from spec_applied.json), NOT simply the latest snapshot. This ensures that if a
    run fails mid-way, the next run correctly re-applies all unapplied changes.

    harness.py writes spec_applied.json at the end of each successful run.

spec_delta.json schema:
{
  "from_version": "1.0.0",
  "to_version":   "1.1.0",
  "is_first_run": false,
  "changed_sections":   ["4.3", "10"],
  "unchanged_sections": ["1", "2", ...],
  "new_sections":       [],
  "removed_sections":   [],
  "affected_files":     ["src/components/AnomalyFeed.tsx", ...],
  "unaffected_files":   ["src/types/sensor.ts", ...],
  "rerun_steps": {
    "scaffold": true, "plan": true, "implement": true, "test": true, "judge": true
  },
  "section_summaries":  { "4.3": "added sensorId prop", "10": "AC-4 point count fix" }
}

spec.changelog format (appended, newest last):
    ## [1.1.0] — 2026-04-25
    ### Changed
    - §4.3 AnomalyFeed: added sensorId prop
    - §10 AC-4: updated point count from 2880 to 2016
    ### Affected files
    - src/components/AnomalyFeed.tsx
    - tests/components/AnomalyFeed.test.tsx

spec_applied.json schema:
{
  "last_applied_version": "1.1.0",
  "applied_at": "2026-04-25T10:30:00Z",
  "applied_steps": ["scaffold", "plan", "implement", "test"],
  "final_status": "PASS",
  "run_history": [
    { "version": "1.0.0", "applied_at": "...", "status": "PASS", "steps": [...] },
    { "version": "1.1.0", "applied_at": "...", "status": "PASS", "steps": [...] }
  ]
}

Usage:
    python pipeline/spec_diff.py              # called by harness before any step
    python pipeline/spec_diff.py --show       # print delta, no writes
    python pipeline/spec_diff.py --from 1.0.0 # force compare against specific version
    python pipeline/spec_diff.py --history    # print aggregated changelog
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT         = Path(__file__).parent.parent
SPEC_PATH    = ROOT / "spec.md"
HISTORY_DIR  = ROOT / "scaffold" / "spec_history"
DELTA_OUT    = ROOT / "scaffold" / "spec_delta.json"
APPLIED_PATH = ROOT / "scaffold" / "spec_applied.json"
CHANGELOG    = ROOT / "spec.changelog"


# ════════════════════════════════════════════════════════════════════════════
# Section parser
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
    # Match section headers: ## 0. Title or ## 4. Title or ### 4.3 Title
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
# File→section mapping
# ════════════════════════════════════════════════════════════════════════════

# Maps section keys to src/test files they govern.
# Built from spec's "File:" declarations + known structural sections.
_STATIC_SECTION_FILE_MAP: dict[str, list[str]] = {
    # Component specs
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
    # Shared types — affects all files that import from sensor.ts
    "5":   ["src/types/sensor.ts"],
    # Constants — affects hooks that import from demoConstants
    "6":   ["src/data/demoConstants.ts"],
    # File tree — scaffold re-run if this changes
    "7":   [],   # handled separately as scaffold_trigger
    # AC changes affect test files and judge
    "10":  [],   # handled separately as test_trigger
    # App shell
    "3":   ["src/App.tsx", "src/main.tsx"],
}

# Sections that trigger scaffold re-run regardless
_SCAFFOLD_TRIGGER_SECTIONS = {"7", "8"}
# Sections that trigger test re-run but not re-implement
_TEST_ONLY_TRIGGER_SECTIONS = {"10"}
# Sections that can be ignored for code generation
_IGNORED_SECTIONS = {"0", "1", "2", "9", "11"}


def _extract_file_map_from_spec(sections: dict[str, SpecSection]) -> dict[str, list[str]]:
    """
    Augment static map by scanning spec sections for `**File:** src/...` declarations.
    This handles new components added to spec without updating the static map.
    """
    file_map = {k: list(v) for k, v in _STATIC_SECTION_FILE_MAP.items()}
    file_re  = re.compile(r"\*\*File:\*\*\s+`(src/[^`]+)`")

    for key, section in sections.items():
        found = file_re.findall(section.content)
        for fp in found:
            existing = file_map.setdefault(key, [])
            if fp not in existing:
                existing.append(fp)
                # Auto-add test file
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
    """
    Given changed section keys, return (affected_files, unaffected_files).
    Propagates: if §5 (types) changes, all files that import types are affected.
    """
    affected: set[str] = set()

    for key in changed:
        if key in _IGNORED_SECTIONS or key in _SCAFFOLD_TRIGGER_SECTIONS:
            continue
        for fp in file_map.get(key, []):
            affected.add(fp)

    # Propagation: §5 types change → all hooks + components affected
    if "5" in changed:
        for fp in all_known_files:
            if fp.startswith("src/hooks/") or fp.startswith("src/components/"):
                affected.add(fp)
            # corresponding test files
            test = fp.replace("src/", "tests/", 1)
            test = re.sub(r"\.(tsx?)$", r".test.\1", test)
            test = re.sub(r"\.(ts)$",   r".test.\1", test)
            affected.add(test)

    # §6 constants change → hooks affected (they import constants)
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

    scaffold = (
        bool(affected_files)
        or any(k in _SCAFFOLD_TRIGGER_SECTIONS for k in changed)
    )
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
# History management
# ════════════════════════════════════════════════════════════════════════════

def _save_snapshot(version: str, text: str) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{version}.md"
    path.write_text(text)


def _load_latest_snapshot(exclude_version: str) -> tuple[str | None, str | None]:
    """Return (version, text) of the most recent snapshot != exclude_version."""
    if not HISTORY_DIR.exists():
        return None, None
    snapshots = sorted(HISTORY_DIR.glob("*.md"), key=lambda p: p.stem)
    for snap in reversed(snapshots):
        ver = snap.stem
        if ver != exclude_version:
            return ver, snap.read_text()
    return None, None


def _load_snapshot(version: str) -> str | None:
    path = HISTORY_DIR / f"{version}.md"
    return path.read_text() if path.exists() else None


# ════════════════════════════════════════════════════════════════════════════
# spec_applied.json — last successfully applied version
# ════════════════════════════════════════════════════════════════════════════

def load_applied() -> dict | None:
    """Load spec_applied.json. Returns None if not present (first run)."""
    if not APPLIED_PATH.exists():
        return None
    try:
        return json.loads(APPLIED_PATH.read_text())
    except Exception:
        return None


def get_last_applied_version() -> str | None:
    """Return the version string of the last successfully applied run."""
    applied = load_applied()
    return applied.get("last_applied_version") if applied else None


def write_applied(
    version: str,
    steps: list[str],
    status: str,
) -> None:
    """
    Called by harness at end of successful run.
    Appends to run_history and updates last_applied_version.
    """
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
    """Print run history from spec_applied.json."""
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
# spec.changelog — aggregated human-readable changelog
# ════════════════════════════════════════════════════════════════════════════

def _append_changelog(delta: "SpecDelta") -> None:
    """
    Append a git-style changelog entry to spec.changelog.
    Also saves a per-version .changelog.md to spec_history/.
    """
    from datetime import datetime, timezone
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines: list[str] = [
        f"## [{delta.to_version}] — {date_str}",
        f"_(from {delta.from_version or 'initial'})_",
        "",
    ]

    # Changed sections
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
                title = ""
                # Try to get section title from current spec
                lines.append(f"- §{sec}{': ' + note if note else ''}")
            lines.append("")
        if delta.removed_sections:
            lines.append("### Removed")
            for sec in delta.removed_sections:
                lines.append(f"- §{sec}")
            lines.append("")
    else:
        lines += ["### No section changes detected", ""]

    # Affected files
    if delta.affected_files:
        lines.append("### Affected files")
        for fp in delta.affected_files:
            lines.append(f"- `{fp}`")
        lines.append("")

    # Re-run steps
    rerun = [k for k, v in delta.rerun_steps.items() if v]
    if rerun:
        lines.append(f"### Steps re-run: {', '.join(rerun)}")
        lines.append("")

    lines.append("---")
    lines.append("")
    entry = "\n".join(lines)

    # Append to aggregated spec.changelog
    existing = CHANGELOG.read_text() if CHANGELOG.exists() else ""
    CHANGELOG.write_text(existing + entry)

    # Also save per-version changelog to history dir
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    cl_path = HISTORY_DIR / f"{delta.to_version}.changelog.md"
    cl_path.write_text(entry)


def print_changelog(n: int = 0) -> None:
    """Print aggregated changelog. n=0 means all."""
    if not CHANGELOG.exists():
        print("[spec_diff] No changelog yet.")
        return
    content = CHANGELOG.read_text()
    if not n:
        print(content)
        return
    # Print last n entries (split on "## [")
    entries = re.split(r"(?=^## \[)", content, flags=re.MULTILINE)
    entries = [e for e in entries if e.strip()]
    for entry in entries[-n:]:
        print(entry)


# ════════════════════════════════════════════════════════════════════════════
# Section summary generator (one-liner per changed section)
# ════════════════════════════════════════════════════════════════════════════

def _summarise_change(key: str, old_content: str, new_content: str) -> str:
    """
    Produce a terse one-line human-readable summary of what changed in a section.
    Uses simple heuristics — no LLM call.
    """
    old_lines = set(old_content.splitlines())
    new_lines = set(new_content.splitlines())
    added   = [l.strip() for l in (new_lines - old_lines) if l.strip()]
    removed = [l.strip() for l in (old_lines - new_lines) if l.strip()]

    # Prioritise interface/type/prop changes
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
# Core diff logic
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecDelta:
    from_version:       str | None
    to_version:         str
    is_first_run:       bool
    changed_sections:   list[str]
    unchanged_sections: list[str]
    new_sections:       list[str]          # sections in new spec not in old
    removed_sections:   list[str]          # sections in old spec not in new
    affected_files:     list[str]
    unaffected_files:   list[str]
    rerun_steps:        dict[str, bool]
    section_summaries:  dict[str, str]     # key → one-liner


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

    # Detect changed, new, removed sections
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

    # All sections changed on first run
    if is_first_run:
        changed   = sorted(current_secs.keys(), key=lambda k: [int(x) for x in k.split(".")])
        unchanged = []

    # Build file map from current spec
    file_map = _extract_file_map_from_spec(current_secs)

    # Gather all known files
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

    # ── --history: show changelog + run history, no diff ─────────────────────
    if args.history:
        print_changelog(n=args.last)
        print_run_history()
        return

    if not SPEC_PATH.exists():
        print(f"[spec_diff] ERROR: {SPEC_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    current_text = SPEC_PATH.read_text()
    current_ver  = parse_spec_version(current_text)

    # ── Determine baseline version ────────────────────────────────────────────
    # Priority: --from flag > last applied version > latest snapshot
    # Using last APPLIED (not just latest snapshot) ensures failed runs don't
    # cause the next run to miss unapplied changes.
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
            # Fall back to latest snapshot (handles case where applied == current)
            _, prev_text = _load_latest_snapshot(exclude_version=current_ver)
            baseline_source = "latest snapshot"

    print(f"[spec_diff] Baseline: {baseline_source}")

    delta = compute_delta(current_text, prev_text)

    # ── Print summary ─────────────────────────────────────────────────────────
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

    # ── Write outputs ─────────────────────────────────────────────────────────
    DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DELTA_OUT.write_text(json.dumps(asdict(delta), indent=2))
    print(f"[spec_diff] Delta     → {DELTA_OUT}")

    # Raw snapshot
    _save_snapshot(current_ver, current_text)
    print(f"[spec_diff] Snapshot  → {HISTORY_DIR}/{current_ver}.md")

    # Changelog entry (only on actual version change, not same-version re-runs)
    if delta.from_version != delta.to_version or delta.is_first_run:
        _append_changelog(delta)
        print(f"[spec_diff] Changelog → {CHANGELOG}  "
              f"(entry for {delta.to_version})")


if __name__ == "__main__":
    main()
