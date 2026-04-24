"""
pipeline/spec_diff.py
Spec diff engine — detect what changed between spec versions and map to affected files.

Reads:
    spec.md                          ← current spec
    scaffold/spec_history/           ← previous spec snapshots (auto-created)

Writes:
    scaffold/spec_delta.json         ← delta for this run (consumed by harness + downstream)
    scaffold/spec_history/<ver>.md   ← snapshot of current spec for future diffs

spec_delta.json schema:
{
  "from_version": "1.0.0",        // null if first run
  "to_version":   "1.1.0",
  "is_first_run": false,
  "changed_sections": [           // list of section keys that differ
    "4.3",
    "10"
  ],
  "unchanged_sections": ["1", "2", "3", "4.1", "4.2", "4.4", "4.5", "4.6", "5", "6", "7", "9"],
  "affected_files": [             // src/ files that must be re-scaffolded/re-planned/re-implemented
    "src/components/AnomalyFeed.tsx",
    "tests/components/AnomalyFeed.test.tsx"
  ],
  "unaffected_files": [           // src/ files that can be copied from previous run
    "src/types/sensor.ts",
    "src/data/demoConstants.ts",
    ...
  ],
  "rerun_steps": {                // which pipeline steps must re-run
    "scaffold": true,             // true if any affected_file is new or removed
    "plan":     true,             // true if any affected_file changed
    "implement": true,            // true if any affected_file changed
    "test":     true,             // always true if implement=true
    "judge":    true              // always true if test=true
  },
  "section_summaries": {          // one-line summary of each changed section
    "4.3": "AnomalyFeed: added sensorId prop",
    "10":  "AC-4: updated point count from 2880 to 2016"
  }
}

Usage (called by harness before any other step):
    python pipeline/spec_diff.py

Usage (standalone, for inspection):
    python pipeline/spec_diff.py --show
    python pipeline/spec_diff.py --show --from 1.0.0
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
    args = parser.parse_args()

    if not SPEC_PATH.exists():
        print(f"[spec_diff] ERROR: {SPEC_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    current_text = SPEC_PATH.read_text()
    current_ver  = parse_spec_version(current_text)

    # Load previous snapshot
    if args.from_version:
        prev_text = _load_snapshot(args.from_version)
        if prev_text is None:
            print(f"[spec_diff] ERROR: snapshot {args.from_version} not found in {HISTORY_DIR}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        _, prev_text = _load_latest_snapshot(exclude_version=current_ver)

    delta = compute_delta(current_text, prev_text)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n[spec_diff] {delta.from_version or '(none)'} → {delta.to_version}")
    if delta.is_first_run:
        print("[spec_diff] First run — full pipeline required.")
    else:
        print(f"[spec_diff] Changed sections  : {delta.changed_sections or '(none)'}")
        print(f"[spec_diff] New sections      : {delta.new_sections or '(none)'}")
        print(f"[spec_diff] Removed sections  : {delta.removed_sections or '(none)'}")
        print(f"[spec_diff] Affected files    : {len(delta.affected_files)}")
        for fp in delta.affected_files:
            note = delta.section_summaries.get(
                next((k for k, files in _STATIC_SECTION_FILE_MAP.items() if fp in files), ""),
                ""
            )
            print(f"    {fp}" + (f"  ← {note}" if note else ""))
        print(f"[spec_diff] Unaffected files  : {len(delta.unaffected_files)}")
        print(f"[spec_diff] Re-run steps      : "
              f"{[k for k, v in delta.rerun_steps.items() if v]}")

    if args.show:
        return

    # ── Save delta ─────────────────────────────────────────────────────────
    DELTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    DELTA_OUT.write_text(json.dumps(asdict(delta), indent=2))
    print(f"[spec_diff] Delta written → {DELTA_OUT}")

    # ── Save current spec as snapshot ──────────────────────────────────────
    _save_snapshot(current_ver, current_text)
    print(f"[spec_diff] Snapshot saved → {HISTORY_DIR}/{current_ver}.md")


if __name__ == "__main__":
    main()
