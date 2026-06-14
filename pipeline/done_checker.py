"""
pipeline/done_checker.py
========================
Standalone "done gate" — verifies whether the current codebase reflects the
active spec version well enough to be marked applied.

Design goal: pragmatic, not aesthetic. PASS/FAIL is based only on
"spec requires X → does evidence show X?". Code style, patterns, and
best practices are NOT evaluated. judge.py handles code quality review.

Run this AFTER:
  1. executor has generated/updated src/
  2. Human has tested and fixed manually
  3. absorber has been refreshed (codebase_map.md is current)

────────────────────────────────────────────────────────────────
Inputs consumed
────────────────────────────────────────────────────────────────

  spec (specwright_spec_<slug>.md)
    Parsed dynamically — no hardcoded sections.
    Extracts: spec version, all ## sections, AC items (numbered list
    under ## Acceptance Criteria), requirement IDs (CLR-*, CON-*,
    NFR-*, NEW-*, SEC-*, PER-*).

    Spec sections are detected from the output of 04_specwright.py which
    uses REQUIRED SECTIONS: Overview, Goals, Non-Goals, Architecture,
    Tech Stack, Data Models, API Contracts, Workflow & State Machine,
    Error Handling, Non-Functional Requirements, Out of Scope,
    Acceptance Criteria, Open Questions.
    done_checker handles any spec regardless of which sections are
    present — it parses dynamically, not by expected-name lookup.

  codebase_map.md  (absorber output)
    LLM narrative of current src/ state. Used as context for the LLM
    to understand what is implemented without reading every file.
    If missing: proceeds with spec + diff only (with a warning).

  git diff
    Actual code changes to cross-reference against spec sections.
    Default: uncommitted changes (git diff HEAD).
    Falls back to HEAD~1..HEAD if no uncommitted changes found.
    Override with --diff-base <ref>.

────────────────────────────────────────────────────────────────
Outputs written
────────────────────────────────────────────────────────────────

  done_checker/done_checker_result.md   (short-term, OVERWRITE)
    Structured result for both human reading and agent consumption.
    Sections: Status, Summary, Failed Items (with fix suggestions),
    Passed Items, Requirement ID Findings, Open Questions.
    Each failed item includes: item_id, verdict, finding, file,
    symbol, line_hint, fix — sufficient for downstream debugger.

  done_checker/done_checker_log.json    (long-term, APPEND)
    One entry per run: timestamp, spec_version, overall verdict,
    passed/total counts, failed item IDs.

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python done_checker.py --project my-app
    Full check. Reads spec + codebase_map.md + git diff HEAD.
    Writes result.md + log.json.
    Exit 0 = PASSED · Exit 1 = NOT_PASSED · Exit 2 = ERROR

  python done_checker.py --project my-app --diff-base main
  python done_checker.py --project my-app --diff-base HEAD~5
  python done_checker.py --project my-app --diff-base abc1234
    Diff against given ref instead of uncommitted changes.
    Use --diff-base main after merging a feature branch.
    Use --diff-base HEAD~N when code was committed incrementally.

  python done_checker.py --project my-app --sections "Acceptance Criteria,Error Handling"
    Check only the named spec section(s). Comma-separated.
    Useful for targeted re-checks after fixing specific failures.
    Section names are matched case-insensitively (substring match).

  python done_checker.py --project my-app --dry-run
    Builds briefing and prints to stdout. No LLM call, no writes.
    Use to verify context quality before spending tokens.

  python done_checker.py --project my-app --show-last
    Print the most recent done_checker_result.md without re-running.

────────────────────────────────────────────────────────────────
Artifact impact by command
────────────────────────────────────────────────────────────────

  Command                  result.md    log.json
  ─────────────────────── ──────────── ──────────
  (normal run)             OVERWRITE    APPEND
  --dry-run                –            –
  --show-last              –            –

────────────────────────────────────────────────────────────────
Loop cadence
────────────────────────────────────────────────────────────────

  [human tests + fixes manually]
      ↓
  python done_checker.py --project <name>
      ↓ NOT_PASSED → read Failed Items → fix with debugger
      ↓ PASSED
  python 05_spectracker.py --project <name> --mark-applied --status PASS

  done_checker prompts y/n after PASSED and can call spectracker automatically.

────────────────────────────────────────────────────────────────
Agentic consumer note (for debugger / patcher downstream)
────────────────────────────────────────────────────────────────

done_checker_result.md is structured for agent consumption.
Each failed item in ## Failed Items contains:

  item_id   : AC-07 / CLR-010 / NFR-offline-1 / etc.
  verdict   : FAIL | PARTIAL
  finding   : what is wrong (1–2 sentences)
  file      : src/path/to/file.ts  (most likely location)
  symbol    : functionName / ComponentName  (if identifiable)
  line_hint : approximate line number or null
  fix       : concrete suggestion (what to add/change/remove)
  blocking  : true if this prevents PASSED verdict

Downstream debugger workflow:
  1. Read ## Failed Items from done_checker_result.md
  2. For each blocking item: open file at line_hint, apply fix
  3. Re-run done_checker.py to verify

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_MD,
    DONE_CHECKER_RESULT,
    DONE_CHECKER_LOG,
    SRC_DIR,
    artifact_root,
    ensure_dirs,
    get_project_slug,
    get_spec_path,
)
from modules.call_llm import call_llm_json  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_summary as print_cost_summary  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402


ROLE = "done_checker"

MAX_SPEC_CHARS     = 120_000
MAX_CODEBASE_CHARS =  80_000
MAX_DIFF_CHARS     =  60_000
MAX_BRIEFING_CHARS = 240_000


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _done_dir() -> Path:
    return Path(str(DONE_CHECKER_RESULT)).parent

def _result_path() -> Path:
    return Path(str(DONE_CHECKER_RESULT))

def _log_path() -> Path:
    return Path(str(DONE_CHECKER_LOG))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="done_checker.py",
        description="Verify current codebase reflects the active spec version.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python done_checker.py --project my-app
              python done_checker.py --project my-app --diff-base main
              python done_checker.py --project my-app --diff-base HEAD~5
              python done_checker.py --project my-app --sections "Acceptance Criteria"
              python done_checker.py --project my-app --dry-run
              python done_checker.py --project my-app --show-last
        """),
    )
    p.add_argument("--project", default=None,
                   help="Project name. Sets PIPELINE_PROJECT.")
    p.add_argument("--diff-base", default=None, metavar="GIT_REF",
                   help=(
                       "Git ref to diff against. Default: uncommitted changes "
                       "(git diff HEAD). Examples: main, HEAD~5, abc1234."
                   ))
    p.add_argument("--sections", default=None, metavar="SECTIONS",
                   help=(
                       "Comma-separated spec section names to check (case-insensitive "
                       "substring match). Default: all sections. "
                       "Example: 'Acceptance Criteria,Error Handling'"
                   ))
    p.add_argument("--dry-run", action="store_true",
                   help="Build briefing and print it; skip LLM call and writes.")
    p.add_argument("--show-last", action="store_true",
                   help="Print most recent done_checker_result.md and exit.")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error(
            "PIPELINE_PROJECT is not set. "
            "Use --project <name> or export PIPELINE_PROJECT=<name>."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safe readers
# ─────────────────────────────────────────────────────────────────────────────

def _read_text(path: Path, label: str) -> str:
    if not path.exists():
        print(f"[done_checker][warn] {label} not found: {path}")
        return ""
    track_read(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[done_checker][warn] Could not read {label}: {e}")
        return ""


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    print(f"[done_checker] Truncating {label}: {len(text):,} → {limit:,} chars")
    return text[:limit] + f"\n\n[truncated: {len(text):,} chars total]"


# ─────────────────────────────────────────────────────────────────────────────
# Git diff
# ─────────────────────────────────────────────────────────────────────────────

def _get_git_diff(base: str | None, cwd: Path) -> tuple[str, list[str]]:
    """Return (diff_text, changed_files). Tries multiple strategies."""
    if not cwd.exists():
        cwd = Path.cwd()

    def _run(cmd: list[str]) -> str:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)
        return r.stdout or ""

    if base:
        diff_text     = _run(["git", "diff", f"{base}...HEAD"])
        changed_files = _run(["git", "diff", "--name-only", f"{base}...HEAD"]).splitlines()
    else:
        diff_text     = _run(["git", "diff", "HEAD"])
        changed_files = _run(["git", "diff", "--name-only", "HEAD"]).splitlines()

        # Fallback: nothing uncommitted → use last commit
        if not diff_text.strip():
            diff_text2     = _run(["git", "diff", "HEAD~1", "HEAD"])
            changed_files2 = _run(["git", "diff", "--name-only", "HEAD~1", "HEAD"]).splitlines()
            if diff_text2.strip():
                print("[done_checker] No uncommitted changes — using HEAD~1..HEAD diff")
                diff_text     = diff_text2
                changed_files = changed_files2

    return diff_text, [f.strip() for f in changed_files if f.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Spec parsing — fully dynamic, no hardcoded section names
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_REQ_ID_RE  = re.compile(r"\b(CLR-\d+|CON-\d+|NFR-[A-Za-z0-9_-]+|NEW-\d+|SEC-\d+|PER-\d+)\b")


def _parse_spec_version(text: str) -> str:
    m = re.search(r"^#\s*Version:\s*(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def _parse_ac_items(spec_text: str) -> list[dict[str, str]]:
    """
    Dynamically find the Acceptance Criteria section (case-insensitive) and
    extract numbered items. Works regardless of section order or extra sections.
    """
    # Find any ## heading containing "acceptance criteria" (case-insensitive)
    ac_match = re.search(
        r"^##\s+[^\n]*acceptance criteria[^\n]*\s*\n(.*?)(?=^##\s|\Z)",
        spec_text, re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if not ac_match:
        return []

    section = ac_match.group(1)
    items   = []

    # Pattern: "N. **AC-XX**: description"  (possibly multi-line until next item)
    for m in re.finditer(
        r"^\d+\.\s+\*\*(AC-\d+)\*\*:?\s*(.+?)(?=^\d+\.\s+\*\*AC-|\Z)",
        section, re.MULTILINE | re.DOTALL,
    ):
        items.append({
            "id":   m.group(1).strip(),
            "text": m.group(2).strip().replace("\n", " "),
        })

    return items


def _parse_all_sections(spec_text: str, filter_names: list[str] | None) -> list[dict[str, str]]:
    """
    Extract all ## sections as {heading, content} dicts.
    If filter_names is given, only return sections whose heading contains
    any filter string (case-insensitive substring match).
    """
    sections = []
    matches  = list(_SECTION_RE.finditer(spec_text))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(spec_text)
        content = spec_text[start:end].strip()

        if filter_names:
            if not any(f.lower() in heading.lower() for f in filter_names):
                continue

        sections.append({"heading": heading, "content": content})

    return sections


def _extract_req_ids(spec_text: str) -> list[str]:
    return sorted(set(_REQ_ID_RE.findall(spec_text)))


# ─────────────────────────────────────────────────────────────────────────────
# Briefing builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_briefing(
    spec_text:       str,
    codebase_map:    str,
    diff_text:       str,
    changed_files:   list[str],
    ac_items:        list[dict[str, str]],
    req_ids:         list[str],
    sections:        list[dict[str, str]],
    spec_version:    str,
    filter_names:    list[str] | None,
    diff_base:       str | None,
) -> str:
    parts: list[str] = []

    # Section 0 — context
    diff_desc = (
        f"`git diff {diff_base}...HEAD`" if diff_base
        else "`git diff HEAD` (uncommitted + staged changes)"
    )
    files_block = (
        "\n".join(f"  - {f}" for f in changed_files)
        if changed_files else "  (none detected)"
    )
    filter_note = (
        f"**Section filter active:** {', '.join(filter_names)}"
        if filter_names else "**Checking all spec sections.**"
    )
    parts.append(textwrap.dedent(f"""\
        ## 0. Done-check context

        **Spec version:** {spec_version}
        **Git diff source:** {diff_desc}
        **Changed files ({len(changed_files)}):**
        {files_block}

        **Requirement IDs in spec ({len(req_ids)}):**
        {', '.join(req_ids) or '(none found)'}

        {filter_note}
    """).strip())

    # Section 1 — spec
    parts.append(
        "## 1. Spec\n\n"
        "```markdown\n"
        + _truncate(spec_text, MAX_SPEC_CHARS, "spec")
        + "\n```"
    )

    # Section 2 — codebase map
    if codebase_map:
        parts.append(
            "## 2. Codebase map (absorber narrative)\n\n"
            + _truncate(codebase_map, MAX_CODEBASE_CHARS, "codebase_map.md")
        )
    else:
        parts.append(
            "## 2. Codebase map\n\n"
            "_[Not found — run absorber first for best results. "
            "Evaluation proceeds from spec + diff only.]_"
        )

    # Section 3 — git diff
    if diff_text.strip():
        parts.append(
            "## 3. Git diff\n\n"
            "```diff\n"
            + _truncate(diff_text, MAX_DIFF_CHARS, "git diff")
            + "\n```"
        )
    else:
        parts.append(
            "## 3. Git diff\n\n"
            "_[No diff found. If code is already committed, "
            "re-run with --diff-base HEAD~N.]_"
        )

    # Section 4 — AC items to check
    if ac_items:
        ac_lines = [f"- **{i['id']}**: {i['text']}" for i in ac_items]
        parts.append(
            "## 4. Acceptance Criteria to verify\n\n"
            + "\n".join(ac_lines)
        )
    else:
        parts.append(
            "## 4. Acceptance Criteria\n\n"
            "_[No numbered AC items found — evaluate spec sections holistically.]_"
        )

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a pragmatic implementation reviewer. Your ONLY job is to determine
whether the current codebase has implemented what the spec says it should.

You are NOT evaluating:
  - Code style or formatting
  - Architectural elegance
  - Test coverage beyond what AC explicitly requires
  - Performance beyond explicit NFR values
  - Best practices not stated in the spec

You ARE evaluating:
  - "Spec says X must exist" → does evidence (diff or codebase map) show it?
  - "AC requires specific value Y" (e.g. rgba(220,50,50,0.55), ≤500ms, ≤10 items) → does code match?
  - "AC says this MUST NOT happen" → does evidence suggest it does happen?
  - "Spec defines an interface/function signature" → does the implementation match?

Verdicts:
  PASS    — evidence shows the requirement is met. Minor cosmetic issues are fine.
  PARTIAL — implementation exists but clearly misses one aspect of the requirement,
            OR spec is ambiguous and implementation covers a reasonable interpretation.
            Mark blocking: false if the gap is minor and does not prevent the feature
            from working. Mark blocking: true if the gap meaningfully breaks the intent.
  FAIL    — spec explicitly requires something AND codebase has no evidence of it,
            OR a specific required value is wrong,
            OR a MUST NOT constraint is violated.

You will receive:
  Section 0 — context (spec version, changed files, requirement IDs)
  Section 1 — full spec text
  Section 2 — codebase_map.md (LLM narrative of current src/ state)
  Section 3 — git diff (actual code changes)
  Section 4 — Acceptance Criteria items to evaluate

Evaluate EVERY AC item in Section 4.
Also evaluate requirement IDs (CLR-*, CON-*, NFR-*, NEW-*, SEC-*, PER-*)
that appear in the diff scope or codebase map. Skip IDs with no relevant evidence.

Return raw JSON only (no markdown fences):
{
  "spec_version": "<version string>",
  "overall": "PASSED" | "NOT_PASSED",
  "summary": "<2-3 sentence plain English summary>",
  "passed_count": <N>,
  "total_count": <N>,
  "items": [
    {
      "item_id":   "AC-07",
      "verdict":   "PASS" | "PARTIAL" | "FAIL",
      "finding":   "<one sentence: what was checked and what was found>",
      "file":      "src/lib/opfs.ts",
      "symbol":    "writeAudioToOpfs",
      "line_hint": 42,
      "fix":       "<concrete suggestion; null if PASS>",
      "blocking":  true
    }
  ],
  "req_id_findings": [
    {
      "item_id":   "CLR-010",
      "verdict":   "PASS" | "PARTIAL" | "FAIL",
      "finding":   "<one sentence>",
      "file":      "src/lib/opfs.ts",
      "symbol":    null,
      "line_hint": null,
      "fix":       null,
      "blocking":  false
    }
  ],
  "open_questions": [
    "<any ambiguity that prevented confident assessment>"
  ]
}

Overall verdict:
  PASSED     — all items PASS, or all PARTIAL have blocking: false
  NOT_PASSED — any item is FAIL, or any PARTIAL has blocking: true

Be direct. If you cannot find evidence for something, say so — do not invent it.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Result writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_result_md(result: dict[str, Any], spec_version: str) -> str:
    overall      = result.get("overall", "NOT_PASSED")
    summary      = result.get("summary", "")
    passed_count = result.get("passed_count", 0)
    total_count  = result.get("total_count", 0)
    items        = result.get("items", [])
    req_findings = result.get("req_id_findings", [])
    open_qs      = result.get("open_questions", [])
    ts           = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    L: list[str] = []

    L += [
        "# Done Check Result", "",
        f"**Status:** {overall}  ({passed_count}/{total_count} items passed)",
        f"**Spec version:** {spec_version}",
        f"**Checked at:** {ts}",
        "",
        "## Summary", "",
        summary, "",
    ]

    # Failed / partial — primary section for agent consumption
    failed = [i for i in items if i.get("verdict") in ("FAIL", "PARTIAL")]
    if failed:
        L += ["## Failed Items", ""]
        for item in failed:
            iid      = item.get("item_id", "?")
            verdict  = item.get("verdict", "?")
            finding  = item.get("finding", "")
            file_    = item.get("file") or "unknown"
            symbol   = item.get("symbol")
            line_h   = item.get("line_hint")
            fix      = item.get("fix") or "No suggestion available."
            blocking = item.get("blocking", True)
            nb_tag   = " _(non-blocking)_" if not blocking else ""

            loc = f"`{file_}`"
            if symbol:
                loc += f" → `{symbol}`"
            if line_h:
                loc += f" (line ~{line_h})"

            L += [
                f"### {iid} — {verdict}{nb_tag}", "",
                f"**Finding:** {finding}",
                f"**Location:** {loc}",
                f"**Fix:** {fix}",
                "",
            ]

    # Passed — compact
    passed = [i for i in items if i.get("verdict") == "PASS"]
    if passed:
        L += ["## Passed Items", ""]
        L += [f"- **{i['item_id']}**: {i.get('finding', '')}" for i in passed]
        L += [""]

    # Requirement ID findings
    if req_findings:
        L += ["## Requirement ID Findings (CLR / CON / NFR / NEW)", ""]
        for item in req_findings:
            iid     = item.get("item_id", "?")
            verdict = item.get("verdict", "?")
            finding = item.get("finding", "")
            fix     = item.get("fix")
            L.append(f"- **{iid}** [{verdict}]: {finding}")
            if fix:
                L.append(f"  → Fix: {fix}")
        L += [""]

    # Open questions
    if open_qs:
        L += ["## Open Questions (LLM uncertainty)", ""]
        L += [f"- {q}" for q in open_qs]
        L += [""]

    # Footer — action prompt
    L += ["---", ""]
    slug = os.environ.get("PIPELINE_PROJECT", "<name>")
    if overall == "PASSED":
        L += [
            "## ✅ All items passed",
            "",
            "To mark this spec version as applied:",
            "```",
            f"python pipeline/05_spectracker.py --project {slug} --mark-applied --status PASS",
            "```", "",
        ]
    else:
        fail_n    = sum(1 for i in items if i.get("verdict") == "FAIL")
        partial_b = sum(1 for i in items
                        if i.get("verdict") == "PARTIAL" and i.get("blocking", True))
        L += [
            f"## ❌ Not passed  ({fail_n} failures, {partial_b} blocking partials)",
            "",
            "Fix the items in **Failed Items** above, then re-run:",
            "```",
            f"python pipeline/done_checker.py --project {slug}",
            "```", "",
        ]

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(result: dict[str, Any], spec_version: str, diff_base: str | None) -> None:
    log = _log_path()
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text())
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    entry = {
        "checked_at":   datetime.now(timezone.utc).isoformat(),
        "spec_version": spec_version,
        "overall":      result.get("overall"),
        "passed_count": result.get("passed_count", 0),
        "total_count":  result.get("total_count", 0),
        "diff_base":    diff_base or "HEAD (uncommitted)",
        "failed_ids": [
            i.get("item_id") for i in result.get("items", [])
            if i.get("verdict") in ("FAIL", "PARTIAL") and i.get("blocking", True)
        ],
    }
    existing.append(entry)

    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"entries": existing}, indent=2, ensure_ascii=False))
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# Long-term artifact commit — y/n keep log entry
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_commit_log() -> None:
    """
    Ask user whether to keep the log entry just appended to done_checker_log.json.
    Mirrors the pattern used by post_interactive._maybe_commit_run_log().
    Called after print_artifact_summary / print_cost_summary (in finally),
    before _human_apply_gate.
    """
    log = _log_path()
    if not log.exists():
        return

    try:
        data     = json.loads(log.read_text(encoding="utf-8"))
        entries  = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        return

    if not entries:
        return

    artifact_name = log.name
    try:
        ans = input(f"  Keep this entry in {artifact_name}? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"  [done_checker] Entry kept in {artifact_name} (non-interactive).")
        return

    if ans in ("n", "no"):
        entries.pop()
        try:
            log.write_text(
                json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [done_checker] Entry discarded — {artifact_name} unchanged.")
        except Exception as exc:
            print(f"  [done_checker][warn] Could not revert {artifact_name}: {exc}")
    else:
        print(f"  [done_checker] Entry kept in {artifact_name} (total: {len(entries)}).")


# ─────────────────────────────────────────────────────────────────────────────
# Human gate — y/n prompt after PASSED → auto-run spectracker
# ─────────────────────────────────────────────────────────────────────────────

def _human_apply_gate(spec_version: str) -> None:
    slug = os.environ.get("PIPELINE_PROJECT", "")
    print()
    print("=" * 60)
    print(f"  ✅ PASSED — spec version {spec_version}")
    print("=" * 60)
    print()
    print("  All checked items pass. This spec version appears to be")
    print("  fully implemented in the current codebase.")
    print()
    print("  Mark as applied in spectracker?")
    print("    [y] Yes — run spectracker --mark-applied --status PASS  (recommended)")
    print("    [n] No  — skip (mark later manually)")
    print()

    try:
        choice = input("  Choice [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "y"   # non-interactive → default apply

    if choice in ("", "y", "yes"):
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "05_spectracker.py"),
            "--mark-applied", "--status", "PASS",
        ]
        if slug:
            cmd += ["--project", slug]
        print(f"\n  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("  [done_checker] ✓ Spec version marked as applied.")
        else:
            print("  [done_checker] spectracker exited non-zero — check output above.")
    else:
        print()
        print("  Skipped. To mark later:")
        print(f"    python pipeline/05_spectracker.py --project {slug or '<name>'} "
              "--mark-applied --status PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()

    print("=" * 60)
    print("  DONE CHECKER")
    print("=" * 60)
    print()

    # ── --show-last ──────────────────────────────────────────────────────────
    if args.show_last:
        p = _result_path()
        if not p.exists():
            print("[done_checker] No previous result found.")
            sys.exit(2)
        print(p.read_text())
        sys.exit(0)

    # ── Load inputs ──────────────────────────────────────────────────────────
    spec_path    = get_spec_path()
    spec_text    = _read_text(spec_path, "spec")
    codebase_map = _read_text(Path(str(ABSORBER_CODEBASE_MD)), "codebase_map.md")

    if not spec_text:
        print(f"[done_checker][error] Spec not found: {spec_path}", file=sys.stderr)
        sys.exit(2)

    if not codebase_map:
        print("[done_checker][warn] codebase_map.md not found.")
        print("  Run absorber first for best results. Proceeding with spec + diff only.\n")

    # ── Git diff ─────────────────────────────────────────────────────────────
    src_parent = Path(str(SRC_DIR)).parent
    cwd        = src_parent if src_parent.exists() else Path.cwd()
    diff_text, changed_files = _get_git_diff(args.diff_base, cwd)

    if not diff_text.strip():
        print("[done_checker][warn] No git diff found.")
        print("  If code is committed, try: --diff-base HEAD~N\n")

    print(f"  Changed files in diff: {len(changed_files)}")
    for f in changed_files[:12]:
        print(f"    {f}")
    if len(changed_files) > 12:
        print(f"    ... and {len(changed_files) - 12} more")
    print()

    # ── Parse spec (dynamic) ─────────────────────────────────────────────────
    spec_version  = _parse_spec_version(spec_text)
    ac_items      = _parse_ac_items(spec_text)
    req_ids       = _extract_req_ids(spec_text)
    filter_names  = (
        [s.strip() for s in args.sections.split(",")]
        if args.sections else None
    )
    sections      = _parse_all_sections(spec_text, filter_names)

    print(f"  Spec version:     {spec_version}")
    print(f"  Sections found:   {len(sections)}"
          + (f"  (filter: {filter_names})" if filter_names else ""))
    print(f"  AC items found:   {len(ac_items)}")
    print(f"  Requirement IDs:  {len(req_ids)}")
    print()

    # ── Build briefing ───────────────────────────────────────────────────────
    briefing = _build_briefing(
        spec_text      = spec_text,
        codebase_map   = codebase_map,
        diff_text      = diff_text,
        changed_files  = changed_files,
        ac_items       = ac_items,
        req_ids        = req_ids,
        sections       = sections,
        spec_version   = spec_version,
        filter_names   = filter_names,
        diff_base      = args.diff_base,
    )
    briefing = briefing[:MAX_BRIEFING_CHARS]
    print(f"  Briefing size: {len(briefing):,} chars")
    print()

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[done_checker] DRY RUN — briefing follows (no LLM call):\n")
        print(briefing)
        sys.exit(0)

    # ── LLM call (call_llm_json handles retries + JSON parse) ────────────────
    print("[done_checker] Calling LLM …")
    t0 = time.time()

    try:
        result, cost = call_llm_json(
            role        = ROLE,
            system      = _SYSTEM,
            user        = briefing,
            max_tokens  = 8192,
            caller_file = __file__,
            label       = "done_checker",
        )
    except Exception as e:
        print(f"[done_checker][error] LLM call failed: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"  Elapsed: {time.time() - t0:.1f}s  |  cost: ${cost:.4f}")
    print()

    overall      = result.get("overall", "NOT_PASSED")
    passed_count = result.get("passed_count", 0)
    total_count  = result.get("total_count", 0)
    print(f"  Overall: {overall}  ({passed_count}/{total_count} passed)")
    print()

    exit_code = 0
    try:
        # ── Write artifacts ──────────────────────────────────────────────────
        _done_dir().mkdir(parents=True, exist_ok=True)

        result_md   = _write_result_md(result, spec_version)
        result_path = _result_path()

        # Apply md_header (created vs modified)
        result_with_header = apply_md_header(
            content = result_md,
            path    = result_path,
            owner   = "done_checker.py",
        )
        result_path.write_text(result_with_header, encoding="utf-8")
        track_write(result_path)
        print(f"  Written:  {result_path}")

        _append_log(result, spec_version, args.diff_base)
        print(f"  Appended: {_log_path()}")

        if overall != "PASSED":
            blocking = [
                i.get("item_id") for i in result.get("items", [])
                if i.get("verdict") in ("FAIL", "PARTIAL") and i.get("blocking", True)
            ]
            print(f"\nNOT_PASSED — {len(blocking)} blocking item(s): {blocking}")
            print(f"See: {result_path}")
            print()
            slug = os.environ.get("PIPELINE_PROJECT", "<name>")
            print(f"Fix failures, then re-run:")
            print(f"  python pipeline/done_checker.py --project {slug}")
            exit_code = 1

    except Exception as exc:
        print(f"[done_checker][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 2

    finally:
        print()
        print_artifact_summary("[done_checker]")
        print()
        print_cost_summary("[done_checker]")

    # Long-term artifact commit prompt (after summary, before human gate)
    if exit_code in (0, 1):   # chỉ hỏi khi log đã được append (không hỏi khi exception)
        _maybe_commit_log()

    # Human gate: nếu PASSED → hỏi mark applied → auto-run spectracker
    if exit_code == 0:
        _human_apply_gate(spec_version)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()