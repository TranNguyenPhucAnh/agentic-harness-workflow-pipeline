"""
pipeline/11_judge.py
====================
Step 11 — Judge / Validator.

Runs after verification/tests have passed. Aggregates pipeline artifacts into a
single briefing, sends it to the judge model for final review, and writes:

  artifacts_<slug>/sessions/<NNN>/execution/judge_overwrite_verdict_raw.json
  artifacts_<slug>/sessions/<NNN>/reports/judge_verdict_summary.md

When PIPELINE_SESSION is not set, paths.py falls back to the legacy layout:

  artifacts_<slug>/execution/judge_overwrite_verdict_raw.json
  artifacts_<slug>/reports/judge_verdict_summary.md

Supports both:
  - FULL/PARTIAL flow:
      specwright_spec_<slug>.md / scaffolder_compressed_spec.md,
      planner_full_execution_plan.json,
      scaffolder_codebase_skeleton.json,
      debugger_overwrite_test_summary.json,
      executor_overwrite_manifest.json,
      source files,
      test files,
      spectracker_overwrite_version_delta.json.

  - MINI targeted flow:
      clarificator_requirement_synthesis.md,
      enricher_overwrite_enriched_prompt.md,
      planner_mini_execution_plan.json,
      planner_mini_impact_analysis.json,
      executor_overwrite_manifest.json,
      debugger_overwrite_test_summary.json,
      and only the target/implemented files.

Direct execution:
  python 11_judge.py --project my-app
  python 11_judge.py --project my-app --session 1
  PIPELINE_PROJECT=my-app python 11_judge.py
  PIPELINE_PROJECT=my-app PIPELINE_SESSION=001 python 11_judge.py

Required environment:
  OPENROUTER_API_KEY=<your-key>

At the end of each run, prints:
  - artifacts/files read
  - artifacts/files created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# === WRITE AUTHORITY: judge ===
# OWNS  : artifacts_<slug>/sessions/<NNN>/execution/judge_overwrite_verdict_raw.json
#         artifacts_<slug>/sessions/<NNN>/reports/judge_verdict_summary.md
# READS : artifacts_<slug>/specwright_spec_<slug>.md
#         artifacts_<slug>/sessions/<NNN>/cache/scaffolder_compressed_spec.md
#         artifacts_<slug>/sessions/<NNN>/state/planner_full_execution_plan.json
#         artifacts_<slug>/sessions/<NNN>/state/planner_mini_execution_plan.json
#         artifacts_<slug>/sessions/<NNN>/state/planner_mini_impact_analysis.json
#         artifacts_<slug>/sessions/<NNN>/state/scaffolder_codebase_skeleton.json
#         artifacts_<slug>/sessions/<NNN>/state/clarificator_requirement_synthesis.md
#         artifacts_<slug>/sessions/<NNN>/cache/spectracker_overwrite_version_delta.json
#         artifacts_<slug>/sessions/<NNN>/execution/enricher_overwrite_enriched_prompt.md
#         artifacts_<slug>/sessions/<NNN>/execution/executor_overwrite_manifest.json
#         artifacts_<slug>/sessions/<NNN>/execution/debugger_overwrite_test_summary.json
#         artifacts_<slug>/knowledge/current/archivist_spec_gaps.md
#         artifacts_<slug>/src/**
#         artifacts_<slug>/tests/**

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ARCHIVIST_SPEC_GAPS,
    CLARIFIED_REQ,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    ENRICHER_OVERWRITE_PROMPT,
    EXECUTOR_OVERWRITE_MANIFEST,
    JUDGE_OVERWRITE_VERDICT_RAW,
    JUDGE_VERDICT_SUMMARY,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_IMPACT,
    PLANNER_MINI_PLAN,
    SCAFFOLD_JSON,
    SCAFFOLDER_COMPRESSED_SPEC,
    SPECTRACKER_VERSION_DELTA,
    SRC_DIR,
    TESTS_DIR,
    artifact_root,
    ensure_dirs,
    get_project_name,
    get_project_slug,
    get_session_id,
    get_spec_path,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
MAX_BRIEFING_CHARS = 900_000
MAX_FILE_CHARS = 80_000


# ─────────────────────────────────────────────────────────────────────────────
# Artifact/file access tracking
# ─────────────────────────────────────────────────────────────────────────────

_ARTIFACTS_READ: set[str] = set()
_ARTIFACTS_WRITTEN: set[str] = set()


def _track_read(path: Any) -> None:
    _ARTIFACTS_READ.add(str(path))


def _track_write(path: Any) -> None:
    _ARTIFACTS_WRITTEN.add(str(path))


def _print_artifact_access_summary() -> None:
    print("[11] Artifacts/files read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[11]   READ  {item}")
    else:
        print("[11]   READ  (none)")

    print("[11] Artifacts/files created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[11]   WRITE {item}")
    else:
        print("[11]   WRITE (none)")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a senior software engineer acting as a final code reviewer and sign-off authority.

You will receive a complete pipeline briefing. Section 0 tells you whether this is:
- FULL run
- PARTIAL run
- MINI targeted run

For FULL runs:
- Review all implemented files equally.

For PARTIAL runs:
- Focus your review on re-implemented files.
- For reused files, only flag issues if they directly interact with changed files.
- Do NOT block approval for pre-existing issues in reused files.

For MINI targeted runs:
- Review ONLY the target files listed in planner_mini_execution_plan.target_files and files written
  in executor_overwrite_manifest.files.
- Do NOT block approval for unrelated pre-existing issues outside the mini target scope.
- If you notice a problem outside the target scope, mention it as a non-blocking
  note or as "requires broader follow-up", unless it directly breaks the targeted task.
- Judge whether the targeted change satisfies the clarified requirement and respects
  planner_mini_execution_plan constraints.

Review dimensions:
A. REQUIREMENT / SPEC COMPLIANCE
   - FULL/PARTIAL: spec compliance, acceptance criteria.
   - MINI: clarified requirement + planner_mini_execution_plan compliance.
B. CODE QUALITY
   - Correctness, maintainability, type/syntax safety, no obvious regressions.
C. TEST / VERIFIER QUALITY
   - Are tests or verification meaningful? Are failures ignored?
D. ARCHITECTURE / SCOPE SAFETY
   - Clean dependencies, correct file boundaries, no unauthorized broad changes.
E. GAPS / RISKS
   - Missing coverage, edge cases, production risks.

Return a structured JSON object — raw JSON only, no markdown fences:
{
  "verdict": "APPROVED" | "APPROVED_WITH_NOTES" | "NEEDS_REVISION",
  "run_type": "full" | "partial" | "mini",
  "summary": "2-3 sentence executive summary",
  "sections": {
    "requirement_compliance": { "score": 1-5, "notes": "...", "scope": "..." },
    "code_quality":           { "score": 1-5, "notes": "..." },
    "test_quality":           { "score": 1-5, "notes": "..." },
    "architecture_scope":     { "score": 1-5, "notes": "..." },
    "gaps_risks":             { "notes": "..." }
  },
  "blocking_issues": [ "issue 1" ],
  "non_blocking_notes": [ "note 1" ],
  "partial_run_notes": "observations about reused files for partial runs, else null",
  "mini_run_notes": "observations about target scope for mini runs, else null",
  "sign_off": "judge model + timestamp placeholder"
}

Scoring:
- 5 = excellent
- 4 = good
- 3 = acceptable
- 2 = needs work
- 1 = failing

Verdict rules:
- APPROVED: no blocking issues, average score >= 3.5.
- APPROVED_WITH_NOTES: no blocking issues, but notable non-blocking issues/risks.
- NEEDS_REVISION: one or more blocking issues found.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI / project/session setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="11_judge.py",
        description="Run final judge review over pipeline artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 11_judge.py --project my-app
              python 11_judge.py --project my-app --session 1
              PIPELINE_PROJECT=my-app python 11_judge.py
              PIPELINE_PROJECT=my-app PIPELINE_SESSION=001 python 11_judge.py

              python 11_judge.py --project my-app --model deepseek/deepseek-v3.2
        """),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Optional session id for direct execution. Sets PIPELINE_SESSION. "
            "Example: --session 1 resolves to sessions/001."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", DEFAULT_MODEL),
        help=f"Judge model id. Default: env JUDGE_MODEL or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max-briefing-chars",
        type=int,
        default=MAX_BRIEFING_CHARS,
        help=f"Maximum briefing size before truncation. Default: {MAX_BRIEFING_CHARS}.",
    )
    return parser


def _configure_project(
    project: str | None,
    session: str | None,
    parser: argparse.ArgumentParser,
) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project

    if session is not None:
        raw = str(session).strip()
        if not raw:
            parser.error("--session cannot be empty.")
        try:
            os.environ["PIPELINE_SESSION"] = f"{int(raw):03d}"
        except ValueError:
            parser.error("--session must be an integer, e.g. --session 1.")

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 11_judge.py directly."
    )


def _require_openrouter_key(parser: argparse.ArgumentParser) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        parser.error(
            "OPENROUTER_API_KEY is not set. Export OPENROUTER_API_KEY=<your-key> and retry."
        )
    return api_key


# ─────────────────────────────────────────────────────────────────────────────
# Safe loaders
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(path: Any, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        _track_read(path)
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        print(f"[11][warn] Could not parse JSON {path}: {exc}", file=sys.stderr)
        return default


def _read_text(path: Any) -> str:
    if not path.exists():
        return ""
    try:
        _track_read(path)
        return path.read_text(errors="replace")
    except Exception as exc:
        print(f"[11][warn] Could not read {path}: {exc}", file=sys.stderr)
        return ""


def _load_impl_record() -> dict[str, Any]:
    rec = _read_json(EXECUTOR_OVERWRITE_MANIFEST, {})
    return rec if isinstance(rec, dict) else {}


def _load_plan_mini() -> dict[str, Any]:
    plan = _read_json(PLANNER_MINI_PLAN, {})
    return plan if isinstance(plan, dict) else {}


def _load_analysis_mini() -> dict[str, Any]:
    analysis = _read_json(PLANNER_MINI_IMPACT, {})
    return analysis if isinstance(analysis, dict) else {}


def _load_delta() -> dict[str, Any] | None:
    delta = _read_json(SPECTRACKER_VERSION_DELTA, None)
    return delta if isinstance(delta, dict) else None


def _load_spec_optional() -> str:
    if SCAFFOLDER_COMPRESSED_SPEC.exists():
        return _read_text(SCAFFOLDER_COMPRESSED_SPEC)

    spec_path = get_spec_path()
    return _read_text(spec_path)


def _load_test_report() -> dict[str, Any]:
    report = _read_json(DEBUGGER_OVERWRITE_TEST_SUMMARY, {})
    return report if isinstance(report, dict) else {}


def _detect_scope() -> str:
    rec = _load_impl_record()
    scope = rec.get("scope")
    if scope in {"full", "mini"}:
        return scope

    report = _load_test_report()
    scope = report.get("scope")
    if scope in {"full", "mini"}:
        return scope

    if PLANNER_MINI_PLAN.exists() or PLANNER_MINI_IMPACT.exists():
        return "mini"

    return "full"


# ─────────────────────────────────────────────────────────────────────────────
# Path / file collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_rel(raw: str) -> Path:
    normalized = raw.replace("\\", "/").strip()
    rel = Path(normalized)

    if not normalized:
        raise ValueError("empty path")

    if rel.is_absolute():
        raise ValueError(f"absolute path not allowed: {raw}")

    if any(part == ".." for part in rel.parts):
        raise ValueError(f"path traversal not allowed: {raw}")

    return rel


def _resolve_artifact_path(rel: str) -> Path:
    safe = _safe_rel(rel)
    raw = safe.as_posix()

    if raw.startswith("src/"):
        return SRC_DIR / raw[len("src/"):]
    if raw.startswith("tests/"):
        return TESTS_DIR / raw[len("tests/"):]

    return artifact_root() / safe


def _extract_file_list(value: Any) -> list[str]:
    """
    Normalize file list from either:
      ["src/a.ts"]
    or:
      [{"path": "src/a.ts", ...}]
    or:
      [{"file_path": "src/a.ts", ...}]
    or:
      [{"file": "src/a.ts", ...}]
    """
    files: list[str] = []

    if not isinstance(value, list):
        return files

    for item in value:
        if isinstance(item, str):
            files.append(item)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("file_path") or item.get("file")
            if isinstance(path, str):
                files.append(path)

    return sorted(set(files))


def _mini_target_files(plan_mini: dict[str, Any], impl_record: dict[str, Any]) -> list[str]:
    files: set[str] = set()
    files.update(_extract_file_list(plan_mini.get("target_files", [])))
    files.update(_extract_file_list(impl_record.get("files", [])))
    return sorted(files)


def _read_file_for_briefing(path: Path) -> str:
    if not path.exists():
        return f"[file not found: {path}]"

    _track_read(path)
    text = path.read_text(errors="replace")
    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + f"\n\n[truncated: {len(text)} chars total]"
    return text


def _lang_for_path(rel: str) -> str:
    ext = Path(rel).suffix.lower()
    mapping = {
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".py": "python",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".txt": "text",
        ".ini": "ini",
        ".cfg": "ini",
        ".conf": "text",
        ".sh": "bash",
    }
    return mapping.get(ext, "")


def _format_file_block(rel: str, content: str, label: str = "") -> str:
    lang = _lang_for_path(rel)
    suffix = f" _({label})_" if label else ""
    return f"### {rel}{suffix}\n```{lang}\n{content}\n```"


def _collect_files_by_rel_paths(paths: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}

    for rel in sorted(set(paths)):
        try:
            path = _resolve_artifact_path(rel)
        except Exception as exc:
            files[rel] = f"[invalid path: {exc}]"
            continue

        files[rel] = _read_file_for_briefing(path)

    return files


def _collect_ts_files(root: Any, prefix: str) -> dict[str, str]:
    files: dict[str, str] = {}

    if not root.exists():
        return files

    for ext in ("*.ts", "*.tsx"):
        for path in sorted(root.rglob(ext)):
            rel = prefix + "/" + str(path.relative_to(root)).replace("\\", "/")
            files[rel] = _read_file_for_briefing(path)

    return files


def _collect_changed_src_files() -> dict[str, str]:
    """
    Collect source files whose content differs from scaffold stubs.
    Falls back to all TS/TSX source files.
    """
    stub_map: dict[str, str] = {}

    if SCAFFOLD_JSON.exists():
        scaffold = _read_json(SCAFFOLD_JSON, {})
        if isinstance(scaffold, dict):
            maybe_stub_map = scaffold.get("stub_map", {})
            if isinstance(maybe_stub_map, dict):
                stub_map = {
                    str(k): str(v)
                    for k, v in maybe_stub_map.items()
                }

            # Fallback for current scaffolder schema: derive stub_map from files[].
            files = scaffold.get("files", [])
            if not stub_map and isinstance(files, list):
                for entry in files:
                    if (
                        isinstance(entry, dict)
                        and not entry.get("is_test")
                        and isinstance(entry.get("file_path"), str)
                        and isinstance(entry.get("code"), str)
                    ):
                        stub_map[entry["file_path"]] = entry["code"]

    changed: dict[str, str] = {}

    if not SRC_DIR.exists():
        return changed

    for ext in ("*.ts", "*.tsx"):
        for path in sorted(SRC_DIR.rglob(ext)):
            rel = "src/" + str(path.relative_to(SRC_DIR)).replace("\\", "/")
            current = _read_file_for_briefing(path)
            stub = stub_map.get(rel, "")

            if not stub or current.strip() != stub.strip():
                changed[rel] = current

    if changed:
        return changed

    return _collect_ts_files(SRC_DIR, "src")


def _affected_src_set(delta: dict[str, Any] | None) -> set[str]:
    if delta is None or delta.get("is_first_run", True):
        return set()

    affected = delta.get("affected_files", [])
    if not isinstance(affected, list):
        return set()

    return {
        str(item)
        for item in affected
        if isinstance(item, str) and item.startswith("src/")
    }


# ─────────────────────────────────────────────────────────────────────────────
# Briefing sections
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_or_partial_context(delta: dict[str, Any] | None) -> tuple[str, bool, set[str]]:
    affected_set = _affected_src_set(delta)
    is_partial = bool(affected_set)

    if is_partial and delta:
        fv = delta.get("from_version") or "?"
        tv = delta.get("to_version", "?")
        changed_secs = delta.get("changed_sections", [])
        summaries = delta.get("section_summaries", {})
        if not isinstance(changed_secs, list):
            changed_secs = []
        if not isinstance(summaries, dict):
            summaries = {}

        lines = [
            "## 0. Run context",
            "",
            "**This is a PARTIAL run** — spec changed from "
            f"`{fv}` to `{tv}`.",
            "",
            f"Changed spec sections: `{changed_secs}`",
            "",
            "**Changed sections:**",
        ]

        for sec in changed_secs:
            note = summaries.get(str(sec), summaries.get(sec, ""))
            lines.append(f"- §{sec}: {note}")

        lines += [
            "",
            "**Files re-implemented this run — primary review focus:**",
        ]

        for fp in sorted(affected_set):
            lines.append(f"- `{fp}`")

        skipped = _extract_file_list(_load_impl_record().get("skipped_delta", []))
        if skipped:
            lines += [
                "",
                "**Files reused from previous run — secondary review only:**",
            ]
            for fp in skipped:
                lines.append(f"- `{fp}`")

        lines += [
            "",
            "**Review instructions:**",
            "- Focus spec-compliance and logic review on re-implemented files.",
            "- For reused files, only flag issues if they interact with changed files.",
            "- Do NOT block approval for issues in reused files that predate this run.",
        ]

        return "\n".join(lines), True, affected_set

    return (
        "## 0. Run context\n\n"
        "**This is a FULL run** — review all implemented files equally.",
        False,
        set(),
    )


def _build_mini_context(
    plan_mini: dict[str, Any],
    analysis_mini: dict[str, Any],
    impl_record: dict[str, Any],
) -> str:
    target_files = _mini_target_files(plan_mini, impl_record)

    lines = [
        "## 0. Run context",
        "",
        "**This is a MINI targeted run.**",
        "",
        "**Primary review focus:**",
        "- Files listed in `planner_mini_execution_plan.target_files`",
        "- Files written in `executor_overwrite_manifest.files`",
        "",
        "**Scope rule:**",
        "- Do NOT block approval for unrelated pre-existing issues outside target scope.",
        "- If an outside-scope issue directly breaks the targeted task, mention it clearly.",
        "- If a required fix would broaden scope beyond target files, mark it as a follow-up.",
        "",
        "**Target / implemented files:**",
    ]

    if target_files:
        for fp in target_files:
            lines.append(f"- `{fp}`")
    else:
        lines.append("- _(none found)_")

    mode = impl_record.get("mode", "unknown")
    lines += [
        "",
        f"Implementation mode: `{mode}`",
    ]

    if analysis_mini:
        lines += [
            "",
            "**Planner impact analysis is included below and should be used to evaluate scope safety.**",
        ]

    return "\n".join(lines)


def _append_full_sections(parts: list[str], is_partial: bool, affected_set: set[str]) -> None:
    # 1. Spec
    spec = _load_spec_optional()
    if spec:
        parts.append("## 1. Canonical Spec\n\n" + spec)
    else:
        parts.append("## 1. Canonical Spec\n\n_[missing]_")

    # 1b. Spec gaps / addendum
    addendum = _read_text(ARCHIVIST_SPEC_GAPS)
    if addendum:
        parts.append(
            "## 1b. Archivist Spec Gaps\n\n"
            + addendum
        )

    # 2. Planner plan
    if PLANNER_FULL_PLAN.exists():
        plan = _read_json(PLANNER_FULL_PLAN, {})
        parts.append(
            "## 2. Planner Full Execution Plan\n\n"
            f"```json\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n```"
        )
    else:
        parts.append(
            "## 2. Planner Full Execution Plan\n\n"
            "_Not available._"
        )

    # 3. Implementation record
    impl_record = _load_impl_record()
    if impl_record:
        parts.append(
            "## 3. Executor Overwrite Manifest\n\n"
            f"```json\n{json.dumps(impl_record, indent=2, ensure_ascii=False)}\n```"
        )

    # 4. Test report
    test_report = _load_test_report()
    if test_report:
        parts.append(
            "## 4. Debugger Overwrite Test Summary\n\n"
            f"```json\n{json.dumps(test_report, indent=2, ensure_ascii=False)}\n```"
        )

    # 5. Source files
    if is_partial and affected_set:
        primary = _collect_files_by_rel_paths(sorted(affected_set))
        src_block = "\n\n".join(
            _format_file_block(fp, code, "re-implemented")
            for fp, code in primary.items()
        )
        parts.append(
            f"## 5. Re-implemented Source Files ({len(primary)} files)\n\n{src_block}"
        )

        skipped = _extract_file_list(impl_record.get("skipped_delta", []))
        if skipped:
            secondary = _collect_files_by_rel_paths(skipped)
            secondary_block = "\n\n".join(
                _format_file_block(fp, _signature_preview(code), "reused, signature preview")
                for fp, code in secondary.items()
            )
            parts.append(
                f"## 5b. Reused Files — signature preview ({len(secondary)} files)\n\n"
                + secondary_block
            )
    else:
        src_files = _collect_changed_src_files()
        src_block = "\n\n".join(
            _format_file_block(fp, code)
            for fp, code in src_files.items()
        )
        parts.append(
            f"## 5. Implemented Source Files ({len(src_files)} files)\n\n{src_block}"
        )

    # 6. Test files
    test_files = _collect_ts_files(TESTS_DIR, "tests")
    test_block = "\n\n".join(
        _format_file_block(fp, code)
        for fp, code in test_files.items()
    )
    parts.append(f"## 6. Test Files ({len(test_files)} files)\n\n{test_block}")


def _append_mini_sections(
    parts: list[str],
    plan_mini: dict[str, Any],
    analysis_mini: dict[str, Any],
    impl_record: dict[str, Any],
) -> None:
    clarified = _read_text(CLARIFIED_REQ)
    enriched = _read_text(ENRICHER_OVERWRITE_PROMPT)
    test_report = _load_test_report()

    if clarified:
        parts.append("## 1. Clarificator Requirement Synthesis\n\n" + clarified)
    else:
        parts.append("## 1. Clarificator Requirement Synthesis\n\n_[missing]_")

    if enriched:
        parts.append("## 1b. Enricher Overwrite Enriched Prompt\n\n" + enriched)

    if plan_mini:
        parts.append(
            "## 2. Planner Mini Execution Plan\n\n"
            f"```json\n{json.dumps(plan_mini, indent=2, ensure_ascii=False)}\n```"
        )
    else:
        parts.append("## 2. Planner Mini Execution Plan\n\n_[missing]_")

    if analysis_mini:
        parts.append(
            "## 3. Planner Mini Impact Analysis\n\n"
            f"```json\n{json.dumps(analysis_mini, indent=2, ensure_ascii=False)}\n```"
        )
    else:
        parts.append("## 3. Planner Mini Impact Analysis\n\n_[missing]_")

    if impl_record:
        parts.append(
            "## 4. Executor Overwrite Manifest\n\n"
            f"```json\n{json.dumps(impl_record, indent=2, ensure_ascii=False)}\n```"
        )

    if test_report:
        parts.append(
            "## 5. Debugger Overwrite Test Summary\n\n"
            f"```json\n{json.dumps(test_report, indent=2, ensure_ascii=False)}\n```"
        )

    target_files = _mini_target_files(plan_mini, impl_record)
    file_map = _collect_files_by_rel_paths(target_files)

    file_blocks = "\n\n".join(
        _format_file_block(fp, content, "mini target/implemented")
        for fp, content in file_map.items()
    )

    parts.append(
        f"## 6. Mini Target / Implemented Files ({len(file_map)} files)\n\n"
        + (file_blocks or "_No target files collected._")
    )


def _signature_preview(content: str, max_lines: int = 40) -> str:
    lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        if "throw new Error" in stripped:
            continue

        if any(token in stripped for token in (
            "export ",
            "function ",
            "class ",
            "interface ",
            "type ",
            "const ",
            "def ",
        )):
            lines.append(line)

        if len(lines) >= max_lines:
            lines.append("... [signature preview truncated]")
            break

    return "\n".join(lines) if lines else "\n".join(content.splitlines()[:max_lines])


def build_briefing(max_chars: int = MAX_BRIEFING_CHARS) -> str:
    parts: list[str] = []

    scope = _detect_scope()
    impl_record = _load_impl_record()

    if scope == "mini":
        plan_mini = _load_plan_mini()
        analysis_mini = _load_analysis_mini()

        parts.append(_build_mini_context(plan_mini, analysis_mini, impl_record))
        _append_mini_sections(parts, plan_mini, analysis_mini, impl_record)
    else:
        delta = _load_delta()
        context, is_partial, affected_set = _build_full_or_partial_context(delta)
        parts.append(context)
        _append_full_sections(parts, is_partial, affected_set)

    briefing = "\n\n---\n\n".join(parts)

    if len(briefing) > max_chars:
        briefing = (
            briefing[:max_chars]
            + f"\n\n[BRIEFING TRUNCATED at {max_chars:,} chars; original size {len(briefing):,}]"
        )

    return briefing


# ─────────────────────────────────────────────────────────────────────────────
# API call
# ─────────────────────────────────────────────────────────────────────────────

def call_deepseek_judge(
    briefing: str,
    *,
    api_key: str,
    model: str,
) -> tuple[str, list[Any] | None, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": briefing},
        ],
        "reasoning": {"enabled": True},
        "temperature": 0.1,
        "max_tokens": 16000,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"[11] Calling judge model: {model} …")

    last_error = None

    with httpx.Client(timeout=300) as client:
        for attempt in range(2):
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            usage = data.get("usage", {})
            prompt_t = usage.get("prompt_tokens", "?")
            completion_t = usage.get("completion_tokens", "?")
            print(f"[11] Tokens: prompt={prompt_t}, completion={completion_t}")

            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            reasoning_details = msg.get("reasoning_details")

            if tool_calls:
                raise RuntimeError(
                    f"Judge returned tool_calls instead of text: {tool_calls}"
                )

            if content and content.strip():
                return content.strip(), reasoning_details, usage

            last_error = f"Empty content. finish_reason={finish_reason}, message={msg}"
            print(f"[11][warn] {last_error}", file=sys.stderr)

            if attempt == 0:
                print("[11] Retrying in 3s …", file=sys.stderr)
                time.sleep(3)

    raise RuntimeError(f"Judge failed after retries: {last_error}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            print("[11][error] No JSON object found in judge response.", file=sys.stderr)
            print(f"[11][error] Raw first 1000 chars:\n{text[:1000]}", file=sys.stderr)
            sys.exit(1)

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            print(f"[11][error] JSON parse failed: {exc}", file=sys.stderr)
            print(f"[11][error] Raw first 1000 chars:\n{text[:1000]}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(parsed, dict):
        print("[11][error] Judge JSON top-level is not an object.", file=sys.stderr)
        sys.exit(1)

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Report renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_report(review: dict[str, Any], *, model: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = review.get("verdict", "UNKNOWN")
    run_type = review.get("run_type", _detect_scope())

    verdict_icon = {
        "APPROVED": "✅",
        "APPROVED_WITH_NOTES": "⚠️",
        "NEEDS_REVISION": "❌",
    }.get(verdict, "❓")

    lines = [
        "# Judge Verdict Summary — Final Review",
        f"_Generated: {now}_",
        f"_Model: {model}_",
        f"_Project: **{get_project_name()}**_",
        f"_Project slug: **{get_project_slug()}**_",
        f"_Session: **{get_session_id() or 'legacy/no-session'}**_",
        f"_Run type: **{run_type}**_",
        "",
        f"## Verdict: {verdict_icon} {verdict}",
        "",
        f"> {review.get('summary', '')}",
        "",
        "## Scores",
        "",
        "| Dimension | Score | Notes |",
        "|---|---|---|",
    ]

    sections = review.get("sections", {})
    if not isinstance(sections, dict):
        sections = {}

    dimension_labels = {
        "requirement_compliance": "Requirement / Spec Compliance",
        "spec_compliance": "Spec Compliance",
        "code_quality": "Code Quality",
        "test_quality": "Test / Verifier Quality",
        "architecture_scope": "Architecture / Scope",
        "architecture": "Architecture",
        "gaps_risks": "Gaps / Risks",
    }

    emitted = set()
    for key, label in dimension_labels.items():
        sec = sections.get(key)
        if not isinstance(sec, dict):
            continue

        score = sec.get("score", "—")
        notes = str(sec.get("notes", "—")).replace("\n", " ")
        score_str = f"{score}/5" if isinstance(score, int) else str(score)

        lines.append(f"| {label} | {score_str} | {notes} |")
        emitted.add(key)

    if not emitted:
        lines.append("| — | — | No structured scores returned |")

    blocking = review.get("blocking_issues", [])
    lines += ["", "## Blocking Issues", ""]
    if isinstance(blocking, list) and blocking:
        for issue in blocking:
            lines.append(f"- ❌ {issue}")
    else:
        lines.append("_None — all checks passed._")

    notes_list = review.get("non_blocking_notes", [])
    lines += ["", "## Non-blocking Notes", ""]
    if isinstance(notes_list, list) and notes_list:
        for note in notes_list:
            lines.append(f"- ℹ️ {note}")
    else:
        lines.append("_None._")

    partial_notes = review.get("partial_run_notes")
    if partial_notes:
        lines += ["", "## Partial Run Notes", "", str(partial_notes), ""]

    mini_notes = review.get("mini_run_notes")
    if mini_notes:
        lines += ["", "## Mini Run Notes", "", str(mini_notes), ""]

    lines += [
        "",
        "---",
        f"**Sign-off:** {review.get('sign_off', model)}",
    ]

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_raw_verdict(
    *,
    model: str,
    scope: str,
    briefing_chars: int,
    usage: dict[str, Any],
    raw_response: str,
    reasoning_details: list[Any] | None,
) -> None:
    JUDGE_OVERWRITE_VERDICT_RAW.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_OVERWRITE_VERDICT_RAW.write_text(
        json.dumps(
            {
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "project": get_project_name(),
                "project_slug": get_project_slug(),
                "session_id": get_session_id(),
                "scope_detected": scope,
                "briefing_chars": briefing_chars,
                "usage": usage,
                "response": raw_response,
                "reasoning_details": reasoning_details,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _track_write(JUDGE_OVERWRITE_VERDICT_RAW)
    print(f"[11] Raw response + reasoning saved → {JUDGE_OVERWRITE_VERDICT_RAW}")


def _write_report(report_md: str) -> None:
    JUDGE_VERDICT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_VERDICT_SUMMARY.write_text(report_md, encoding="utf-8")
    _track_write(JUDGE_VERDICT_SUMMARY)
    print(f"\n[11] Judge verdict summary written → {JUDGE_VERDICT_SUMMARY}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, args.session, parser)

    # Important: project/session env must be configured before ensure_dirs().
    ensure_dirs()

    exit_code = 0

    try:
        api_key = _require_openrouter_key(parser)

        scope = _detect_scope()
        print(f"[11] Project: {get_project_name()} ({get_project_slug()})")
        print(f"[11] Session: {get_session_id() or 'legacy/no-session'}")
        print(f"[11] Scope detected: {scope}")
        print("[11] Building pipeline briefing …")

        briefing = build_briefing(max_chars=args.max_briefing_chars)
        print(f"[11] Briefing size: {len(briefing):,} chars")

        raw_response, reasoning_details, usage = call_deepseek_judge(
            briefing,
            api_key=api_key,
            model=args.model,
        )

        _write_raw_verdict(
            model=args.model,
            scope=scope,
            briefing_chars=len(briefing),
            usage=usage,
            raw_response=raw_response,
            reasoning_details=reasoning_details,
        )

        review = _parse_json(raw_response)

        if "run_type" not in review:
            review["run_type"] = "mini" if scope == "mini" else "full"

        report_md = render_report(review, model=args.model)
        _write_report(report_md)

        print(f"\n{'=' * 60}")
        print(report_md)
        print(f"{'=' * 60}")

        verdict = review.get("verdict", "")
        if verdict == "NEEDS_REVISION":
            print("[11] Judge verdict: NEEDS_REVISION — blocking issues found.", file=sys.stderr)
            exit_code = 1
        else:
            print(f"[11] Judge verdict: {verdict} ✅")

    except Exception as exc:
        print(f"[11][error] Judge failed: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
