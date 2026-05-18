"""
pipeline/11_judge.py
====================
Step 11 — Judge / Validator.

Runs after verification/tests have passed. Aggregates pipeline artifacts into a
single briefing, sends it to the judge model for final review, and writes:

  artifacts_<slug>/judge/verdict_raw.json      (short-term, overwrite)
  artifacts_<slug>/judge/verdict_summary.md    (short-term, overwrite)
  artifacts_<slug>/judge/verdict_log.json      (long-term, append)

Supports both:
  - FULL/PARTIAL flow:
      spec/specwright_spec_<slug>.md,
      planner/full_plan.json,
      scaffolder/blueprint.json,
      debugger/test_summary.json,
      executor/manifest.json,
      output/src/ files,
      output/tests/ files,
      spectracker/version_delta.json.

  - MINI targeted flow:
      clarificator/session.json,
      enricher/enriched_prompt.md,
      planner/mini_plan.json (includes impact field),
      executor/manifest.json,
      debugger/test_summary.json,
      and only the target/implemented files.

Direct execution:
  python 11_judge.py --project my-app
  PIPELINE_PROJECT=my-app python 11_judge.py

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


# === WRITE AUTHORITY: judge ===
# OWNS  : artifacts_<slug>/judge/verdict_raw.json
#         artifacts_<slug>/judge/verdict_summary.md
#         artifacts_<slug>/judge/verdict_log.json
# READS : artifacts_<slug>/spec/specwright_spec_<slug>.md
#         artifacts_<slug>/planner/full_plan.json
#         artifacts_<slug>/planner/mini_plan.json
#         artifacts_<slug>/scaffolder/blueprint.json
#         artifacts_<slug>/clarificator/session.json
#         artifacts_<slug>/spectracker/version_delta.json
#         artifacts_<slug>/enricher/enriched_prompt.md
#         artifacts_<slug>/executor/manifest.json
#         artifacts_<slug>/debugger/test_summary.json
#         artifacts_<slug>/archivist/spec_gaps.md
#         artifacts_<slug>/output/src/**
#         artifacts_<slug>/output/tests/**

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ARCHIVIST_SPEC_GAPS,
    CLARIFIED_REQ,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    ENRICHER_OVERWRITE_PROMPT,
    EXECUTOR_OVERWRITE_MANIFEST,
    JUDGE_OVERWRITE_VERDICT_RAW,
    JUDGE_VERDICT_LOG,
    JUDGE_VERDICT_SUMMARY,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    SCAFFOLD_JSON,
    SPECTRACKER_VERSION_DELTA,
    SRC_DIR,
    TESTS_DIR,
    artifact_root,
    ensure_dirs,
    get_project_name,
    get_project_slug,
    get_spec_path,
)
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402


# Model identity resolved from artifacts/models.py role "judge".
# Reasoning is enabled by default for judge via REASONING_OVERRIDES in models.py.
MAX_BRIEFING_CHARS = 900_000
MAX_FILE_CHARS = 80_000


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

ROLE = "judge"

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
              PIPELINE_PROJECT=my-app python 11_judge.py

              # Model is resolved from artifacts/models.py role "judge".
              # To change: edit ROLES["judge"] in models.py.
        """),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
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
    parser: argparse.ArgumentParser,
) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 11_judge.py directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe loaders
# ─────────────────────────────────────────────────────────────────────────────

def _read_json(path: Any, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        track_read(path)
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        print(f"[11][warn] Could not parse JSON {path}: {exc}", file=sys.stderr)
        return default


def _read_text(path: Any) -> str:
    if not path.exists():
        return ""
    try:
        track_read(path)
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


def _load_analysis_mini(plan_mini: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract impact analysis from mini_plan["impact"] field (merged by planner)."""
    if plan_mini is None:
        plan_mini = _load_plan_mini()
    impact = plan_mini.get("impact", {})
    return impact if isinstance(impact, dict) else {}


def _load_delta() -> dict[str, Any] | None:
    delta = _read_json(SPECTRACKER_VERSION_DELTA, None)
    return delta if isinstance(delta, dict) else None


def _load_spec_optional() -> str:
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

    if PLANNER_MINI_PLAN.exists():
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
    if raw.startswith("output/src/"):
        return SRC_DIR / raw[len("output/src/"):]
    if raw.startswith("tests/"):
        return TESTS_DIR / raw[len("tests/"):]
    if raw.startswith("output/tests/"):
        return TESTS_DIR / raw[len("output/tests/"):]

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

    track_read(path)
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
            rel = "output/src/" + str(path.relative_to(SRC_DIR)).replace("\\", "/")
            current = _read_file_for_briefing(path)
            stub = stub_map.get(rel, "")

            if not stub or current.strip() != stub.strip():
                changed[rel] = current

    if changed:
        return changed

    return _collect_ts_files(SRC_DIR, "output/src")


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
    test_files = _collect_ts_files(TESTS_DIR, "output/tests")
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
        analysis_mini = _load_analysis_mini(plan_mini)

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

def call_judge(
    briefing: str,
) -> tuple[str, list[Any] | None, dict[str, Any]]:
    """
    Call the judge model via the central model registry.
    Model identity, provider, and reasoning flag are resolved from
    artifacts/models.py role "judge". Reasoning is ON by default for judge.
    """
    model_id = get_model(ROLE)
    print(f"[11] Calling judge model: {model_id} …")

    last_error = None

    for attempt in range(2):
        resp = call_model(
            ROLE,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": briefing},
            ],
            temperature=0.1,
            max_tokens=32768,
        )

        usage = resp.usage
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=model_id, provider=get_provider(ROLE))
            print_call(__file__, pt, ct, call_cost)

        choice = resp.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None)
        finish_reason = getattr(choice, "finish_reason", None)
        reasoning_details = getattr(msg, "reasoning_details", None)

        if tool_calls:
            raise RuntimeError(
                f"Judge returned tool_calls instead of text: {tool_calls}"
            )

        if content and content.strip():
            usage_dict = dict(usage) if usage else {}
            return content.strip(), reasoning_details, usage_dict

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

def render_report(review: dict[str, Any]) -> str:
    model = get_model(ROLE)
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
    scope: str,
    briefing_chars: int,
    usage: dict[str, Any],
    raw_response: str,
    reasoning_details: list[Any] | None,
) -> None:
    model = get_model(ROLE)
    JUDGE_OVERWRITE_VERDICT_RAW.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_OVERWRITE_VERDICT_RAW.write_text(
        json.dumps(
            {
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "project": get_project_name(),
                "project_slug": get_project_slug(),
                "scope_detected": scope,
                "briefing_chars": briefing_chars,
                "usage": usage,
                "response": raw_response,
                "reasoning_details": reasoning_details,
            },
            indent=2,
            ensure_ascii=False,
            default=lambda o: vars(o) if hasattr(o, '__dict__') else str(o),
        ),
        encoding="utf-8",
    )
    track_write(JUDGE_OVERWRITE_VERDICT_RAW)
    print(f"[11] Raw response + reasoning saved → {JUDGE_OVERWRITE_VERDICT_RAW}")


def _append_verdict_log(review: dict[str, Any], scope: str) -> None:
    """Append a trimmed verdict entry to the long-term verdict_log.json."""
    log_path = JUDGE_VERDICT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if log_path.exists():
            track_read(log_path)
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        else:
            existing = {}
    except Exception:
        existing = {}

    if not isinstance(existing, dict):
        existing = {}

    entries: list[dict[str, Any]] = existing.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    entry: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": get_project_name(),
        "project_slug": get_project_slug(),
        "scope": scope,
        "verdict": review.get("verdict"),
        "run_type": review.get("run_type"),
        "summary": review.get("summary"),
        "sections": review.get("sections"),
        "blocking_issues": review.get("blocking_issues", []),
        "non_blocking_notes": review.get("non_blocking_notes", []),
    }

    entries.append(entry)
    log_path.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False,
                   default=lambda o: vars(o) if hasattr(o, '__dict__') else str(o)),
        encoding="utf-8",
    )
    track_write(log_path)
    print(f"[11] Verdict log appended → {log_path}")


def _write_report(report_md: str, review: dict[str, Any], scope: str) -> None:
    JUDGE_VERDICT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_VERDICT_SUMMARY.write_text(apply_md_header(report_md, JUDGE_VERDICT_SUMMARY, owner="11_judge.py"), encoding="utf-8")
    track_write(JUDGE_VERDICT_SUMMARY)
    print(f"\n[11] Judge verdict summary written → {JUDGE_VERDICT_SUMMARY}")
    _append_verdict_log(review, scope)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: project/session env must be configured before ensure_dirs().
    ensure_dirs()

    exit_code = 0

    try:
        scope = _detect_scope()
        print(f"[11] Project: {get_project_name()} ({get_project_slug()})")
        print(f"[11] Scope detected: {scope}")
        print("[11] Building pipeline briefing …")

        briefing = build_briefing(max_chars=args.max_briefing_chars)
        print(f"[11] Briefing size: {len(briefing):,} chars")

        raw_response, reasoning_details, usage = call_judge(briefing)

        _write_raw_verdict(
            scope=scope,
            briefing_chars=len(briefing),
            usage=usage,
            raw_response=raw_response,
            reasoning_details=reasoning_details,
        )

        review = _parse_json(raw_response)

        if "run_type" not in review:
            review["run_type"] = "mini" if scope == "mini" else "full"

        report_md = render_report(review)
        _write_report(report_md, review, scope)

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
        print_summary("[11]")
        print_artifact_summary("[11]")
        prompt_next_step(ROLE, prefix="[11]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
