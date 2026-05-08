"""
pipeline/06_judge_deepseek.py
=============================
Step 6 — DeepSeek V3.2 as Judge / Validator.

Runs after verification/tests have passed. Aggregates pipeline artifacts into a
single briefing, sends it to the judge model for final review, and writes:

  artifacts_<slug>/reports/judge_report.md
  artifacts_<slug>/run/judge_raw.json

Supports both:
  - FULL/PARTIAL flow:
      spec.md/spec_compressed.md, plan.json, scaffold.json, test_report.json,
      impl_record.json, source files, test files, spec_delta.json.
  - MINI targeted flow:
      clarified_requirement.md, enriched_prompt.md, plan_mini.json,
      analysis_mini.json, impl_record.json, test_report.json, and only the
      target/implemented files.

Direct execution:
  python 06_judge_deepseek.py --project my-app
  PIPELINE_PROJECT=my-app python 06_judge_deepseek.py

Required environment:
  OPENROUTER_API_KEY=<your-key>

For taxonomy details see docs/artifacts.md
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

# === WRITE AUTHORITY: 06_judge_deepseek ===
# OWNS  : artifacts_<slug>/run/judge_raw.json
#         artifacts_<slug>/reports/judge_report.md
# READS : spec/plans/impl/test/source artifacts

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ANALYSIS_MINI,
    CLARIFIED_REQ,
    ENRICHED_PROMPT,
    IMPL_RECORD,
    JUDGE_RAW,
    JUDGE_REPORT,
    PLAN_JSON as GLM_PLAN_PATH,
    PLAN_MINI,
    SCAFFOLD_JSON,
    SPEC_ADDENDUM,
    SPEC_COMPRESSED,
    SPEC_DELTA,
    SPEC_PATH,
    SRC_DIR,
    TEST_REPORT,
    TESTS_DIR,
    artifact_root,
    ensure_dirs,
)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"
MAX_BRIEFING_CHARS = 900_000
MAX_FILE_CHARS = 80_000


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
- Review ONLY the target files listed in plan_mini.target_files and files written
  in impl_record.files.
- Do NOT block approval for unrelated pre-existing issues outside the mini target scope.
- If you notice a problem outside the target scope, mention it as a non-blocking
  note or as "requires broader follow-up", unless it directly breaks the targeted task.
- Judge whether the targeted change satisfies the clarified requirement and respects
  plan_mini constraints.

Review dimensions:
A. REQUIREMENT / SPEC COMPLIANCE
   - FULL/PARTIAL: spec compliance, acceptance criteria.
   - MINI: clarified requirement + plan_mini compliance.
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
  "sign_off": "DeepSeek V3.2 + timestamp placeholder"
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
# CLI / project setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final judge review over pipeline artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 06_judge_deepseek.py --project my-app
              PIPELINE_PROJECT=my-app python 06_judge_deepseek.py

              python 06_judge_deepseek.py --project my-app --model deepseek/deepseek-v3.2
        """),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
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
    parser: argparse.ArgumentParser,
) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 06_judge_deepseek.py directly."
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

def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        print(f"[06][warn] Could not parse JSON {path}: {exc}", file=sys.stderr)
        return default


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(errors="replace")
    except Exception as exc:
        print(f"[06][warn] Could not read {path}: {exc}", file=sys.stderr)
        return ""


def _load_impl_record() -> dict[str, Any]:
    rec = _read_json(IMPL_RECORD, {})
    return rec if isinstance(rec, dict) else {}


def _load_plan_mini() -> dict[str, Any]:
    plan = _read_json(PLAN_MINI, {})
    return plan if isinstance(plan, dict) else {}


def _load_analysis_mini() -> dict[str, Any]:
    analysis = _read_json(ANALYSIS_MINI, {})
    return analysis if isinstance(analysis, dict) else {}


def _load_delta() -> dict[str, Any] | None:
    delta = _read_json(SPEC_DELTA, None)
    return delta if isinstance(delta, dict) else None


def _load_spec_optional() -> str:
    if SPEC_COMPRESSED.exists():
        return _read_text(SPEC_COMPRESSED)
    return _read_text(SPEC_PATH)


def _load_test_report() -> dict[str, Any]:
    report = _read_json(TEST_REPORT, {})
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

    if PLAN_MINI.exists() or ANALYSIS_MINI.exists():
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


def _collect_ts_files(root: Path, prefix: str) -> dict[str, str]:
    files: dict[str, str] = {}

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

    changed: dict[str, str] = {}

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
        "- Files listed in `plan_mini.target_files`",
        "- Files written in `impl_record.files`",
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

    return "\n".join(lines)


def _append_full_sections(parts: list[str], is_partial: bool, affected_set: set[str]) -> None:
    # 1. Spec
    spec = _load_spec_optional()
    if spec:
        parts.append("## 1. spec.md\n\n" + spec)
    else:
        parts.append("## 1. spec.md\n\n_[missing]_")

    # 1b. Spec addendum
    addendum = _read_text(SPEC_ADDENDUM)
    if addendum:
        parts.append(
            "## 1b. Spec Addendum\n\n"
            + addendum
        )

    # 2. GLM plan
    if GLM_PLAN_PATH.exists():
        plan = _read_json(GLM_PLAN_PATH, {})
        parts.append(
            "## 2. GLM 5.1 Architectural Plan\n\n"
            f"```json\n{json.dumps(plan, indent=2)}\n```"
        )
    else:
        parts.append(
            "## 2. GLM 5.1 Architectural Plan\n\n"
            "_Not available._"
        )

    # 3. Implementation record
    impl_record = _load_impl_record()
    if impl_record:
        parts.append(
            "## 3. Implementation Record\n\n"
            f"```json\n{json.dumps(impl_record, indent=2)}\n```"
        )

    # 4. Test report
    test_report = _load_test_report()
    if test_report:
        parts.append(
            "## 4. Test / Verification Report\n\n"
            f"```json\n{json.dumps(test_report, indent=2)}\n```"
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
    enriched = _read_text(ENRICHED_PROMPT)
    test_report = _load_test_report()

    if clarified:
        parts.append("## 1. Clarified Requirement\n\n" + clarified)
    else:
        parts.append("## 1. Clarified Requirement\n\n_[missing]_")

    if enriched:
        parts.append("## 1b. Enriched Prompt\n\n" + enriched)

    if plan_mini:
        parts.append(
            "## 2. plan_mini.json\n\n"
            f"```json\n{json.dumps(plan_mini, indent=2)}\n```"
        )
    else:
        parts.append("## 2. plan_mini.json\n\n_[missing]_")

    if analysis_mini:
        parts.append(
            "## 3. analysis_mini.json\n\n"
            f"```json\n{json.dumps(analysis_mini, indent=2)}\n```"
        )
    else:
        parts.append("## 3. analysis_mini.json\n\n_[missing]_")

    if impl_record:
        parts.append(
            "## 4. impl_record.json\n\n"
            f"```json\n{json.dumps(impl_record, indent=2)}\n```"
        )

    if test_report:
        parts.append(
            "## 5. Test / Verification Report\n\n"
            f"```json\n{json.dumps(test_report, indent=2)}\n```"
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

    print(f"[06] Calling judge model: {model} …")

    last_error = None

    with httpx.Client(timeout=300) as client:
        for attempt in range(2):
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            usage = data.get("usage", {})
            prompt_t = usage.get("prompt_tokens", "?")
            completion_t = usage.get("completion_tokens", "?")
            print(f"[06] Tokens: prompt={prompt_t}, completion={completion_t}")

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
            print(f"[06][warn] {last_error}", file=sys.stderr)

            if attempt == 0:
                print("[06] Retrying in 3s …", file=sys.stderr)
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
            print("[06][error] No JSON object found in judge response.", file=sys.stderr)
            print(f"[06][error] Raw first 1000 chars:\n{text[:1000]}", file=sys.stderr)
            sys.exit(1)

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as exc:
            print(f"[06][error] JSON parse failed: {exc}", file=sys.stderr)
            print(f"[06][error] Raw first 1000 chars:\n{text[:1000]}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(parsed, dict):
        print("[06][error] Judge JSON top-level is not an object.", file=sys.stderr)
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
        "# Judge Report — Final Review",
        f"_Generated: {now}_",
        f"_Model: {model}_",
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: project env must be configured before ensure_dirs().
    ensure_dirs()

    api_key = _require_openrouter_key(parser)

    scope = _detect_scope()
    print(f"[06] Scope detected: {scope}")
    print("[06] Building pipeline briefing …")

    briefing = build_briefing(max_chars=args.max_briefing_chars)
    print(f"[06] Briefing size: {len(briefing):,} chars")

    raw_response, reasoning_details, usage = call_deepseek_judge(
        briefing,
        api_key=api_key,
        model=args.model,
    )

    JUDGE_RAW.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_RAW.write_text(json.dumps({
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope_detected": scope,
        "briefing_chars": len(briefing),
        "usage": usage,
        "response": raw_response,
        "reasoning_details": reasoning_details,
    }, indent=2))
    print(f"[06] Raw response + reasoning saved → {JUDGE_RAW}")

    review = _parse_json(raw_response)

    if "run_type" not in review:
        review["run_type"] = "mini" if scope == "mini" else "full"

    report_md = render_report(review, model=args.model)

    JUDGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_REPORT.write_text(report_md)

    print(f"\n[06] Judge report written → {JUDGE_REPORT}")
    print(f"\n{'=' * 60}")
    print(report_md)
    print(f"{'=' * 60}")

    verdict = review.get("verdict", "")
    if verdict == "NEEDS_REVISION":
        print("[06] Judge verdict: NEEDS_REVISION — blocking issues found.", file=sys.stderr)
        sys.exit(1)

    print(f"[06] Judge verdict: {verdict} ✅")


if __name__ == "__main__":
    main()
