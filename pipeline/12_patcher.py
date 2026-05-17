"""
pipeline/12_patcher.py
======================
Step 12 — Fix blocking issues identified by the judge.

Called automatically by harness.py when 11_judge.py exits with verdict
NEEDS_REVISION. Human approval is not required for judge-classified blocking
issues, but this script must still enforce scope safety.

Supports:
  - FULL flow:
      Uses canonical spec + judge finding +
      affected output/src/ files. Patch scope defaults to output/src/** only.
      Never patches tests.

  - MINI targeted flow:
      Uses clarificator/session.json +
      planner/mini_plan.json (includes impact field) +
      executor/manifest.json + judge finding.
      Patch scope is strictly limited to planner_mini_execution_plan.target_files.

Writes:
  artifacts_<slug>/patcher/fix_summary.md      (short-term, overwrite)
  artifacts_<slug>/patcher/attempt_log.json    (long-term, append)
  target source/config/query files when allowed

Reads:
  artifacts_<slug>/judge/verdict_raw.json
  artifacts_<slug>/executor/manifest.json
  artifacts_<slug>/debugger/test_summary.json
  artifacts_<slug>/planner/mini_plan.json
  artifacts_<slug>/planner/full_plan.json
  artifacts_<slug>/clarificator/session.json
  artifacts_<slug>/enricher/enriched_prompt.md
  artifacts_<slug>/archivist/knowledge_log.md
  artifacts_<slug>/spec/specwright_spec_<slug>.md
  artifacts_<slug>/output/src/**

Direct execution:
  python 12_patcher.py --project my-app
  PIPELINE_PROJECT=my-app python 12_patcher.py

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# === WRITE AUTHORITY: patcher ===
# OWNS  : artifacts_<slug>/patcher/fix_summary.md
#         artifacts_<slug>/patcher/attempt_log.json
#         allowed source/target files only
# READS : artifacts_<slug>/judge/verdict_raw.json
#         artifacts_<slug>/executor/manifest.json
#         artifacts_<slug>/debugger/test_summary.json
#         artifacts_<slug>/planner/mini_plan.json
#         artifacts_<slug>/planner/full_plan.json
#         artifacts_<slug>/clarificator/session.json
#         artifacts_<slug>/enricher/enriched_prompt.md
#         artifacts_<slug>/archivist/knowledge_log.md
#         artifacts_<slug>/spec/specwright_spec_<slug>.md
#         artifacts_<slug>/output/src/**

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ARCHIVIST_KNOWLEDGE_LOG,
    CLARIFIED_REQ,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    ENRICHER_OVERWRITE_PROMPT,
    EXECUTOR_OVERWRITE_MANIFEST,
    JUDGE_OVERWRITE_VERDICT_RAW,
    PATCHER_ATTEMPT_LOG,
    PATCHER_OVERWRITE_FIX_SUMMARY,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    SRC_DIR,
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

_PATCHER_MODEL           = get_model("patcher")            # surface fixes
_PATCHER_SECONDARY_MODEL = get_model("patcher_secondary")  # logic/hook/data + mini scope

# Model roles resolved from artifacts/models.py:
#   "patcher"           — surface/component fixes (qwen)
#   "patcher_secondary" — logic/hook/data fixes + mini scope (minimax)
ROLE = "patcher"  # primary role — used for post_interactive next-step suggestion
MAX_FILE_CHARS = 80_000


@dataclass
class JudgeFinding:
    description: str
    severity: str
    files: list[str]
    section: str = ""


@dataclass
class FixRecord:
    finding: str
    files: list[str]
    patched: bool
    files_written: list[str]
    note: str
    escalated: bool = False
    escalated_to: str = ""
    rejected_files: list[str] | None = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="12_patcher.py",
        description="Apply scoped fixes for judge blocking issues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 12_patcher.py --project my-app
              PIPELINE_PROJECT=my-app python 12_patcher.py

              python 12_patcher.py --project my-app --verbose
              python 12_patcher.py --project my-app --skip-confirm
              python 12_patcher.py --project my-app --fix-non-blocking
        """),
    )
    parser.add_argument("--project", default=None, help="Project name. Sets PIPELINE_PROJECT.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fix-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix blocking issues. Default: true.",
    )
    parser.add_argument(
        "--fix-non-blocking",
        action="store_true",
        default=False,
        help="Also attempt to fix non-blocking notes. Default: false.",
    )
    parser.add_argument(
        "--skip-confirm",
        "--skip-vitest",
        dest="skip_confirm",
        action="store_true",
        help="Skip post-fix verification/confirmation.",
    )
    return parser


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project

    if os.environ.get("PIPELINE_PROJECT"):
        return

    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 12_patcher.py directly."
    )



def _read_json(path: Any, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        track_read(path)
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        print(f"[12][warn] Could not parse JSON {path}: {exc}", file=sys.stderr)
        return default


def _read_text(path: Any) -> str:
    if not path.exists():
        return ""
    try:
        track_read(path)
        return path.read_text(errors="replace")
    except Exception as exc:
        print(f"[12][warn] Could not read {path}: {exc}", file=sys.stderr)
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


def _load_test_report() -> dict[str, Any]:
    report = _read_json(DEBUGGER_OVERWRITE_TEST_SUMMARY, {})
    return report if isinstance(report, dict) else {}


def _current_scope() -> str:
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


def _mini_allowed_files() -> set[str]:
    return set(_extract_file_list(_load_plan_mini().get("target_files", [])))


def _implemented_files() -> set[str]:
    return set(_extract_file_list(_load_impl_record().get("files", [])))


def _safe_rel(raw: str) -> Path:
    normalized = raw.replace("\\", "/").strip().lstrip("/")
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

    return artifact_root() / safe


def _is_disallowed_pipeline_path(rel: str) -> bool:
    normalized = rel.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        return True
    if normalized == "spec.md":
        return True
    if normalized.startswith("specwright_spec_") and normalized.endswith(".md"):
        return True

    blocked_prefixes = (
        "artifacts_",
        "state/",
        "cache/",
        "execution/",
        "run/",
        "knowledge/",
        "reports/",
        "sessions/",
        "session_runs/",
        "absorber/",
        "clarificator/",
        "enricher/",
        "spectracker/",
        "scaffolder/",
        "planner/",
        "executor/",
        "debugger/",
        "reporter/",
        "judge/",
        "patcher/",
        "archivist/",
        "spec/",
        "output/tests/",
    )
    return normalized.startswith(blocked_prefixes)


def _path_exists(rel: str) -> bool:
    try:
        return _resolve_artifact_path(rel).exists()
    except Exception:
        return False


def _read_file_for_prompt(rel: str) -> str:
    try:
        path = _resolve_artifact_path(rel)
    except Exception as exc:
        return f"[invalid path: {exc}]"

    if not path.exists():
        return f"[file not found: {path}]"

    track_read(path)
    text = path.read_text(errors="replace")
    if len(text) > MAX_FILE_CHARS:
        return text[:MAX_FILE_CHARS] + f"\n\n[truncated: {len(text)} chars total]"
    return text


def _lang_for_path(rel: str) -> str:
    ext = Path(rel).suffix.lower()
    return {
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
    }.get(ext, "")


def _format_file_block(rel: str) -> str:
    code = _read_file_for_prompt(rel)
    lang = _lang_for_path(rel)
    return f"### {rel}\n```{lang}\n{code}\n```"


def _is_test_path(rel: str) -> bool:
    lowered = rel.lower()
    return rel.startswith("tests/") or rel.startswith("output/tests/") or ".test." in lowered or ".spec." in lowered


def _model_call(
    role: str,
    messages: list[dict[str, str]],
    max_tokens: int = 32768,
) -> str:
    """
    Thin wrapper around call_model() with retry and token logging.
    role must be registered in artifacts/models.py ROLES.
    """
    model_id = get_model(role)
    for attempt in range(2):
        resp = call_model(role, messages, temperature=0.1, max_tokens=max_tokens)
        usage = getattr(resp, "usage", None)
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=model_id, provider=get_provider(role))
            print_call(__file__, pt, ct, call_cost, label=f"[12] {model_id}")

        content = resp.choices[0].message.content
        if content and content.strip():
            return content.strip()

        if attempt == 0:
            print(f"    [warn] empty response from {model_id}, retrying in 3s …", file=sys.stderr)
            time.sleep(3)

    return ""


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())
    return text


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = _strip_json_fences(raw)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())

    if not isinstance(parsed, dict):
        raise ValueError("JSON top-level is not an object")

    return parsed


def load_judge_verdict() -> dict[str, Any]:
    if not JUDGE_OVERWRITE_VERDICT_RAW.exists():
        print(f"[12] ERROR: {JUDGE_OVERWRITE_VERDICT_RAW} not found.", file=sys.stderr)
        print("[12] Run 11_judge.py first.", file=sys.stderr)
        sys.exit(1)

    raw_data = _read_json(JUDGE_OVERWRITE_VERDICT_RAW, {})
    if not isinstance(raw_data, dict):
        print(f"[12] ERROR: invalid judge_overwrite_verdict_raw.json shape.", file=sys.stderr)
        sys.exit(1)

    raw_resp = raw_data.get("response", "")
    if not isinstance(raw_resp, str) or not raw_resp.strip():
        print("[12] ERROR: judge_overwrite_verdict_raw.json has empty response.", file=sys.stderr)
        sys.exit(1)

    try:
        return _parse_json_object(raw_resp)
    except Exception as exc:
        print(f"[12] ERROR: could not parse judge response JSON: {exc}", file=sys.stderr)
        print(f"[12] Raw first 1000 chars:\n{raw_resp[:1000]}", file=sys.stderr)
        sys.exit(1)


_RE_ANY_REL_PATH = re.compile(
    r"(?:src|queries|dags|config|configs|sql|scripts|app|lib|tests)/"
    r"[\w.\-/]+\."
    r"(?:ts|tsx|js|jsx|py|sql|json|ya?ml|toml|md|txt|ini|cfg|conf|sh)"
)
_RE_SRC_PATH = re.compile(r"src/[\w.\-/]+\.(?:ts|tsx|js|jsx)")

_KEYWORD_FILES: dict[str, list[str]] = {
    "useSensorData": ["src/hooks/useSensorData.ts", "src/data/demoConstants.ts"],
    "useReplay": ["src/hooks/useReplay.ts"],
    "demoConstants": ["src/data/demoConstants.ts"],
    "useMemo": ["src/hooks/useSensorData.ts"],
    "requestAnimationFrame": ["src/hooks/useReplay.ts"],
    "rAF": ["src/hooks/useReplay.ts"],
    "setInterval": ["src/hooks/useReplay.ts"],
    "anomaly": ["src/hooks/useSensorData.ts"],
    "jumpToNext": ["src/hooks/useReplay.ts"],
    "windowStart": ["src/hooks/useReplay.ts"],
    "duplicate": ["src/hooks/useSensorData.ts", "src/data/demoConstants.ts"],
}

_COMPONENT_SCAN_KEYWORDS = {
    "theme",
    "tailwind",
    "bg-white",
    "text-gray-800",
    "dark theme",
    "light theme",
    "colour",
    "color",
}


def _infer_full_files(text: str) -> list[str]:
    found: list[str] = []

    for match in _RE_SRC_PATH.findall(text):
        if match not in found and _path_exists(match):
            found.append(match)

    text_lower = text.lower()
    for keyword, files in _KEYWORD_FILES.items():
        if keyword.lower() in text_lower:
            for rel in files:
                if rel not in found and _path_exists(rel):
                    found.append(rel)

    if any(keyword in text_lower for keyword in _COMPONENT_SCAN_KEYWORDS):
        comp_dir = SRC_DIR / "components"
        if comp_dir.exists():
            for path in sorted(comp_dir.rglob("*.tsx")):
                rel = "output/src/" + str(path.relative_to(SRC_DIR)).replace("\\", "/")
                if rel not in found:
                    found.append(rel)

    return found


def _infer_mini_files(text: str) -> tuple[list[str], list[str]]:
    allowed = _mini_allowed_files()
    implemented = _implemented_files()

    candidates: set[str] = set(_RE_ANY_REL_PATH.findall(text))
    if not candidates:
        candidates.update(implemented)
        candidates.update(allowed)

    mapped: list[str] = []
    rejected: list[str] = []

    for rel in sorted(candidates):
        if rel in allowed:
            mapped.append(rel)
        else:
            rejected.append(rel)

    return mapped, rejected


def _section_notes(verdict: dict[str, Any]) -> dict[str, str]:
    sections = verdict.get("sections", {})
    if not isinstance(sections, dict):
        return {}

    notes: dict[str, str] = {}
    for key, value in sections.items():
        if isinstance(value, dict):
            notes[key] = str(value.get("notes", ""))
    return notes


def extract_findings(
    verdict: dict[str, Any],
    *,
    scope: str,
) -> tuple[list[JudgeFinding], list[JudgeFinding], list[str]]:
    section_notes = _section_notes(verdict)
    scope_rejections: list[str] = []

    def map_files(description: str, section_hint: str = "") -> list[str]:
        combined = description + " " + section_notes.get(section_hint, "")

        if scope == "mini":
            mapped, rejected = _infer_mini_files(combined)
            for rel in rejected:
                scope_rejections.append(
                    f"Judge finding references `{rel}`, which is outside "
                    "planner_mini_execution_plan.target_files."
                )
            return mapped

        return _infer_full_files(combined)

    blocking: list[JudgeFinding] = []
    for desc in verdict.get("blocking_issues", []):
        desc_text = str(desc)
        section_hint = ""

        for sec_name, notes in section_notes.items():
            first_words = desc_text.lower().split()[:4]
            if first_words and any(word in notes.lower() for word in first_words):
                section_hint = sec_name
                break

        blocking.append(
            JudgeFinding(
                description=desc_text,
                severity="blocking",
                files=map_files(desc_text, section_hint),
                section=section_hint,
            )
        )

    non_blocking: list[JudgeFinding] = []
    for desc in verdict.get("non_blocking_notes", []):
        desc_text = str(desc)
        non_blocking.append(
            JudgeFinding(
                description=desc_text,
                severity="non_blocking",
                files=map_files(desc_text),
            )
        )

    return blocking, non_blocking, scope_rejections


def _load_full_context() -> str:
    spec = _read_text(get_spec_path())

    parts = ["## Canonical spec\n\n" + (spec or "[canonical spec missing]")]

    plan = _read_json(PLANNER_FULL_PLAN, None)
    if isinstance(plan, dict):
        parts.append(
            "## planner_full_execution_plan.json\n\n```json\n"
            + json.dumps(plan, indent=2, ensure_ascii=False)
            + "\n```"
        )

    kb = _read_text(ARCHIVIST_KNOWLEDGE_LOG)
    if kb:
        parts.append("## archivist_knowledge_log.md\n\n" + kb)

    return "\n\n---\n\n".join(parts)


def _load_mini_context() -> str:
    parts: list[str] = []

    clarified = _read_text(CLARIFIED_REQ)
    parts.append("## clarificator_requirement_synthesis.md\n\n" + (clarified or "[missing]"))

    enriched = _read_text(ENRICHER_OVERWRITE_PROMPT)
    if enriched:
        parts.append("## enricher_overwrite_enriched_prompt.md\n\n" + enriched)

    plan = _load_plan_mini()
    if plan:
        parts.append(
            "## planner/mini_plan.json\n\n```json\n"
            + json.dumps(plan, indent=2, ensure_ascii=False)
            + "\n```"
        )
    else:
        parts.append("## planner/mini_plan.json\n\n[missing]")

    analysis = _load_analysis_mini(plan)
    if analysis:
        parts.append(
            "## planner/mini_plan.json — impact field\n\n```json\n"
            + json.dumps(analysis, indent=2, ensure_ascii=False)
            + "\n```"
        )

    impl = _load_impl_record()
    if impl:
        parts.append(
            "## executor_overwrite_manifest.json\n\n```json\n"
            + json.dumps(impl, indent=2, ensure_ascii=False)
            + "\n```"
        )

    kb = _read_text(ARCHIVIST_KNOWLEDGE_LOG)
    if kb:
        parts.append("## archivist_knowledge_log.md\n\n" + kb)

    return "\n\n---\n\n".join(parts)


def _load_run_context() -> str:
    return _load_mini_context() if _current_scope() == "mini" else _load_full_context()


JUDGE_FIX_SYSTEM_FULL_MINIMAX = """\
You are a senior TypeScript engineer fixing issues identified by a code reviewer.

Rules:
- Fix ONLY what the judge finding describes.
- Do not refactor unrelated code.
- Patch source files only.
- Never modify tests.
- TypeScript strict; avoid `any`.
- Preserve public interfaces unless explicitly required.
- Output raw JSON only, no markdown fences.

Return:
{
  "files": [
    {
      "file_path": "src/hooks/useSensorData.ts",
      "code": "<full corrected file content>",
      "change_summary": "one sentence"
    }
  ],
  "root_cause": "one sentence",
  "fix_summary": "one sentence"
}
"""

JUDGE_FIX_SYSTEM_FULL_QWEN = """\
You are a senior TypeScript/React developer fixing a surface issue identified by a judge.

Rules:
- Fix only the judge finding.
- Do not touch unrelated code.
- Patch source files only.
- Never modify tests.
- Tailwind only when styling is involved.
- TypeScript strict; avoid `any`.
- Output raw JSON only, no markdown fences.

Return:
{
  "files": [
    {
      "file_path": "src/components/AnomalyFeed.tsx",
      "code": "<full corrected file content>",
      "change_summary": "one sentence"
    }
  ],
  "fix_summary": "one sentence"
}
"""

JUDGE_FIX_SYSTEM_MINI = """\
You are fixing a blocking issue from a judge in a MINI targeted run.

Critical scope rule:
- You may ONLY write files explicitly listed in planner_mini_execution_plan.target_files.
- Do NOT add new files.
- Do NOT modify tests unless tests are explicitly listed in target_files.
- If the fix requires a file outside target_files, return files=[].

Context:
- clarificator_requirement_synthesis.md
- planner_mini_execution_plan.json
- planner_mini_impact_analysis.json
- executor_overwrite_manifest.json
- judge finding
- allowed affected files

Rules:
- Fix only what the judge finding describes.
- Preserve the mini task boundary.
- Keep changes minimal.
- Respect the file's language/format.
- Output raw JSON only, no markdown fences.

Return:
{
  "files": [
    {
      "file_path": "queries/example.sql",
      "code": "<full corrected file content>",
      "change_summary": "one sentence"
    }
  ],
  "root_cause": "one sentence",
  "fix_summary": "one sentence"
}
"""

_SURFACE_KEYWORDS = {
    "theme",
    "tailwind",
    "colour",
    "color",
    "bg-",
    "text-",
    "dark",
    "light",
    "class",
    "aria",
    "selector",
    "label",
}


def _choose_agent_and_prompt(finding: JudgeFinding, *, scope: str) -> tuple[str, str]:
    if scope == "mini":
        return "patcher_secondary", JUDGE_FIX_SYSTEM_MINI

    text_lower = finding.description.lower()
    if any(kw in text_lower for kw in _SURFACE_KEYWORDS):
        return "patcher", JUDGE_FIX_SYSTEM_FULL_QWEN

    return "patcher_secondary", JUDGE_FIX_SYSTEM_FULL_MINIMAX


def _parse_fix_response(raw: str, label: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        return _parse_json_object(raw)
    except Exception as exc:
        print(f"    [12] JSON parse failed for {label}: {exc}", file=sys.stderr)
        print(f"    [12] Raw first 500 chars: {raw[:500]}", file=sys.stderr)
        return None


def _allowed_to_write(rel: str, *, scope: str) -> tuple[bool, str]:
    try:
        rel = _safe_rel(rel).as_posix()
    except Exception as exc:
        return False, str(exc)

    if _is_disallowed_pipeline_path(rel):
        return False, "pipeline artifact paths are rejected by patcher"

    if _is_test_path(rel):
        return False, "test file writes are rejected by judge-fix step"

    if scope == "mini":
        if rel not in _mini_allowed_files():
            return False, f"`{rel}` is outside planner_mini_execution_plan.target_files"
        return True, ""

    if not (rel.startswith("src/") or rel.startswith("output/src/")):
        return False, "full judge-fix may only patch output/src/**"

    return True, ""


def fix_finding(
    finding: JudgeFinding,
    verdict: dict[str, Any],
    *,
    verbose: bool,
) -> FixRecord:
    scope = _current_scope()

    if not finding.files:
        note = "no files mapped"
        if scope == "mini":
            note += " — finding did not map to planner_mini_execution_plan.target_files"
        else:
            note += " — check heuristics or add explicit path in judge finding"

        print(f"  [12] No files mapped for: {finding.description[:80]}… — skipping")
        return FixRecord(
            finding=finding.description,
            files=[],
            patched=False,
            files_written=[],
            note=note,
            escalated=True,
            escalated_to="human",
        )

    rejected: list[str] = []
    allowed_files: list[str] = []

    for rel in finding.files:
        ok, reason = _allowed_to_write(rel, scope=scope)
        if ok:
            allowed_files.append(rel)
        else:
            rejected.append(f"{rel}: {reason}")

    if rejected:
        print("  [12] Scope violation before model call — auto-fix rejected:", file=sys.stderr)
        for item in rejected:
            print(f"       - {item}", file=sys.stderr)

        return FixRecord(
            finding=finding.description,
            files=finding.files,
            patched=False,
            files_written=[],
            note="scope violation; auto-fix rejected",
            escalated=True,
            escalated_to="human",
            rejected_files=rejected,
        )

    role, system = _choose_agent_and_prompt(finding, scope=scope)

    files_block = "\n\n".join(_format_file_block(rel) for rel in allowed_files)

    sections_block = ""
    if finding.section:
        sec = verdict.get("sections", {}).get(finding.section, {})
        if isinstance(sec, dict) and sec.get("notes"):
            sections_block = f"\n### Judge section notes ({finding.section})\n{sec['notes']}\n"

    allowed_block = ""
    if scope == "mini":
        allowed_block = (
            "\n### Mini allowed write set — STRICT\n"
            + "\n".join(f"- `{rel}`" for rel in sorted(_mini_allowed_files()))
            + "\n"
        )

    user_content = (
        f"### Run context\n\n{_load_run_context()}\n\n"
        f"### Judge finding to fix\n{finding.description}\n"
        f"{sections_block}\n"
        f"{allowed_block}\n"
        f"### Affected files you may edit\n\n{files_block}"
    )

    print(f"  [12] → {get_model(role)} fixing: {finding.description[:90]}…")
    if verbose:
        print(f"  [12]   scope: {scope}")
        print(f"  [12]   files: {allowed_files}")

    raw = _model_call(
        role,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    ).strip()

    if not raw:
        return FixRecord(finding.description, finding.files, False, [], "model returned empty response")

    patch = _parse_fix_response(raw, finding.description[:40])
    if not patch:
        return FixRecord(finding.description, finding.files, False, [], "JSON parse failed")

    written: list[str] = []
    rejected_after_model: list[str] = []

    patch_files = patch.get("files", [])
    if not isinstance(patch_files, list):
        patch_files = []

    for entry in patch_files:
        if not isinstance(entry, dict):
            continue

        out_rel = str(entry.get("file_path", "")).strip()
        code = entry.get("code")

        ok, reason = _allowed_to_write(out_rel, scope=scope)
        if not ok:
            msg = f"{out_rel}: {reason}"
            rejected_after_model.append(msg)
            print(f"  [12] ⚠ Scope violation: {msg} — rejected", file=sys.stderr)
            continue

        if not isinstance(code, str):
            msg = f"{out_rel}: missing code string"
            rejected_after_model.append(msg)
            print(f"  [12] ⚠ Invalid patch: {msg} — rejected", file=sys.stderr)
            continue

        out_path = _resolve_artifact_path(out_rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        track_write(out_path)

        written.append(out_rel)
        change = entry.get("change_summary", "")
        print(f"  [12] ✓ Wrote {out_rel}" + (f" — {change}" if change else ""))

    fix_summary = str(patch.get("fix_summary", ""))
    root_cause = str(patch.get("root_cause", ""))
    note = f"{root_cause} | {fix_summary}" if root_cause else fix_summary

    if rejected_after_model and not written:
        return FixRecord(
            finding=finding.description,
            files=finding.files,
            patched=False,
            files_written=[],
            note=note or "all model patches rejected by scope guard",
            escalated=True,
            escalated_to="human",
            rejected_files=rejected_after_model,
        )

    return FixRecord(
        finding=finding.description,
        files=finding.files,
        patched=bool(written),
        files_written=written,
        note=note,
        escalated=bool(rejected_after_model and not written),
        escalated_to="human" if rejected_after_model and not written else "",
        rejected_files=rejected_after_model or None,
    )


def run_vitest_confirm() -> tuple[bool, str]:
    print("\n[12] Running Vitest to confirm fixes …")
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=artifact_root(),
        capture_output=True,
        text=True,
    )
    output = result.stdout + "\n" + result.stderr
    passed = result.returncode == 0
    summary = next(
        (
            line.strip()
            for line in output.splitlines()
            if ("passed" in line or "failed" in line) and "test" in line.lower()
        ),
        "no summary line found",
    )
    print(f"[12] Vitest {'✓' if passed else '✗'} {summary}")
    return passed, output


def _verify_python(path: Path) -> tuple[bool, str]:
    try:
        track_read(path)
        py_compile.compile(str(path), doraise=True)
        return True, "py_compile OK"
    except Exception as exc:
        return False, f"py_compile failed: {exc}"


def _verify_json(path: Path) -> tuple[bool, str]:
    try:
        track_read(path)
        json.loads(path.read_text(errors="replace"))
        return True, "JSON parse OK"
    except Exception as exc:
        return False, f"JSON parse failed: {exc}"


def _verify_toml(path: Path) -> tuple[bool, str]:
    try:
        import tomllib
        track_read(path)
        tomllib.loads(path.read_text(errors="replace"))
        return True, "TOML parse OK"
    except Exception as exc:
        return False, f"TOML parse failed: {exc}"


def _verify_yaml(path: Path) -> tuple[bool, str]:
    track_read(path)
    try:
        import yaml  # type: ignore
    except Exception:
        text = path.read_text(errors="replace")
        if "\t" in text:
            return False, "YAML basic check failed: tabs found; install PyYAML for full parse"
        return True, "YAML basic check OK; PyYAML not installed"

    try:
        yaml.safe_load(path.read_text(errors="replace"))
        return True, "YAML parse OK"
    except Exception as exc:
        return False, f"YAML parse failed: {exc}"


def run_mini_confirm(files_written: list[str]) -> tuple[bool, str]:
    print("\n[12] Running mini verifier to confirm fixes …")

    if not files_written:
        return False, "no files written"

    checks: list[str] = []
    all_ok = True
    ts_like = False

    for rel in sorted(set(files_written)):
        path = _resolve_artifact_path(rel)
        ext = path.suffix.lower()

        if not path.exists():
            ok, msg = False, f"file not found: {path}"
        elif ext == ".py":
            ok, msg = _verify_python(path)
        elif ext == ".json":
            ok, msg = _verify_json(path)
        elif ext in {".yaml", ".yml"}:
            ok, msg = _verify_yaml(path)
        elif ext == ".toml":
            ok, msg = _verify_toml(path)
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            ts_like = True
            track_read(path)
            ok, msg = True, "TS/JS file written; Vitest will run if package.json exists"
        else:
            track_read(path)
            ok, msg = True, "basic existence check OK"

        status = "PASS" if ok else "FAIL"
        checks.append(f"{status} {rel}: {msg}")
        print(f"[12][mini] {status} {rel} — {msg}")
        all_ok = all_ok and ok

    package_json = artifact_root() / "package.json"
    if ts_like and package_json.exists():
        vitest_ok, vitest_output = run_vitest_confirm()
        all_ok = all_ok and vitest_ok
        checks.append("VITEST " + ("PASS" if vitest_ok else "FAIL"))
        checks.append(vitest_output[-1200:])
    elif ts_like:
        checks.append("VITEST SKIPPED: package.json not found")

    return all_ok, "\n".join(checks)


def run_confirm(*, scope: str, files_written: list[str], skip_confirm: bool) -> tuple[bool, str]:
    if skip_confirm:
        print("\n[12] Skipping confirmation (--skip-confirm)")
        return True, "skipped"

    if scope == "mini":
        return run_mini_confirm(files_written)

    return run_vitest_confirm()



def write_overwrite_fix_summary(
    *,
    scope: str,
    verdict: dict[str, Any],
    blocking: list[JudgeFinding],
    non_blocking: list[JudgeFinding],
    fix_records: list[FixRecord],
    confirm_passed: bool,
    confirm_summary: str,
    scope_rejections: list[str],
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_patched = sum(1 for record in fix_records if record.patched)
    n_escalated = sum(1 for record in fix_records if record.escalated)

    lines = [
        "# Patcher Fix Summary",
        f"_Generated: {now}_",
        "",
        f"- Project: `{get_project_name()}`",
        f"- Project slug: `{get_project_slug()}`",
        f"- Scope: `{scope}`",
        f"- Judge verdict: `{verdict.get('verdict')}`",
        f"- Blocking issues: {len(blocking)}",
        f"- Non-blocking notes: {len(non_blocking)}",
        f"- Fix attempts: {len(fix_records)}",
        f"- Fixes patched: {n_patched}",
        f"- Escalated: {n_escalated}",
        f"- Confirmation: {'✅ PASS' if confirm_passed else '❌ FAIL'}",
        "",
    ]

    if scope_rejections:
        lines += ["## Scope rejections", ""]
        for item in sorted(set(scope_rejections)):
            lines.append(f"- {item}")
        lines.append("")

    lines += ["## Fix records", ""]
    if not fix_records:
        lines.append("_No fixes attempted._")
    else:
        for record in fix_records:
            icon = "✅" if record.patched else "❌"
            esc = " — ESCALATED" if record.escalated else ""
            lines += [
                f"### {icon} {record.finding[:120]}{esc}",
                "",
                f"- Mapped files: {', '.join(record.files) if record.files else '—'}",
                f"- Files written: {', '.join(record.files_written) if record.files_written else '—'}",
                f"- Note: {record.note or '—'}",
            ]
            if record.rejected_files:
                lines.append("- Rejected files:")
                for rejected in record.rejected_files:
                    lines.append(f"  - {rejected}")
            lines.append("")

    lines += [
        "## Confirmation summary",
        "",
        "```text",
        confirm_summary[-2000:] if isinstance(confirm_summary, str) else str(confirm_summary),
        "```",
        "",
    ]

    PATCHER_OVERWRITE_FIX_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    PATCHER_OVERWRITE_FIX_SUMMARY.write_text(apply_md_header("\n".join(lines).rstrip() + "\n", PATCHER_OVERWRITE_FIX_SUMMARY, owner="12_patcher.py"), encoding="utf-8")
    track_write(PATCHER_OVERWRITE_FIX_SUMMARY)

    print(f"[12] Overwrite fix summary written → {PATCHER_OVERWRITE_FIX_SUMMARY}")


def append_attempt_log(report: dict[str, Any]) -> None:
    PATCHER_ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)

    existing: list[Any] = []
    if PATCHER_ATTEMPT_LOG.exists():
        try:
            track_read(PATCHER_ATTEMPT_LOG)
            loaded = json.loads(PATCHER_ATTEMPT_LOG.read_text(errors="replace"))
            if isinstance(loaded, dict):
                existing = loaded.get("entries", [])
            elif isinstance(loaded, list):
                # migrate legacy bare-list format
                existing = loaded
        except Exception:
            existing = []

    existing.append(report)
    PATCHER_ATTEMPT_LOG.write_text(
        json.dumps({"entries": existing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(PATCHER_ATTEMPT_LOG)

    print(f"\n[12] Attempt log appended → {PATCHER_ATTEMPT_LOG}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: project env must be configured before ensure_dirs().
    ensure_dirs()

    exit_code = 0

    try:
        scope = _current_scope()
        print(f"[12] Project: {get_project_name()} ({get_project_slug()})")
        print(f"[12] Scope detected: {scope}")

        verdict = load_judge_verdict()

        if verdict.get("verdict") == "APPROVED":
            print("[12] Judge already APPROVED — nothing to fix.")

            report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "project": get_project_name(),
                "project_slug": get_project_slug(),
                "scope": scope,
                "judge_verdict": verdict.get("verdict"),
                "blocking_count": 0,
                "non_blocking_count": 0,
                "fix_attempts": 0,
                "fixes_patched": 0,
                "escalated_count": 0,
                "confirm_passed": True,
                "confirm_summary": "judge already approved",
                "scope_rejections": [],
                "records": [],
            }
            append_attempt_log(report)
            write_overwrite_fix_summary(
                scope=scope,
                verdict=verdict,
                blocking=[],
                non_blocking=[],
                fix_records=[],
                confirm_passed=True,
                confirm_summary="judge already approved",
                scope_rejections=[],
            )
            return

        print(f"[12] Judge verdict: {verdict.get('verdict')}")
        print(f"[12] Summary: {str(verdict.get('summary', ''))[:160]}")

        blocking, non_blocking, scope_rejections = extract_findings(verdict, scope=scope)

        if scope_rejections:
            print("\n[12] Mini scope warnings from judge findings:")
            for item in sorted(set(scope_rejections)):
                print(f"  ⚠ {item}")

        print(f"\n[12] Blocking issues ({len(blocking)}):")
        for finding in blocking:
            print(f"  • {finding.description[:100]}")
            print(f"    files: {finding.files or ['(no files mapped)']}")

        print(f"\n[12] Non-blocking notes ({len(non_blocking)}):")
        for finding in non_blocking:
            role, _ = _choose_agent_and_prompt(finding, scope=scope)
            print(f"  • {finding.description[:100]}")
            print(f"    files: {finding.files or ['(no files mapped)']}, model: {get_model(role)}")

        to_fix: list[JudgeFinding] = []
        if args.fix_blocking:
            to_fix.extend(blocking)
        if args.fix_non_blocking:
            to_fix.extend(non_blocking)

        fix_records: list[FixRecord] = []

        if not to_fix:
            print("\n[12] Nothing to fix.")
        else:
            print(f"\n[12] Fixing {len(to_fix)} finding(s) …")
            for finding in to_fix:
                fix_records.append(
                    fix_finding(
                        finding,
                        verdict,
                        verbose=args.verbose,
                    )
                )

        files_written = sorted({rel for record in fix_records for rel in record.files_written})

        if fix_records:
            confirm_passed, confirm_summary = run_confirm(
                scope=scope,
                files_written=files_written,
                skip_confirm=args.skip_confirm,
            )
        elif args.skip_confirm:
            confirm_passed, confirm_summary = True, "skipped"
        else:
            confirm_passed, confirm_summary = False, "no fixes attempted"

        n_patched = sum(1 for record in fix_records if record.patched)
        n_escalated = sum(1 for record in fix_records if record.escalated)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": get_project_name(),
            "project_slug": get_project_slug(),
            "scope": scope,
            "judge_verdict": verdict.get("verdict"),
            "blocking_count": len(blocking),
            "non_blocking_count": len(non_blocking),
            "fix_attempts": len(fix_records),
            "fixes_patched": n_patched,
            "escalated_count": n_escalated,
            "confirm_passed": confirm_passed,
            "confirm_summary": confirm_summary[-2000:] if isinstance(confirm_summary, str) else confirm_summary,
            "scope_rejections": sorted(set(scope_rejections)),
            "records": [asdict(record) for record in fix_records],
        }
        append_attempt_log(report)

        write_overwrite_fix_summary(
            scope=scope,
            verdict=verdict,
            blocking=blocking,
            non_blocking=non_blocking,
            fix_records=fix_records,
            confirm_passed=confirm_passed,
            confirm_summary=confirm_summary,
            scope_rejections=scope_rejections,
        )

        print(f"\n{'=' * 50}")
        print("  STEP 12 SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Project:            {get_project_name()} ({get_project_slug()})")
        print(f"  Scope:              {scope}")
        print(f"  Blocking issues:    {len(blocking)}")
        print(f"  Fixes applied:      {n_patched}/{len(fix_records)}")
        print(f"  Escalated:          {n_escalated}")
        print(f"  Confirm after fix:  {'✓ PASS' if confirm_passed else '✗ FAIL'}")

        for record in fix_records:
            icon = "✅" if record.patched else "❌"
            esc = " ESCALATED" if record.escalated else ""
            print(f"  {icon}{esc} {record.finding[:80]}")
            for written in record.files_written:
                print(f"     → {written}")
            if record.rejected_files:
                for rejected in record.rejected_files:
                    print(f"     rejected: {rejected}")
            if record.note:
                print(f"     note: {record.note[:120]}")

        failed_or_escalated = [
            record
            for record in fix_records
            if not record.patched or record.escalated
        ]

        if failed_or_escalated:
            print(f"\n[12] ⚠ {len(failed_or_escalated)} fix(es) failed/rejected/escalated:")
            for record in failed_or_escalated:
                print(f"     • {record.finding[:100]}")
                print(f"       {record.note}")

        if not confirm_passed and not args.skip_confirm:
            print("\n[12] Confirmation failed after judge fixes — human review needed.")
            exit_code = 1

        if any(record.escalated for record in fix_records):
            print("\n[12] Some fixes escalated due to scope/safety constraints.", file=sys.stderr)
            exit_code = 1

        if exit_code == 0:
            print("\n[12] Done. harness.py can re-run judge to confirm improvement.")

    except Exception as exc:
        print(f"[12][error] Patcher failed: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        print_summary("[12]")
        print_artifact_summary("[12]")
        prompt_next_step(ROLE, prefix="[12]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
