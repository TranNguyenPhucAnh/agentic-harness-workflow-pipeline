"""
pipeline/07_planner.py
======================
Step 7 — Planner (reasoning-heavy, no code output).

This script supports two scopes:

FULL SCOPE
──────────
Spec-driven planner for the full pipeline.

Reads:
    artifacts_<slug>/specwright_spec_<slug>.md
    artifacts_<slug>/state/scaffolder_codebase_skeleton.json

Writes:
    artifacts_<slug>/state/planner_full_execution_plan.json

Consumed by:
    pipeline/08_executor.py --scope full --use-planner-plan


MINI SCOPE
──────────
Targeted planner for small daily-driver tasks.

Reads:
    artifacts_<slug>/state/clarificator_requirement_synthesis.md
    artifacts_<slug>/execution/enricher_overwrite_enriched_prompt.md        optional
    artifacts_<slug>/execution/clarificator_overwrite_raw.json              optional
    artifacts_<slug>/knowledge/current/archivist_knowledge_log.md           optional
    artifacts_<slug>/knowledge/current/absorber_codebase_map.md             optional
    artifacts_<slug>/knowledge/current/absorber_config_map.json             optional
    artifacts_<slug>/knowledge/current/absorber_blame_map.md                optional
    artifacts_<slug>/knowledge/current/patcher_findings_snapshot.md         optional
    artifacts_<slug>/knowledge/current/archivist_spec_gaps.md               optional

Writes:
    artifacts_<slug>/state/planner_mini_execution_plan.json
    artifacts_<slug>/state/planner_mini_impact_analysis.json

Consumed by:
    pipeline/08_executor.py --scope mini --use-planner-plan

Does NOT write any src/ files. 08_executor.py is the sole executor.

At the end of each run, prints:
    - artifacts read
    - artifacts created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


# === WRITE AUTHORITY: planner ===
# OWNS full:
#   artifacts_<slug>/state/planner_full_execution_plan.json
#
# OWNS mini:
#   artifacts_<slug>/state/planner_mini_execution_plan.json
#   artifacts_<slug>/state/planner_mini_impact_analysis.json
#
# READS full:
#   artifacts_<slug>/specwright_spec_<slug>.md
#   artifacts_<slug>/state/scaffolder_codebase_skeleton.json
#
# READS mini:
#   artifacts_<slug>/state/clarificator_requirement_synthesis.md
#   artifacts_<slug>/execution/enricher_overwrite_enriched_prompt.md
#   artifacts_<slug>/execution/clarificator_overwrite_raw.json
#   artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
#   artifacts_<slug>/knowledge/current/absorber_codebase_map.md
#   artifacts_<slug>/knowledge/current/absorber_config_map.json
#   artifacts_<slug>/knowledge/current/absorber_blame_map.md
#   artifacts_<slug>/knowledge/current/patcher_findings_snapshot.md
#   artifacts_<slug>/knowledge/current/archivist_spec_gaps.md

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from artifacts.paths import (  # noqa: E402
    ABSORBER_BLAME_MAP,
    ABSORBER_CODEBASE_MAP,
    ABSORBER_CONFIG_MAP,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    CLARIFICATOR_OVERWRITE_RAW,
    CLARIFIED_REQ,
    ENRICHER_OVERWRITE_PROMPT,
    PATCHER_FINDINGS_SNAPSHOT,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_IMPACT,
    PLANNER_MINI_PLAN,
    SCAFFOLD_JSON,
    ensure_dirs,
    get_spec_path,
)


ROLE = "planner"


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
    print("[07] Artifacts read:")
    if _ARTIFACTS_READ:
        for item in sorted(_ARTIFACTS_READ):
            print(f"[07]   READ  {item}")
    else:
        print("[07]   READ  (none)")

    print("[07] Artifacts created/updated/overwritten/appended:")
    if _ARTIFACTS_WRITTEN:
        for item in sorted(_ARTIFACTS_WRITTEN):
            print(f"[07]   WRITE {item}")
    else:
        print("[07]   WRITE (none)")


# ════════════════════════════════════════════════════════════════════════════
# Full-scope prompt
# ════════════════════════════════════════════════════════════════════════════

FULL_SYSTEM_PROMPT = """\
You are a senior software architect acting as a PLANNER.
You will receive a spec and a scaffold JSON (stub files with signatures only).

Your job is NOT to write code.
Your job is to reason carefully and produce an implementation plan.

## Step 0 — Identify the stack FIRST

Before planning any file, read the spec and scaffold and extract the project stack.

Record this as a top-level "stack" object in your JSON output.

The stack should include:
- Primary language and version if known, e.g. TypeScript 5.x, Python 3.12, Go 1.22
- Runtime / bundler / platform, e.g. Vite 5, Node 20, Bun, uWSGI, Docker
- Main framework(s), e.g. React 18, FastAPI, Vue 3, Django, Express
- CSS / styling system if any, e.g. Tailwind CSS v3, CSS Modules, Styled Components
- Test runner if any, e.g. Vitest, pytest, Jest, Go test
- Key libraries that affect implementation patterns, e.g. Zustand, React Query, Pydantic, SQLAlchemy

If the project is a monorepo or mixed stack, represent that explicitly, for example:
{
  "frontend": {
    "language": "TypeScript 5.x",
    "runtime": "Vite 5",
    "framework": "React 18",
    "styling": "Tailwind CSS v3",
    "test_runner": "Vitest",
    "key_libs": ["Zustand", "React Query"]
  },
  "backend": {
    "language": "Python 3.12",
    "runtime": "Uvicorn",
    "framework": "FastAPI",
    "styling": null,
    "test_runner": "pytest",
    "key_libs": ["Pydantic", "SQLAlchemy"]
  }
}

## Step 1 — Plan each non-test stub file

For each non-test stub file, output a task object describing:
- What the file does and its role in the system
- Ordered list of implementation sub-tasks, in dependency-aware order
- Key types / interfaces / schemas / models this file depends on, with source file
- Gotchas or edge cases the implementer must handle
- Styling hints for visual/UI components if applicable

## Step 2 — Stack-specific gotchas

For EACH file, "gotchas" must include framework/language/runtime quirks relevant to the detected stack and to THIS file.

Do not give generic advice.
Ask yourself:

"What would a developer who knows the detected stack warn their colleague about before implementing this specific file?"

Examples of good stack-derived gotchas:
- React 18+: useEffect can run twice in StrictMode, so effects need cleanup/idempotence
- Vite: use import.meta.env instead of process.env for client env vars
- Python/FastAPI: use async def only when awaiting async I/O; do not block the event loop
- Pydantic v2: use model_config / field validators instead of v1 Config/validator patterns
- Vue 3 Composition API: destructuring reactive() can lose reactivity
- Go: propagate context cancellation to avoid goroutine leaks
- SQLAlchemy async: do not mix sync Session with async engine/session

The point is to derive these from the spec's stack, not from hardcoded React/Vite assumptions.

Return a single JSON object — NO markdown fences, raw JSON only:
{
  "plan_version": "1.0.0",
  "scope": "full",
  "stack": {
    "language": "TypeScript 5.x",
    "runtime": "Vite 5",
    "framework": "React 18",
    "styling": "Tailwind CSS v3",
    "test_runner": "Vitest",
    "key_libs": ["Zustand", "React Query"]
  },
  "tasks": [
    {
      "file_path": "src/hooks/useSensorData.ts",
      "role": "one-sentence role description",
      "depends_on": ["src/types/sensor.ts", "src/data/demoConstants.ts"],
      "sub_tasks": [
        "1. Generate base SensorPoint array using POINTS_PER_DAY constant ...",
        "2. Inject anomaly clusters at morning (07-09h) and evening (18-21h) ...",
        "3. ..."
      ],
      "gotchas": [
        "decisionScore must be negative for anomaly points (-0.05 to -0.45)",
        "React state updates derived from timers must be cleaned up in useEffect cleanup"
      ],
      "tailwind_hints": null
    }
  ],
  "implementation_order": [
    "src/types/sensor.ts",
    "src/data/demoConstants.ts",
    "src/hooks/useSensorData.ts",
    "src/hooks/useReplay.ts",
    "src/components/SummaryStickyBar.tsx",
    "src/components/ReplayControls.tsx",
    "src/components/AnomalyFeed.tsx",
    "src/components/ModelGates.tsx",
    "src/App.tsx",
    "src/main.tsx"
  ],
  "global_notes": "any cross-cutting concerns the implementer should know"
}

Rules:
- Reason as deeply as needed — this is your reasoning budget well spent.
- Be specific: reference exact constant names, prop names, type names, schema names, and file paths from the spec/scaffold.
- implementation_order must respect dependency order.
- tailwind_hints: include for visual components if the detected stack uses Tailwind; otherwise provide relevant styling hints or null.
- Do not assume TypeScript, React, Vite, Tailwind, or Vitest unless the spec/scaffold actually indicates them.
- Do not use implementation patterns from a different stack than the one detected.
- Output raw JSON only. Absolutely no markdown fences or preamble text.
"""


# ════════════════════════════════════════════════════════════════════════════
# Mini-scope prompt
# ════════════════════════════════════════════════════════════════════════════

MINI_SYSTEM_PROMPT = """\
You are a senior software architect acting as a TARGETED MINI PLANNER.

You are planning a small, focused change to an existing project.

Your job is NOT to write code.
Your job is to analyze the request and produce:
1. A targeted implementation plan: planner_mini_execution_plan.json
2. A lightweight impact/risk analysis: planner_mini_impact_analysis.json

You must be conservative:
- Do NOT broaden the task.
- Do NOT recommend full rewrites.
- Do NOT modify unrelated files.
- Preserve existing public APIs unless the request explicitly requires an API change.
- Prefer the smallest safe set of target files.
- If unsure whether a file must change, put it in planner_mini_impact_analysis.recommendations or warnings, NOT target_files.

You will receive:
- The clarified user request.
- Optional enriched prompt.
- Optional clarificator session metadata.
- Optional project knowledge.
- Optional codebase/config/blame maps.
- Optional previous patcher findings.
- Optional known spec gaps.

Return ONE raw JSON object with exactly this top-level shape:
{
  "planner_mini_execution_plan": {
    "plan_version": "1.0.0",
    "scope": "mini",
    "task_summary": "Short description of the requested change",
    "constraints": [
      "Do not modify unrelated files",
      "Preserve existing public APIs unless necessary"
    ],
    "target_files": [
      {
        "path": "src/components/Header.tsx",
        "action": "MODIFY",
        "reason": "Need to update CTA button behavior",
        "instructions": [
          "Locate the existing CTA button",
          "Change click handler to open pricing modal",
          "Keep styling unchanged"
        ],
        "risk": "medium"
      }
    ],
    "test_suggestions": [
      {
        "type": "unit",
        "path": "src/components/Header.test.tsx",
        "reason": "Cover CTA click behavior"
      }
    ],
    "out_of_scope": [
      "Do not refactor navbar layout"
    ]
  },
  "planner_mini_impact_analysis": {
    "scope": "mini",
    "possible_changes": [
      {
        "path": "src/components/Header.tsx",
        "change": "Update CTA click behavior",
        "confidence": "high"
      }
    ],
    "impacts": [
      {
        "area": "UI interaction",
        "description": "CTA now opens pricing modal instead of navigating directly",
        "severity": "medium"
      }
    ],
    "warnings": [
      "If no modal infrastructure exists, implementer may need to create it only if listed in target_files"
    ],
    "conflicts": [],
    "recommendations": [
      "Run the relevant component tests after patching"
    ]
  }
}

Allowed target_files.action values:
- "MODIFY" for existing files
- "CREATE" for new files
- "DELETE" only if the request explicitly asks to remove a file
- "RENAME" only if the request explicitly asks to rename/move a file

Allowed risk values:
- "low"
- "medium"
- "high"

Rules:
- target_files must contain only files that should be changed by the implementer.
- Use repo-relative paths only.
- Do not include artifact paths such as artifacts_*/, state/*, execution/*, run/*, cache/*, knowledge/*, reports/*.
- Do not include spec.md or specwright_spec_<slug>.md unless the user explicitly asks to update the canonical spec.
- If a test file should be added/updated, include it either in target_files or test_suggestions depending on whether implementation should patch it.
- Keep instructions concrete and actionable.
- planner_mini_impact_analysis should explain risks and impacts, but it must not authorize broad rewrites.
- Output raw JSON only. No markdown fences. No explanation outside JSON.
"""


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="07_planner.py",
        description=(
            "Planner step. Writes planner_full_execution_plan.json "
            "or planner_mini_execution_plan.json + planner_mini_impact_analysis.json."
        ),
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
        "--scope",
        choices=["full", "mini"],
        default="full",
        help=(
            "Planner scope. full writes state/planner_full_execution_plan.json; "
            "mini writes state/planner_mini_execution_plan.json and "
            "state/planner_mini_impact_analysis.json."
        ),
    )
    return parser


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
        "PIPELINE_PROJECT=<name> before running 07_planner.py directly."
    )


# ════════════════════════════════════════════════════════════════════════════
# Generic helpers
# ════════════════════════════════════════════════════════════════════════════

def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _read_optional(path: Any, *, max_chars: int | None = None) -> str:
    """
    Read a LazyPath/Path if it exists. Returns empty string on missing/empty.
    """
    try:
        if not path.exists():
            return ""

        _track_read(path)

        if hasattr(path, "read_text"):
            text = path.read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")

        text = text.strip()
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + f"\n\n<!-- truncated at {max_chars} chars -->"
        return text
    except Exception as exc:
        print(f"[07] WARNING: could not read {path}: {exc}", file=sys.stderr)
        return ""


def _read_json_optional(path: Any, *, max_chars: int | None = None) -> str:
    """
    Read a JSON-ish file and return pretty text for prompt injection.
    If parsing fails, returns raw text.
    """
    raw = _read_optional(path, max_chars=max_chars)
    if not raw:
        return ""
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except Exception:
        return raw


def _parse_json(raw: str, label: str) -> dict:
    """Extract JSON from model output robustly (handles accidental fences)."""
    raw = raw.strip()

    # Strip common markdown fences if present.
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{label} parsed as {type(parsed).__name__}, expected object.")
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost JSON object.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if not isinstance(parsed, dict):
                raise RuntimeError(f"{label} parsed as {type(parsed).__name__}, expected object.")
            return parsed
        except json.JSONDecodeError as exc:
            print(f"[07] JSON parse failed for {label}: {exc}", file=sys.stderr)
            print(f"[07] Raw output (first 1000 chars):\n{raw[:1000]}", file=sys.stderr)
            raise RuntimeError(f"Could not parse JSON from {label}") from exc

    print(f"[07] No JSON object found in {label}.", file=sys.stderr)
    print(f"[07] Raw output (first 1000 chars):\n{raw[:1000]}", file=sys.stderr)
    raise RuntimeError(f"No JSON object found in {label}")


# ════════════════════════════════════════════════════════════════════════════
# Core model call
# ════════════════════════════════════════════════════════════════════════════

def _call_planner_json(
    *,
    system_prompt: str,
    user_message: str,
    label: str,
    temperature: float = 0.2,
    max_tokens: int = 32768,
) -> dict:
    """
    Call the planner role (config in artifacts/models.py) and return parsed JSON.

    Retries once on transient failures.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    model    = get_model(ROLE)
    provider = get_provider(ROLE)
    print(f"[07] Calling model: {model} (provider: {provider}) — {label} …")

    last_error: Exception | None = None

    for attempt in range(2):
        try:
            resp = call_model(
                ROLE,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            usage = getattr(resp, "usage", None)
            if usage:
                pt        = getattr(usage, "prompt_tokens",     0) or 0
                ct        = getattr(usage, "completion_tokens", 0) or 0
                call_cost = record_usage(usage, model=model, provider=provider)
                print_call(__file__, pt, ct, call_cost)

            choice  = resp.choices[0]
            message = choice.message
            content = message.content
            tool_calls = getattr(message, "tool_calls", None)
            finish_reason = getattr(choice, "finish_reason", None)

            if tool_calls:
                raise RuntimeError(f"Model returned tool_calls instead of text: {tool_calls}")

            if not content or not content.strip():
                raise RuntimeError(
                    f"Model returned empty content. finish_reason={finish_reason}, "
                    f"message={message}"
                )

            return _parse_json(content.strip(), label=label)

        except Exception as exc:
            last_error = exc
            print(f"[07] {label} failed: {exc}", file=sys.stderr)

            if attempt == 0:
                print("[07] Retrying in 3s …", file=sys.stderr)
                time.sleep(3)

    raise RuntimeError(f"{label} failed after retries: {last_error}")


# ════════════════════════════════════════════════════════════════════════════
# Full-scope planner
# ════════════════════════════════════════════════════════════════════════════

def _load_full_spec() -> str:
    spec_path = get_spec_path()
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Missing canonical spec: {spec_path}\n"
            "Run specwright first, for example:\n"
            "  python harness.py --project <name> --scope full"
        )

    _track_read(spec_path)
    return spec_path.read_text(encoding="utf-8")


def _load_scaffold() -> dict:
    if not SCAFFOLD_JSON.exists():
        raise FileNotFoundError(
            f"Missing scaffold: {SCAFFOLD_JSON}\n"
            "Run scaffolder first, for example:\n"
            "  python harness.py --project <name> --scope full --scaffold"
        )

    _track_read(SCAFFOLD_JSON)
    return json.loads(SCAFFOLD_JSON.read_text(encoding="utf-8"))


def call_full_planner(spec: str, stub_files: list[dict]) -> dict:
    user_message = (
        f"### canonical spec\n\n{spec}\n\n"
        f"### scaffold stub files\n\n"
        f"{json.dumps(stub_files, indent=2, ensure_ascii=False)}"
    )

    return _call_planner_json(
        system_prompt=FULL_SYSTEM_PROMPT,
        user_message=user_message,
        label="full planner response",
        temperature=0.2,
        max_tokens=32768,
    )


def validate_full_plan(plan: dict, stub_files: list[dict]) -> None:
    """
    Warn if any stub file is missing from the plan and report detected stack.
    """
    tasks = plan.get("tasks", [])
    if not isinstance(tasks, list):
        print("[07] WARNING: plan.tasks is not a list", file=sys.stderr)
        tasks = []

    planned = {
        task.get("file_path")
        for task in tasks
        if isinstance(task, dict) and task.get("file_path")
    }

    for file_entry in stub_files:
        fp = file_entry.get("file_path")
        if fp and fp not in planned:
            print(f"[07] WARNING: stub file not covered by plan: {fp}")

    required_keys = {"plan_version", "tasks", "implementation_order"}
    missing = required_keys - set(plan.keys())
    if missing:
        print(f"[07] WARNING: plan missing keys: {missing}")

    if "scope" not in plan:
        plan["scope"] = "full"

    if "stack" not in plan:
        print("[07] WARNING: plan missing 'stack' — framework quirks may be generic")
    else:
        print(f"[07] Stack detected: {json.dumps(plan['stack'], indent=2, ensure_ascii=False)}")


def run_full_scope() -> None:
    spec = _load_full_spec()
    scaffold = _load_scaffold()

    files = scaffold.get("files", [])
    if not isinstance(files, list):
        raise RuntimeError(
            "Invalid scaffolder_codebase_skeleton.json: expected top-level key "
            "'files' to be a list."
        )

    stub_files = [
        file_entry
        for file_entry in files
        if isinstance(file_entry, dict) and not file_entry.get("is_test")
    ]

    print("[07] Scope: full")
    print(f"[07] Planning {len(stub_files)} non-test stub file(s) …")

    plan = call_full_planner(spec, stub_files)
    validate_full_plan(plan, stub_files)

    PLANNER_FULL_PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLANNER_FULL_PLAN.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _track_write(PLANNER_FULL_PLAN)

    print(f"[07] Full plan written → {PLANNER_FULL_PLAN}")
    print(f"[07] Tasks in plan: {len(plan.get('tasks', []))}")
    print(f"[07] Implementation order: {plan.get('implementation_order', [])}")
    print("[07] Done. Pass --use-planner-plan to 08_executor.py to use this plan.")


# ════════════════════════════════════════════════════════════════════════════
# Mini-scope planner
# ════════════════════════════════════════════════════════════════════════════

def _load_mini_request() -> str:
    """
    Load the mini request.

    Preferred:
      1. execution/enricher_overwrite_enriched_prompt.md
      2. state/clarificator_requirement_synthesis.md

    enricher_overwrite_enriched_prompt is optional and may include richer context
    generated by the enricher. clarificator_requirement_synthesis is the
    canonical minimum input for mini planning.
    """
    enriched = _read_optional(ENRICHER_OVERWRITE_PROMPT, max_chars=20_000)
    clarified = _read_optional(CLARIFIED_REQ, max_chars=20_000)

    if enriched and clarified:
        return (
            "### Enriched prompt\n"
            f"{enriched}\n\n"
            "### Clarified requirement synthesis\n"
            f"{clarified}"
        )

    if enriched:
        return enriched

    if clarified:
        return clarified

    raise FileNotFoundError(
        "Mini planner requires a clarified request.\n"
        f"Missing both:\n"
        f"  - {ENRICHER_OVERWRITE_PROMPT}\n"
        f"  - {CLARIFIED_REQ}\n\n"
        "Run clarificator/enricher first, for example:\n"
        "  python harness.py --project <name> --scope mini --clarify\n"
        "or provide/create state/clarificator_requirement_synthesis.md."
    )


def _load_mini_context_bundle() -> str:
    """
    Load optional knowledge/context files for mini planning.
    Missing files are ignored.
    """
    sections: list[str] = []

    sources: list[tuple[str, Any, str, int]] = [
        (
            "execution/clarificator_overwrite_raw.json",
            CLARIFICATOR_OVERWRITE_RAW,
            "json",
            20_000,
        ),
        (
            "knowledge/current/archivist_knowledge_log.md",
            ARCHIVIST_KNOWLEDGE_LOG,
            "text",
            30_000,
        ),
        (
            "knowledge/current/absorber_codebase_map.md",
            ABSORBER_CODEBASE_MAP,
            "text",
            35_000,
        ),
        (
            "knowledge/current/absorber_config_map.json",
            ABSORBER_CONFIG_MAP,
            "json",
            20_000,
        ),
        (
            "knowledge/current/absorber_blame_map.md",
            ABSORBER_BLAME_MAP,
            "text",
            20_000,
        ),
        (
            "knowledge/current/patcher_findings_snapshot.md",
            PATCHER_FINDINGS_SNAPSHOT,
            "text",
            20_000,
        ),
        (
            "knowledge/current/archivist_spec_gaps.md",
            ARCHIVIST_SPEC_GAPS,
            "text",
            15_000,
        ),
    ]

    for label, path, kind, cap in sources:
        text = (
            _read_json_optional(path, max_chars=cap)
            if kind == "json"
            else _read_optional(path, max_chars=cap)
        )
        if text:
            sections.append(f"### {label}\n{text}")

    return "\n\n".join(sections)


def call_mini_planner(request: str, context_bundle: str) -> dict:
    context_block = (
        f"\n\n## Project context bundle\n\n{context_bundle}"
        if context_bundle
        else "\n\n## Project context bundle\n\n(no knowledge context available)"
    )

    user_message = (
        "## Clarified mini request\n\n"
        f"{request}"
        f"{context_block}"
    )

    return _call_planner_json(
        system_prompt=MINI_SYSTEM_PROMPT,
        user_message=user_message,
        label="mini planner response",
        temperature=0.15,
        max_tokens=32768,
    )


def _normalize_target_action(action: Any) -> str:
    value = str(action or "MODIFY").strip().upper()
    allowed = {"MODIFY", "CREATE", "DELETE", "RENAME"}
    return value if value in allowed else "MODIFY"


def _normalize_risk(risk: Any) -> str:
    value = str(risk or "medium").strip().lower()
    allowed = {"low", "medium", "high"}
    return value if value in allowed else "medium"


def _is_disallowed_target_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        return True

    blocked_prefixes = (
        "artifacts_",
        "state/",
        "cache/",
        "execution/",
        "run/",       # legacy artifact dir, still blocked for safety
        "knowledge/",
        "reports/",
    )

    if normalized in {"spec.md"}:
        return True
    if normalized.startswith("specwright_spec_") and normalized.endswith(".md"):
        return True
    if normalized.startswith(blocked_prefixes):
        return True
    if "/../" in f"/{normalized}/" or normalized.startswith("../"):
        return True
    return False


def validate_and_normalize_mini_result(result: dict) -> tuple[dict, dict]:
    """
    Validate and normalize model output into
    (planner_mini_execution_plan, planner_mini_impact_analysis).

    Backward compatibility:
    - Accepts old top-level keys "plan_mini" and "analysis_mini".
    - Accepts direct mini plan object if it has scope=mini and target_files.
    """
    if "planner_mini_execution_plan" in result:
        plan = result.get("planner_mini_execution_plan") or {}
        analysis = result.get("planner_mini_impact_analysis") or {}
    elif "plan_mini" in result:
        plan = result.get("plan_mini") or {}
        analysis = result.get("analysis_mini") or {}
    elif result.get("scope") == "mini" and "target_files" in result:
        plan = result
        analysis = {}
    else:
        raise RuntimeError(
            "Mini planner response missing top-level "
            "'planner_mini_execution_plan'."
        )

    if not isinstance(plan, dict):
        raise RuntimeError("planner_mini_execution_plan must be a JSON object.")
    if not isinstance(analysis, dict):
        analysis = {}

    plan.setdefault("plan_version", "1.0.0")
    plan["scope"] = "mini"
    plan.setdefault("task_summary", "")
    plan.setdefault("constraints", [])
    plan.setdefault("target_files", [])
    plan.setdefault("test_suggestions", [])
    plan.setdefault("out_of_scope", [])

    if not isinstance(plan["constraints"], list):
        plan["constraints"] = [str(plan["constraints"])]
    if not isinstance(plan["target_files"], list):
        raise RuntimeError("planner_mini_execution_plan.target_files must be a list.")
    if not isinstance(plan["test_suggestions"], list):
        plan["test_suggestions"] = []
    if not isinstance(plan["out_of_scope"], list):
        plan["out_of_scope"] = [str(plan["out_of_scope"])]

    normalized_targets: list[dict] = []
    for raw_entry in plan["target_files"]:
        if not isinstance(raw_entry, dict):
            print(f"[07] WARNING: skipping invalid target_files entry: {raw_entry!r}")
            continue

        path = str(raw_entry.get("path", "")).replace("\\", "/").strip().lstrip("/")
        if _is_disallowed_target_path(path):
            print(f"[07] WARNING: dropping disallowed target path: {path!r}")
            continue

        instructions = raw_entry.get("instructions", [])
        if isinstance(instructions, str):
            instructions = [instructions]
        if not isinstance(instructions, list):
            instructions = []

        normalized_targets.append(
            {
                "path": path,
                "action": _normalize_target_action(raw_entry.get("action")),
                "reason": str(raw_entry.get("reason", "")).strip(),
                "instructions": [
                    str(item).strip()
                    for item in instructions
                    if str(item).strip()
                ],
                "risk": _normalize_risk(raw_entry.get("risk")),
            }
        )

    plan["target_files"] = normalized_targets

    if not plan["target_files"]:
        print("[07] WARNING: mini plan has no target_files. Implementer may have nothing to do.")

    analysis.setdefault("scope", "mini")
    analysis["scope"] = "mini"
    analysis.setdefault("generated_at", _utc_now_iso())
    analysis.setdefault("possible_changes", [])
    analysis.setdefault("impacts", [])
    analysis.setdefault("warnings", [])
    analysis.setdefault("conflicts", [])
    analysis.setdefault("recommendations", [])

    for key in ("possible_changes", "impacts", "warnings", "conflicts", "recommendations"):
        if not isinstance(analysis.get(key), list):
            analysis[key] = [str(analysis[key])]

    return plan, analysis


def run_mini_scope() -> None:
    print("[07] Scope: mini")
    print("[07] Building targeted mini plan …")

    request = _load_mini_request()
    context_bundle = _load_mini_context_bundle()

    print(f"[07] Request context: {len(request)} chars")
    if context_bundle:
        print(f"[07] Knowledge context: {len(context_bundle)} chars")
    else:
        print("[07] Knowledge context: none")

    result = call_mini_planner(request, context_bundle)
    plan_mini, impact_analysis = validate_and_normalize_mini_result(result)

    PLANNER_MINI_PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLANNER_MINI_IMPACT.parent.mkdir(parents=True, exist_ok=True)

    PLANNER_MINI_PLAN.write_text(
        json.dumps(plan_mini, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _track_write(PLANNER_MINI_PLAN)

    PLANNER_MINI_IMPACT.write_text(
        json.dumps(impact_analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _track_write(PLANNER_MINI_IMPACT)

    print(f"[07] Mini plan written            → {PLANNER_MINI_PLAN}")
    print(f"[07] Mini impact analysis written → {PLANNER_MINI_IMPACT}")
    print(f"[07] Target files: {len(plan_mini.get('target_files', []))}")
    for entry in plan_mini.get("target_files", []):
        print(
            f"  - {entry.get('action', 'MODIFY'):<6} "
            f"{entry.get('path')}  risk={entry.get('risk')}"
        )
    print("[07] Done. Pass --scope mini --use-planner-plan to 08_executor.py.")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    exit_code = 0

    try:
        parser = _build_parser()
        args = parser.parse_args()

        _configure_project(args.project, parser)

        # Important: do not call ensure_dirs() at import-time.
        # PIPELINE_PROJECT must be available before artifact paths are resolved.
        ensure_dirs()

        if args.scope == "mini":
            run_mini_scope()
        else:
            run_full_scope()

    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[07] ERROR: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        print_summary("[07]")
        _print_artifact_access_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
