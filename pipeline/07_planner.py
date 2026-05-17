"""
pipeline/07_planner.py
======================
Step 7 — Planner (reasoning-heavy, no code output).

This script supports two scopes:

FULL SCOPE
──────────
Spec-driven planner for the full pipeline.

Reads:
    artifacts_<slug>/spec/specwright_spec_<slug>.md
    artifacts_<slug>/scaffolder/blueprint.json

Writes:
    artifacts_<slug>/planner/full_plan.json        (short-term, overwrite)
    artifacts_<slug>/planner/plan_log.json         (long-term, append)

Consumed by:
    pipeline/08_executor.py --scope full --use-planner-plan


MINI SCOPE
──────────
Targeted planner for small daily-driver tasks.

Reads:
    artifacts_<slug>/clarificator/session.json             (field: requirement_synthesis)
    artifacts_<slug>/enricher/enriched_prompt.md           optional
    artifacts_<slug>/clarificator/session.json             optional (full object as context)
    artifacts_<slug>/archivist/knowledge_log.md            optional
    artifacts_<slug>/absorber/codebase_map.md              optional
    artifacts_<slug>/patcher/attempt_log.json              optional (last entry)
    artifacts_<slug>/archivist/spec_gaps.md                optional

Writes:
    artifacts_<slug>/planner/mini_plan.json        (short-term, overwrite — { "plan": {...}, "impact": {...} })
    artifacts_<slug>/planner/plan_log.json         (long-term, append)

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
#   artifacts_<slug>/planner/full_plan.json          (short-term, overwrite)
#   artifacts_<slug>/planner/plan_log.json           (long-term, append)
#
# OWNS mini:
#   artifacts_<slug>/planner/mini_plan.json          (short-term, overwrite — {plan, impact})
#   artifacts_<slug>/planner/plan_log.json           (long-term, append — shared with full)
#
# READS full:
#   artifacts_<slug>/spec/specwright_spec_<slug>.md
#   artifacts_<slug>/scaffolder/blueprint.json
#
# READS mini:
#   artifacts_<slug>/clarificator/session.json           (field: requirement_synthesis)
#   artifacts_<slug>/enricher/enriched_prompt.md
#   artifacts_<slug>/clarificator/session.json           (full object as context bundle)
#   artifacts_<slug>/archivist/knowledge_log.md
#   artifacts_<slug>/absorber/codebase_map.md
#   artifacts_<slug>/patcher/attempt_log.json            (last entry only)
#   artifacts_<slug>/archivist/spec_gaps.md

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402
from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_MAP,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    CLARIFICATOR_SESSION,
    CLARIFIED_REQ,
    ENRICHER_OVERWRITE_PROMPT,
    PATCHER_ATTEMPT_LOG,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    PLANNER_PLAN_LOG,
    SCAFFOLD_JSON,
    ensure_dirs,
    get_spec_path,
)


ROLE = "planner"


# ════════════════════════════════════════════════════════════════════════════
# Artifact access tracking
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

## Step 1 — Plan EVERY file in the scaffold

RULE — FULL COVERAGE:
Every file listed in implementation_order MUST have a task entry with sub_tasks.
No file may appear only as a depends_on reference without its own task entry.
This includes: types files, error/exception definitions, config files, entry points,
constants files, and any other file that appears in the scaffold.

For each file, output a task object with:
- behavior_summary: 1-2 sentences describing what this file does in the context of
  the overall system — written so the implementer understands the file's role WITHOUT
  needing to read the spec. Be concrete: name the callers, the data it produces, the
  invariants it maintains.
- role: one-sentence functional label (same as before)
- depends_on: list of files this file imports from
- sub_tasks: ordered implementation steps, specific enough that the implementer does
  not need the spec. For types/interfaces files: list every type, interface, enum, and
  constant to define with their fields and value constraints.
- gotchas: edge cases and framework quirks the implementer must handle
- notes: cross-cutting concerns from global context that apply ONLY to this file
  (see Rule — Notes Distribution below)
- tailwind_hints: styling hints for visual components, or null

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
      "file_path": "src/types/sensor.ts",
      "behavior_summary": "Defines all shared TypeScript types used across the app. Every other file imports from here — it has no runtime logic, only type declarations.",
      "role": "Shared type definitions for sensor data, anomaly records, and replay state.",
      "depends_on": [],
      "sub_tasks": [
        "1. Define SensorPoint interface with fields: timestamp (number), temperature (number), humidity (number), pressure (number), decisionScore (number).",
        "2. Define AnomalyCluster interface with fields: startIndex (number), endIndex (number), severity ('low'|'medium'|'high').",
        "3. Define ReplayState type as union: 'idle' | 'playing' | 'paused' | 'done'.",
        "4. Export all types — no default export."
      ],
      "gotchas": [
        "TypeScript strict mode: every field must be explicitly typed; avoid implicit any.",
        "Do not add runtime values (classes, constants) here — types only."
      ],
      "notes": [],
      "tailwind_hints": null
    },
    {
      "file_path": "src/hooks/useSensorData.ts",
      "behavior_summary": "Generates and manages the sensor dataset for the dashboard. Called once by App.tsx on mount. Consumers (AnomalyFeed, ReplayControls) read data from this hook via context or props.",
      "role": "React hook producing the demo sensor dataset with injected anomaly clusters.",
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
      "notes": [],
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
  ]
}

RULE — NOTES DISTRIBUTION:
Do NOT output a global_notes string. Instead, for each cross-cutting concern,
inject it into the notes[] array of ONLY the task(s) it directly applies to.
A note that applies to 3 files goes into those 3 tasks, not into a global field.
notes[] for a task should contain only concerns directly relevant to that file.
If a task has no relevant cross-cutting notes, set notes to [].

Rules:
- Reason as deeply as needed — this is your reasoning budget well spent.
- Be specific: reference exact constant names, prop names, type names, schema names, and file paths from the spec/scaffold.
- Every file in implementation_order must have a task entry — no exceptions.
- implementation_order must respect dependency order.
- behavior_summary must be self-contained: the implementer should understand the
  file's role without reading the spec. Name callers, consumers, and invariants.
- sub_tasks for types/interfaces/errors files must enumerate every type/class to
  define with their fields and value constraints — not just "define types".
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
- Do not include artifact paths such as artifacts_*/, planner/*, executor/*, debugger/*, judge/*, patcher/*, archivist/*, absorber/*, clarificator/*, enricher/*, spectracker/*, scaffolder/*, spec/*, or legacy paths state/*, execution/*, run/*, cache/*, knowledge/*, reports/*.
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
            "Planner step. Writes planner/full_plan.json "
            "or planner/mini_plan.json (with impact merged). "
            "Appends to planner/plan_log.json in both modes."
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
            "Planner scope. full writes planner/full_plan.json; "
            "mini writes planner/mini_plan.json ({plan, impact} merged). "
            "Both modes append to planner/plan_log.json."
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

        track_read(path)

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

    track_read(spec_path)
    return spec_path.read_text(encoding="utf-8")


def _load_scaffold() -> dict:
    if not SCAFFOLD_JSON.exists():
        raise FileNotFoundError(
            f"Missing scaffold: {SCAFFOLD_JSON}\n"
            "Run scaffolder first, for example:\n"
            "  python harness.py --project <name> --scope full --scaffold"
        )

    track_read(SCAFFOLD_JSON)
    return json.loads(SCAFFOLD_JSON.read_text(encoding="utf-8"))


def _extract_stub_files(scaffold: dict) -> list[dict]:
    """
    Extract non-test files from the new module-centric blueprint schema.

    blueprint.json schema:
      { modules: [{ module, purpose, files: [{ path, kind }] }] }

    kind values: "source" | "test" | "config" | "migration"
    Non-test = kind != "test".

    Returns a flat list of { file_path, kind } dicts for the full planner prompt.
    """
    modules = scaffold.get("modules", [])
    if not isinstance(modules, list):
        # Legacy flat-file schema fallback (is_test field)
        files = scaffold.get("files", [])
        if not isinstance(files, list):
            raise RuntimeError(
                "Invalid scaffold: expected top-level 'modules' list "
                "(new schema) or 'files' list (legacy schema)."
            )
        print("[07] WARNING: scaffold uses legacy flat-file schema — consider regenerating.")
        return [
            f for f in files
            if isinstance(f, dict) and not f.get("is_test")
        ]

    stub_files: list[dict] = []
    for mod in modules:
        if not isinstance(mod, dict):
            continue
        for file_entry in mod.get("files", []):
            if not isinstance(file_entry, dict):
                continue
            if file_entry.get("kind", "source") != "test":
                stub_files.append({
                    "file_path": file_entry.get("path", ""),
                    "kind": file_entry.get("kind", "source"),
                    "module": mod.get("module", ""),
                    "module_purpose": mod.get("purpose", ""),
                })
    return stub_files


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

    stub_files = _extract_stub_files(scaffold)
    if not stub_files:
        raise RuntimeError(
            "No non-test files found in scaffold blueprint. "
            "Ensure scaffolder/blueprint.json has modules with source files."
        )

    print("[07] Scope: full")
    print(f"[07] Planning {len(stub_files)} non-test stub file(s) …")

    plan = call_full_planner(spec, stub_files)
    validate_full_plan(plan, stub_files)

    PLANNER_FULL_PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLANNER_FULL_PLAN.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(PLANNER_FULL_PLAN)

    append_plan_log(scope="full", plan=plan, impact=None)

    print(f"[07] Full plan written → {PLANNER_FULL_PLAN}")
    print(f"[07] Tasks in plan: {len(plan.get('tasks', []))}")
    print(f"[07] Implementation order: {plan.get('implementation_order', [])}")
    print("[07] Done. Pass --use-planner-plan to 08_executor.py to use this plan.")


# ════════════════════════════════════════════════════════════════════════════
# Plan log (long-term, shared by full + mini)
# ════════════════════════════════════════════════════════════════════════════

def append_plan_log(*, scope: str, plan: dict, impact: dict | None) -> None:
    """
    Append one entry to planner/plan_log.json (long-term audit trail).
    Called at the end of both full and mini runs.

    Entry schema:
      {
        "scope": "full" | "mini",
        "generated_at": "<iso>",
        "plan_version": "...",
        "task_summary": "..." (mini only),
        "task_count": N (full) | target_file_count (mini),
        "impact_warnings": [...] (mini only),
      }
    """
    entry: dict = {
        "scope": scope,
        "generated_at": _utc_now_iso(),
        "plan_version": plan.get("plan_version", "1.0.0"),
    }

    if scope == "full":
        entry["task_count"] = len(plan.get("tasks", []))
        entry["implementation_order"] = plan.get("implementation_order", [])
    else:
        entry["task_summary"] = plan.get("task_summary", "")
        entry["task_count"] = len(plan.get("target_files", []))
        if impact:
            entry["impact_warnings"] = impact.get("warnings", [])
            entry["impact_conflicts"] = impact.get("conflicts", [])

    log_path = PLANNER_PLAN_LOG
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if log_path.exists():
            raw = log_path.read_text(encoding="utf-8").strip()
            try:
                data = json.loads(raw)
                entries = data.get("entries", []) if isinstance(data, dict) else data
                if not isinstance(entries, list):
                    entries = []
            except Exception:
                entries = []
        else:
            entries = []

        entries.append(entry)
        log_path.write_text(
            json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        track_write(log_path)
        print(f"[07] Plan log appended ({len(entries)} entries) → {log_path}")
    except Exception as exc:
        print(f"[07] WARNING: could not append plan log: {exc}", file=sys.stderr)


# ════════════════════════════════════════════════════════════════════════════
# Mini-scope planner
# ════════════════════════════════════════════════════════════════════════════

def _load_mini_request() -> str:
    """
    Load the mini request.

    Preferred:
      1. enricher/enriched_prompt.md  (richer context from enricher)
      2. clarificator/session.json    (field: requirement_synthesis)

    Both are tried; if both exist, both are included.
    """
    enriched = _read_optional(ENRICHER_OVERWRITE_PROMPT, max_chars=20_000)
    clarified = _read_clarificator_synthesis(max_chars=20_000)

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
        f"  - {CLARIFIED_REQ} (field: requirement_synthesis)\n\n"
        "Run clarificator/enricher first, for example:\n"
        "  python harness.py --project <name> --scope mini --clarify\n"
        "or provide clarificator/session.json with requirement_synthesis field."
    )


def _read_clarificator_synthesis(*, max_chars: int | None = None) -> str:
    """
    Extract requirement_synthesis field from clarificator/session.json.
    Falls back to full JSON text if field is absent.
    Returns empty string if file missing or unreadable.
    """
    try:
        if not CLARIFIED_REQ.exists():
            return ""
        track_read(CLARIFIED_REQ)
        raw = CLARIFIED_REQ.read_text(encoding="utf-8").strip()
        if not raw:
            return ""
        data = json.loads(raw)
        synthesis = data.get("requirement_synthesis", "")
        text = synthesis.strip() if isinstance(synthesis, str) else ""
        if not text:
            # Fallback: whole session as context
            text = raw
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + f"\n\n<!-- truncated at {max_chars} chars -->"
        return text
    except Exception as exc:
        print(f"[07] WARNING: could not read clarificator session: {exc}", file=sys.stderr)
        return ""


def _load_mini_context_bundle() -> str:
    """
    Load optional knowledge/context files for mini planning.
    Missing files are ignored.

    Sources (new layout):
      clarificator/session.json     — full session object (decisions, conflicts, tier_counts …)
      archivist/knowledge_log.md    — accumulated knowledge
      absorber/codebase_map.md      — merged codebase + config + blame map
      patcher/attempt_log.json      — last entry only (replaces PATCHER_FINDINGS_SNAPSHOT)
      archivist/spec_gaps.md        — known spec gaps
    """
    sections: list[str] = []

    # ── clarificator full session (context only — synthesis already in request) ──
    session_text = _read_json_optional(CLARIFICATOR_SESSION, max_chars=20_000)
    if session_text:
        sections.append("### clarificator/session.json\n" + session_text)

    # ── archivist knowledge log ──
    knowledge = _read_optional(ARCHIVIST_KNOWLEDGE_LOG, max_chars=30_000)
    if knowledge:
        sections.append("### archivist/knowledge_log.md\n" + knowledge)

    # ── absorber codebase map (merged config + blame) ──
    codebase = _read_optional(ABSORBER_CODEBASE_MAP, max_chars=35_000)
    if codebase:
        sections.append("### absorber/codebase_map.md\n" + codebase)

    # ── patcher last attempt entry ──
    patcher_last = _read_patcher_last_entry(max_chars=20_000)
    if patcher_last:
        sections.append("### patcher/attempt_log.json (last entry)\n" + patcher_last)

    # ── spec gaps ──
    spec_gaps = _read_optional(ARCHIVIST_SPEC_GAPS, max_chars=15_000)
    if spec_gaps:
        sections.append("### archivist/spec_gaps.md\n" + spec_gaps)

    return "\n\n".join(sections)


def _read_patcher_last_entry(*, max_chars: int | None = None) -> str:
    """
    Read the last entry from patcher/attempt_log.json.
    Returns empty string if file missing, empty, or unreadable.
    Replaces the removed PATCHER_FINDINGS_SNAPSHOT.
    """
    try:
        if not PATCHER_ATTEMPT_LOG.exists():
            return ""
        track_read(PATCHER_ATTEMPT_LOG)
        raw = PATCHER_ATTEMPT_LOG.read_text(encoding="utf-8").strip()
        if not raw:
            return ""
        data = json.loads(raw)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if not isinstance(entries, list) or not entries:
            return ""
        last = entries[-1]
        text = json.dumps(last, indent=2, ensure_ascii=False)
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + f"\n\n<!-- truncated at {max_chars} chars -->"
        return text
    except Exception as exc:
        print(f"[07] WARNING: could not read patcher attempt log: {exc}", file=sys.stderr)
        return ""


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

    # Artifact folder prefixes — implementer must never touch these
    blocked_prefixes = (
        "artifacts_",
        "planner/",
        "executor/",
        "debugger/",
        "reporter/",
        "judge/",
        "patcher/",
        "archivist/",
        "absorber/",
        "clarificator/",
        "enricher/",
        "spectracker/",
        "scaffolder/",
        "spec/",
        # Legacy dirs — still blocked for safety
        "state/",
        "cache/",
        "execution/",
        "run/",
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
    Validate and normalize model output into (plan, impact).

    The caller merges these into mini_plan.json as {"plan": plan, "impact": impact}.

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

    # Merge plan + impact into single mini_plan.json  { "plan": {...}, "impact": {...} }
    mini_plan_merged = {
        "plan": plan_mini,
        "impact": impact_analysis,
    }

    PLANNER_MINI_PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLANNER_MINI_PLAN.write_text(
        json.dumps(mini_plan_merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(PLANNER_MINI_PLAN)

    append_plan_log(scope="mini", plan=plan_mini, impact=impact_analysis)

    print(f"[07] Mini plan written → {PLANNER_MINI_PLAN}")
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
        print_artifact_summary("[07]")
        prompt_next_step(ROLE, prefix="[07]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
