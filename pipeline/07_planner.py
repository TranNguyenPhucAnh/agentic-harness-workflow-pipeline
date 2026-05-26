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
#   artifacts_<slug>/spec/specwright_spec_<slug>.md  (upstream-aware, specwright)
#   artifacts_<slug>/scaffolder/blueprint.json       (upstream-aware, scaffolder)
#
# READS mini:
#   artifacts_<slug>/clarificator/session.json       (upstream-aware, clarificator, field: requirement_synthesis + full object as context bundle)
#   artifacts_<slug>/enricher/enriched_prompt.md     (upstream-aware, enricher, optional)
#   artifacts_<slug>/absorber/codebase_map.md        (upstream-aware, absorber)
#   artifacts_<slug>/archivist/knowledge_log.md      (knowledge-aware, archivist)
#   artifacts_<slug>/archivist/spec_gaps.md          (knowledge-aware, archivist)
#   artifacts_<slug>/patcher/attempt_log.json        (last entry only)


sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage, summary as cost_summary  # noqa: E402
from modules.call_llm import call_llm_json
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
# Dependency semantics / alias guards
# ════════════════════════════════════════════════════════════════════════════

KNOWN_BUNDLED_PLUGIN_ALIASES: dict[str, dict[str, str]] = {
    "@wavesurfer/regions": {
        "owner_package": "wavesurfer.js",
        "canonical_feature": "Regions plugin",
        "import_path": "wavesurfer.js/dist/plugins/regions.esm.js",
    },
    "wavesurferregions": {
        "owner_package": "wavesurfer.js",
        "canonical_feature": "Regions plugin",
        "import_path": "wavesurfer.js/dist/plugins/regions.esm.js",
    },
}

_PACKAGE_LIKE_RE = re.compile(r"^(?:@[a-z0-9._-]+/[a-z0-9._-]+|[a-z0-9._-]+)$", re.I)


def _normalize_dep_name(value: Any) -> str:
    return str(value or "").strip()


def _collect_planned_text_fragments(plan: dict) -> list[str]:
    fragments: list[str] = []

    stack = plan.get("stack")
    if isinstance(stack, dict):
        for v in stack.values():
            if isinstance(v, str):
                fragments.append(v)
            elif isinstance(v, list):
                fragments.extend(str(x) for x in v)
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str):
                        fragments.append(vv)
                    elif isinstance(vv, list):
                        fragments.extend(str(x) for x in vv)

    for task in plan.get("tasks", []):
        if not isinstance(task, dict):
            continue

        for key in ("file_path", "behavior_summary", "role"):
            val = task.get(key)
            if isinstance(val, str):
                fragments.append(val)

        for key in ("sub_tasks", "gotchas", "notes", "tailwind_hints"):
            val = task.get(key)
            if isinstance(val, list):
                fragments.extend(str(x) for x in val)
            elif isinstance(val, str):
                fragments.append(val)

        dependency_hints = task.get("dependency_hints")
        if isinstance(dependency_hints, list):
            fragments.extend(str(x) for x in dependency_hints)
        elif isinstance(dependency_hints, str):
            fragments.append(dependency_hints)

    return fragments


def _validate_dependency_semantics(plan: dict, stub_files: list[dict]) -> None:
    fragments = _collect_planned_text_fragments(plan)
    lower_fragments = [f.lower() for f in fragments]

    has_regions_plugin_semantics = any(
        ("regionsplugin.create" in f)
        or ("regions plugin" in f)
        or ("regionsplugin" in f)
        or ("wavesurfer.js/dist/plugins/regions.esm.js" in f)
        for f in lower_fragments
    )

    promoted_aliases = []
    for alias in KNOWN_BUNDLED_PLUGIN_ALIASES:
        if any(alias.lower() in f for f in lower_fragments):
            promoted_aliases.append(alias)

    if has_regions_plugin_semantics and promoted_aliases:
        aliases = ", ".join(sorted(set(promoted_aliases)))
        print(
            "[07] WARNING: planner appears to promote bundled plugin alias(es) "
            f"to standalone dependency/package names: {aliases}",
            file=sys.stderr,
        )
        print(
            "[07] WARNING: expected canonical dependency is 'wavesurfer.js' and "
            "Regions should stay a bundled plugin imported via "
            "'wavesurfer.js/dist/plugins/regions.esm.js'.",
            file=sys.stderr,
        )


def _rewrite_stub_semantics(stub_files: list[dict]) -> list[dict]:
    rewritten: list[dict] = []

    for stub in stub_files:
        if not isinstance(stub, dict):
            continue

        item = dict(stub)
        item.setdefault("dependency_hints", [])

        if item.get("file_path") == "package.json":
            hints = item.get("dependency_hints")
            if not isinstance(hints, list):
                hints = []

            normalized_hints = []
            for hint in hints:
                if not isinstance(hint, dict):
                    continue

                h = dict(hint)
                name = _normalize_dep_name(h.get("name"))
                if name in KNOWN_BUNDLED_PLUGIN_ALIASES:
                    alias_info = KNOWN_BUNDLED_PLUGIN_ALIASES[name]
                    h["kind"] = "bundled_plugin"
                    h["owner_package"] = alias_info["owner_package"]
                    h["canonical_feature"] = alias_info["canonical_feature"]
                    h["import_path"] = alias_info["import_path"]

                normalized_hints.append(h)

            item["dependency_hints"] = normalized_hints

        rewritten.append(item)

    return rewritten


# ════════════════════════════════════════════════════════════════════════════
# Artifact access tracking
# ════════════════════════════════════════════════════════════════════════════
# Full-scope prompt
# ════════════════════════════════════════════════════════════════════════════

FULL_SYSTEM_PROMPT = """\
You are a senior software architect acting as a PLANNER.

You will receive:
  1. A canonical spec (Markdown)
  2. Stub files — each entry includes:
       file_path, kind, module, module_purpose,
       exports[]             — exact public symbols this file must export
       quirks[]              — known implementation gotchas from the scaffold author
       acceptance_criteria[] — AC IDs this file must satisfy
       dependency_hints[]    — optional structured dependency metadata from the scaffold

Each dependency_hints entry may include:
- name
- kind: "npm_package" | "bundled_plugin" | "subpath_import" | "tooling" | "ui_primitive"
- owner_package
- canonical_feature
- import_path
  3. open_questions[]  — resolved design decisions (id, question, affects, impact)
  4. module_implementation_order[] — suggested module ordering from the scaffold

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

IMPORTANT:
- Do NOT promote bundled plugins, subpath imports, or feature names into standalone npm packages.
- If scaffold metadata says something is a bundled plugin of another package, record it under the owning package in stack.key_libs or mention it as a plugin detail, NOT as an independent installable library.
- Example: the Wavesurfer Regions plugin belongs to "wavesurfer.js" and should be described as a bundled plugin imported via "wavesurfer.js/dist/plugins/regions.esm.js", not as "@wavesurfer/regions" or "wavesurferregions" dependency.

If the project is a monorepo or mixed stack, represent that explicitly, for example:
{
  "frontend": {
    "language": "TypeScript 5.x",
    "runtime": "Vite 5",
    "framework": "React 18",
    "styling": "Tailwind CSS v4",
    "test_runner": "Vitest",
    "key_libs": ["Zustand", "Dexie", "WaveSurfer.js 7"]
  }
}

## Step 1 — Plan EVERY file in the scaffold

RULE — FULL COVERAGE:
Every file listed in implementation_order MUST have a task entry with sub_tasks.
No file may appear only as a depends_on reference without its own task entry.
This includes: types files, error/exception definitions, config files, entry points,
constants files, and any other file that appears in the scaffold.

RULE — CONSUME SCAFFOLD METADATA:
Each stub file entry carries exports[], quirks[], and acceptance_criteria[] from the
scaffold author. These encode resolved design decisions — treat them as ground truth.

- exports[]: every symbol listed MUST appear in sub_tasks as something to implement
  and export. Do not invent exports not listed; do not omit listed exports.
- quirks[]: every quirk listed MUST be reflected verbatim or paraphrased in the
  task's gotchas[] or sub_tasks. Do not silently drop scaffold quirks.
- acceptance_criteria[]: copy the AC IDs into the task's acceptance_criteria field
  so the implementer knows what to verify.
- dependency_hints[]: treat these as dependency ground truth when present.
  - "npm_package" means an actual installable dependency.
  - "bundled_plugin" means a capability shipped inside another package; do NOT list it as a standalone dependency.
  - "subpath_import" means import via the given import_path from the owning package; do NOT invent a package name from it.

RULE — OPEN QUESTIONS:
For each open question provided, its resolution/impact MUST be reflected in the
gotchas[] or notes[] of every task whose file_path appears in the question's
affects[] list. Do not re-open resolved questions — treat the impact text as final.

RULE — CONFIG FILES:
Config files (kind === "config") need task entries too.
Their sub_tasks describe what to write (content/setup), not runtime logic.
Their gotchas should cover tooling quirks (plugin order, extends chains, etc.).
tailwind_hints is null for config files.

For package.json specifically:
- Only list actual installable npm packages in its dependency-related sub_tasks.
- If a feature is provided by a bundled plugin, write it as an implementation/import note tied to the owning package, not as a separate dependency to install.
- Never normalize or sanitize a plugin alias into a fake package name.

RULE — IMPLEMENTATION ORDER:
Use the module_implementation_order hint to sequence implementation_order.
Within a module, order files by their internal dependencies.
Config files come last unless another file depends on them at build time.

For each file, output a task object with:
- file_path
- kind: "source" | "config"
- behavior_summary: 1-2 sentences describing what this file does in the context of
  the overall system — written so the implementer understands the file's role WITHOUT
  needing to read the spec. Be concrete: name the callers, the data it produces, the
  invariants it maintains.
- role: one-sentence functional label
- acceptance_criteria: list of AC IDs copied from the scaffold (empty list if none)
- depends_on: list of files this file imports from
- dependency_hints: copy through any relevant structured dependency metadata for this file, especially package.json
- sub_tasks: ordered implementation steps, specific enough that the implementer does
  not need the spec. Every export listed in the scaffold's exports[] MUST appear here.
  For types/interfaces files: list every type, interface, enum, and constant to define
  with their fields and value constraints.
  For config files: describe the exact content to write.
- gotchas: edge cases and framework quirks. Every quirk from the scaffold's quirks[]
  MUST appear here (verbatim or paraphrased). Add stack-derived gotchas on top.
- notes: cross-cutting concerns from open_questions that apply to THIS file only.
  If an open question's affects[] includes this file_path, its resolution goes here.
  If no relevant open questions, set notes to [].
- tailwind_hints: styling hints for visual components, or null

## Step 2 — Stack-specific gotchas

For EACH file, "gotchas" must include framework/language/runtime quirks relevant
to the detected stack AND to this specific file. Do not give generic advice.
Ask yourself: "What would a developer who knows the detected stack warn their
colleague about before implementing this specific file?"

Examples of good stack-derived gotchas:
- React 18+: useEffect can run twice in StrictMode — effects need cleanup/idempotence
- Vite: use import.meta.env instead of process.env for client env vars
- Python/FastAPI: use async def only when awaiting async I/O; do not block the event loop
- Pydantic v2: use model_config / field validators instead of v1 Config/validator patterns
- Vue 3 Composition API: destructuring reactive() can lose reactivity
- Go: propagate context cancellation to avoid goroutine leaks
- SQLAlchemy async: do not mix sync Session with async engine/session

Derive these from the spec's stack, not from hardcoded assumptions.

Return a single JSON object — NO markdown fences, raw JSON only:
{
  "plan_version": "1.0.0",
  "scope": "full",
  "stack": { ... },
  "tasks": [
    {
      "file_path": "src/types.ts",
      "kind": "source",
      "behavior_summary": "Defines all shared TypeScript types used across the app. Every other file imports from here — no runtime logic, only type declarations.",
      "role": "Shared type definitions for all domain objects.",
      "acceptance_criteria": [],
      "depends_on": [],
      "sub_tasks": [
        "1. Define SrtCue interface with fields: id (string), start (number), end (number), text (string).",
        "2. Define Tag type as union: 'vocab' | 'slang' | 'idiom' | 'speed' | 'intonation' | 'accent' | 'context'.",
        "3. Export all types — no default export."
      ],
      "gotchas": [
        "All id fields are client-generated UUID strings via crypto.randomUUID(), NOT auto-increment integers (CON-001).",
        "TypeScript strict mode: every field must be explicitly typed; avoid implicit any."
      ],
      "notes": [
        "OQ-10: A boolean field usesOpfsFallback may need to be added to the Episode interface — see open_questions for resolution."
      ],
      "tailwind_hints": null
    }
  ],
  "implementation_order": [
    "src/types.ts",
    "src/config.ts",
    "src/db.ts",
    "..."
  ]
}

RULE — NOTES DISTRIBUTION:
Do NOT output a global_notes string. For each cross-cutting concern, inject it into
the notes[] of ONLY the task(s) it directly applies to.
notes[] for a task should contain only concerns directly relevant to that file.
If a task has no relevant cross-cutting notes, set notes to [].

Rules:
- Reason as deeply as needed — this is your reasoning budget well spent.
- Be specific: reference exact constant names, prop names, type names, schema names,
  and file paths from the spec/scaffold.
- Every file in implementation_order must have a task entry — no exceptions.
- implementation_order must respect dependency order.
- behavior_summary must be self-contained: the implementer should understand the
  file's role without reading the spec. Name callers, consumers, and invariants.
- sub_tasks for types/interfaces/errors files must enumerate every type/class to
  define with their fields and value constraints — not just "define types".
- tailwind_hints: include for visual components if the detected stack uses Tailwind;
  otherwise provide relevant styling hints or null.
- Do not assume TypeScript, React, Vite, Tailwind, or Vitest unless the spec/scaffold
  actually indicates them.
- Do not use implementation patterns from a different stack than the one detected.
- Never convert plugin aliases or import-path hints into invented package names.
- If the scaffold/spec mentions Wavesurfer Regions as a plugin, keep it as a plugin of wavesurfer.js; never output "@wavesurfer/regions" or "wavesurferregions" as a package to install unless the scaffold explicitly marks it as an npm_package.
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


def _python_dict_to_json(raw: str) -> dict | None:
    """
    Parse Python dict literal output (single-quoted keys/values) that
    some models emit instead of valid JSON. Uses ast.literal_eval (safe).
    """
    import ast
    try:
        parsed = ast.literal_eval(raw.strip())
        if isinstance(parsed, dict):
            return json.loads(json.dumps(parsed))
    except Exception:
        pass
    return None


def _parse_json(raw: str, label: str) -> dict:
    """
    Extract JSON from model output robustly.

    Pass 1 — direct json.loads (standard JSON)
    Pass 2 — find outermost {...} then json.loads
    Pass 3 — ast.literal_eval (Python dict with single quotes)
    """
    raw = raw.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())

    # Pass 1: direct parse
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{label} parsed as {type(parsed).__name__}, expected object.")
        return parsed
    except json.JSONDecodeError:
        pass

    # Pass 2: find outermost {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if not isinstance(parsed, dict):
                raise RuntimeError(f"{label} parsed as {type(parsed).__name__}, expected object.")
            return parsed
        except json.JSONDecodeError:
            pass

    # Pass 3: Python dict literal (single-quoted keys/values)
    result = _python_dict_to_json(raw)
    if result is not None:
        print(f"[07][warn] {label}: model returned Python dict syntax — parsed via ast.literal_eval.")
        return result

    # Also try pass 3 on the extracted {...} block
    if match:
        result = _python_dict_to_json(match.group())
        if result is not None:
            print(f"[07][warn] {label}: model returned Python dict syntax — parsed via ast.literal_eval.")
            return result

    print(f"[07] ERROR: Could not parse JSON from {label}", file=sys.stderr)
    print(f"[07] Raw output (first 1000 chars):\n{raw[:1000]}", file=sys.stderr)
    raise RuntimeError(f"Could not parse JSON from {label}")


# ════════════════════════════════════════════════════════════════════════════
# Core model call
# ════════════════════════════════════════════════════════════════════════════

# _call_planner_json removed — use call_llm_json() from modules.call_llm



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
    Extract non-test files from the module-centric blueprint schema,
    including top-level config_files.

    blueprint.json schema (new):
    {
      "modules": [{
        "module": "...",
        "purpose": "...",
        "depends_on": [...],
        "files": [{
          "path": "...",
          "kind": "source" | "test" | "config" | "migration",
          "exports": [...],
          "quirks": [...],
          "acceptance_criteria": [...]
        }]
      }],
      "config_files": [{
        "path": "...",
        "kind": "config",
        "note": "...",
        "dependencies": [...]
      }],
      "implementation_order": [...],
      "open_questions": [...]
    }

    Returns a flat list of file dicts for the full planner prompt.
    """
    modules = scaffold.get("modules", [])

    if not isinstance(modules, list):
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

            if file_entry.get("kind", "source") == "test":
                continue

            stub_files.append({
                "file_path": file_entry.get("path", ""),
                "kind": file_entry.get("kind", "source"),
                "module": mod.get("module", ""),
                "module_purpose": mod.get("purpose", ""),
                "exports": file_entry.get("exports", []),
                "quirks": file_entry.get("quirks", []),
                "acceptance_criteria": file_entry.get("acceptance_criteria", []),
                "dependency_hints": file_entry.get("dependency_hints", []),
            })

    for cfg in scaffold.get("config_files", []):
        if not isinstance(cfg, dict):
            continue

        path = cfg.get("path", "")
        if not path:
            continue

        dependency_hints = cfg.get("dependencies", [])
        if not isinstance(dependency_hints, list):
            dependency_hints = []

        stub_files.append({
            "file_path": path,
            "kind": "config",
            "module": "config_files",
            "module_purpose": "Project configuration and tooling",
            "exports": [],
            "quirks": [cfg["note"]] if cfg.get("note") else [],
            "acceptance_criteria": [],
            "dependency_hints": dependency_hints,
        })

    return _rewrite_stub_semantics(stub_files)



def call_full_planner(spec: str, stub_files: list[dict], scaffold: dict) -> dict:
    # Build optional context sections from scaffold top-level fields
    open_questions = scaffold.get("open_questions", [])
    module_order = scaffold.get("implementation_order", [])

    context_sections: list[str] = []

    if module_order:
        context_sections.append(
            "### module_implementation_order\n"
            + json.dumps(module_order, ensure_ascii=False)
        )

    if open_questions:
        context_sections.append(
            "### open_questions\n"
            + json.dumps(open_questions, indent=2, ensure_ascii=False)
        )

    context_block = ("\n\n" + "\n\n".join(context_sections)) if context_sections else ""

    semantic_guard = {
        "dependency_rules": [
            "Do not promote bundled plugins to standalone npm packages.",
            "Do not invent package names from plugin aliases or import-path fragments.",
            "For Wavesurfer v7 Regions, canonical owner package is wavesurfer.js.",
            "Canonical import path for Regions plugin: wavesurfer.js/dist/plugins/regions.esm.js",
        ],
        "known_bundled_plugin_aliases": KNOWN_BUNDLED_PLUGIN_ALIASES,
    }

    user_message = (
        f"### canonical spec\n\n{spec}\n\n"
        f"### scaffold stub files\n\n"
        f"{json.dumps(stub_files, indent=2, ensure_ascii=False)}\n\n"
        f"### semantic guardrails\n\n"
        f"{json.dumps(semantic_guard, indent=2, ensure_ascii=False)}"
        f"{context_block}"
    )

    print(f"[07] Calling model: {get_model(ROLE)} (provider: {get_provider(ROLE)}) — full planner response …")
    result, _ = call_llm_json(
        ROLE,
        FULL_SYSTEM_PROMPT,
        user_message,
        temperature=0.2,
        max_tokens=32768,
        retries=2,
        backoff=False,
        caller_file=__file__,
        label="[07] full planner response",
    )
    return _parse_json(str(result), label="full planner response")


def validate_full_plan(plan: dict, stub_files: list[dict]) -> None:
    """
    Warn if any stub file is missing from the plan, report detected stack,
    and run semantic validation against bundled-plugin/package confusion.
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

    _validate_dependency_semantics(plan, stub_files)



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

    plan = call_full_planner(spec, stub_files, scaffold)
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
        "model": "...",
        "cost": ...,
        "task_summary": "..." (mini only),
        "task_count": N (full) | target_file_count (mini),
        "impact_warnings": [...] (mini only),
      }
    """
    
    token_sum = cost_summary()
    cost_total = token_sum.get("total_cost_usd") if isinstance(token_sum, dict) else None

    entry: dict = {
        "scope": scope,
        "generated_at": _utc_now_iso(),
        "plan_version": plan.get("plan_version", "1.0.0"),
        "model": get_model(ROLE),
        "cost": cost_total,
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
            track_read(log_path)
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

    print(f"[07] Calling model: {get_model(ROLE)} (provider: {get_provider(ROLE)}) — mini planner response …")
    result, _ = call_llm_json(
        ROLE,
        MINI_SYSTEM_PROMPT,
        user_message,
        temperature=0.15,
        max_tokens=32768,
        retries=2,
        backoff=False,
        caller_file=__file__,
        label="[07] mini planner response",
    )
    return _parse_json(str(result), label="mini planner response")


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