"""
pipeline/08_executor.py
=======================
Step 8 — EXECUTOR.

This script supports two scopes:

FULL SCOPE
──────────
Spec/scaffold-driven implementation.

Modes:
    Default, no planner plan:
        Single API call for all requested non-test stub files.

    With planner plan (--use-planner-plan):
        Reads artifacts_<slug>/planner/full_plan.json produced
        by 07_planner.py.
        For each file, injects the matching planner task:
            - role
            - depends_on
            - sub_tasks
            - gotchas
            - tailwind_hints / styling hints
        Files are generated in implementation_order from the plan.

Delta mode:
    --only-files src/foo.py,src/bar.ts
        Only those files are regenerated.
        Other restored files in src/ are used as import/context references.

Reads:
    artifacts_<slug>/spec/specwright_spec_<slug>.md
    artifacts_<slug>/scaffolder/blueprint.json
    artifacts_<slug>/planner/full_plan.json                         optional, with --use-planner-plan

Writes:
    artifacts_<slug>/output/src/**                                 non-test files only
    artifacts_<slug>/executor/manifest.json                        (short-term, overwrite)
    artifacts_<slug>/executor/manifest_log.json                    (long-term, append)


MINI SCOPE
──────────
Targeted implementation for small daily-driver tasks.

Reads:
    artifacts_<slug>/planner/mini_plan.json                        (field: plan + impact merged)
    target files listed in mini_plan.json["plan"]["target_files"]  optional existing content

Writes:
    only files listed in mini_plan.json target_files
    artifacts_<slug>/executor/manifest.json                        (short-term, overwrite)
    artifacts_<slug>/executor/manifest_log.json                    (long-term, append)

Rules:
    - Does NOT read the canonical spec.
    - Does NOT require scaffolder blueprint.json.
    - Does NOT do broad rewrites.
    - Does NOT create/modify files outside mini_plan target_files.
    - Does NOT write artifact folder paths (executor/, planner/, absorber/, etc.).

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


# === WRITE AUTHORITY: executor ===
# OWNS full:
#   artifacts_<slug>/executor/manifest.json          (short-term, overwrite)
#   artifacts_<slug>/executor/manifest_log.json      (long-term, append)
#   artifacts_<slug>/output/src/**
#
# OWNS mini:
#   artifacts_<slug>/executor/manifest.json          (short-term, overwrite)
#   artifacts_<slug>/executor/manifest_log.json      (long-term, append)
#   files explicitly listed in mini_plan.json target_files
#
# READS full:
#   artifacts_<slug>/scaffolder/blueprint.json
#   artifacts_<slug>/planner/full_plan.json          (optional, with --use-planner-plan)
#   artifacts_<slug>/spec/specwright_spec_<slug>.md  (only in single-call fallback)
#
# READS mini:
#   artifacts_<slug>/planner/mini_plan.json          (fields: plan + impact)

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage, summary as cost_summary  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402
from artifacts.paths import (  # noqa: E402
    EXECUTOR_MANIFEST_LOG,
    EXECUTOR_OVERWRITE_MANIFEST,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    SCAFFOLD_JSON,
    SRC_DIR,
    artifact_root,
    ensure_dirs,
    get_spec_path,
)


ROLE = "executor"


# ════════════════════════════════════════════════════════════════════════════
# Artifact access tracking
# ════════════════════════════════════════════════════════════════════════════
# Generic source helpers
# ════════════════════════════════════════════════════════════════════════════

SOURCE_EXTENSIONS = {
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".py",
    ".go",
    ".java",
    ".kt",
    ".kts",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".vue",
    ".svelte",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".md",
}

FENCE_LANGUAGE_BY_EXT = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".py": "python",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".vue": "vue",
    ".svelte": "svelte",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".md": "markdown",
}


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fence_language(file_path: str) -> str:
    return FENCE_LANGUAGE_BY_EXT.get(Path(file_path).suffix.lower(), "")


def _code_fence(file_path: str, code: str) -> str:
    lang = _fence_language(file_path)
    if lang:
        return f"```{lang}\n{code}\n```"
    return f"```\n{code}\n```"


def _is_probably_source_file(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_EXTENSIONS


def _read_json_file(path: Any, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    track_read(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return data


def _read_optional_text(path: Any, *, max_chars: int | None = None) -> str:
    try:
        if not path.exists():
            return ""
        track_read(path)
        text = path.read_text(encoding="utf-8")
        if max_chars is not None and len(text) > max_chars:
            return text[:max_chars] + f"\n\n<!-- truncated at {max_chars} chars -->"
        return text
    except Exception as exc:
        print(f"[08] WARNING: could not read {path}: {exc}", file=sys.stderr)
        return ""


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

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
        "PIPELINE_PROJECT=<name> before running 08_executor.py directly."
    )


# ════════════════════════════════════════════════════════════════════════════
# Safe path helpers
# ════════════════════════════════════════════════════════════════════════════

def _safe_src_output_path(file_path: str) -> Path | None:
    """
    Convert scaffold file_path into an output path under SRC_DIR.

    Accepts:
        src/foo/bar.ts  -> SRC_DIR/foo/bar.ts
        foo/bar.ts      -> SRC_DIR/foo/bar.ts

    Rejects traversal/outside paths.
    """
    normalized = file_path.replace("\\", "/").strip().lstrip("/")
    rel = normalized[len("src/"):] if normalized.startswith("src/") else normalized
    out_path = SRC_DIR / rel

    try:
        src_root = Path(SRC_DIR).resolve()
        resolved = out_path.resolve()
    except FileNotFoundError:
        src_root = Path(SRC_DIR).absolute()
        resolved = out_path.absolute()

    if resolved != src_root and src_root not in resolved.parents:
        return None

    return out_path


def _normalize_repo_rel_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("/")


def _is_disallowed_artifact_rel_path(rel: str) -> bool:
    """
    Mini mode must not patch pipeline artifacts or state by accident.
    """
    normalized = _normalize_repo_rel_path(rel)

    if not normalized:
        return True

    if normalized.startswith("../") or "/../" in f"/{normalized}/":
        return True

    if normalized == "spec.md":
        return True

    if normalized.startswith("specwright_spec_") and normalized.endswith(".md"):
        return True

    blocked_prefixes = (
        "artifacts_",
        # artifact module folders — mini must not touch these
        "executor/",
        "planner/",
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
        # legacy dirs — still blocked for safety
        "state/",
        "cache/",
        "execution/",
        "run/",
        "knowledge/",
        "reports/",
    )
    return normalized.startswith(blocked_prefixes)


def _safe_artifact_output_path(rel_path: str) -> Path | None:
    """
    Convert a mini plan repo/project-relative path into a safe path under
    artifact_root().

    Accepts examples:
        src/components/Header.tsx
        tests/Header.test.tsx
        dags/ingest.py
        queries/daily.sql

    Rejects:
        ../x
        /absolute/path
        state/*
        cache/*
        execution/*
        run/*
        knowledge/*
        reports/*
        artifacts_*
        spec.md
        specwright_spec_<slug>.md
    """
    rel = _normalize_repo_rel_path(rel_path)

    if _is_disallowed_artifact_rel_path(rel):
        return None

    root = artifact_root()
    out_path = root / rel

    try:
        root_resolved = root.resolve()
        out_resolved = out_path.resolve()
    except FileNotFoundError:
        root_resolved = root.absolute()
        out_resolved = out_path.absolute()

    if out_resolved != root_resolved and root_resolved not in out_resolved.parents:
        return None

    return out_path


# ════════════════════════════════════════════════════════════════════════════
# Spec / prompt helpers — full scope
# ════════════════════════════════════════════════════════════════════════════

def _load_spec() -> str:
    spec_path = get_spec_path()
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Missing canonical spec: {spec_path}\n"
            "Run specwright first."
        )

    track_read(spec_path)
    return spec_path.read_text(encoding="utf-8")


def _format_stack_section(stack: dict[str, Any] | None) -> str:
    if not stack:
        return ""

    return f"""
## Project stack — follow exactly

The architect detected this stack from the spec/scaffold:

{json.dumps(stack, indent=2, ensure_ascii=False)}

Rules:
- Apply the idioms, imports, runtime APIs, framework conventions, and testing assumptions of this stack.
- Do NOT assume TypeScript, React, Vite, Tailwind, Vitest, Python, FastAPI, Vue, Go, or any other stack unless present in this stack/spec/scaffold.
- Do NOT use patterns from unrelated stacks.
- If this is a mixed-stack or monorepo project, infer the correct sub-stack from the current file path.
- Preserve the language, module system, file layout, public APIs, exported names, and dependency style implied by the scaffold.
"""


def build_system_prompt_single(
    instructions: str,
    stack: dict[str, Any] | None = None,
) -> str:
    """
    Single-call prompt used when no planner plan is available.
    """
    stack_section = _format_stack_section(stack)

    return f"""\
You are a senior software developer implementing source files.
{stack_section}
You will receive:
1. A technical spec
2. A scaffold JSON with stub files
3. Model-specific implementation instructions

Your task:
- Implement ONLY the missing function bodies, component bodies, classes, modules, handlers, or source bodies in the non-test source files.
- Return a JSON object with this exact schema:
  {{
    "files": [
      {{
        "file_path": "src/path/to/file.ext",
        "code": "<complete file content>"
      }}
    ]
  }}
- Do NOT modify test files (kind: "test" in the scaffold).
- Do NOT add new files beyond what is in the scaffold.
- Follow the detected project stack strictly.
- Preserve the scaffold's language, module system, imports style, public APIs, exported names, function signatures, and file paths.
- The code field must contain the COMPLETE file content for each file.
- Do not introduce libraries, frameworks, styling systems, runtime APIs, or test tools that are not in the spec/scaffold/stack.
- Output raw JSON only. No markdown fences. No explanation text.

Model-specific instructions:
{instructions}"""


def build_system_prompt_per_file(
    stack: dict[str, Any] | None = None,
) -> str:
    """
    Per-file generation prompt — one file per API call.
    """
    stack_section = _format_stack_section(stack)

    return f"""\
You are a senior software developer implementing ONE source file.
{stack_section}
You will receive:
1. A behavior summary describing this file's role in the system — treat as authoritative
2. A task plan from a senior architect with sub-tasks, gotchas, and notes — follow it carefully
3. Optional dependency/context files already implemented — for import/API reference
4. The stub for the SINGLE file you must implement

Your task:
- Implement this ONE file only.
- Return a JSON object with this EXACT schema:
  {{
    "file_path": "src/path/to/file.ext",
    "code": "<complete file content>"
  }}
- Follow the detected project stack strictly.
- Preserve the scaffold's language, module system, imports style, public APIs, exported names, function signatures, and file path.
- Keep compatibility with dependency/context files shown in the prompt.
- Do not introduce libraries, frameworks, styling systems, runtime APIs, or test tools that are not in the spec/scaffold/stack.
- The code field must be the COMPLETE file content.
- Output raw JSON only. Absolutely no markdown fences or explanation text.
"""


# ════════════════════════════════════════════════════════════════════════════
# Prompt helpers — mini scope
# ════════════════════════════════════════════════════════════════════════════

def build_system_prompt_mini_file() -> str:
    return """\
You are a senior software developer applying a TARGETED MINI PATCH to one file.

You will receive:
1. The complete mini plan.
2. Optional impact analysis.
3. The specific target-file instruction.
4. Existing content of the target file, if it exists.

Your task:
- Patch ONLY the requested target file.
- Do NOT modify, create, delete, rename, or mention unrelated files.
- Do NOT perform a broad rewrite unless the target-file instructions explicitly require it.
- Preserve existing public APIs, exports, imports, formatting style, and behavior unless the mini plan explicitly says to change them.
- Keep the change as small and safe as possible.
- If action is CREATE, return the complete new file content.
- If action is MODIFY, return the complete updated file content.
- If action is DELETE, return {"path": "...", "delete": true}.
- If action is RENAME, return the complete content for the destination path only if a destination is explicitly provided by the plan.
- Return raw JSON only. No markdown fences. No explanation text.

For MODIFY/CREATE/RENAME, use this exact schema:
{
  "path": "repo/relative/path.ext",
  "content": "<complete file content>"
}

For DELETE, use this exact schema:
{
  "path": "repo/relative/path.ext",
  "delete": true
}
"""


def _format_mini_target_instruction(target: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append(f"Path: {target.get('path')}")
    lines.append(f"Action: {target.get('action', 'MODIFY')}")
    lines.append(f"Risk: {target.get('risk', 'medium')}")

    reason = target.get("reason")
    if reason:
        lines.append(f"Reason: {reason}")

    instructions = target.get("instructions", [])
    if isinstance(instructions, str):
        instructions = [instructions]

    if instructions:
        lines.append("Instructions:")
        for item in instructions:
            lines.append(f"- {item}")

    # Optional future-compatible fields.
    for key in ("to_path", "destination", "rename_to"):
        if target.get(key):
            lines.append(f"{key}: {target[key]}")

    return "\n".join(lines)


def _build_mini_user_message(
    *,
    plan: dict[str, Any],
    analysis: dict[str, Any] | None,
    target: dict[str, Any],
    existing_content: str | None,
) -> str:
    path = str(target.get("path", ""))

    if existing_content is None:
        existing_block = "### Existing target file\n\n*(file does not exist yet)*"
    else:
        existing_block = (
            f"### Existing target file: {path}\n\n"
            f"{_code_fence(path, existing_content)}"
        )

    analysis_block = (
        f"\n\n### Mini impact analysis\n\n{json.dumps(analysis, indent=2, ensure_ascii=False)}"
        if analysis
        else ""
    )

    return (
        f"### Mini execution plan\n\n"
        f"{json.dumps(plan, indent=2, ensure_ascii=False)}"
        f"{analysis_block}\n\n"
        f"### Target-file instruction\n\n"
        f"{_format_mini_target_instruction(target)}\n\n"
        f"{existing_block}\n\n"
        f"Return only the JSON object for this one target file."
    )


# ════════════════════════════════════════════════════════════════════════════
# Core model call
# ════════════════════════════════════════════════════════════════════════════

def _call_executor(system: str, user_message: str) -> str:
    """
    Call the executor role (config in artifacts/models.py).

    Retries once on transient failures. Returns raw text content.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    model    = get_model(ROLE)
    provider = get_provider(ROLE)

    last_error: Exception | None = None

    for attempt in range(2):
        try:
            resp = call_model(
                ROLE,
                messages=messages,
                temperature=0.15,
                max_tokens=32768,
            )

            usage = getattr(resp, "usage", None)
            if usage:
                pt        = getattr(usage, "prompt_tokens",     0) or 0
                ct        = getattr(usage, "completion_tokens", 0) or 0
                call_cost = record_usage(usage, model=model, provider=provider)
                print_call(__file__, pt, ct, call_cost)

            choice     = resp.choices[0]
            message    = choice.message
            content    = message.content
            tool_calls = getattr(message, "tool_calls", None)
            finish_reason = getattr(choice, "finish_reason", None)

            if tool_calls:
                raise RuntimeError(f"Model returned tool_calls instead of text: {tool_calls}")

            if not content or not content.strip():
                raise RuntimeError(
                    f"Model returned empty content. "
                    f"finish_reason={finish_reason}, message={message}"
                )

            return content.strip()

        except Exception as exc:
            last_error = exc
            print(f"[08] [{model}] {exc}", file=sys.stderr)

            if attempt == 0:
                print("[08] Retrying in 3s …", file=sys.stderr)
                time.sleep(3)

    raise RuntimeError(f"Executor call failed after retries: {last_error}")


# ════════════════════════════════════════════════════════════════════════════
# JSON extraction
# ════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str, label: str) -> dict[str, Any]:
    """
    Parse JSON from model response.

    Handles accidental markdown fences and accidental surrounding prose.
    Raises RuntimeError on failure.
    """
    raw = raw.strip()

    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Parsed JSON for {label} is not an object.")
        return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Parsed JSON for {label} is not an object.")
            return parsed
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"JSON parse failed for {label}: {exc}\n"
                f"Raw, first 500 chars:\n{raw[:500]}"
            ) from exc

    raise RuntimeError(f"No JSON object found in response for {label}.")


# ════════════════════════════════════════════════════════════════════════════
# Plan / task formatting — full scope
# ════════════════════════════════════════════════════════════════════════════

def _build_task_block(task: dict[str, Any] | None) -> str:
    """Format planner task as a prompt section."""
    if not task:
        return ""

    lines: list[str] = ["### Implementation plan from architect\n"]

    behavior_summary = task.get("behavior_summary")
    if behavior_summary:
        lines.append(f"**File purpose:** {behavior_summary}\n")

    role = task.get("role")
    if role:
        lines.append(f"**Role:** {role}\n")

    deps = task.get("depends_on", [])
    if deps:
        lines.append(f"**Depends on:** {', '.join(str(dep) for dep in deps)}\n")

    sub_tasks = task.get("sub_tasks", [])
    if sub_tasks:
        lines.append("**Sub-tasks, implement in this order:**")
        for sub_task in sub_tasks:
            lines.append(f"  {sub_task}")
        lines.append("")

    gotchas = task.get("gotchas", [])
    if gotchas:
        lines.append("**Gotchas / edge cases:**")
        for gotcha in gotchas:
            lines.append(f"  - {gotcha}")
        lines.append("")

    notes = task.get("notes", [])
    if notes:
        lines.append("**Cross-cutting notes:**")
        for note in notes:
            lines.append(f"  - {note}")
        lines.append("")

    tailwind_hints = task.get("tailwind_hints")
    if tailwind_hints:
        lines.append(f"**Styling hints:** {tailwind_hints}\n")

    styling_hints = task.get("styling_hints")
    if styling_hints and styling_hints != tailwind_hints:
        lines.append(f"**Additional styling hints:** {styling_hints}\n")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Context selection — full scope
# ════════════════════════════════════════════════════════════════════════════

def _compact_reference_code(code: str, max_chars: int = 1500) -> str:
    """
    Compact code when used as broad API reference.
    """
    compact_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        compact_lines.append(line)

    compact = "\n".join(compact_lines)
    return compact[:max_chars]


def _is_shared_reference_path(file_path: str) -> bool:
    markers = [
        "types/",
        "type/",
        "models/",
        "model/",
        "schemas/",
        "schema/",
        "constants/",
        "constant/",
        "data/",
        "config/",
        "settings/",
        "utils/",
        "lib/",
    ]
    normalized = file_path.replace("\\", "/")
    return any(marker in normalized for marker in markers)


def _is_entrypoint_like(file_path: str) -> bool:
    name = Path(file_path).name.lower()
    return name in {
        "app.tsx",
        "app.jsx",
        "app.vue",
        "app.svelte",
        "main.ts",
        "main.tsx",
        "main.js",
        "main.jsx",
        "index.ts",
        "index.tsx",
        "index.js",
        "index.jsx",
        "server.py",
        "main.py",
        "app.py",
        "main.go",
        "server.go",
    }


def _build_context_block(
    file_path: str,
    task: dict[str, Any] | None,
    already_written: dict[str, str],
) -> str:
    """
    Build dependency/context section for the current file.
    """
    if not already_written:
        return ""

    deps = set(task.get("depends_on", [])) if task else set()
    relevant: dict[str, str] = {}
    label = "Dependencies already implemented — for import/API reference"

    if deps:
        relevant = {
            fp: code
            for fp, code in already_written.items()
            if fp in deps
        }

    if _is_entrypoint_like(file_path):
        label = "API reference, compacted — full implementations omitted"
        relevant = {
            fp: _compact_reference_code(code)
            for fp, code in already_written.items()
            if _is_shared_reference_path(fp)
            or "/hooks/" in fp.replace("\\", "/")
            or "/services/" in fp.replace("\\", "/")
            or "/routes/" in fp.replace("\\", "/")
        }

    if not relevant:
        relevant = {
            fp: code
            for fp, code in already_written.items()
            if _is_shared_reference_path(fp)
        }
        label = "Shared references already implemented"

    if not relevant:
        return ""

    context_block = f"### {label}\n"

    for fp, code in relevant.items():
        context_block += f"\n#### {fp}\n{_code_fence(fp, code)}\n"

    return context_block


# ════════════════════════════════════════════════════════════════════════════
# Per-file generation — full scope
# ════════════════════════════════════════════════════════════════════════════

```python
def implement_file(
    stub: dict[str, Any],
    task: dict[str, Any] | None,
    already_written: dict[str, str],
    stack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    file_path = stub["file_path"]
    task_block = _build_task_block(task)
    context_block = _build_context_block(file_path, task, already_written)

    stub_code = stub.get("code", "")
    user_msg = (
        f"{task_block}\n"
        f"{context_block}\n"
        f"### Stub file to implement: {file_path}\n"
        f"{_code_fence(file_path, stub_code)}"
    )

    approx_tokens = len(user_msg) // 4
    if approx_tokens > 28000:
        print(
            f"[08] ⚠ Large prompt for {file_path}: ~{approx_tokens:,} tokens "
            f"(limit ~32k). Response may be truncated.",
            file=sys.stderr,
        )

    print(f"[08]   → Implementing {file_path} …")
    raw = _call_executor(build_system_prompt_per_file(stack=stack), user_msg)
    result = _parse_json(raw, file_path)

    # Be tolerant if model accidentally returns the single-call shape.
    if "files" in result and isinstance(result["files"], list):
        for entry in result["files"]:
            if isinstance(entry, dict) and entry.get("file_path") == file_path:
                return entry
        if result["files"] and isinstance(result["files"][0], dict):
            return result["files"][0]

    return result


# ════════════════════════════════════════════════════════════════════════════
# Ordering helpers — full scope
# ════════════════════════════════════════════════════════════════════════════

def order_stubs(
    stub_files: list[dict[str, Any]],
    plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Sort stub files by implementation_order from plan, or keep scaffold order."""
    if not plan:
        return stub_files

    order = plan.get("implementation_order", [])
    order_map = {fp: i for i, fp in enumerate(order)}

    return sorted(
        stub_files,
        key=lambda file_entry: order_map.get(file_entry["file_path"], 999_999),
    )


# ════════════════════════════════════════════════════════════════════════════
# Single-call fallback — full scope
# ════════════════════════════════════════════════════════════════════════════

def implement_all_single_call(
    spec: str,
    stub_files: list[dict[str, Any]],
    instructions: str,
    stack: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Single-call mode — all requested files in one request.
    """
    user_msg = (
        f"### canonical spec\n\n{spec}\n\n"
        f"### scaffold, stub files to implement\n\n"
        f"{json.dumps(stub_files, indent=2, ensure_ascii=False)}"
    )

    model    = get_model(ROLE)
    provider = get_provider(ROLE)
    print(f"[08] Calling model: {model} (provider: {provider}), single-call mode …")

    raw = _call_executor(
        build_system_prompt_single(instructions=instructions, stack=stack),
        user_msg,
    )
    result = _parse_json(raw, "single-call")

    files = result.get("files", [])
    if not isinstance(files, list):
        raise RuntimeError("single-call result missing list field: files")

    return [entry for entry in files if isinstance(entry, dict)]


# ════════════════════════════════════════════════════════════════════════════
# Delta mode restored context — full scope
# ════════════════════════════════════════════════════════════════════════════

def _load_restored_files(only_set: set[str]) -> dict[str, str]:
    """
    Read already-restored src/ files into memory so they can be used as
    import/reference context for the executor in delta mode.

    These are build outputs, not pipeline artifacts, so they are not included
    in the artifact access summary.
    """
    restored: dict[str, str] = {}

    if not SRC_DIR.exists():
        return restored

    all_paths = sorted(
        path
        for path in SRC_DIR.rglob("*")
        if path.is_file() and _is_probably_source_file(path)
    )

    for path in all_paths:
        rel = "src/" + str(path.relative_to(SRC_DIR)).replace("\\", "/")

        if rel in only_set:
            continue

        try:
            restored[rel] = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[08] WARNING: could not read restored file {rel}: {exc}")

    return restored


# ════════════════════════════════════════════════════════════════════════════
# Output writing — full scope
# ════════════════════════════════════════════════════════════════════════════

def _write_generated_entry(entry: dict[str, Any]) -> str | None:
    """
    Write one generated full-scope entry to SRC_DIR.

    Returns written file_path, or None if skipped.
    """
    fp = entry.get("file_path")
    code = entry.get("code")

    if not fp or not isinstance(fp, str):
        print(f"[08] SKIP malformed entry without file_path: {entry}", file=sys.stderr)
        return None

    if code is None or not isinstance(code, str):
        print(f"[08] SKIP malformed entry without code: {fp}", file=sys.stderr)
        return None

    out_path = _safe_src_output_path(fp)
    if out_path is None:
        print(f"[08] SKIP outside src/: {fp}", file=sys.stderr)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")

    # src/** is a build output owned by executor; include it in write summary.
    track_write(out_path)

    print(f"[08] WROTE {out_path}")
    return fp


# ════════════════════════════════════════════════════════════════════════════
# Mini scope
# ════════════════════════════════════════════════════════════════════════════

def _load_mini_plan() -> dict[str, Any]:
    """
    Load mini_plan.json and extract the "plan" field.

    mini_plan.json schema (new): { "plan": {...}, "impact": {...} }
    Backward compat: if no "plan" key, treat entire object as the plan.
    """
    if not PLANNER_MINI_PLAN.exists():
        raise FileNotFoundError(
            f"Missing mini plan: {PLANNER_MINI_PLAN}\n"
            "Run planner first:\n"
            "  python harness.py --project <name> --scope mini --plan"
        )

    merged = _read_json_file(PLANNER_MINI_PLAN, "planner/mini_plan.json")

    # New merged format: { "plan": {...}, "impact": {...} }
    if "plan" in merged and isinstance(merged["plan"], dict):
        plan = merged["plan"]
    else:
        # Backward compat: old format was the plan object itself
        print("[08] WARNING: mini_plan.json has no 'plan' key — assuming legacy flat format.")
        plan = merged

    if plan.get("scope") != "mini":
        print("[08] WARNING: mini_plan plan has no scope='mini'. Continuing.")

    targets = plan.get("target_files", [])
    if not isinstance(targets, list):
        raise RuntimeError("mini_plan target_files must be a list.")

    return plan


def _load_mini_analysis() -> dict[str, Any] | None:
    """
    Load impact analysis from the merged mini_plan.json "impact" field.

    New format: mini_plan.json = { "plan": {...}, "impact": {...} }
    Falls back gracefully if file missing or impact field absent.
    """
    if not PLANNER_MINI_PLAN.exists():
        return None
    try:
        merged = json.loads(PLANNER_MINI_PLAN.read_text(encoding="utf-8"))
        if not isinstance(merged, dict):
            return None
        impact = merged.get("impact")
        return impact if isinstance(impact, dict) else None
    except Exception as exc:
        print(
            f"[08] WARNING: could not read impact from mini_plan.json: {exc}",
            file=sys.stderr,
        )
        return None


def _mini_allowed_target_paths(plan: dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    for entry in plan.get("target_files", []):
        if isinstance(entry, dict):
            path = _normalize_repo_rel_path(str(entry.get("path", "")))
            if path:
                allowed.add(path)
            for key in ("to_path", "destination", "rename_to"):
                if entry.get(key):
                    allowed.add(_normalize_repo_rel_path(str(entry[key])))
    return allowed


def _normalize_mini_target(entry: dict[str, Any]) -> dict[str, Any] | None:
    path = _normalize_repo_rel_path(str(entry.get("path", "")))
    if not path:
        print(f"[08] WARNING: skipping mini target without path: {entry}", file=sys.stderr)
        return None

    if _is_disallowed_artifact_rel_path(path):
        print(f"[08] WARNING: skipping disallowed mini target: {path}", file=sys.stderr)
        return None

    action = str(entry.get("action", "MODIFY")).strip().upper()
    if action not in {"MODIFY", "CREATE", "DELETE", "RENAME"}:
        action = "MODIFY"

    normalized = dict(entry)
    normalized["path"] = path
    normalized["action"] = action
    normalized["risk"] = str(entry.get("risk", "medium")).strip().lower() or "medium"

    return normalized


def _read_existing_target(path: str) -> str | None:
    """
    Reads target source/build-output file, not a pipeline artifact. It is not
    tracked in artifact read summary.
    """
    out_path = _safe_artifact_output_path(path)
    if out_path is None:
        return None
    if not out_path.exists():
        return None
    try:
        return out_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[08] WARNING: target file is not UTF-8 text: {path}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"[08] WARNING: could not read target {path}: {exc}", file=sys.stderr)
        return None


def _implement_mini_target(
    *,
    plan: dict[str, Any],
    analysis: dict[str, Any] | None,
    target: dict[str, Any],
) -> dict[str, Any]:
    path = str(target["path"])
    action = str(target.get("action", "MODIFY")).upper()

    existing = _read_existing_target(path)

    if action == "MODIFY" and existing is None:
        print(
            f"[08] WARNING: MODIFY target does not exist yet; treating as CREATE: {path}",
            file=sys.stderr,
        )

    user_msg = _build_mini_user_message(
        plan=plan,
        analysis=analysis,
        target=target,
        existing_content=existing,
    )

    approx_tokens = len(user_msg) // 4
    if approx_tokens > 28000:
        print(
            f"[08] ⚠ Large mini prompt for {path}: ~{approx_tokens:,} tokens "
            f"(limit ~32k). Response may be truncated.",
            file=sys.stderr,
        )

    print(f"[08]   → Mini patch {action:<6} {path} …")
    raw = _call_executor(build_system_prompt_mini_file(), user_msg)
    result = _parse_json(raw, f"mini target {path}")

    # Tolerate full-scope key names.
    if "file_path" in result and "path" not in result:
        result["path"] = result["file_path"]
    if "code" in result and "content" not in result:
        result["content"] = result["code"]

    return result


def _write_mini_result(
    result: dict[str, Any],
    *,
    target: dict[str, Any],
    allowed_paths: set[str],
) -> str | None:
    """
    Write one mini result.

    Returns written/deleted/renamed path marker, or None if skipped.
    """
    requested_path = _normalize_repo_rel_path(str(target.get("path", "")))
    result_path = _normalize_repo_rel_path(str(result.get("path", requested_path)))

    if result_path not in allowed_paths:
        print(
            f"[08] SKIP mini result outside plan target set: {result_path}",
            file=sys.stderr,
        )
        return None

    out_path = _safe_artifact_output_path(result_path)
    if out_path is None:
        print(f"[08] SKIP unsafe mini output path: {result_path}", file=sys.stderr)
        return None

    action = str(target.get("action", "MODIFY")).upper()

    if action == "DELETE" or result.get("delete") is True:
        if out_path.exists():
            out_path.unlink()
            print(f"[08] DELETED {result_path}")
        else:
            print(f"[08] DELETE no-op, file not found: {result_path}")

        track_write(out_path)
        return result_path

    content = result.get("content")
    if content is None or not isinstance(content, str):
        print(f"[08] SKIP mini result without content: {result_path}", file=sys.stderr)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    track_write(out_path)

    print(f"[08] WROTE {result_path}")
    return result_path


def run_mini_scope(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Execute targeted mini implementation from planner/mini_plan.json.

    Returns (written, failed_files, record_extra).
    """
    print("[08] Scope: mini")

    plan = _load_mini_plan()
    analysis = _load_mini_analysis()

    raw_targets = plan.get("target_files", [])
    targets: list[dict[str, Any]] = []

    for raw in raw_targets:
        if not isinstance(raw, dict):
            print(f"[08] WARNING: skipping invalid target entry: {raw!r}", file=sys.stderr)
            continue
        normalized = _normalize_mini_target(raw)
        if normalized:
            targets.append(normalized)

    if args.only_files.strip():
        only_set = {
            _normalize_repo_rel_path(path)
            for path in args.only_files.split(",")
            if path.strip()
        }
        before = len(targets)
        targets = [target for target in targets if target["path"] in only_set]
        print(f"[08] Mini --only-files: {len(targets)}/{before} target(s) selected.")

    print(f"[08] Mini targets: {len(targets)}")
    for target in targets:
        print(
            f"[08]   {target.get('action', 'MODIFY'):<6} "
            f"{target.get('path')}  risk={target.get('risk', 'medium')}"
        )

    allowed_paths = _mini_allowed_target_paths(plan)

    written: list[str] = []
    failed_files: list[str] = []

    for target in targets:
        fp = str(target["path"])

        try:
            result = _implement_mini_target(
                plan=plan,
                analysis=analysis,
                target=target,
            )
            written_fp = _write_mini_result(
                result,
                target=target,
                allowed_paths=allowed_paths,
            )
            if written_fp:
                written.append(written_fp)
            else:
                failed_files.append(fp)

        except Exception as exc:
            print(f"[08] FAILED mini target {fp}: {exc}", file=sys.stderr)
            failed_files.append(fp)

    record_extra = {
        "scope": "mini",
        "plan": "planner/mini_plan.json",
        "impact_analysis": "planner/mini_plan.json (field: impact)",
        "task_summary": plan.get("task_summary", ""),
        "target_files": [target.get("path") for target in targets],
    }

    return written, failed_files, record_extra


# ════════════════════════════════════════════════════════════════════════════
# Full scope
# ════════════════════════════════════════════════════════════════════════════

def run_full_scope(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, Any]]:
    print("[08] Scope: full")

    scaffold = _read_json_file(SCAFFOLD_JSON, "scaffolder/blueprint.json")

    instrs = scaffold.get("implementation_instructions", {}).get("for_executor", "")
    scaffold_stack = scaffold.get("stack")

    # ── Extract non-test stubs from new module-centric schema ─────────────────
    modules = scaffold.get("modules")
    if isinstance(modules, list):
        # New schema: modules[].files[].{path, kind}
        all_stubs: list[dict[str, Any]] = []
        for mod in modules:
            if not isinstance(mod, dict):
                continue
            for file_entry in mod.get("files", []):
                if not isinstance(file_entry, dict):
                    continue
                if file_entry.get("kind", "source") != "test":
                    all_stubs.append({
                        "file_path": file_entry.get("path", ""),
                        "kind": file_entry.get("kind", "source"),
                        "module": mod.get("module", ""),
                    })
    else:
        # Legacy flat schema fallback: files[].{file_path, is_test}
        files = scaffold.get("files", [])
        if not isinstance(files, list):
            raise RuntimeError(
                "scaffolder/blueprint.json must contain 'modules' list (new schema) "
                "or 'files' list (legacy schema)."
            )
        print("[08] WARNING: scaffold uses legacy flat-file schema — consider regenerating.")
        all_stubs = [
            file_entry
            for file_entry in files
            if isinstance(file_entry, dict) and not file_entry.get("is_test")
        ]

    # ── Delta filtering ───────────────────────────────────────────────────────
    only_set: set[str] = set()

    if args.only_files.strip():
        only_set = {
            fp.strip()
            for fp in args.only_files.split(",")
            if fp.strip()
        }

        stub_files = [
            file_entry
            for file_entry in all_stubs
            if file_entry.get("file_path") in only_set
        ]

        skipped = [
            file_entry["file_path"]
            for file_entry in all_stubs
            if file_entry.get("file_path") not in only_set
        ]

        print(
            f"[08] Delta mode — {len(stub_files)} file(s) to implement, "
            f"{len(skipped)} unaffected skipped."
        )

        for fp in skipped:
            print(f"[08]   SKIP unaffected: {fp}")

        requested_missing = sorted(
            only_set - {file_entry["file_path"] for file_entry in all_stubs}
        )
        for fp in requested_missing:
            print(f"[08] WARNING: --only-files path not found in scaffold: {fp}")

    else:
        stub_files = all_stubs

    # ── Load planner plan if requested ────────────────────────────────────────
    plan: dict[str, Any] | None = None
    task_index: dict[str, dict[str, Any]] = {}
    stack: dict[str, Any] | None = scaffold_stack if isinstance(scaffold_stack, dict) else None

    if args.use_planner_plan:
        if not PLANNER_FULL_PLAN.exists():
            raise FileNotFoundError(
                f"ERROR: --use-planner-plan set but full planner plan not found: {PLANNER_FULL_PLAN}\n"
                "Run 07_planner.py --scope full first."
            )

        plan = _read_json_file(PLANNER_FULL_PLAN, "planner/full_plan.json")
        task_index = {
            task["file_path"]: task
            for task in plan.get("tasks", [])
            if isinstance(task, dict) and "file_path" in task
        }

        plan_stack = plan.get("stack")
        if isinstance(plan_stack, dict):
            stack = plan_stack

        print(
            f"[08] Planner full plan loaded — {len(task_index)} tasks, "
            f"order: {plan.get('implementation_order', [])}"
        )

        if stack:
            print(f"[08] Stack in use:\n{json.dumps(stack, indent=2, ensure_ascii=False)}")
        else:
            print("[08] WARNING: planner plan has no 'stack'; executor will use generic prompt.")

    else:
        print("[08] No planner plan — falling back to single-call mode (loading spec).")

        if stack:
            print(
                f"[08] Stack from scaffold in use:\n"
                f"{json.dumps(stack, indent=2, ensure_ascii=False)}"
            )

    # ── Execute ───────────────────────────────────────────────────────────────
    written: list[str] = []
    failed_files: list[str] = []

    if plan:
        ordered = order_stubs(stub_files, plan)

        already_written: dict[str, str] = (
            _load_restored_files(only_set)
            if only_set
            else {}
        )

        if already_written:
            print(
                f"[08] Import context seeded with "
                f"{len(already_written)} restored file(s)."
            )

        for stub in ordered:
            fp = stub["file_path"]
            task = task_index.get(fp)

            try:
                entry = implement_file(
                    stub=stub,
                    task=task,
                    already_written=already_written,
                    stack=stack,
                )
            except Exception as exc:
                print(f"[08] FAILED to implement {fp}: {exc}", file=sys.stderr)
                failed_files.append(fp)
                continue

            written_fp = _write_generated_entry(entry)

            if written_fp:
                already_written[written_fp] = entry["code"]
                written.append(written_fp)
            else:
                failed_files.append(fp)

    else:
        spec = _load_spec()
        try:
            entries = implement_all_single_call(
                spec=spec,
                stub_files=stub_files,
                instructions=instrs,
                stack=stack,
            )
        except Exception as exc:
            print(f"[08] FAILED single-call generation: {exc}", file=sys.stderr)
            failed_files.extend([file_entry["file_path"] for file_entry in stub_files])
            entries = []

        for entry in entries:
            written_fp = _write_generated_entry(entry)
            if written_fp:
                written.append(written_fp)

    skipped_delta = (
        sorted({file_entry["file_path"] for file_entry in all_stubs} - only_set)
        if only_set
        else []
    )

    mode = "per-file-with-planner-plan" if plan else "single-call"
    if only_set:
        mode += "-delta"

    record_extra = {
        "scope": "full",
        "mode": mode,
        "skipped_delta": skipped_delta,
        "stack": stack,
        "plan": "planner/full_plan.json" if plan else None,
    }

    return written, failed_files, record_extra


# ════════════════════════════════════════════════════════════════════════════
# Run record
# ════════════════════════════════════════════════════════════════════════════

def append_manifest_log(
    *,
    scope: str,
    mode: str,
    written: list[str],
    failed_files: list[str],
    extra: dict[str, Any],
) -> None:
    """
    Append one entry to executor/manifest_log.json (long-term audit trail).

    Entry schema:
      { scope, mode, generated_at, files_written, failed_count, cost, task_summary? }
    """
    # Extract cost from the current run's token summary
    token_sum = cost_summary()
    cost_total = token_sum.get("total_cost_usd") if isinstance(token_sum, dict) else None

    entry: dict[str, Any] = {
        "scope": scope,
        "mode": mode,
        "generated_at": _utc_now_iso(),
        "files_written": len(written),
        "failed_count": len(failed_files),
        "cost": cost_total,
    }
    if extra.get("task_summary"):
        entry["task_summary"] = extra["task_summary"]
    if extra.get("stack"):
        entry["stack"] = extra["stack"]

    log_path = EXECUTOR_MANIFEST_LOG
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
        print(f"[08] Manifest log appended ({len(entries)} entries) → {log_path}")
    except Exception as exc:
        print(f"[08] WARNING: could not append manifest log: {exc}", file=sys.stderr)


def _write_impl_record(
    *,
    scope: str,
    mode: str,
    written: list[str],
    failed_files: list[str],
    extra: dict[str, Any],
) -> None:
    token_sum = cost_summary()
    cost_total = token_sum.get("total_cost_usd") if isinstance(token_sum, dict) else None

    record = {
        "executor_role": ROLE,
        "executor_model": get_model(ROLE),
        "scope": scope,
        "mode": mode,
        "generated_at": _utc_now_iso(),
        "files": written,
        "failed_files": failed_files,
        "token_summary": token_sum,
        "cost": cost_total,
    }
    record.update(extra)

    EXECUTOR_OVERWRITE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EXECUTOR_OVERWRITE_MANIFEST.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(EXECUTOR_OVERWRITE_MANIFEST)
    print(f"[08] Executor manifest → {EXECUTOR_OVERWRITE_MANIFEST}")

    append_manifest_log(
        scope=scope,
        mode=mode,
        written=written,
        failed_files=failed_files,
        extra=extra,
    )


# ════════════════════════════════════════════════════════════════════════════
# CLI / Main
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="08_executor.py",
        description="Executor. Implements full scaffold plan or mini targeted plan.",
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
        help="Executor scope. full uses scaffolder/blueprint.json; mini uses planner/mini_plan.json.",
    )

    parser.add_argument(
        "--use-planner-plan",
        action="store_true",
        help=(
            "Full scope: inject planner/full_plan.json as per-file "
            "implementation guidance. Mini scope: accepted for harness compatibility; "
            "mini always uses planner/mini_plan.json."
        ),
    )

    parser.add_argument(
        "--only-files",
        default="",
        help=(
            "Comma-separated paths to implement. "
            "Full scope: delta mode against scaffold. "
            "Mini scope: filters mini_plan target_files."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: do not call ensure_dirs() at import-time.
    # PIPELINE_PROJECT must be available before artifact paths are resolved.
    ensure_dirs()

    exit_code = 0

    try:
        if args.scope == "mini":
            written, failed_files, extra = run_mini_scope(args)
            mode = "mini-targeted"
            if args.only_files.strip():
                mode += "-filtered"
            extra["mode"] = mode
            _write_impl_record(
                scope="mini",
                mode=mode,
                written=written,
                failed_files=failed_files,
                extra=extra,
            )

        else:
            written, failed_files, extra = run_full_scope(args)
            mode = str(extra.get("mode", "single-call"))
            _write_impl_record(
                scope="full",
                mode=mode,
                written=written,
                failed_files=failed_files,
                extra=extra,
            )

    except Exception as exc:
        print(f"[08] ERROR: {exc}", file=sys.stderr)

        # Best-effort failure record.
        try:
            _write_impl_record(
                scope=args.scope,
                mode="failed-before-execution",
                written=[],
                failed_files=[],
                extra={"error": str(exc)},
            )
        except Exception:
            pass

        exit_code = 1

    else:
        if failed_files:
            print(
                f"[08] Done with {len(written)} file(s) written, "
                f"{len(failed_files)} failed: {failed_files}",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            print(f"[08] Done — {len(written)} file(s) written.")

    finally:
        print_summary("[08]")
        print_artifact_summary("[08]")
        prompt_next_step(ROLE, prefix="[08]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
