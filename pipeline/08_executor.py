"""
pipeline/08_executor.py
=======================
Step 8 — EXECUTOR.

FULL SCOPE
──────────
Reads planner/full_plan.json exclusively.
full_plan.json must contain: tasks[], implementation_order[], stack{}.

Delta mode:
    --only-files src/foo.py,src/bar.ts

Reads:
    artifacts_<slug>/planner/full_plan.json
    artifacts_<slug>/output/src/**   (delta mode only — context seeding, not tracked)

Writes:
    artifacts_<slug>/output/src/**
    artifacts_<slug>/executor/manifest.json        (short-term, overwrite)
    artifacts_<slug>/executor/manifest_log.json    (long-term, append)


MINI SCOPE
──────────
Reads planner/mini_plan.json exclusively.

Reads:
    artifacts_<slug>/planner/mini_plan.json
    target files listed in mini_plan.json["plan"]["target_files"]

Writes:
    only files listed in mini_plan.json target_files
    artifacts_<slug>/executor/manifest.json        (short-term, overwrite)
    artifacts_<slug>/executor/manifest_log.json    (long-term, append)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
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
#   artifacts_<slug>/planner/full_plan.json
#   artifacts_<slug>/output/src/**                  (delta mode only — context seeding, not tracked)
#
# READS mini:
#   artifacts_<slug>/planner/mini_plan.json


sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.models import get_model, get_provider          # noqa: E402
from modules.artifact_tracking import (                        # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary, summary as cost_summary  # noqa: E402
from modules.call_llm import call_llm                          # noqa: E402
from modules.post_interactive import prompt_next_step          # noqa: E402
from artifacts.paths import (                                  # noqa: E402
    EXECUTOR_MANIFEST_LOG,
    EXECUTOR_OVERWRITE_MANIFEST,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    SRC_DIR,
    artifact_root,
    ensure_dirs,
)


ROLE = "executor"


# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

SOURCE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".py", ".go", ".java", ".kt", ".kts", ".rs", ".rb", ".php",
    ".cs", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".vue", ".svelte", ".html", ".css", ".scss", ".sass", ".less",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".sql",
    ".sh", ".bash", ".zsh", ".fish", ".md",
}

FENCE_LANGUAGE_BY_EXT = {
    ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx",
    ".mjs": "javascript", ".cjs": "javascript",
    ".py": "python", ".go": "go", ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".c": "c", ".h": "c", ".hpp": "cpp",
    ".vue": "vue", ".svelte": "svelte", ".html": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".sql": "sql",
    ".sh": "bash", ".bash": "bash", ".zsh": "zsh", ".fish": "fish",
    ".md": "markdown",
}


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fence_language(file_path: str) -> str:
    return FENCE_LANGUAGE_BY_EXT.get(Path(file_path).suffix.lower(), "")


def _code_fence(file_path: str, code: str) -> str:
    lang = _fence_language(file_path)
    return f"```{lang}\n{code}\n```" if lang else f"```\n{code}\n```"


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


def _normalize_repo_rel_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("/")


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
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
    src/foo/bar.ts  → SRC_DIR/foo/bar.ts
    foo/bar.ts      → SRC_DIR/foo/bar.ts
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


def _is_disallowed_artifact_rel_path(rel: str) -> bool:
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
        "executor/", "planner/", "debugger/", "reporter/", "judge/",
        "patcher/", "archivist/", "absorber/", "clarificator/", "enricher/",
        "spectracker/", "scaffolder/", "spec/",
        "state/", "cache/", "execution/", "run/", "knowledge/", "reports/",
    )
    return normalized.startswith(blocked_prefixes)


def _safe_artifact_output_path(rel_path: str) -> Path | None:
    rel = _normalize_repo_rel_path(rel_path)
    if _is_disallowed_artifact_rel_path(rel):
        return None

    root = artifact_root()
    out_path = root / rel

    try:
        root_resolved = root.resolve()
        out_resolved  = out_path.resolve()
    except FileNotFoundError:
        root_resolved = root.absolute()
        out_resolved  = out_path.absolute()

    if out_resolved != root_resolved and root_resolved not in out_resolved.parents:
        return None
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# Prompts — full scope
# ════════════════════════════════════════════════════════════════════════════

def _format_stack_section(stack: dict[str, Any] | None) -> str:
    if not stack:
        return ""
    return f"""
## Project stack — follow exactly

{json.dumps(stack, indent=2, ensure_ascii=False)}

Rules:
- Apply the idioms, imports, runtime APIs, and framework conventions of this stack.
- Do NOT assume any stack element not listed above.
- Preserve the language, module system, file layout, public APIs, and exported names implied by the plan.
"""


def build_system_prompt_per_file(stack: dict[str, Any] | None = None) -> str:
    stack_section = _format_stack_section(stack)
    return f"""\
You are a senior software developer implementing ONE source file.
{stack_section}
You will receive:
1. A behavior summary and task plan from a senior architect — follow it carefully
2. Optional dependency/context files already implemented — for import/API reference
3. The file path you must implement

Your task:
- Implement this ONE file only.
- Return a JSON object with this EXACT schema:
  {{
    "file_path": "src/path/to/file.ext",
    "code": "<complete file content>"
  }}
- Follow the project stack strictly.
- Keep compatibility with dependency/context files shown in the prompt.
- Do not introduce libraries, frameworks, or tools not in the stack.
- The code field must be the COMPLETE file content.
- Output raw JSON only. No markdown fences. No explanation text.

FILE PLACEMENT RULES (critical — wrong placement breaks the build):
- Source files (components, hooks, stores, lib, types) → file_path MUST start with "src/"
  Example: "src/components/Button.tsx", "src/hooks/useStore.ts"
- Root-level config files → file_path MUST NOT have "src/" prefix:
  package.json, index.html, vite.config.ts, tsconfig.json, tsconfig.app.json,
  tsconfig.node.json, components.json, .env.example, vercel.json
  Example: "tsconfig.app.json" (NOT "src/tsconfig.app.json")

TSCONFIG REQUIREMENTS (when implementing tsconfig.app.json or tsconfig.json):
- compilerOptions.paths must include: {{"@/*": ["./src/*"]}}
- compilerOptions.baseUrl must be: "."
- exclude must include: "node_modules"
- include must be: ["src/**/*"]
"""


# ════════════════════════════════════════════════════════════════════════════
# Prompts — mini scope
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
- Do NOT perform a broad rewrite unless explicitly required.
- Preserve existing public APIs, exports, imports, and behavior unless the plan says otherwise.
- Keep the change as small and safe as possible.
- Return raw JSON only. No markdown fences. No explanation text.

For MODIFY/CREATE/RENAME:
{
  "path": "repo/relative/path.ext",
  "content": "<complete file content>"
}

For DELETE:
{
  "path": "repo/relative/path.ext",
  "delete": true
}
"""


def _format_mini_target_instruction(target: dict[str, Any]) -> str:
    lines: list[str] = [
        f"Path: {target.get('path')}",
        f"Action: {target.get('action', 'MODIFY')}",
        f"Risk: {target.get('risk', 'medium')}",
    ]
    if target.get("reason"):
        lines.append(f"Reason: {target['reason']}")
    instructions = target.get("instructions", [])
    if isinstance(instructions, str):
        instructions = [instructions]
    if instructions:
        lines.append("Instructions:")
        for item in instructions:
            lines.append(f"- {item}")
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
    existing_block = (
        "### Existing target file\n\n*(file does not exist yet)*"
        if existing_content is None
        else f"### Existing target file: {path}\n\n{_code_fence(path, existing_content)}"
    )
    analysis_block = (
        f"\n\n### Mini impact analysis\n\n"
        f"{json.dumps(analysis, indent=2, ensure_ascii=False)}"
        if analysis else ""
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
# JSON extraction
# ════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str, label: str) -> dict[str, Any]:
    """
    Extract JSON from model output.
    Pass 1 — direct json.loads
    Pass 2 — find outermost {...} then json.loads
    Pass 3 — ast.literal_eval (Python dict with single-quoted keys/values)
    """
    import ast as _ast

    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())

    # Pass 1: direct parse
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Parsed JSON for {label} is not an object.")
        return parsed
    except json.JSONDecodeError:
        pass

    # Pass 2: find outermost {...}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Parsed JSON for {label} is not an object.")
            return parsed
        except json.JSONDecodeError:
            pass

    # Pass 3: Python dict literal (single-quoted keys — some models emit this)
    def _try_ast(s: str) -> dict[str, Any] | None:
        try:
            parsed = _ast.literal_eval(s.strip())
            if isinstance(parsed, dict):
                return json.loads(json.dumps(parsed))
        except Exception:
            pass
        return None

    for candidate in (raw, match.group() if match else None):
        if candidate is None:
            continue
        result = _try_ast(candidate)
        if result is not None:
            print(f"[08][warn] {label}: model returned Python dict syntax — parsed via ast.literal_eval.")
            return result

    raise RuntimeError(
        f"No JSON object found in response for {label}.\n"
        f"Raw, first 500 chars:\n{raw[:500]}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Task / context formatting — full scope
# ════════════════════════════════════════════════════════════════════════════

def _build_task_block(task: dict[str, Any] | None) -> str:
    if not task:
        return ""

    lines: list[str] = ["### Implementation plan from architect\n"]

    if task.get("behavior_summary"):
        lines.append(f"**File purpose:** {task['behavior_summary']}\n")
    if task.get("role"):
        lines.append(f"**Role:** {task['role']}\n")

    deps = task.get("depends_on", [])
    if deps:
        lines.append(f"**Depends on:** {', '.join(str(d) for d in deps)}\n")

    sub_tasks = task.get("sub_tasks", [])
    if sub_tasks:
        lines.append("**Sub-tasks, implement in this order:**")
        for st in sub_tasks:
            lines.append(f"  {st}")
        lines.append("")

    gotchas = task.get("gotchas", [])
    if gotchas:
        lines.append("**Gotchas / edge cases:**")
        for g in gotchas:
            lines.append(f"  - {g}")
        lines.append("")

    notes = task.get("notes", [])
    if notes:
        lines.append("**Cross-cutting notes:**")
        for n in notes:
            lines.append(f"  - {n}")
        lines.append("")

    tailwind_hints = task.get("tailwind_hints")
    if tailwind_hints:
        lines.append(f"**Styling hints:** {tailwind_hints}\n")

    styling_hints = task.get("styling_hints")
    if styling_hints and styling_hints != tailwind_hints:
        lines.append(f"**Additional styling hints:** {styling_hints}\n")

    return "\n".join(lines)


def _compact_reference_code(code: str, max_chars: int = 1500) -> str:
    lines = [
        line for line in code.splitlines()
        if line.strip() and not line.strip().startswith(("//", "#"))
    ]
    return "\n".join(lines)[:max_chars]


def _is_shared_reference_path(file_path: str) -> bool:
    markers = [
        "types/", "type/", "models/", "model/",
        "schemas/", "schema/", "constants/", "constant/",
        "data/", "config/", "settings/", "utils/", "lib/",
    ]
    normalized = file_path.replace("\\", "/")
    return any(m in normalized for m in markers)


def _is_entrypoint_like(file_path: str) -> bool:
    return Path(file_path).name.lower() in {
        "app.tsx", "app.jsx", "app.vue", "app.svelte",
        "main.ts", "main.tsx", "main.js", "main.jsx",
        "index.ts", "index.tsx", "index.js", "index.jsx",
        "server.py", "main.py", "app.py", "main.go", "server.go",
    }


def _build_context_block(
    file_path: str,
    task: dict[str, Any] | None,
    already_written: dict[str, str],
) -> str:
    if not already_written:
        return ""

    deps = set(task.get("depends_on", [])) if task else set()

    if deps:
        relevant = {fp: code for fp, code in already_written.items() if fp in deps}
        label = "Dependencies already implemented — for import/API reference"
    elif _is_entrypoint_like(file_path):
        relevant = {
            fp: _compact_reference_code(code)
            for fp, code in already_written.items()
            if _is_shared_reference_path(fp)
            or any(
                seg in fp.replace("\\", "/")
                for seg in ("/hooks/", "/services/", "/routes/")
            )
        }
        label = "API reference, compacted — full implementations omitted"
    else:
        relevant = {
            fp: code
            for fp, code in already_written.items()
            if _is_shared_reference_path(fp)
        }
        label = "Shared references already implemented"

    if not relevant:
        return ""

    block = f"### {label}\n"
    for fp, code in relevant.items():
        block += f"\n#### {fp}\n{_code_fence(fp, code)}\n"
    return block


# ════════════════════════════════════════════════════════════════════════════
# Per-file generation — full scope
# ════════════════════════════════════════════════════════════════════════════

def implement_file(
    file_path: str,
    task: dict[str, Any] | None,
    already_written: dict[str, str],
    stack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_block    = _build_task_block(task)
    context_block = _build_context_block(file_path, task, already_written)

    user_msg = (
        f"{task_block}\n"
        f"{context_block}\n"
        f"### File to implement: {file_path}\n"
    )

    approx_tokens = len(user_msg) // 4
    if approx_tokens > 28000:
        print(
            f"[08] ⚠ Large prompt for {file_path}: ~{approx_tokens:,} tokens "
            f"(limit ~32k). Response may be truncated.",
            file=sys.stderr,
        )

    print(f"[08]   → Implementing {file_path} …")
    raw, _ = call_llm(
        ROLE,
        build_system_prompt_per_file(stack=stack),
        user_msg,
        temperature=0.15,
        max_tokens=32768,
        retries=2,
        backoff=False,
        caller_file=__file__,
        label=f"[08] {get_model(ROLE)}",
    )
    result = _parse_json(raw, file_path)

    # Tolerate model accidentally returning multi-file shape
    if "files" in result and isinstance(result["files"], list):
        for entry in result["files"]:
            if isinstance(entry, dict) and entry.get("file_path") == file_path:
                return entry
        if result["files"] and isinstance(result["files"][0], dict):
            return result["files"][0]

    return result


# ════════════════════════════════════════════════════════════════════════════
# Delta mode restored context — full scope
# ════════════════════════════════════════════════════════════════════════════

def _load_restored_files(only_set: set[str]) -> dict[str, str]:
    """
    Read existing src/ files for import/reference context in delta mode.
    Build outputs — not tracked in artifact summary.
    """
    restored: dict[str, str] = {}
    if not SRC_DIR.exists():
        return restored

    for path in sorted(
        p for p in SRC_DIR.rglob("*")
        if p.is_file() and _is_probably_source_file(p)
    ):
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

# Files that belong at output/ root rather than output/src/
_ROOT_LEVEL_FILES = {
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "index.html", "vite.config.ts", "vite.config.js",
    "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json",
    "tsconfig.base.json", "jest.config.ts", "jest.config.js",
    "vitest.config.ts", "vitest.config.js",
    "eslint.config.js", "eslint.config.ts", ".eslintrc.json", ".eslintrc.js",
    "prettier.config.js", ".prettierrc", ".prettierrc.json",
    "postcss.config.js", "postcss.config.cjs",
    "tailwind.config.js", "tailwind.config.ts", "tailwind.config.cjs",
    "components.json", "drizzle.config.ts",
    ".env", ".env.example", ".env.local",
    "README.md", "Dockerfile", "docker-compose.yml",
    "nginx.conf",
}


def _resolve_output_path(fp: str) -> Path | None:
    """
    Resolve fp to an absolute output path.

    Routing logic:
    - Root-level config files (package.json, tsconfig.app.json, index.html, …)
      → output/ root regardless of what prefix the model used
    - Everything else → output/src/
    """
    normalized = _normalize_repo_rel_path(fp)
    filename   = Path(normalized).name

    # Root-level config files → always output/ root
    if filename in _ROOT_LEVEL_FILES:
        output_dir = SRC_DIR.parent   # artifacts_<slug>/output/
        out_path   = output_dir / filename
        # Safety check — must stay within output/
        try:
            if output_dir.resolve() not in out_path.resolve().parents and                out_path.resolve() != output_dir.resolve():
                pass  # filename is just a name, always safe
        except Exception:
            pass
        return out_path

    # Everything else → output/src/
    return _safe_src_output_path(fp)


def _write_generated_entry(entry: dict[str, Any]) -> str | None:
    fp   = entry.get("file_path")
    code = entry.get("code")

    if not fp or not isinstance(fp, str):
        print(f"[08] SKIP malformed entry without file_path: {entry}", file=sys.stderr)
        return None
    if code is None or not isinstance(code, str):
        print(f"[08] SKIP malformed entry without code: {fp}", file=sys.stderr)
        return None

    out_path = _resolve_output_path(fp)
    if out_path is None:
        print(f"[08] SKIP unsafe output path: {fp}", file=sys.stderr)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    track_write(out_path)
    print(f"[08] WROTE {out_path}")
    return fp


# ════════════════════════════════════════════════════════════════════════════
# Full scope
# ════════════════════════════════════════════════════════════════════════════

def _load_full_plan() -> dict[str, Any]:
    if not PLANNER_FULL_PLAN.exists():
        raise FileNotFoundError(
            f"Missing full plan: {PLANNER_FULL_PLAN}\n"
            "Run 07_planner.py --scope full first."
        )
    return _read_json_file(PLANNER_FULL_PLAN, "planner/full_plan.json")


def run_full_scope(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, Any]]:
    print("[08] Scope: full")

    plan = _load_full_plan()

    stack: dict[str, Any] | None = (
        plan["stack"] if isinstance(plan.get("stack"), dict) else None
    )
    if stack:
        print(f"[08] Stack:\n{json.dumps(stack, indent=2, ensure_ascii=False)}")
    else:
        print("[08] WARNING: full_plan.json has no 'stack' — using generic prompt.")

    task_index: dict[str, dict[str, Any]] = {
        task["file_path"]: task
        for task in plan.get("tasks", [])
        if isinstance(task, dict) and "file_path" in task
    }

    # implementation_order is the canonical file list
    implementation_order: list[str] = plan.get("implementation_order", [])
    all_file_paths: list[str] = implementation_order or sorted(task_index.keys())

    if not all_file_paths:
        raise RuntimeError(
            "full_plan.json has no 'implementation_order' and no 'tasks' — "
            "cannot determine which files to implement."
        )

    print(
        f"[08] Full plan loaded — {len(task_index)} tasks, "
        f"{len(all_file_paths)} files in order"
    )

    # ── Delta filtering ───────────────────────────────────────────────────────
    only_set: set[str] = set()

    if args.only_files.strip():
        only_set   = {fp.strip() for fp in args.only_files.split(",") if fp.strip()}
        file_paths = [fp for fp in all_file_paths if fp in only_set]
        skipped    = [fp for fp in all_file_paths if fp not in only_set]

        print(
            f"[08] Delta mode — {len(file_paths)} file(s) to implement, "
            f"{len(skipped)} skipped."
        )
        for fp in skipped:
            print(f"[08]   SKIP unaffected: {fp}")
        for fp in sorted(only_set - set(all_file_paths)):
            print(f"[08] WARNING: --only-files path not in plan: {fp}")
    else:
        file_paths = all_file_paths

    # ── Execute ───────────────────────────────────────────────────────────────
    already_written: dict[str, str] = (
        _load_restored_files(only_set) if only_set else {}
    )
    if already_written:
        print(f"[08] Import context seeded with {len(already_written)} restored file(s).")

    written:      list[str] = []
    failed_files: list[str] = []

    for fp in file_paths:
        task = task_index.get(fp)
        if task is None:
            print(f"[08] WARNING: no task found for {fp} — implementing with no guidance.")

        try:
            entry = implement_file(
                file_path=fp,
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

    skipped_delta = sorted(set(all_file_paths) - only_set) if only_set else []
    mode = "per-file-with-planner-plan" + ("-delta" if only_set else "")

    return written, failed_files, {
        "scope": "full",
        "mode": mode,
        "skipped_delta": skipped_delta,
        "stack": stack,
        "plan": "planner/full_plan.json",
    }


# ════════════════════════════════════════════════════════════════════════════
# Mini scope
# ════════════════════════════════════════════════════════════════════════════

def _load_mini_plan() -> dict[str, Any]:
    if not PLANNER_MINI_PLAN.exists():
        raise FileNotFoundError(
            f"Missing mini plan: {PLANNER_MINI_PLAN}\n"
            "Run planner first: python harness.py --project <name> --scope mini --plan"
        )
    merged = _read_json_file(PLANNER_MINI_PLAN, "planner/mini_plan.json")

    if "plan" in merged and isinstance(merged["plan"], dict):
        plan = merged["plan"]
    else:
        print("[08] WARNING: mini_plan.json has no 'plan' key — assuming legacy flat format.")
        plan = merged

    if plan.get("scope") != "mini":
        print("[08] WARNING: mini_plan plan has no scope='mini'. Continuing.")

    if not isinstance(plan.get("target_files"), list):
        raise RuntimeError("mini_plan target_files must be a list.")

    return plan


def _load_mini_analysis() -> dict[str, Any] | None:
    if not PLANNER_MINI_PLAN.exists():
        return None
    try:
        merged = json.loads(PLANNER_MINI_PLAN.read_text(encoding="utf-8"))
        if not isinstance(merged, dict):
            return None
        impact = merged.get("impact")
        return impact if isinstance(impact, dict) else None
    except Exception as exc:
        print(f"[08] WARNING: could not read impact from mini_plan.json: {exc}", file=sys.stderr)
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
    normalized["path"]   = path
    normalized["action"] = action
    normalized["risk"]   = str(entry.get("risk", "medium")).strip().lower() or "medium"
    return normalized


def _read_existing_target(path: str) -> str | None:
    out_path = _safe_artifact_output_path(path)
    if out_path is None or not out_path.exists():
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
    path   = str(target["path"])
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
    raw, _ = call_llm(
        ROLE,
        build_system_prompt_mini_file(),
        user_msg,
        temperature=0.15,
        max_tokens=32768,
        retries=2,
        backoff=False,
        caller_file=__file__,
        label=f"[08] {get_model(ROLE)}",
    )
    result = _parse_json(raw, f"mini target {path}")

    # Tolerate full-scope key names
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
    requested_path = _normalize_repo_rel_path(str(target.get("path", "")))
    result_path    = _normalize_repo_rel_path(str(result.get("path", requested_path)))

    if result_path not in allowed_paths:
        print(f"[08] SKIP mini result outside plan target set: {result_path}", file=sys.stderr)
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
    print("[08] Scope: mini")

    plan     = _load_mini_plan()
    analysis = _load_mini_analysis()

    targets: list[dict[str, Any]] = []
    for raw in plan.get("target_files", []):
        if not isinstance(raw, dict):
            print(f"[08] WARNING: skipping invalid target entry: {raw!r}", file=sys.stderr)
            continue
        normalized = _normalize_mini_target(raw)
        if normalized:
            targets.append(normalized)

    if args.only_files.strip():
        only_set = {
            _normalize_repo_rel_path(p)
            for p in args.only_files.split(",")
            if p.strip()
        }
        before  = len(targets)
        targets = [t for t in targets if t["path"] in only_set]
        print(f"[08] Mini --only-files: {len(targets)}/{before} target(s) selected.")

    print(f"[08] Mini targets: {len(targets)}")
    for t in targets:
        print(f"[08]   {t.get('action', 'MODIFY'):<6} {t.get('path')}  risk={t.get('risk', 'medium')}")

    allowed_paths = _mini_allowed_target_paths(plan)
    written:      list[str] = []
    failed_files: list[str] = []

    for target in targets:
        fp = str(target["path"])
        try:
            result     = _implement_mini_target(plan=plan, analysis=analysis, target=target)
            written_fp = _write_mini_result(result, target=target, allowed_paths=allowed_paths)
            if written_fp:
                written.append(written_fp)
            else:
                failed_files.append(fp)
        except Exception as exc:
            print(f"[08] FAILED mini target {fp}: {exc}", file=sys.stderr)
            failed_files.append(fp)

    return written, failed_files, {
        "scope":        "mini",
        "plan":         "planner/mini_plan.json",
        "task_summary": plan.get("task_summary", ""),
        "target_files": [t.get("path") for t in targets],
    }


# ════════════════════════════════════════════════════════════════════════════
# Manifest writing
# ════════════════════════════════════════════════════════════════════════════

def _append_manifest_log(
    *,
    scope: str,
    mode: str,
    written: list[str],
    failed_files: list[str],
    extra: dict[str, Any],
) -> None:
    token_sum  = cost_summary()
    cost_total = token_sum.get("total_cost_usd") if isinstance(token_sum, dict) else None

    entry: dict[str, Any] = {
        "scope":          scope,
        "mode":           mode,
        "generated_at":   _utc_now_iso(),
        "files_written":  len(written),
        "failed_count":   len(failed_files),
        "cost":           cost_total,
    }
    if extra.get("task_summary"):
        entry["task_summary"] = extra["task_summary"]
    if extra.get("stack"):
        entry["stack"] = extra["stack"]

    log_path = EXECUTOR_MANIFEST_LOG
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entries: list[Any] = []
        if log_path.exists():
            track_read(log_path)
            try:
                data    = json.loads(log_path.read_text(encoding="utf-8").strip())
                entries = data.get("entries", []) if isinstance(data, dict) else data
                if not isinstance(entries, list):
                    entries = []
            except Exception:
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
    token_sum  = cost_summary()
    cost_total = token_sum.get("total_cost_usd") if isinstance(token_sum, dict) else None

    record: dict[str, Any] = {
        "executor_role":  ROLE,
        "executor_model": get_model(ROLE),
        "scope":          scope,
        "mode":           mode,
        "generated_at":   _utc_now_iso(),
        "files":          written,
        "failed_files":   failed_files,
        "token_summary":  token_sum,
        "cost":           cost_total,
    }
    record.update(extra)

    EXECUTOR_OVERWRITE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EXECUTOR_OVERWRITE_MANIFEST.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(EXECUTOR_OVERWRITE_MANIFEST)
    print(f"[08] Executor manifest → {EXECUTOR_OVERWRITE_MANIFEST}")

    _append_manifest_log(
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
        description="Executor. Full scope reads full_plan.json only. Mini scope reads mini_plan.json only.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name. Sets PIPELINE_PROJECT before resolving artifact paths.",
    )
    parser.add_argument(
        "--scope",
        choices=["full", "mini"],
        default="full",
        help="full: reads planner/full_plan.json. mini: reads planner/mini_plan.json.",
    )
    parser.add_argument(
        "--only-files",
        default="",
        help=(
            "Comma-separated paths to implement. "
            "Full scope: delta mode. Mini scope: filters target_files."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()

    exit_code = 0

    try:
        if args.scope == "mini":
            written, failed_files, extra = run_mini_scope(args)
            mode = "mini-targeted" + ("-filtered" if args.only_files.strip() else "")
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
            mode = str(extra.get("mode", "per-file-with-planner-plan"))
            _write_impl_record(
                scope="full",
                mode=mode,
                written=written,
                failed_files=failed_files,
                extra=extra,
            )

    except Exception as exc:
        print(f"[08] ERROR: {exc}", file=sys.stderr)
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