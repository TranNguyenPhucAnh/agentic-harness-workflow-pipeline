"""
pipeline/03a_implement_qwen.py
Step 3a — Qwen 3.6 Plus as EXECUTOR.

Modes:
    Default, no GLM plan:
        Single API call for all requested non-test stub files.

    With GLM plan (--use-glm-plan):
        Reads artifacts_<slug>/state/plan.json produced by 03b_implement_glm.py.
        For each file, injects the matching GLM task:
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

Writes:
    artifacts_<slug>/src/**                  non-test files only
    artifacts_<slug>/run/impl_record.json

Reads:
    artifacts_<slug>/spec.md or cache/spec_compressed.md
    artifacts_<slug>/state/scaffold.json
    artifacts_<slug>/state/plan.json         optional, with --use-glm-plan

For taxonomy details see docs/artifacts.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.6-plus"


# === WRITE AUTHORITY: 03a_implement_qwen ===
# OWNS  : artifacts_<slug>/run/impl_record.json
#         artifacts_<slug>/src/**
# READS : artifacts_<slug>/spec.md, artifacts_<slug>/state/scaffold.json,
#         artifacts_<slug>/state/plan.json

import sys as _sys

_sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    SPEC_PATH,
    CACHE_DIR,
    SCAFFOLD_JSON,
    PLAN_JSON as GLM_PLAN,
    IMPL_RECORD,
    SRC_DIR,
    ensure_dirs,
)

ensure_dirs()


# ── Generic source helpers ────────────────────────────────────────────────────

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


def _fence_language(file_path: str) -> str:
    return FENCE_LANGUAGE_BY_EXT.get(Path(file_path).suffix.lower(), "")


def _code_fence(file_path: str, code: str) -> str:
    lang = _fence_language(file_path)
    if lang:
        return f"```{lang}\n{code}\n```"
    return f"```\n{code}\n```"


def _is_probably_source_file(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_EXTENSIONS


def _safe_src_output_path(file_path: str) -> Path | None:
    """
    Convert scaffold file_path into an output path under SRC_DIR.

    Accepts:
        src/foo/bar.ts  -> SRC_DIR/foo/bar.ts
        foo/bar.ts      -> SRC_DIR/foo/bar.ts

    Rejects traversal/outside paths.
    """
    rel = file_path[len("src/") :] if file_path.startswith("src/") else file_path
    out_path = SRC_DIR / rel

    try:
        src_root = SRC_DIR.resolve()
        resolved = out_path.resolve()
    except FileNotFoundError:
        src_root = SRC_DIR.absolute()
        resolved = out_path.absolute()

    if resolved != src_root and src_root not in resolved.parents:
        return None

    return out_path


# ── Spec / prompt helpers ─────────────────────────────────────────────────────

def _load_spec() -> str:
    compressed = CACHE_DIR / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else SPEC_PATH.read_text()


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
    Single-call prompt used when no GLM plan is available.

    This is intentionally generic. Any stack-specific constraints should come
    from the spec/scaffold/instructions/stack object, not from this prompt.
    """
    stack_section = _format_stack_section(stack)

    return f"""\
You are a senior software developer implementing source files.
{stack_section}
You will receive:
1. A technical spec, spec.md
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
- Do NOT modify test files, where is_test is true.
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

    This is stack-aware through the optional stack object generated by GLM.
    """
    stack_section = _format_stack_section(stack)

    return f"""\
You are a senior software developer implementing ONE source file.
{stack_section}
You will receive:
1. The technical spec, spec.md, for full project context
2. Optional dependency/context files that were already implemented
3. Optional task plan produced by a senior architect — follow it carefully
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


# ── API call ──────────────────────────────────────────────────────────────────

def _extract_chat_text_response(data: dict[str, Any], label: str) -> str:
    choice = data["choices"][0]
    msg = choice["message"]

    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    finish_reason = choice.get("finish_reason")

    if tool_calls:
        raise RuntimeError(
            f"{label} returned tool_calls instead of text: {tool_calls}"
        )

    if not content or not content.strip():
        raise RuntimeError(
            f"{label} returned empty content. "
            f"finish_reason={finish_reason}, message={msg}"
        )

    return content.strip()


def _call_qwen(system: str, user_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.15,
        "max_tokens": 32768,
    }

    last_error: Exception | None = None

    with httpx.Client(timeout=180) as client:
        for attempt in range(2):
            try:
                r = client.post(OPENROUTER_URL, headers=headers, json=payload)
                r.raise_for_status()

                try:
                    data = r.json()
                except json.JSONDecodeError as e:
                    body_preview = r.text[:1000] if r.text else "<empty body>"
                    raise RuntimeError(
                        f"OpenRouter returned non-JSON response: {e}\n"
                        f"Response body, first 1000 chars:\n{body_preview}"
                    ) from e

                usage = data.get("usage", {})
                prompt_t = usage.get("prompt_tokens", "?")
                completion_t = usage.get("completion_tokens", "?")
                print(f"[qwen] Tokens: prompt={prompt_t}, completion={completion_t}")

                return _extract_chat_text_response(data, label="Qwen")

            except httpx.HTTPStatusError as e:
                body_preview = (
                    e.response.text[:1000]
                    if e.response is not None and e.response.text
                    else "<empty body>"
                )
                last_error = RuntimeError(
                    f"HTTP error from OpenRouter: {e}\n"
                    f"Response body, first 1000 chars:\n{body_preview}"
                )
                print(f"[qwen] {last_error}", file=sys.stderr)

            except (httpx.HTTPError, RuntimeError) as e:
                last_error = e
                print(f"[qwen] {e}", file=sys.stderr)

            if attempt == 0:
                print("[qwen] Retrying in 3s …", file=sys.stderr)
                time.sleep(3)

    raise RuntimeError(f"Qwen call failed after retries: {last_error}")


# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_json(raw: str, label: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response.

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
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"JSON parse failed for {label}: {e}\n"
                f"Raw, first 500 chars:\n{raw[:500]}"
            ) from e

    raise RuntimeError(f"No JSON object found in response for {label}.")


# ── Plan / task formatting ────────────────────────────────────────────────────

def _build_task_block(task: dict[str, Any] | None) -> str:
    """Format GLM plan task as a prompt section."""
    if not task:
        return ""

    lines: list[str] = ["### Implementation plan from architect\n"]

    role = task.get("role")
    if role:
        lines.append(f"**Role:** {role}\n")

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

    # Backward-compatible name from older React/Tailwind-focused planner.
    tailwind_hints = task.get("tailwind_hints")
    if tailwind_hints:
        lines.append(f"**Styling hints:** {tailwind_hints}\n")

    # Future-compatible generic field if planner starts emitting it.
    styling_hints = task.get("styling_hints")
    if styling_hints and styling_hints != tailwind_hints:
        lines.append(f"**Additional styling hints:** {styling_hints}\n")

    return "\n".join(lines)


# ── Context selection ─────────────────────────────────────────────────────────

def _compact_reference_code(code: str, max_chars: int = 1500) -> str:
    """
    Compact code when used as broad API reference.

    This is intentionally simple and language-agnostic:
    - remove blank lines
    - remove obvious full-line comments
    - cap length
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
    """
    Small shared files that are often useful as context.

    Kept broad enough for frontend/backend/mixed stacks.
    """
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
    """
    Files that often wire the app together and can benefit from compact broad API refs.
    """
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

    Priority:
    1. Explicit task depends_on files.
    2. For entrypoint-like files: compact shared refs from many files.
    3. Fallback: shared references such as types/models/schemas/constants/config.
    """
    if not already_written:
        return ""

    deps = set(task.get("depends_on", [])) if task else set()

    relevant: dict[str, str] = {}

    if deps:
        relevant = {
            fp: code
            for fp, code in already_written.items()
            if fp in deps
        }

    label = "Dependencies already implemented — for import/API reference"

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


# ── Per-file generation ───────────────────────────────────────────────────────

def implement_file(
    spec: str,
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
        f"### spec.md\n\n{spec}\n\n"
        f"{context_block}\n"
        f"{task_block}\n"
        f"### Stub file to implement: {file_path}\n"
        f"{_code_fence(file_path, stub_code)}"
    )

    approx_tokens = len(user_msg) // 4
    if approx_tokens > 28000:
        print(
            f"[03a] ⚠ Large prompt for {file_path}: ~{approx_tokens:,} tokens "
            f"(limit ~32k). Response may be truncated.",
            file=sys.stderr,
        )

    print(f"[03a]   → Implementing {file_path} …")
    raw = _call_qwen(build_system_prompt_per_file(stack=stack), user_msg)
    result = _parse_json(raw, file_path)

    # Be tolerant if model accidentally returns the single-call shape.
    if "files" in result and isinstance(result["files"], list):
        for entry in result["files"]:
            if entry.get("file_path") == file_path:
                return entry
        if result["files"]:
            return result["files"][0]

    return result


# ── Ordering helpers ──────────────────────────────────────────────────────────

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
        key=lambda f: order_map.get(f["file_path"], 999_999),
    )


# ── Single-call fallback ──────────────────────────────────────────────────────

def implement_all_single_call(
    spec: str,
    stub_files: list[dict[str, Any]],
    instructions: str,
    stack: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Single-call mode — all requested files in one request.

    Used when no GLM plan is requested.
    """
    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold, stub files to implement\n\n"
        f"{json.dumps(stub_files, indent=2, ensure_ascii=False)}"
    )

    print("[03a] Calling Qwen 3.6 Plus, single-call mode …")

    raw = _call_qwen(
        build_system_prompt_single(instructions=instructions, stack=stack),
        user_msg,
    )
    result = _parse_json(raw, "single-call")

    files = result.get("files", [])
    if not isinstance(files, list):
        raise RuntimeError("single-call result missing list field: files")

    return files


# ── Delta mode restored context ────────────────────────────────────────────────

def _load_restored_files(only_set: set[str]) -> dict[str, str]:
    """
    Read already-restored src/ files into memory so they can be used as
    import/reference context for Qwen in delta mode.

    Loads only files NOT in only_set, i.e. unaffected/restored files.
    """
    restored: dict[str, str] = {}

    if not SRC_DIR.exists():
        return restored

    all_paths = sorted(
        p for p in SRC_DIR.rglob("*")
        if p.is_file() and _is_probably_source_file(p)
    )

    for p in all_paths:
        rel = "src/" + str(p.relative_to(SRC_DIR)).replace("\\", "/")

        if rel in only_set:
            continue

        try:
            restored[rel] = p.read_text()
        except Exception as e:
            print(f"[03a] WARNING: could not read restored file {rel}: {e}")

    return restored


# ── Output writing ────────────────────────────────────────────────────────────

def _write_generated_entry(entry: dict[str, Any]) -> str | None:
    """
    Write one generated entry to SRC_DIR.

    Returns written file_path, or None if skipped.
    """
    fp = entry.get("file_path")
    code = entry.get("code")

    if not fp or not isinstance(fp, str):
        print(f"[03a] SKIP malformed entry without file_path: {entry}", file=sys.stderr)
        return None

    if code is None or not isinstance(code, str):
        print(f"[03a] SKIP malformed entry without code: {fp}", file=sys.stderr)
        return None

    out_path = _safe_src_output_path(fp)
    if out_path is None:
        print(f"[03a] SKIP outside src/: {fp}", file=sys.stderr)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)

    print(f"[03a] WROTE {out_path}")
    return fp


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-glm-plan",
        action="store_true",
        help="Inject artifacts/state/plan.json as per-file implementation guidance.",
    )

    parser.add_argument(
        "--only-files",
        default="",
        help=(
            "Comma-separated src/ paths to implement, delta mode. "
            "All other stubs are skipped and assumed already restored by harness."
        ),
    )

    args = parser.parse_args()

    spec = _load_spec()
    scaffold = json.loads(SCAFFOLD_JSON.read_text())

    instrs = scaffold.get("implementation_instructions", {}).get("for_qwen", "")
    scaffold_stack = scaffold.get("stack")

    all_stubs: list[dict[str, Any]] = [
        f for f in scaffold["files"]
        if not f.get("is_test")
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
            f for f in all_stubs
            if f["file_path"] in only_set
        ]

        skipped = [
            f["file_path"]
            for f in all_stubs
            if f["file_path"] not in only_set
        ]

        print(
            f"[03a] Delta mode — {len(stub_files)} file(s) to implement, "
            f"{len(skipped)} unaffected skipped."
        )

        for fp in skipped:
            print(f"[03a]   SKIP unaffected: {fp}")

        requested_missing = sorted(
            only_set - {f["file_path"] for f in all_stubs}
        )
        for fp in requested_missing:
            print(f"[03a] WARNING: --only-files path not found in scaffold: {fp}")

    else:
        stub_files = all_stubs

    # ── Load GLM plan if requested ────────────────────────────────────────────

    plan: dict[str, Any] | None = None
    task_index: dict[str, dict[str, Any]] = {}
    stack: dict[str, Any] | None = scaffold_stack if isinstance(scaffold_stack, dict) else None

    if args.use_glm_plan:
        if not GLM_PLAN.exists():
            print(
                "[03a] ERROR: --use-glm-plan set but artifacts/state/plan.json not found.",
                file=sys.stderr,
            )
            print("             Run 03b_implement_glm.py first.", file=sys.stderr)
            sys.exit(1)

        plan = json.loads(GLM_PLAN.read_text())
        task_index = {
            t["file_path"]: t
            for t in plan.get("tasks", [])
            if isinstance(t, dict) and "file_path" in t
        }

        plan_stack = plan.get("stack")
        if isinstance(plan_stack, dict):
            stack = plan_stack

        print(
            f"[03a] GLM plan loaded — {len(task_index)} tasks, "
            f"order: {plan.get('implementation_order', [])}"
        )

        if stack:
            print(f"[03a] Stack in use:\n{json.dumps(stack, indent=2, ensure_ascii=False)}")
        else:
            print(
                "[03a] WARNING: GLM plan has no 'stack'; executor will use generic prompt."
            )

    else:
        print("[03a] No GLM plan — using single-call mode.")

        if stack:
            print(
                f"[03a] Stack from scaffold in use:\n"
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
                f"[03a] Import context seeded with "
                f"{len(already_written)} restored file(s)."
            )

        for stub in ordered:
            fp = stub["file_path"]
            task = task_index.get(fp)

            try:
                entry = implement_file(
                    spec=spec,
                    stub=stub,
                    task=task,
                    already_written=already_written,
                    stack=stack,
                )
            except Exception as e:
                print(f"[03a] FAILED to implement {fp}: {e}", file=sys.stderr)
                failed_files.append(fp)
                continue

            written_fp = _write_generated_entry(entry)

            if written_fp:
                already_written[written_fp] = entry["code"]
                written.append(written_fp)

    else:
        try:
            entries = implement_all_single_call(
                spec=spec,
                stub_files=stub_files,
                instructions=instrs,
                stack=stack,
            )
        except Exception as e:
            print(f"[03a] FAILED single-call generation: {e}", file=sys.stderr)
            failed_files.extend([f["file_path"] for f in stub_files])
            entries = []

        for entry in entries:
            written_fp = _write_generated_entry(entry)
            if written_fp:
                written.append(written_fp)

    # ── Run record ────────────────────────────────────────────────────────────

    mode = "per-file-with-glm-plan" if plan else "single-call"
    if only_set:
        mode += "-delta"

    skipped_delta = (
        sorted({f["file_path"] for f in all_stubs} - only_set)
        if only_set
        else []
    )

    IMPL_RECORD.parent.mkdir(parents=True, exist_ok=True)
    IMPL_RECORD.write_text(
        json.dumps(
            {
                "model": "qwen",
                "mode": mode,
                "files": written,
                "skipped_delta": skipped_delta,
                "failed_files": failed_files,
                "stack": stack,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if failed_files:
        print(
            f"[03a] Done with {len(written)} files written, "
            f"{len(failed_files)} failed: {failed_files}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[03a] Done — {len(written)} files written.")


if __name__ == "__main__":
    main()
