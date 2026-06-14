"""
pipeline/08b_test_writer.py
===========================
Step 8b — TEST WRITER.

On-off standalone step. Writes real behavior-driven unit tests into
output/tests/ by consuming the actual source files + full_plan.json.

Intended workflow:
  1. executor (08) writes src/ files
  2. User fixes compile errors, then re-runs absorber (01)
  3. test_writer (08b) scans src/, reads full_plan.json, writes real tests
  4. debugger (09) picks up from there and runs/repairs them

Design decisions:
  - Reads full_plan.json for intent: AC IDs, behavior_summary, sub_tasks,
    gotchas, depends_on.
  - Reads actual src/ files to know real exports, hook signatures, type
    names — NOT blueprint stubs.
  - Reads absorber/codebase_map.md as structural overview so the model
    understands the whole project before writing any one test file.
  - One LLM call per test file. Each call gets:
      * stack from full_plan.json
      * codebase_map.md (overview, ~3k chars max)
      * full task for the corresponding src file (ACs, sub_tasks, gotchas)
      * actual source file content
      * existing stub content (if any) as baseline
  - Writes to output/tests/ only — never touches output/src/.
  - Overwrites existing stubs; skips files already containing real tests
    (heuristic: no "not implemented" throw) unless --force is set.
  - Produces test_writer/manifest.json (short-term) and
    test_writer/manifest_log.json (long-term, append).

Reads:
  artifacts_<slug>/planner/full_plan.json
  artifacts_<slug>/absorber/codebase_map.md
  artifacts_<slug>/output/src/**              (actual source)
  artifacts_<slug>/output/tests/**            (existing stubs, optional)

Writes:
  artifacts_<slug>/output/tests/**
  artifacts_<slug>/test_writer/manifest.json        (short-term, overwrite)
  artifacts_<slug>/test_writer/manifest_log.json    (long-term, append)

Direct execution:
  python pipeline/08b_test_writer.py --project my-app
  python pipeline/08b_test_writer.py --project my-app --only-files src/hooks/useReplay.ts
  python pipeline/08b_test_writer.py --project my-app --force
  PIPELINE_PROJECT=my-app python pipeline/08b_test_writer.py
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.models import get_model                          # noqa: E402
from artifacts.paths import (                                   # noqa: E402
    ABSORBER_CODEBASE_MD,
    PLANNER_FULL_PLAN,
    SRC_DIR,
    TESTS_DIR,
    artifact_root,
    ensure_dirs,
)
from modules.artifact_tracking import (                         # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary, summary as cost_summary # noqa: E402
from modules.call_llm import call_llm_json                     # noqa: E402
from modules.post_interactive import prompt_next_step          # noqa: E402

ROLE = "test_writer"  # own model slot in artifacts/models.py — swap independently

TAG = "[08b]"

# ════════════════════════════════════════════════════════════════════════════
# Paths — test_writer owns these two artifacts
# ════════════════════════════════════════════════════════════════════════════

def _tw_manifest() -> Path:
    return artifact_root() / "test_writer" / "manifest.json"

def _tw_manifest_log() -> Path:
    return artifact_root() / "test_writer" / "manifest_log.json"

def _ensure_tw_dir() -> None:
    (artifact_root() / "test_writer").mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _fence(file_path: str, code: str) -> str:
    ext_map = {
        ".ts": "typescript", ".tsx": "tsx",
        ".js": "javascript", ".jsx": "jsx",
        ".py": "python",
    }
    lang = ext_map.get(Path(file_path).suffix.lower(), "")
    return f"```{lang}\n{code}\n```"


def _read_text(path: Path, *, track: bool = True) -> str:
    if not path.exists():
        return ""
    if track:
        track_read(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _is_stub(content: str) -> bool:
    """Heuristic: file is still a scaffold stub if it throws 'not implemented'."""
    return bool(
        re.search(r'throw new Error\(["\']not implemented', content, re.IGNORECASE)
        or re.search(r'raise NotImplementedError', content)
        or re.search(r't\.Fatal\("not implemented"\)', content)
    )


# ════════════════════════════════════════════════════════════════════════════
# Plan loading
# ════════════════════════════════════════════════════════════════════════════

def _load_full_plan() -> dict[str, Any]:
    if not PLANNER_FULL_PLAN.exists():
        raise FileNotFoundError(
            f"Missing full plan: {PLANNER_FULL_PLAN}\n"
            "Run 07_planner.py --scope full first."
        )
    track_read(PLANNER_FULL_PLAN)
    data = json.loads(PLANNER_FULL_PLAN.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("full_plan.json must be a JSON object")
    return data


def _build_task_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map src file_path → task dict."""
    return {
        task["file_path"]: task
        for task in plan.get("tasks", [])
        if isinstance(task, dict) and "file_path" in task
    }


def _infer_test_path(src_rel: str) -> str:
    """
    src/hooks/useReplay.ts  → tests/hooks/useReplay.test.ts
    src/components/Foo.tsx  → tests/components/Foo.test.tsx
    src/utils/format.ts     → tests/utils/format.test.ts
    """
    p = Path(src_rel)
    # strip leading src/ if present
    parts = p.parts
    if parts and parts[0] == "src":
        inner = Path(*parts[1:])
    else:
        inner = p

    stem   = inner.stem
    suffix = inner.suffix  # .ts / .tsx / .py …
    test_name = f"{stem}.test{suffix}"
    return str(Path("tests") / inner.parent / test_name)


# ════════════════════════════════════════════════════════════════════════════
# Source scanning
# ════════════════════════════════════════════════════════════════════════════

_TESTABLE_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".py", ".go"}
_SKIP_PATTERNS = [
    re.compile(r"^(index|main|app|vite\.config|tsconfig|package)\.", re.IGNORECASE),
    re.compile(r"\.d\.ts$"),
    re.compile(r"\.config\.(ts|js|cjs|mjs)$"),
    re.compile(r"\.stories\.(ts|tsx|js|jsx)$"),
]


def _should_test(src_rel: str, task: dict[str, Any] | None) -> bool:
    """Return True if this src file deserves a test."""
    name = Path(src_rel).name.lower()
    if any(pat.search(name) for pat in _SKIP_PATTERNS):
        return False
    if Path(src_rel).suffix not in _TESTABLE_EXTENSIONS:
        return False
    # if plan says kind=config or no ACs/subtasks, skip
    if task:
        kind = task.get("kind", "source")
        if kind in {"config", "migration"}:
            return False
        has_acs      = bool(task.get("acceptance_criteria"))
        has_subtasks = bool(task.get("sub_tasks"))
        if not has_acs and not has_subtasks:
            return False
    return True


def _scan_src_files() -> list[str]:
    """Return list of src-relative paths like 'src/hooks/useReplay.ts'."""
    src_root = Path(SRC_DIR)
    if not src_root.exists():
        return []
    out: list[str] = []
    for p in sorted(src_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src_root.parent)  # keeps 'src/' prefix
        out.append(rel.as_posix())
    return out


# ════════════════════════════════════════════════════════════════════════════
# System prompt
# ════════════════════════════════════════════════════════════════════════════

def _compute_relative_import(test_path: str, src_rel: str) -> str:
    """
    Compute the relative import path FROM a test file TO its source file.

    test_path : "tests/hooks/useReplay.test.ts"
    src_rel   : "src/hooks/useReplay.ts"
    result    : "../../src/hooks/useReplay"   (no extension)

    Vitest resolves imports relative to the test file. The @/ alias is
    configured for Vite/src only — it does NOT work inside tests/ at runtime.
    Always use explicit relative paths.
    """
    test_dir = Path(test_path).parent          # tests/hooks
    src_path = Path(src_rel).with_suffix("")   # src/hooks/useReplay

    # Both paths are relative to artifact output root.
    # Compute relative path from test_dir → src_path.
    # Since both are relative, anchor them to a common imaginary root.
    anchor   = Path(".")
    abs_test = anchor / test_dir               # ./tests/hooks
    abs_src  = anchor / src_path               # ./src/hooks/useReplay

    rel = os.path.relpath(abs_src, abs_test)   # ../../src/hooks/useReplay
    return rel.replace("\\", "/")


def _build_system_prompt(stack: dict[str, Any] | None) -> str:
    stack_block = ""
    if stack:
        stack_block = (
            "\n## Project stack — follow exactly\n\n"
            f"{json.dumps(stack, indent=2, ensure_ascii=False)}\n\n"
            "Rules:\n"
            "- Use only libraries listed in this stack.\n"
            "- Match the module system (ESM/CJS), import style, and test runner shown.\n"
        )

    return f"""\
You are a senior developer writing BEHAVIOR-DRIVEN unit tests for ONE source file.
{stack_block}
You will receive:
1. A codebase overview (structural map)
2. The implementation plan for the source file (acceptance criteria, sub-tasks, gotchas)
3. The actual source file content
4. An existing test stub (may be empty or placeholder-only)
5. The EXACT relative import path you must use to import the source file

## CRITICAL — Import rules (violations cause test runner crashes)

### Rule 1 — Always use the provided relative import path
The prompt gives you: `RELATIVE IMPORT PATH: ../../src/hooks/useReplay`
Use it EXACTLY as given. Example:
  ✓  import {{ useReplay }} from '../../src/hooks/useReplay';
  ✗  import {{ useReplay }} from '@/hooks/useReplay';        ← alias broken in tests/
  ✗  import {{ useReplay }} from 'src/hooks/useReplay';      ← missing ../..

### Rule 2 — Never import runtime/UI libs directly
Runtime dependencies (dexie, dexie-react-hooks, wavesurfer.js, zustand stores,
React context providers, browser APIs) MUST be mocked — never imported for real.
  ✓  vi.mock('dexie-react-hooks', () => ({{ useLiveQuery: vi.fn() }}))
  ✗  import {{ useLiveQuery }} from 'dexie-react-hooks';     ← pulls real IndexedDB

### Rule 3 — Mock strategy by file type
- Hooks using dexie/IndexedDB  → vi.mock the dexie import at top of test
- Hooks using zustand store    → vi.mock the store module, return mock state
- Components with wavesurfer   → vi.mock('wavesurfer.js', () => ({{ default: {{ create: vi.fn() }} }}))
- Components with context      → wrap render() in the real or a stub Provider
- Pure utils/helpers           → no mocks needed; test input→output directly

### Rule 4 — Testing library imports
  ✓  import {{ render, screen, fireEvent }} from '@testing-library/react';
  ✓  import {{ renderHook, act }} from '@testing-library/react';
  ✓  import {{ describe, it, expect, vi, beforeEach }} from 'vitest';
  ✗  import {{ renderHook }} from '@testing-library/react-hooks';  ← deprecated

Your task:
- Write COMPLETE, runnable tests covering the acceptance criteria and key behaviors.
- Behavior-driven: test WHAT the unit does, not HOW it is implemented internally.
- Keep tests deterministic — no random data, no real timers, no real network calls.
- For React components: query by role/text/testid, not by class or tag.
- For React hooks: use renderHook + act.
- For pure functions: plain it/expect, no DOM setup.

Return raw JSON only — no markdown fences, no explanation:
{{
  "file_path": "tests/hooks/useReplay.test.ts",
  "code": "<complete test file content>"
}}
"""


# ════════════════════════════════════════════════════════════════════════════
# Type surface extractor
# ════════════════════════════════════════════════════════════════════════════

# Patterns that capture the public type surface of a TS/TSX file.
# We extract these and inject them BEFORE the full source so the model
# sees exact types first, reducing hallucinated property names.

_RE_INTERFACE    = re.compile(
    r'^export\s+(?:default\s+)?interface\s+\w[\w<>, ]*\s*(?:extends[^{]*)?\{[^}]*\}',
    re.MULTILINE | re.DOTALL,
)
_RE_TYPE_ALIAS   = re.compile(
    r'^export\s+type\s+\w+\s*(?:<[^>]*>)?\s*=\s*(?:[^;{]|\{[^}]*\})+;',
    re.MULTILINE | re.DOTALL,
)
_RE_ENUM         = re.compile(
    r'^export\s+(?:const\s+)?enum\s+\w+\s*\{[^}]*\}',
    re.MULTILINE | re.DOTALL,
)
# Function/hook signatures — capture only up to the opening brace (no body)
_RE_FUNC_SIG     = re.compile(
    r'^export\s+(?:default\s+)?(?:async\s+)?function\s+\w+\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?::\s*[^{;]+)?(?=\s*[{;])',
    re.MULTILINE,
)
# Arrow functions / const hooks assigned at top level
_RE_ARROW_SIG    = re.compile(
    r'^export\s+(?:const|let)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=>{]+)?(?=\s*=>)',
    re.MULTILINE,
)
# export { ... } and export type { ... } re-exports
_RE_REEXPORT     = re.compile(
    r'^export\s+(?:type\s+)?\{[^}]+\}(?:\s+from\s+[\'"][^\'"]+[\'"])?;',
    re.MULTILINE,
)
# Plain named exports: export const FOO = ... (just the declaration line)
_RE_CONST_EXPORT = re.compile(
    r'^export\s+(?:const|let|var)\s+\w+\s*(?::\s*[^\n=]+)?(?=\s*=)',
    re.MULTILINE,
)


def _extract_type_surface(src_content: str, src_rel: str) -> str:
    """
    Extract the exported type surface from a TypeScript/TSX source file.

    Returns a condensed string containing:
      - interface definitions (full, with all fields)
      - type aliases
      - enums
      - function/hook signatures (no bodies)
      - export declarations

    This is injected into the test prompt BEFORE the full source so the
    model sees exact property names and types before reading implementation
    details, reducing hallucinated fields and type mismatches.

    Falls back gracefully for non-TS files (returns empty string).
    """
    ext = Path(src_rel).suffix.lower()
    if ext not in {".ts", ".tsx", ".js", ".jsx"}:
        return ""

    chunks: list[str] = []

    def _add(pattern: re.Pattern, label: str) -> None:
        matches = pattern.findall(src_content)
        for m in matches:
            text = m.strip() if isinstance(m, str) else m[0].strip()
            if text:
                chunks.append(text)

    # Interfaces and type aliases — most important for property correctness
    for m in _RE_INTERFACE.finditer(src_content):
        chunks.append(m.group(0).strip())

    for m in _RE_TYPE_ALIAS.finditer(src_content):
        text = m.group(0).strip()
        # Trim very long union types but keep enough to be useful
        if len(text) > 300:
            text = text[:300] + "  /* … */"
        chunks.append(text)

    for m in _RE_ENUM.finditer(src_content):
        chunks.append(m.group(0).strip())

    # Function/hook signatures
    for m in _RE_FUNC_SIG.finditer(src_content):
        sig = m.group(0).strip().rstrip("{").strip()
        chunks.append(sig + ";")

    for m in _RE_ARROW_SIG.finditer(src_content):
        sig = m.group(0).strip().rstrip("=").strip()
        chunks.append(sig + ";")

    for m in _RE_REEXPORT.finditer(src_content):
        chunks.append(m.group(0).strip())

    for m in _RE_CONST_EXPORT.finditer(src_content):
        chunks.append(m.group(0).strip() + ";")

    if not chunks:
        return ""

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return "\n\n".join(unique)




_CODEBASE_MAP_MAX = 3000  # chars injected from codebase_map.md


def _build_user_message(
    *,
    src_rel: str,
    test_path: str,
    src_content: str,
    stub_content: str,
    task: dict[str, Any] | None,
    codebase_map: str,
    relative_import: str,
) -> str:
    map_snippet = codebase_map[:_CODEBASE_MAP_MAX]
    if len(codebase_map) > _CODEBASE_MAP_MAX:
        map_snippet += "\n… (truncated)"

    task_block = ""
    if task:
        acs       = task.get("acceptance_criteria") or []
        subtasks  = task.get("sub_tasks") or []
        gotchas   = task.get("gotchas") or []
        notes     = task.get("notes") or []
        behavior  = task.get("behavior_summary", "")

        lines: list[str] = ["## Implementation plan\n"]
        if behavior:
            lines.append(f"**Purpose:** {behavior}\n")
        if acs:
            lines.append("**Acceptance criteria:**")
            for ac in acs:
                lines.append(f"  - {ac}")
            lines.append("")
        if subtasks:
            lines.append("**Behaviors to test (from sub-tasks):**")
            for st in subtasks:
                lines.append(f"  - {st}")
            lines.append("")
        if gotchas:
            lines.append("**Gotchas / edge cases:**")
            for g in gotchas:
                lines.append(f"  - {g}")
            lines.append("")
        if notes:
            lines.append("**Notes:**")
            for n in notes:
                lines.append(f"  - {n}")
            lines.append("")
        task_block = "\n".join(lines)

    stub_block = (
        "*(no existing stub)*"
        if not stub_content.strip()
        else f"### Existing stub (baseline)\n\n{_fence(test_path, stub_content)}"
    )

    return (
        f"## Codebase overview\n\n{map_snippet}\n\n"
        f"---\n\n"
        f"{task_block}"
        f"---\n\n"
        f"## RELATIVE IMPORT PATH (use this exactly)\n\n"
        f"```\n"
        f"import {{ ... }} from '{relative_import}';\n"
        f"```\n\n"
        f"Do NOT use `@/` aliases or bare `src/` paths — they break in tests/.\n\n"
        f"---\n\n"
        f"## Source file to test: `{src_rel}`\n\n"
        f"{_fence(src_rel, src_content)}\n\n"
        f"---\n\n"
        f"{stub_block}\n\n"
        f"---\n\n"
        f"Write the complete test file for `{test_path}`.\n"
        f"Return raw JSON: {{\"file_path\": \"{test_path}\", \"code\": \"...\"}}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Per-file test generation
# ════════════════════════════════════════════════════════════════════════════

def _rewrite_alias_imports(code: str, test_path: str) -> str:
    """
    Last-resort rewrite: replace any @/foo/bar imports with the correct
    relative path computed from the test file's location.

    This catches cases where the model ignores the import rule and still
    emits @/ aliases. We can't know the exact exported name so we only
    fix the module path, leaving named imports untouched.

    Pattern matched: from '@/some/path'  or  from "@/some/path"
    """
    test_dir = Path(test_path).parent  # e.g. tests/hooks

    def _replace(m: re.Match) -> str:
        quote    = m.group(1)           # ' or "
        alias    = m.group(2)           # some/path (after @/)
        src_path = Path("src") / alias  # src/some/path
        rel      = os.path.relpath(src_path, test_dir).replace("\\", "/")
        return f"from {quote}{rel}{quote}"

    rewritten = re.sub(
        r'from ([\'"])@/([^\'"]+)\1',
        _replace,
        code,
    )
    if rewritten != code:
        print(f"{TAG}   ⚠ Auto-fixed @/ alias import(s) in {test_path}")
    return rewritten


def _write_test_file(test_path_str: str, code: str) -> Path | None:
    """Resolve, validate, write. Returns written path or None on error."""
    rel = test_path_str.replace("\\", "/").strip()

    # must live under tests/
    if not rel.startswith("tests/"):
        print(
            f"{TAG} SKIP scope violation: model tried to write {rel!r} "
            "(must start with 'tests/')",
            file=sys.stderr,
        )
        return None

    inner   = Path(rel[len("tests/"):])
    out     = Path(TESTS_DIR) / inner
    tests_root = Path(TESTS_DIR).resolve()

    try:
        resolved = out.resolve()
    except Exception:
        resolved = out.absolute()

    if tests_root not in resolved.parents and resolved != tests_root:
        print(f"{TAG} SKIP path traversal rejected: {rel}", file=sys.stderr)
        return None

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code, encoding="utf-8")
    track_write(out)
    return out


def generate_test_for_file(
    *,
    src_rel: str,
    test_path: str,
    stack: dict[str, Any] | None,
    task: dict[str, Any] | None,
    codebase_map: str,
    force: bool,
) -> dict[str, Any]:
    """
    Generate and write one test file.
    Returns a result dict: {src, test_path, status, note}.
    """
    src_abs = Path(SRC_DIR) / src_rel.removeprefix("src/")
    src_content = _read_text(src_abs)
    if not src_content:
        return {"src": src_rel, "test_path": test_path, "status": "skipped", "note": "src file not found"}

    # Resolve existing stub
    inner      = test_path.removeprefix("tests/")
    stub_abs   = Path(TESTS_DIR) / inner
    stub_content = _read_text(stub_abs)

    # Skip if already has real tests and not forced
    if stub_content and not _is_stub(stub_content) and not force:
        print(f"{TAG}   SKIP {test_path} — already has real tests (use --force to overwrite)")
        return {"src": src_rel, "test_path": test_path, "status": "skipped", "note": "already implemented"}

    system = _build_system_prompt(stack)
    user   = _build_user_message(
        src_rel=src_rel,
        test_path=test_path,
        src_content=src_content,
        stub_content=stub_content,
        task=task,
        codebase_map=codebase_map,
        relative_import=_compute_relative_import(test_path, src_rel),
    )

    approx_tokens = (len(system) + len(user)) // 4
    if approx_tokens > 28000:
        print(
            f"{TAG}   ⚠ Large prompt for {src_rel}: ~{approx_tokens:,} tokens",
            file=sys.stderr,
        )

    print(f"{TAG}   → Generating test: {test_path} …")
    try:
        result, _ = call_llm_json(
            ROLE,
            system,
            user,
            temperature=0.1,
            max_tokens=16384,
            retries=2,
            backoff=False,
            caller_file=__file__,
            label=f"{TAG} {get_model(ROLE)}",
        )
    except Exception as exc:
        print(f"{TAG}   FAILED to generate {test_path}: {exc}", file=sys.stderr)
        return {"src": src_rel, "test_path": test_path, "status": "failed", "note": str(exc)}

    out_path_claim = result.get("file_path", test_path)
    code = result.get("code")

    if not isinstance(code, str) or not code.strip():
        return {"src": src_rel, "test_path": test_path, "status": "failed", "note": "model returned empty code"}

    code    = _rewrite_alias_imports(code, test_path)
    written = _write_test_file(out_path_claim, code)
    if written:
        print(f"{TAG}   WROTE {written}")
        return {"src": src_rel, "test_path": str(written.relative_to(artifact_root())), "status": "written", "note": ""}
    else:
        return {"src": src_rel, "test_path": test_path, "status": "failed", "note": "write rejected (scope/path error)"}


# ════════════════════════════════════════════════════════════════════════════
# Main run
# ════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    plan         = _load_full_plan()
    stack        = plan.get("stack") if isinstance(plan.get("stack"), dict) else None
    task_index   = _build_task_index(plan)
    codebase_map = _read_text(Path(ABSORBER_CODEBASE_MD))

    if codebase_map:
        print(f"{TAG} Codebase map loaded ({len(codebase_map)} chars)")
    else:
        print(f"{TAG} WARNING: absorber/codebase_map.md not found — model gets less structural context")

    # Determine file list
    if args.only_files:
        src_rels = [f.strip() for f in args.only_files.split(",") if f.strip()]
        print(f"{TAG} --only-files mode: {len(src_rels)} file(s)")
    else:
        src_rels = _scan_src_files()
        print(f"{TAG} Scanned {len(src_rels)} src file(s)")

    written_records: list[dict] = []
    skipped_records: list[dict] = []

    for src_rel in src_rels:
        task      = task_index.get(src_rel)
        test_path = _infer_test_path(src_rel)

        if not _should_test(src_rel, task):
            print(f"{TAG}   SKIP {src_rel} — not testable (config/no ACs/no subtasks)")
            skipped_records.append({"src": src_rel, "test_path": test_path, "reason": "not testable"})
            continue

        result = generate_test_for_file(
            src_rel=src_rel,
            test_path=test_path,
            stack=stack,
            task=task,
            codebase_map=codebase_map,
            force=args.force,
        )

        if result["status"] == "written":
            written_records.append(result)
        else:
            skipped_records.append(result)

    return written_records, skipped_records


# ════════════════════════════════════════════════════════════════════════════
# Manifest writing
# ════════════════════════════════════════════════════════════════════════════

def _write_manifest(written: list[dict], skipped: list[dict]) -> None:
    _ensure_tw_dir()
    token_sum  = cost_summary()
    cost_total = token_sum.get("total_cost_usd") if isinstance(token_sum, dict) else None

    record: dict[str, Any] = {
        "generated_at":    _utc_now_iso(),
        "role":            ROLE,
        "model":           get_model(ROLE),
        "files_written":   len(written),
        "files_skipped":   len(skipped),
        "cost":            cost_total,
        "written":         written,
        "skipped":         skipped,
    }

    manifest = _tw_manifest()
    manifest.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(manifest)
    print(f"\n{TAG} Test-writer manifest → {manifest}")

    # Append to log
    log = _tw_manifest_log()
    entries: list[Any] = []
    if log.exists():
        try:
            track_read(log)
            data    = json.loads(log.read_text(encoding="utf-8"))
            entries = data.get("entries", []) if isinstance(data, dict) else data
            if not isinstance(entries, list):
                entries = []
        except Exception:
            pass

    log_entry = {
        "generated_at":  record["generated_at"],
        "files_written": record["files_written"],
        "files_skipped": record["files_skipped"],
        "cost":          cost_total,
    }
    entries.append(log_entry)
    log.write_text(json.dumps({"entries": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(log)
    print(f"{TAG} Manifest log appended ({len(entries)} entries) → {log}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="08b_test_writer.py",
        description=(
            "Standalone test writer. Reads full_plan.json + actual src/ files "
            "and writes behavior-driven tests into output/tests/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/08b_test_writer.py --project my-app
  python pipeline/08b_test_writer.py --project my-app --force
  python pipeline/08b_test_writer.py --project my-app --only-files src/hooks/useReplay.ts,src/utils/format.ts
  PIPELINE_PROJECT=my-app python pipeline/08b_test_writer.py
""",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name. Sets PIPELINE_PROJECT.",
    )
    parser.add_argument(
        "--only-files",
        default="",
        metavar="SRC_PATH[,SRC_PATH...]",
        help=(
            "Comma-separated src file paths to generate tests for. "
            "Paths should be relative like 'src/hooks/useReplay.ts'. "
            "Omit to process all testable src/ files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite test files that already contain real tests. "
            "By default, only stubs (throw 'not implemented') are overwritten."
        ),
    )
    return parser


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if os.environ.get("PIPELINE_PROJECT"):
        return
    parser.error(
        "PIPELINE_PROJECT is not set. Use --project <name> or export "
        "PIPELINE_PROJECT=<name> before running 08b_test_writer.py directly."
    )


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()
    _ensure_tw_dir()

    exit_code = 0

    try:
        written, skipped = run(args)
        _write_manifest(written, skipped)

        n_written = len(written)
        n_failed  = sum(1 for s in skipped if s.get("status") == "failed")
        n_skipped = len(skipped) - n_failed

        print(
            f"\n{TAG} Done — {n_written} test(s) written, "
            f"{n_skipped} skipped, {n_failed} failed."
        )

        if n_failed:
            print(f"{TAG} {n_failed} file(s) failed — see manifest for details.")
            exit_code = 1

    except Exception as exc:
        print(f"{TAG} ERROR: {exc}", file=sys.stderr)
        try:
            _write_manifest([], [{"status": "failed", "note": str(exc)}])
        except Exception:
            pass
        exit_code = 1

    finally:
        print_summary(TAG)
        print_artifact_summary(TAG)
        prompt_next_step(ROLE, prefix=TAG)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()