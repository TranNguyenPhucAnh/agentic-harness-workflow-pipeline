"""
pipeline/09_debugger.py
=======================
Step 9 — Interactive bug-fix assistant.

Flow:
  Phase 1 — Intake:   gather_bug_report() → bug text (or Vitest output)
  Phase 2 — Locate:   LLM call 1: bug report + codebase_map + blueprint
                       → list of relevant files + reasoning
  Phase 3 — Read:     read contents of those files
  Phase 4 — Patch:    LLM call 2: bug + context + file contents
                       → full rewrite per file
  Phase 5 — Review:   print reasoning + diff preview + Y/N prompt
  Phase 6 — Apply:    if Y → write files

Writes:
  artifacts_<slug>/debugger/test_summary.json   (short-term, overwrite)
  artifacts_<slug>/output/src/**                patched files
  artifacts_<slug>/output/tests/**              patched test files

Reads:
  artifacts_<slug>/blueprint.json
  artifacts_<slug>/absorber/codebase_map.md
  artifacts_<slug>/executor/manifest.json       (scope detection only)
  artifacts_<slug>/archivist/knowledge_log.md
  artifacts_<slug>/output/src/**
  artifacts_<slug>/output/tests/**

Direct execution:
  python 09_debugger.py --project my-app
  python 09_debugger.py --project my-app --text "Button click crashes app"
  python 09_debugger.py --project my-app --auto   # skip Y/N, apply directly
  PIPELINE_PROJECT=my-app python 09_debugger.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_MD,
    ARCHIVIST_KNOWLEDGE_LOG,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    EXECUTOR_OVERWRITE_MANIFEST,
    SRC_DIR,
    TESTS_DIR,
    artifact_root,
    ensure_dirs,
)
from artifacts.models import get_model  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_summary  # noqa: E402
from modules.call_llm import call_llm_messages  # noqa: E402
from modules.drag_and_drop import gather_text_file_bundle  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402

ROLE = "debugger"
_RE_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="09_debugger.py", description="Interactive bug-fix assistant.")
    p.add_argument("--project", default=None, help="Project name.")
    p.add_argument("--text", default=None, metavar="TEXT", help="Bug description inline.")
    p.add_argument("--input", metavar="FILE", nargs="+", help="Bug report file(s).")
    p.add_argument("--auto", action="store_true", help="Skip Y/N review, apply patches directly.")
    p.add_argument("--no-interactive", action="store_true", help="Disable TTY prompts.")
    p.add_argument("--run-tests", action="store_true", help="Run Vitest first, use output as bug report.")
    p.add_argument("--verbose", action="store_true")
    # Legacy args from harness — accepted but ignored
    p.add_argument("--impl", default="primary", help=argparse.SUPPRESS)
    p.add_argument("--max-iter", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--max-cluster-attempts", type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument("--no-repair", action="store_true", help=argparse.SUPPRESS)
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if os.environ.get("PIPELINE_PROJECT"):
        return
    parser.error("PIPELINE_PROJECT is not set. Use --project <name>.")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    track_read(path)
    return path.read_text(errors="replace")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        track_read(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_path(rel: str) -> Path:
    """Resolve a relative path like src/foo.ts or tests/bar.test.ts to absolute."""
    normalized = rel.replace("\\", "/").strip()
    if normalized.startswith("src/"):
        return SRC_DIR / normalized[len("src/"):]
    if normalized.startswith("tests/"):
        return TESTS_DIR / normalized[len("tests/"):]
    return artifact_root() / normalized


def _strip_ansi(text: str) -> str:
    return _RE_ANSI.sub("", text)


def _call_llm(role: str, messages: list[dict[str, str]], max_tokens: int = 8192) -> str:
    content, _ = call_llm_messages(
        role, messages,
        retries=2, backoff=False,
        caller_file=__file__,
        label=f"[09] {get_model(role)}",
        max_tokens=max_tokens,
    )
    # === DEBUG LOG ===
    import sys as _sys
    print(f"[DEBUG debugger] _call_llm content ({len(content)} chars): {repr(content[:300])}", file=_sys.stderr)
    # === END DEBUG ===

    return content


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()

    # Strip markdown fences
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())

    # Strip tool-call hallucination: model đôi khi tự viết <tool_call>...</tool_call>
    # và <tool_response>...</tool_response> trước khi trả JSON thật.
    # Bỏ hết mọi thứ trước dấu { đầu tiên.
    brace_idx = text.find("{")
    if brace_idx > 0:
        text = text[brace_idx:]

    # Nếu có nhiều JSON objects liên tiếp (tool_call JSON + response JSON),
    # chỉ lấy object đầu tiên bằng cách tìm closing brace cân bằng.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tìm outermost balanced { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # reset và tìm tiếp
                    start = None
                    continue

    raise json.JSONDecodeError("No valid JSON object found", text, 0)

def _colored_diff(old: str, new: str, filename: str) -> str:
    """Generate unified diff string."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}", n=3)
    return "".join(diff)


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — Intake
# ════════════════════════════════════════════════════════════════════════════

def _run_vitest_for_report() -> str:
    """Run Vitest and return output as bug report text."""
    artifact_dir = artifact_root()
    output_dir = artifact_dir / "output"
    project_dir = output_dir if (output_dir / "package.json").exists() else artifact_dir

    config_candidates = [
        project_dir / "vitest.config.ts",
        project_dir / "vitest.config.js",
        artifact_dir / "vitest.config.ts",
        artifact_dir / "vitest.config.js",
    ]
    config_flag: list[str] = []
    for c in config_candidates:
        if c.exists():
            config_flag = ["--config", str(c)]
            break

    try:
        result = subprocess.run(
            ["npx", "vitest", "run", "--reporter=verbose"] + config_flag,
            cwd=project_dir,
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "CI": "true", "FORCE_COLOR": "0"},
        )
        combined = _strip_ansi(result.stdout + "\n" + result.stderr)
        if result.returncode == 0:
            print("[09] ✓ All tests pass — nothing to debug.")
            return ""
        return combined
    except Exception as exc:
        return f"Vitest failed to run: {exc}"


def phase1_intake(args: argparse.Namespace) -> str:
    """Gather bug report from CLI args, stdin, or Vitest output."""
    # If --run-tests, use Vitest output as the bug report
    if args.run_tests:
        print("[09] Phase 1 — Running Vitest to gather failure report …")
        report = _run_vitest_for_report()
        if not report:
            return ""
        print(f"[09] Vitest failures captured ({len(report)} chars)")
        return report

    # Otherwise gather from user input
    print("[09] Phase 1 — Gathering bug report …")
    try:
        bundle = gather_text_file_bundle(
            cli_text=args.text,
            cli_files=args.input or [],
            read_file_fn=lambda p: (track_read(p) or "") or p.read_text(errors="replace"),
            prompt_title="Bug report",
            prompt_body=(
                "Describe the bug: what you observed, steps to reproduce, expected behavior.\n"
                "Or drag-and-drop a file with test output / error log.\n"
                "Press Enter twice to submit."
            ),
            attachment_prompt="Attach files if needed",
            default_attachment_only_prompt="Analyze the attached file(s).",
            allow_interactive=not args.no_interactive,
            ask_for_attachments_after_text=True,
        )
        text = bundle.text.strip()
        if text:
            print(f"[09] Bug report received ({len(text)} chars)")
        return text
    except RuntimeError:
        return ""


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Locate (LLM call 1)
# ════════════════════════════════════════════════════════════════════════════

LOCATE_SYSTEM = """\
You are a senior developer analyzing a bug report against a codebase.

Given:
- A bug report (test failures, error messages, or user description)
- A codebase map showing all files and their structure
- A blueprint showing module dependencies, exports, and behavioral quirks

Your job: identify which files need to be read to understand and fix the bug.

Return ONLY valid JSON (no markdown fences, no explanation, no tool calls):
{
  "reasoning": "1-3 sentences explaining your analysis",
  "files_to_read": ["src/hooks/useLoopRecorder.ts", "tests/hooks/useLoopRecorder.test.ts", "src/types.ts"],
  "likely_root_cause": "one sentence"
}

RULES:
- Include BOTH the source file and its test file if relevant.
- Include dependency files (types, utils) that the buggy file imports.
- Maximum 8 files — focus on what's needed to fix the bug.
- Paths must match exactly what appears in the codebase map / blueprint.
- CRITICAL: Output ONLY the raw JSON object. No prose before it, no prose after it.
- CRITICAL: Do NOT simulate reading files. Do NOT write <tool_call>, <tool_response>,
  or any XML/function-call tags. You already have all the information you need in the
  codebase map above. Your only output is the JSON object.
- The JSON must start with {{ and end with }}. Nothing else.
"""

def phase2_locate(bug_report: str) -> dict[str, Any]:
    """LLM call 1: identify which files are relevant to the bug."""
    print("[09] Phase 2 — Locating relevant files …")

    codebase_map = _read_text(ABSORBER_CODEBASE_MD)

    # Load manifest for recently-changed files context
    manifest_raw = ""
    if EXECUTOR_OVERWRITE_MANIFEST.exists():
        manifest_raw = _read_text(EXECUTOR_OVERWRITE_MANIFEST).strip()

    blueprint_path = artifact_root() / "blueprint.json"
    if not blueprint_path.exists():
        blueprint_path = artifact_root() / "planner" / "blueprint.json"
    blueprint_raw = _read_text(blueprint_path) if blueprint_path.exists() else ""

    # Build user prompt.
    # - codebase_map: full structural overview, model needs it to locate files
    # - manifest: only the file list (not full content) — enough for context
    # - blueprint: skip — too large, not needed for file location
    parts = [f"## Bug Report\n\n{bug_report}"]

    if codebase_map:
        parts.append(f"## Codebase Map\n\n{codebase_map}")

    if EXECUTOR_OVERWRITE_MANIFEST.exists():
        try:
            manifest_data = json.loads(manifest_raw) if manifest_raw else {}
            file_list = (
                manifest_data.get("files_written")
                or manifest_data.get("written")
                or []
            )
            if file_list:
                lines = []
                for f in file_list:
                    lines.append(f"  - {f}" if isinstance(f, str) else f"  - {f.get('path', str(f))}")
                parts.append("## Recently implemented files\n\n" + "\n".join(lines))
        except Exception:
            pass  # manifest unreadable — skip silently

    user_content = "\n\n---\n\n".join(parts)

    messages = [
        {"role": "system", "content": LOCATE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    raw = _call_llm("debugger", messages, max_tokens=8192)
    result = _parse_json_response(raw)

    files = result.get("files_to_read", [])
    reasoning = result.get("reasoning", "")
    root_cause = result.get("likely_root_cause", "")

    print(f"[09] Reasoning: {reasoning}")
    print(f"[09] Likely root cause: {root_cause}")
    print(f"[09] Files to read ({len(files)}):")
    for f in files:
        print(f"     • {f}")

    return result


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — Read
# ════════════════════════════════════════════════════════════════════════════

def phase3_read(files: list[str]) -> dict[str, str]:
    """Read the contents of identified files."""
    print(f"[09] Phase 3 — Reading {len(files)} file(s) …")

    contents: dict[str, str] = {}
    for rel in files:
        path = _resolve_path(rel)
        if path.exists():
            content = path.read_text(errors="replace")
            track_read(path)
            contents[rel] = content
            lines = len(content.splitlines())
            print(f"     ✓ {rel} ({lines} lines)")
        else:
            contents[rel] = f"// FILE NOT FOUND: {rel}"
            print(f"     ✗ {rel} (not found)")

    return contents


# ════════════════════════════════════════════════════════════════════════════
# Phase 4 — Patch (LLM call 2)
# ════════════════════════════════════════════════════════════════════════════

PATCH_SYSTEM = """\
You are a senior TypeScript developer fixing a bug.

Given:
- A bug report describing the problem
- The relevant source and test files
- Analysis of the likely root cause

Your job: produce corrected file contents that fix the bug.

Return ONLY valid JSON (no markdown fences, no explanation, no tool calls):
{
  "reasoning": "explain what was wrong and what you changed",
  "patches": [
    {
      "file_path": "src/hooks/useLoopRecorder.ts",
      "action": "rewrite",
      "code": "<full corrected file content>"
    }
  ]
}

RULES:
- "code" must be the COMPLETE file content (not a diff, not a snippet).
- Only include files you actually changed.
- Do NOT change files unnecessarily — minimal fix only.
- If a test is wrong (testing incorrect behavior), fix the test.
- If source code is wrong, fix the source.
- Maximum 4 files per response.
- Do NOT use any tools. Do NOT output <tool_call> or <tool_response> tags. You have no tools available.
- Output ONLY the JSON object, nothing else before or after it.
"""


def phase4_patch(bug_report: str, locate_result: dict[str, Any], file_contents: dict[str, str]) -> dict[str, Any]:
    """LLM call 2: generate patches for the identified files."""
    print("[09] Phase 4 — Generating patches …")

    # Build file content block
    file_blocks = []
    for rel, content in file_contents.items():
        file_blocks.append(f"### {rel}\n```typescript\n{content}\n```")

    user_parts = [
        f"## Bug Report\n\n{bug_report}",
        f"## Analysis\n\nReasoning: {locate_result.get('reasoning', '')}\nLikely root cause: {locate_result.get('likely_root_cause', '')}",
        "## File Contents\n\n" + "\n\n".join(file_blocks),
    ]

    # Load blueprint quirks for relevant modules
    blueprint_path = artifact_root() / "blueprint.json"
    if not blueprint_path.exists():
        blueprint_path = artifact_root() / "planner" / "blueprint.json"
    if blueprint_path.exists():
        try:
            bp = json.loads(blueprint_path.read_text(encoding="utf-8"))
            # Find quirks for files being patched
            quirks_block = _extract_relevant_quirks(bp, list(file_contents.keys()))
            if quirks_block:
                user_parts.append(f"## Blueprint Quirks (correct behavior)\n\n{quirks_block}")
        except Exception:
            pass

    user_content = "\n\n---\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": PATCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    raw = _call_llm("debugger", messages, max_tokens=16384)
    result = _parse_json_response(raw)

    patches = result.get("patches", [])
    reasoning = result.get("reasoning", "")

    print(f"[09] Patch reasoning: {reasoning}")
    print(f"[09] Files to patch: {len(patches)}")
    for p in patches:
        print(f"     • {p.get('file_path', '?')}")

    return result


def _extract_relevant_quirks(bp: dict[str, Any], files: list[str]) -> str:
    """Extract quirks from blueprint for the given files."""
    parts: list[str] = []
    modules = bp.get("modules", [])

    for module in modules:
        for f in module.get("files", []):
            if f.get("path") in files:
                quirks = f.get("quirks", [])
                if quirks:
                    parts.append(f"**{f['path']}**:")
                    for q in quirks:
                        parts.append(f"  - {q}")

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 — Review
# ════════════════════════════════════════════════════════════════════════════

def phase5_review(
    patch_result: dict[str, Any],
    file_contents: dict[str, str],
    auto: bool,
) -> bool:
    """Show diffs and ask for confirmation. Returns True if user approves."""
    patches = patch_result.get("patches", [])
    reasoning = patch_result.get("reasoning", "")

    if not patches:
        print("[09] No patches generated.")
        return False

    print("\n" + "═" * 70)
    print("[09] Phase 5 — Review")
    print("═" * 70)
    print(f"\nReasoning: {reasoning}\n")

    for patch in patches:
        file_path = patch.get("file_path", "")
        new_code = patch.get("code", "")

        if not file_path or not new_code:
            continue

        old_code = file_contents.get(file_path, "")
        diff = _colored_diff(old_code, new_code, file_path)

        if not diff:
            print(f"  {file_path}: no changes")
            continue

        print(f"┌─ {file_path} ─────────────────────────────────")
        # Show abbreviated diff (max 60 lines)
        diff_lines = diff.splitlines()
        if len(diff_lines) > 60:
            for line in diff_lines[:50]:
                print(f"│ {line}")
            print(f"│ ... ({len(diff_lines) - 50} more lines)")
        else:
            for line in diff_lines:
                print(f"│ {line}")
        print(f"└{'─' * 50}")
        print()

    if auto:
        print("[09] --auto mode: applying patches without confirmation.")
        return True

    # Interactive confirmation
    if not sys.stdin.isatty():
        print("[09] Non-interactive mode: applying patches.")
        return True

    try:
        answer = input("[09] Apply these patches? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\n[09] Cancelled.")
        return False


# ════════════════════════════════════════════════════════════════════════════
# Phase 6 — Apply
# ════════════════════════════════════════════════════════════════════════════

def phase6_apply(patch_result: dict[str, Any]) -> list[str]:
    """Write patched files to disk. Returns list of written paths."""
    patches = patch_result.get("patches", [])
    written: list[str] = []

    print("[09] Phase 6 — Applying patches …")

    for patch in patches:
        file_path = patch.get("file_path", "")
        new_code = patch.get("code", "")

        if not file_path or not new_code:
            continue

        out_path = _resolve_path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_code)
        track_write(out_path)
        written.append(file_path)
        print(f"     ✓ {file_path}")

    return written


# ════════════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════════════

def _write_summary(
    bug_report: str,
    locate_result: dict[str, Any],
    patch_result: dict[str, Any],
    applied: bool,
    written_files: list[str],
) -> None:
    summary = {
        "status": "APPLIED" if applied else "REJECTED",
        "bug_report_length": len(bug_report),
        "files_analyzed": locate_result.get("files_to_read", []),
        "likely_root_cause": locate_result.get("likely_root_cause", ""),
        "patch_reasoning": patch_result.get("reasoning", ""),
        "files_patched": written_files,
        "patches_count": len(patch_result.get("patches", [])),
    }

    DEBUGGER_OVERWRITE_TEST_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DEBUGGER_OVERWRITE_TEST_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    track_write(DEBUGGER_OVERWRITE_TEST_SUMMARY)
    print(f"[09] Summary → {DEBUGGER_OVERWRITE_TEST_SUMMARY}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()

    exit_code = 0

    try:
        # Phase 1 — Intake
        bug_report = phase1_intake(args)
        if not bug_report:
            print("[09] No bug report provided and tests pass. Nothing to do.")
            sys.exit(0)

        # Phase 2 — Locate
        locate_result = phase2_locate(bug_report)
        files_to_read = locate_result.get("files_to_read", [])
        if not files_to_read:
            print("[09] LLM could not identify relevant files. Exiting.")
            sys.exit(1)

        # Phase 3 — Read
        file_contents = phase3_read(files_to_read)

        # Phase 4 — Patch
        patch_result = phase4_patch(bug_report, locate_result, file_contents)
        if not patch_result.get("patches"):
            print("[09] No patches generated. Exiting.")
            exit_code = 1
        else:
            # Phase 5 — Review
            approved = phase5_review(patch_result, file_contents, auto=args.auto)

            if approved:
                # Phase 6 — Apply
                written = phase6_apply(patch_result)
                _write_summary(bug_report, locate_result, patch_result, applied=True, written_files=written)
                print(f"\n[09] ✓ Done — {len(written)} file(s) patched.")
            else:
                _write_summary(bug_report, locate_result, patch_result, applied=False, written_files=[])
                print("[09] Patches rejected. No files modified.")
                exit_code = 1

    except Exception as exc:
        print(f"[09] ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1

    finally:
        print_summary("[09]")
        print_artifact_summary("[09]")
        prompt_next_step(ROLE, prefix="[09]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()