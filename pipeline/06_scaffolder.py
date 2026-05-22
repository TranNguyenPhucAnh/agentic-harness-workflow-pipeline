"""
pipeline/06_scaffolder.py
=========================
Step 6 — Generate module-centric blueprint from canonical spec.

This step is intended for the FULL pipeline. Mini runs should normally skip this
step and use the mini planner/implementer flow instead.

Writes (owner: scaffolder):
  artifacts_<slug>/scaffolder/blueprint.json      ← short-term, overwrite
  artifacts_<slug>/scaffolder/skeleton_log.json   ← long-term, append-only
  artifacts_<slug>/output/tests/**                ← test stubs (skeleton only)

Reads:
  artifacts_<slug>/spec/specwright_spec_<slug>.md

Direct execution:
  python 06_scaffolder.py --project my-app
  PIPELINE_PROJECT=my-app python 06_scaffolder.py

Required environment:
  Xem artifacts/models.py — API key được lấy tự động theo role "scaffolder".

At the end of each run, prints:
  - artifacts/files read
  - artifacts/files created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# === WRITE AUTHORITY: scaffolder ===
# OWNS  : artifacts_<slug>/scaffolder/blueprint.json (short-term, overwrite)
#          artifacts_<slug>/scaffolder/skeleton_log.json (long-term, append-only)
#          artifacts_<slug>/output/tests/** (test stubs)
# READS : artifacts_<slug>/spec/specwright_spec_<slug>.md (upstream-aware, specwright)

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.call_llm import call_llm_json
from modules.post_interactive import prompt_next_step  # noqa: E402
from artifacts.paths import (  # noqa: E402
    SCAFFOLD_JSON,
    SCAFFOLDER_SKELETON_LOG,
    TESTS_DIR,
    ensure_dirs,
    get_spec_path,
)


ROLE = "scaffolder"
MAX_OUTPUT_TOKENS = 65536


# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior software architect generating a module-centric blueprint
    from a technical spec.

    You will receive the canonical spec. The spec is the source of truth. Follow
    its stack, file tree, naming, testing framework, and acceptance criteria.

    Your task:
    1. Read the spec carefully, especially the file tree and module structure.
    2. Produce a SINGLE valid JSON object matching the schema below.
    3. The JSON MUST be valid and parseable by json.loads.

    OUTPUT SCHEMA (strict — no other top-level keys allowed):
    {
      "modules": [
        {
          "module": "<module_name>",
          "purpose": "<one-line description of module responsibility>",
          "files": [
            { "path": "src/auth/service.py", "kind": "source" },
            { "path": "tests/auth/test_service.py", "kind": "test" }
          ]
        }
      ]
    }

    RULES:
    - "kind" MUST be one of: "source", "test", "config", "migration"
    - "path" is relative from project root (e.g. src/..., tests/..., config/...)
    - Group files by logical module. Each module has a clear single responsibility.
    - Do NOT include a "code" field. This is a blueprint, not implementation.
    - Do NOT add files not implied by the spec's file tree or architecture.
    - Preserve file paths exactly as specified by the spec.
    - Every source file in the spec MUST appear in exactly one module.
    - Every test file in the spec MUST appear in exactly one module.
    - If the spec defines config files (docker-compose, .env.example, etc.),
      include them with kind "config".
    - If the spec defines migrations, include them with kind "migration".

    JSON requirements:
    - Use double quotes for all strings.
    - Do NOT include comments, trailing commas, or markdown fences.
    - Output raw JSON only.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# CLI / project setup
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="06_scaffolder.py",
        description="Generate module-centric blueprint from the canonical spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python 06_scaffolder.py --project my-app
              PIPELINE_PROJECT=my-app python 06_scaffolder.py

              python 06_scaffolder.py --project my-app --dry-run

            Model/provider config: xem artifacts/models.py, role "scaffolder".
        """),
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
        "--dry-run",
        action="store_true",
        help="Call model and validate blueprint, but do not write files.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for transient model/API failures. Default: 5.",
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
        "PIPELINE_PROJECT=<name> before running 06_scaffolder.py directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model call
# ─────────────────────────────────────────────────────────────────────────────

# call_scaffolder removed — use call_llm_json() from modules.call_llm



# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing / validation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict[str, Any]:
    """
    Robust JSON extraction from model output.

    Handles markdown fences, top-level arrays, and regex fallback.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"[06][warn] Primary JSON parse failed: {exc}", file=sys.stderr)
    else:
        if isinstance(parsed, list):
            print(
                "[06][warn] Model returned top-level array — auto-wrapping into "
                '{"modules": [...]}',
                file=sys.stderr,
            )
            return {"modules": parsed}
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Model returned JSON, but top-level value is not an object or array "
                f"(got {type(parsed).__name__})."
            )
        return parsed

    # Fallback: extract outermost { ... }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            print(
                f"[06][error] JSON parse failed after extracting outer object: {exc}",
                file=sys.stderr,
            )
            print(f"[06][error] Raw output first 1000 chars:\n{cleaned[:1000]}", file=sys.stderr)
            raise SystemExit(1) from exc

        if isinstance(parsed, list):
            return {"modules": parsed}
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Extracted JSON top-level value is not an object or array "
                f"(got {type(parsed).__name__})."
            )
        return parsed

    print("[06][error] No JSON object found in model response.", file=sys.stderr)
    print(f"[06][error] Raw output first 1000 chars:\n{cleaned[:1000]}", file=sys.stderr)
    raise SystemExit(1)


_VALID_KINDS = {"source", "test", "config", "migration"}


def _validate_blueprint(blueprint: dict[str, Any]) -> None:
    """
    Validate blueprint against the module-centric schema.
    Raises ValueError on any schema violation.
    """
    modules = blueprint.get("modules")
    if not isinstance(modules, list):
        raise ValueError('blueprint must contain "modules" as a list')

    if not modules:
        raise ValueError("blueprint modules list is empty")

    total = 0
    source_count = 0
    test_count = 0

    for idx, mod in enumerate(modules):
        if not isinstance(mod, dict):
            raise ValueError(f"modules[{idx}] must be an object")

        module_name = mod.get("module")
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(f"modules[{idx}].module must be a non-empty string")

        purpose = mod.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"modules[{idx}].purpose must be a non-empty string")

        files = mod.get("files")
        if not isinstance(files, list):
            raise ValueError(f"modules[{idx}].files must be a list")

        if not files:
            raise ValueError(f"modules[{idx}].files is empty — each module needs at least one file")

        for fidx, fentry in enumerate(files):
            if not isinstance(fentry, dict):
                raise ValueError(f"modules[{idx}].files[{fidx}] must be an object")

            path = fentry.get("path")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    f"modules[{idx}].files[{fidx}].path must be a non-empty string"
                )

            # Reject path traversal and absolute paths
            normalized = path.replace("\\", "/").strip()
            if Path(normalized).is_absolute():
                raise ValueError(
                    f"modules[{idx}].files[{fidx}].path must be relative: {path}"
                )
            if ".." in Path(normalized).parts:
                raise ValueError(
                    f"modules[{idx}].files[{fidx}].path contains path traversal: {path}"
                )

            kind = fentry.get("kind")
            if kind not in _VALID_KINDS:
                raise ValueError(
                    f"modules[{idx}].files[{fidx}].kind must be one of "
                    f"{sorted(_VALID_KINDS)}, got: {kind!r}"
                )

            total += 1
            if kind == "source":
                source_count += 1
            elif kind == "test":
                test_count += 1

    # Attach summary (computed, not model-provided)
    blueprint["summary"] = {
        "total": total,
        "source": source_count,
        "test": test_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spec version extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_spec_version(spec_content: str) -> str:
    """Extract version string from spec content, fallback to 'unknown'."""
    match = re.search(r"[Vv]ersion[:\s]+([^\s\n]+)", spec_content)
    if match:
        return match.group(1)
    match = re.search(r"^#+.*?(v\d+\S*)", spec_content, re.MULTILINE)
    if match:
        return match.group(1)
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton log (long-term, append-only)
# ─────────────────────────────────────────────────────────────────────────────

def _append_skeleton_log(blueprint: dict[str, Any], spec_version: str) -> None:
    """Append a summary entry to the long-term skeleton log."""
    log_path = Path(str(SCAFFOLDER_SKELETON_LOG))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_version": spec_version,
        "module_count": len(blueprint.get("modules", [])),
        "summary": blueprint.get("summary", {}),
        "modules": [
            {
                "module": m["module"],
                "purpose": m["purpose"],
                "file_count": len(m.get("files", [])),
            }
            for m in blueprint.get("modules", [])
        ],
    }

    # Append to {"entries": [...]} wrapper — consistent with all other pipeline logs.
    entries: list[dict] = []
    if log_path.exists():
        try:
            track_read(SCAFFOLDER_SKELETON_LOG)
            data = json.loads(log_path.read_text())
            if isinstance(data, dict):
                entries = data.get("entries", [])
            elif isinstance(data, list):
                # Migrate legacy raw-list format on first read.
                entries = data
        except (json.JSONDecodeError, OSError):
            pass

    entries.append(entry)
    log_path.write_text(json.dumps({"entries": entries}, indent=2))
    track_write(SCAFFOLDER_SKELETON_LOG)


# ─────────────────────────────────────────────────────────────────────────────
# Test stub writer
# ─────────────────────────────────────────────────────────────────────────────

def _safe_relative_path(raw_path: str) -> Path:
    """Convert to safe relative path. Rejects absolute and traversal paths."""
    normalized = raw_path.replace("\\", "/").strip()
    rel = Path(normalized)

    if not normalized:
        raise ValueError("empty path is not allowed")
    if rel.is_absolute():
        raise ValueError(f"absolute path is not allowed: {raw_path}")
    if any(part == ".." for part in rel.parts):
        raise ValueError(f"path traversal is not allowed: {raw_path}")

    return rel


def _generate_test_stub(path: str) -> str:
    """Generate a minimal test stub based on file extension."""
    if path.endswith(".py"):
        module_hint = Path(path).stem.replace("test_", "")
        return (
            f'"""Tests for {module_hint}."""\n'
            f"\n"
            f"import pytest\n"
            f"\n"
            f"\n"
            f"class Test{module_hint.title().replace('_', '')}:\n"
            f'    """Test suite for {module_hint}."""\n'
            f"\n"
            f"    def test_placeholder(self):\n"
            f'        raise NotImplementedError("test not implemented")\n'
        )
    elif path.endswith((".ts", ".tsx")):
        module_hint = Path(path).stem.replace(".test", "").replace(".spec", "")
        return (
            f'import {{ describe, it, expect }} from "vitest";\n'
            f"\n"
            f'describe("{module_hint}", () => {{\n'
            f'  it("should be implemented", () => {{\n'
            f'    throw new Error("not implemented");\n'
            f"  }});\n"
            f"}});\n"
        )
    elif path.endswith((".js", ".jsx")):
        module_hint = Path(path).stem.replace(".test", "").replace(".spec", "")
        return (
            f'const {{ describe, it, expect }} = require("@jest/globals");\n'
            f"\n"
            f'describe("{module_hint}", () => {{\n'
            f'  it("should be implemented", () => {{\n'
            f'    throw new Error("not implemented");\n'
            f"  }});\n"
            f"}});\n"
        )
    elif path.endswith(".go"):
        return (
            f"package main\n"
            f"\n"
            f'import "testing"\n'
            f"\n"
            f"func TestPlaceholder(t *testing.T) {{\n"
            f'\tt.Fatal("not implemented")\n'
            f"}}\n"
        )
    else:
        return f"// TODO: implement tests\n"


def _write_test_stubs(blueprint: dict[str, Any]) -> int:
    """
    Materialize test file stubs from blueprint.
    Returns count of test files written.
    """
    count = 0
    for mod in blueprint.get("modules", []):
        for fentry in mod.get("files", []):
            if fentry["kind"] != "test":
                continue

            raw_path = fentry["path"]
            rel = _safe_relative_path(raw_path)

            # Strip leading tests/ prefix since TESTS_DIR already points there
            parts = rel.parts
            if parts and parts[0] == "tests":
                rel = Path(*parts[1:]) if len(parts) > 1 else Path(rel.name)

            dest = Path(str(TESTS_DIR)) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            stub = _generate_test_stub(raw_path)
            dest.write_text(stub)
            track_write(dest)

            print(f"[06] [TEST] {dest}")
            count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Write blueprint
# ─────────────────────────────────────────────────────────────────────────────

def write_blueprint(blueprint: dict[str, Any], spec_version: str) -> None:
    """Write blueprint.json and materialize test stubs."""
    # Add metadata
    blueprint["generated_at"] = datetime.now(timezone.utc).isoformat()
    blueprint["spec_version"] = spec_version

    # Write blueprint JSON
    blueprint_path = Path(str(SCAFFOLD_JSON))
    blueprint_path.parent.mkdir(parents=True, exist_ok=True)
    blueprint_path.write_text(json.dumps(blueprint, indent=2))
    track_write(SCAFFOLD_JSON)
    print(f"[06] Blueprint → {SCAFFOLD_JSON}")

    # Materialize test stubs
    test_count = _write_test_stubs(blueprint)
    print(f"[06] Test stubs written: {test_count}")

    # Append to long-term log
    _append_skeleton_log(blueprint, spec_version)

    summary = blueprint.get("summary", {})
    print(
        f"[06] Done. Modules: {len(blueprint['modules'])}, "
        f"Files: {summary.get('total', 0)} "
        f"(source: {summary.get('source', 0)}, test: {summary.get('test', 0)})"
    )


def preview_blueprint(blueprint: dict[str, Any]) -> None:
    """Dry-run: show what would be written without writing."""
    print("[06] --dry-run: blueprint validated. No files written.")
    print(f"[06] Would write blueprint → {SCAFFOLD_JSON}")

    for mod in blueprint.get("modules", []):
        print(f"[06]   Module: {mod['module']} — {mod['purpose']}")
        for fentry in mod.get("files", []):
            print(f"[06]     [{fentry['kind'].upper():6s}] {fentry['path']}")

    summary = blueprint.get("summary", {})
    print(
        f"[06] Total files: {summary.get('total', 0)} "
        f"(source: {summary.get('source', 0)}, test: {summary.get('test', 0)})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    exit_code = 0

    try:
        parser = _build_parser()
        args = parser.parse_args()

        _configure_project(args.project, parser)
        ensure_dirs()

        spec_path = get_spec_path()
        if not spec_path.exists():
            print(f"[06][error] canonical spec not found: {spec_path}", file=sys.stderr)
            print(
                "[06][hint] This is a full-only scaffold step. Ensure specwright "
                "created the canonical spec before running 06_scaffolder.py.",
                file=sys.stderr,
            )
            sys.exit(1)

        track_read(spec_path)
        spec = spec_path.read_text(errors="replace")
        spec_version = _extract_spec_version(spec)

        _model = get_model(ROLE)
        _prov  = get_provider(ROLE)
        print(f"[06] Calling model: {_model} (provider: {_prov}) …")
        blueprint, _ = call_llm_json(
            ROLE,
            SYSTEM_PROMPT,
            f"Here is the canonical spec:\n\n{spec}",
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
            retries=args.max_retries,
            caller_file=__file__,
            label="[06] scaffold",
        )

        # DEBUG
        if blueprint.get("modules"):
            print(
                f"[06][debug] modules[0] keys: {list(blueprint['modules'][0].keys())}",
                file=sys.stderr,
            )

        try:
            _validate_blueprint(blueprint)
        except ValueError as exc:
            print(f"[06][error] Invalid blueprint: {exc}", file=sys.stderr)
            print(
                f"[06][debug] Blueprint first 1000 chars:\n"
                f"{json.dumps(blueprint, indent=2)[:1000]}",
                file=sys.stderr,
            )
            sys.exit(1)

        if args.dry_run:
            preview_blueprint(blueprint)
            return

        write_blueprint(blueprint, spec_version)

    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else 1

    except Exception as exc:
        print(f"[06][error] Scaffolder failed: {exc}", file=sys.stderr)
        exit_code = 1

    finally:
        print_summary("[06]")
        print_artifact_summary("[06]")
        prompt_next_step(ROLE, prefix="[06]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
