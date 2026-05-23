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
    You are a senior software architect generating a comprehensive module-centric
    blueprint from a technical spec. This blueprint is the ONLY artifact that has
    full visibility of the entire spec before any code is written. Downstream
    planner and executor agents work module-by-module and depend entirely on this
    blueprint for context.

    You will receive the canonical spec. The spec is the source of truth. Follow
    its stack, file tree, naming, testing framework, and acceptance criteria exactly.

    Your task:
    1. Read the spec carefully — file tree, module structure, constraints, open
       questions, acceptance criteria, and stack-specific quirks.
    2. Produce a SINGLE valid JSON object matching the schema below.
    3. The JSON MUST be valid and parseable by json.loads.

    ═══════════════════════════════════════════════════════════════
    OUTPUT SCHEMA (strict — no other top-level keys allowed):
    ═══════════════════════════════════════════════════════════════
    {
      "modules": [
        {
          "module": "<module_name>",
          "purpose": "<one-line description of module responsibility>",
          "depends_on": ["<other_module_name>", ...],
          "files": [
            {
              "path": "src/auth/service.ts",
              "kind": "source",
              "exports": ["functionName", "ClassName", "TypeName"],
              "quirks": [
                "Stack-specific gotcha or non-obvious constraint for this file",
                "Another constraint the executor must know before writing code"
              ],
              "acceptance_criteria": ["AC-01", "AC-02"]
            },
            {
              "path": "tests/auth/service.test.ts",
              "kind": "test",
              "tests": ["AC-01", "AC-02"]
            }
          ]
        }
      ],
      "config_files": [
        {
          "path": "index.html",
          "kind": "config",
          "note": "Vite entry point — must exist at project root, not src/"
        },
        {
          "path": "vite.config.ts",
          "kind": "config",
          "note": "Must include vite-plugin-pwa with workbox config"
        }
      ],
      "implementation_order": ["types", "config", "db", "auth", "api", "ui"],
      "open_questions": [
        {
          "id": "OQ-01",
          "question": "Exact wording from spec",
          "affects": ["module_name", "file_path"],
          "impact": "Brief description of what changes depending on the answer"
        }
      ]
    }

    ═══════════════════════════════════════════════════════════════
    RULES — FILES
    ═══════════════════════════════════════════════════════════════
    - "kind" MUST be one of: "source", "test", "config", "migration"
    - "path" is relative from project root (e.g. src/..., tests/..., config/...)
    - Every source file implied by the spec's file tree MUST appear in exactly
      one module.
    - Every test file implied by the spec MUST appear in exactly one module.
      If the spec does not list test files explicitly, infer them from the
      testing framework and source files that have testable logic (lib/*, utils/*,
      services/*, hooks/*). Pure config and entry-point files do not need tests.
    - Config files (vite.config.ts, tsconfig.json, index.html, .env.example,
      docker-compose.yml, vercel.json, netlify.toml, package.json, etc.) go in
      "config_files", NOT in modules.
    - Migration files go in a dedicated "migrations" module with kind "migration".
    - Do NOT include a "code" field anywhere. This is a blueprint, not implementation.
    - Preserve file paths exactly as specified by the spec.

    ═══════════════════════════════════════════════════════════════
    RULES — EXPORTS
    ═══════════════════════════════════════════════════════════════
    - List every named export the spec defines for this file: functions, classes,
      types, interfaces, constants, React components, hooks.
    - Use exact names from the spec. If the spec does not name exports explicitly,
      infer from context (e.g. a file named "srtParser.ts" exports "parseSrt").
    - For test files, use "tests" key (list of AC IDs) instead of "exports".
    - Omit "exports" for config and migration files.

    ═══════════════════════════════════════════════════════════════
    RULES — QUIRKS
    ═══════════════════════════════════════════════════════════════
    Include quirks for any file where the executor could make a wrong assumption.
    Common categories:
    - Stack version differences (e.g. "Tailwind v4 uses CSS-first config, no
      tailwind.config.js")
    - Import path differences between library versions
    - Required browser APIs or headers (e.g. COOP/COEP for OPFS, SharedArrayBuffer)
    - Non-obvious constraints from the spec (e.g. "primary keys are UUID strings,
      NOT auto-increment")
    - Fallback behavior that must be implemented (e.g. "if OPFS unavailable, store
      blob in IndexedDB and set usesOpfsFallback=true")
    - Deployment requirements (e.g. "vercel.json must set COOP/COEP headers for
      OPFS to work in production")
    - Omit quirks array entirely if there are no non-obvious constraints.

    ═══════════════════════════════════════════════════════════════
    RULES — ACCEPTANCE CRITERIA MAPPING
    ═══════════════════════════════════════════════════════════════
    - Map each AC from the spec to the source file most responsible for satisfying it.
    - Map each AC to the test file that verifies it.
    - An AC can appear in multiple files if multiple files contribute to it.
    - Use exact AC IDs from the spec (e.g. "AC-01", "AC-07").
    - If the spec has no AC IDs, omit acceptance_criteria and tests fields.

    ═══════════════════════════════════════════════════════════════
    RULES — DEPENDS_ON
    ═══════════════════════════════════════════════════════════════
    - List module names (not file paths) that must be fully implemented before
      this module can be implemented.
    - This defines the safe implementation order for the executor.
    - Omit or use [] if the module has no dependencies.

    ═══════════════════════════════════════════════════════════════
    RULES — IMPLEMENTATION_ORDER
    ═══════════════════════════════════════════════════════════════
    - List module names in the order they should be implemented.
    - Must be a valid topological sort of the depends_on graph.
    - Config/infra modules (types, config, db schema) come first.
    - UI/component modules come last.

    ═══════════════════════════════════════════════════════════════
    RULES — CONFIG_FILES
    ═══════════════════════════════════════════════════════════════
    Include ALL files needed for the project to start and build:
    - Entry points: index.html (Vite), main.py, cmd/main.go, etc.
    - Build config: vite.config.ts, webpack.config.js, Makefile, etc.
    - TypeScript: tsconfig.json, tsconfig.app.json, tsconfig.node.json
    - CSS framework config: tailwind.config.ts (v3), or note CSS-first for v4
    - Package management: package.json, pnpm-lock.yaml, requirements.txt, go.mod
    - Environment: .env.example (never .env with real values)
    - PWA: public/manifest.json, public/sw.js or note if generated by plugin
    - Deployment: vercel.json, netlify.toml, Dockerfile, docker-compose.yml
    - Linting/formatting: .eslintrc, .prettierrc, biome.json
    - shadcn/ui: components.json (if used)
    - CI: .github/workflows/*.yml (if mentioned in spec)
    Missing any of these for a project that needs them is an error.

    ═══════════════════════════════════════════════════════════════
    RULES — OPEN_QUESTIONS
    ═══════════════════════════════════════════════════════════════
    - Extract open questions directly from the spec (look for "OQ-", "TBD",
      "open question", "unclear", "to be decided" markers).
    - Only include questions that affect implementation decisions (not UX polish).
    - For each question, identify which modules/files are blocked or affected.
    - If the spec has no open questions, use an empty array [].

    ═══════════════════════════════════════════════════════════════
    JSON REQUIREMENTS
    ═══════════════════════════════════════════════════════════════
    - Use double quotes for all strings.
    - Do NOT include comments, trailing commas, or markdown fences.
    - Output raw JSON only — no prose before or after.
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
    Validate blueprint against the extended module-centric schema.
    Raises ValueError on any schema violation.
    Attaches computed summary to blueprint in-place.
    """
    # ── modules ──────────────────────────────────────────────────────────────
    modules = blueprint.get("modules")
    if not isinstance(modules, list):
        raise ValueError('blueprint must contain "modules" as a list')
    if not modules:
        raise ValueError("blueprint modules list is empty")

    total = 0
    source_count = 0
    test_count = 0
    config_count = 0
    migration_count = 0
    seen_paths: set[str] = set()

    for idx, mod in enumerate(modules):
        if not isinstance(mod, dict):
            raise ValueError(f"modules[{idx}] must be an object")

        module_name = mod.get("module")
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(f"modules[{idx}].module must be a non-empty string")

        purpose = mod.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"modules[{idx}].purpose must be a non-empty string")

        # depends_on is optional but must be a list of strings if present
        depends_on = mod.get("depends_on")
        if depends_on is not None:
            if not isinstance(depends_on, list):
                raise ValueError(f"modules[{idx}].depends_on must be a list")
            for didx, dep in enumerate(depends_on):
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError(
                        f"modules[{idx}].depends_on[{didx}] must be a non-empty string"
                    )

        files = mod.get("files")
        if not isinstance(files, list):
            raise ValueError(f"modules[{idx}].files must be a list")
        if not files:
            raise ValueError(
                f"modules[{idx}].files is empty — each module needs at least one file"
            )

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

            # Warn on duplicate paths (not a hard error — model may legitimately
            # share a file across modules in edge cases, but flag it)
            if normalized in seen_paths:
                print(
                    f"[06][warn] Duplicate path across modules: {path}",
                    file=sys.stderr,
                )
            seen_paths.add(normalized)

            kind = fentry.get("kind")
            if kind not in _VALID_KINDS:
                raise ValueError(
                    f"modules[{idx}].files[{fidx}].kind must be one of "
                    f"{sorted(_VALID_KINDS)}, got: {kind!r}"
                )

            # exports: optional list of strings for source files
            exports = fentry.get("exports")
            if exports is not None:
                if not isinstance(exports, list):
                    raise ValueError(
                        f"modules[{idx}].files[{fidx}].exports must be a list"
                    )
                for eidx, exp in enumerate(exports):
                    if not isinstance(exp, str):
                        raise ValueError(
                            f"modules[{idx}].files[{fidx}].exports[{eidx}] must be a string"
                        )

            # quirks: optional list of strings
            quirks = fentry.get("quirks")
            if quirks is not None:
                if not isinstance(quirks, list):
                    raise ValueError(
                        f"modules[{idx}].files[{fidx}].quirks must be a list"
                    )
                for qidx, q in enumerate(quirks):
                    if not isinstance(q, str):
                        raise ValueError(
                            f"modules[{idx}].files[{fidx}].quirks[{qidx}] must be a string"
                        )

            # acceptance_criteria / tests: optional list of strings
            for ac_key in ("acceptance_criteria", "tests"):
                ac_val = fentry.get(ac_key)
                if ac_val is not None:
                    if not isinstance(ac_val, list):
                        raise ValueError(
                            f"modules[{idx}].files[{fidx}].{ac_key} must be a list"
                        )
                    for acidx, ac in enumerate(ac_val):
                        if not isinstance(ac, str):
                            raise ValueError(
                                f"modules[{idx}].files[{fidx}].{ac_key}[{acidx}] "
                                f"must be a string"
                            )

            total += 1
            if kind == "source":
                source_count += 1
            elif kind == "test":
                test_count += 1
            elif kind == "config":
                config_count += 1
            elif kind == "migration":
                migration_count += 1

    # ── config_files ─────────────────────────────────────────────────────────
    config_files = blueprint.get("config_files")
    if config_files is not None:
        if not isinstance(config_files, list):
            raise ValueError('"config_files" must be a list')
        for cidx, cf in enumerate(config_files):
            if not isinstance(cf, dict):
                raise ValueError(f"config_files[{cidx}] must be an object")
            cf_path = cf.get("path")
            if not isinstance(cf_path, str) or not cf_path.strip():
                raise ValueError(f"config_files[{cidx}].path must be a non-empty string")
            cf_kind = cf.get("kind")
            if cf_kind not in _VALID_KINDS:
                raise ValueError(
                    f"config_files[{cidx}].kind must be one of {sorted(_VALID_KINDS)}, "
                    f"got: {cf_kind!r}"
                )
            # note is optional but must be a string if present
            cf_note = cf.get("note")
            if cf_note is not None and not isinstance(cf_note, str):
                raise ValueError(f"config_files[{cidx}].note must be a string")
            config_count += 1
            total += 1

    # ── implementation_order ─────────────────────────────────────────────────
    impl_order = blueprint.get("implementation_order")
    if impl_order is not None:
        if not isinstance(impl_order, list):
            raise ValueError('"implementation_order" must be a list')
        module_names = {m["module"] for m in modules}
        for oidx, name in enumerate(impl_order):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"implementation_order[{oidx}] must be a non-empty string"
                )
            if name not in module_names:
                print(
                    f"[06][warn] implementation_order[{oidx}] '{name}' not found in modules",
                    file=sys.stderr,
                )

    # ── open_questions ────────────────────────────────────────────────────────
    open_questions = blueprint.get("open_questions")
    if open_questions is not None:
        if not isinstance(open_questions, list):
            raise ValueError('"open_questions" must be a list')
        for oqidx, oq in enumerate(open_questions):
            if not isinstance(oq, dict):
                raise ValueError(f"open_questions[{oqidx}] must be an object")
            oq_id = oq.get("id")
            if not isinstance(oq_id, str) or not oq_id.strip():
                raise ValueError(
                    f"open_questions[{oqidx}].id must be a non-empty string"
                )
            oq_q = oq.get("question")
            if not isinstance(oq_q, str) or not oq_q.strip():
                raise ValueError(
                    f"open_questions[{oqidx}].question must be a non-empty string"
                )
            # affects and impact are optional but validated if present
            oq_affects = oq.get("affects")
            if oq_affects is not None and not isinstance(oq_affects, list):
                raise ValueError(f"open_questions[{oqidx}].affects must be a list")
            oq_impact = oq.get("impact")
            if oq_impact is not None and not isinstance(oq_impact, str):
                raise ValueError(f"open_questions[{oqidx}].impact must be a string")

    # ── attach computed summary ───────────────────────────────────────────────
    blueprint["summary"] = {
        "total": total,
        "source": source_count,
        "test": test_count,
        "config": config_count,
        "migration": migration_count,
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

    # Collect open question IDs for the log entry
    oq_ids = [
        oq.get("id", "?")
        for oq in blueprint.get("open_questions") or []
        if isinstance(oq, dict)
    ]

    entry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_version": spec_version,
        "module_count": len(blueprint.get("modules", [])),
        "config_file_count": len(blueprint.get("config_files") or []),
        "open_question_count": len(oq_ids),
        "open_question_ids": oq_ids,
        "implementation_order": blueprint.get("implementation_order") or [],
        "summary": blueprint.get("summary", {}),
        "modules": [
            {
                "module": m["module"],
                "purpose": m["purpose"],
                "depends_on": m.get("depends_on") or [],
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


def _generate_test_stub(path: str, ac_ids: list[str] | None = None) -> str:
    """
    Generate a minimal test stub based on file extension.
    Embeds AC IDs as comments so the executor knows which criteria to cover.
    """
    ac_comment = ""
    if ac_ids:
        ac_comment = f"  // Acceptance criteria: {', '.join(ac_ids)}\n"

    if path.endswith(".py"):
        module_hint = Path(path).stem.replace("test_", "")
        ac_py = f"# Acceptance criteria: {', '.join(ac_ids)}\n" if ac_ids else ""
        return (
            f'"""Tests for {module_hint}."""\n'
            f"{ac_py}"
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
            f'describe("{module_hint}", () => {{\n'
            f"{ac_comment}"
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
            f"{ac_comment}"
            f'  it("should be implemented", () => {{\n'
            f'    throw new Error("not implemented");\n'
            f"  }});\n"
            f"}});\n"
        )
    elif path.endswith(".go"):
        ac_go = f"// Acceptance criteria: {', '.join(ac_ids)}\n" if ac_ids else ""
        return (
            f"package main\n"
            f"\n"
            f'import "testing"\n'
            f"\n"
            f"{ac_go}"
            f"func TestPlaceholder(t *testing.T) {{\n"
            f'\tt.Fatal("not implemented")\n'
            f"}}\n"
        )
    else:
        ac_generic = f"// Acceptance criteria: {', '.join(ac_ids)}\n" if ac_ids else ""
        return f"{ac_generic}// TODO: implement tests\n"


def _write_test_stubs(blueprint: dict[str, Any]) -> int:
    """
    Materialize test file stubs from blueprint.
    Embeds AC IDs from the "tests" field into each stub.
    Returns count of test files written.
    """
    count = 0
    for mod in blueprint.get("modules", []):
        for fentry in mod.get("files", []):
            if fentry["kind"] != "test":
                continue

            raw_path = fentry["path"]
            ac_ids: list[str] = fentry.get("tests") or []

            rel = _safe_relative_path(raw_path)

            # Strip leading tests/ prefix since TESTS_DIR already points there
            parts = rel.parts
            if parts and parts[0] == "tests":
                rel = Path(*parts[1:]) if len(parts) > 1 else Path(rel.name)

            dest = Path(str(TESTS_DIR)) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            stub = _generate_test_stub(raw_path, ac_ids)
            dest.write_text(stub)
            track_write(dest)

            ac_suffix = f" [{', '.join(ac_ids)}]" if ac_ids else ""
            print(f"[06] [TEST] {dest}{ac_suffix}")
            count += 1

    return count


# ─────────────────────────────────────────────────────────────────────────────
# Write blueprint
# ─────────────────────────────────────────────────────────────────────────────

def write_blueprint(blueprint: dict[str, Any], spec_version: str) -> None:
    """Write blueprint.json and materialize test stubs."""
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

    # Print summary
    summary = blueprint.get("summary", {})
    config_files = blueprint.get("config_files") or []
    open_questions = blueprint.get("open_questions") or []
    impl_order = blueprint.get("implementation_order") or []

    print(
        f"[06] Done. Modules: {len(blueprint['modules'])}, "
        f"Files: {summary.get('total', 0)} "
        f"(source: {summary.get('source', 0)}, "
        f"test: {summary.get('test', 0)}, "
        f"config: {summary.get('config', 0)}, "
        f"migration: {summary.get('migration', 0)})"
    )
    if config_files:
        print(f"[06] Config files: {len(config_files)}")
        for cf in config_files:
            print(f"[06]   [CONFIG] {cf['path']}")
            if cf.get("note"):
                print(f"[06]           note: {cf['note']}")
    if open_questions:
        oq_ids = [oq.get("id", "?") for oq in open_questions if isinstance(oq, dict)]
        print(f"[06] Open questions flagged: {', '.join(oq_ids)}")
    if impl_order:
        print(f"[06] Implementation order: {' → '.join(impl_order)}")


def preview_blueprint(blueprint: dict[str, Any]) -> None:
    """Dry-run: show what would be written without writing."""
    print("[06] --dry-run: blueprint validated. No files written.")
    print(f"[06] Would write blueprint → {SCAFFOLD_JSON}")
    print()

    impl_order = blueprint.get("implementation_order") or []
    if impl_order:
        print(f"[06] Implementation order: {' → '.join(impl_order)}")
        print()

    for mod in blueprint.get("modules", []):
        deps = mod.get("depends_on") or []
        dep_str = f" (depends: {', '.join(deps)})" if deps else ""
        print(f"[06]   Module: {mod['module']}{dep_str} — {mod['purpose']}")
        for fentry in mod.get("files", []):
            kind_label = fentry["kind"].upper().ljust(9)
            exports = fentry.get("exports") or []
            tests = fentry.get("tests") or []
            acs = fentry.get("acceptance_criteria") or []
            quirks = fentry.get("quirks") or []

            line = f"[06]     [{kind_label}] {fentry['path']}"
            if exports:
                line += f"  exports: {', '.join(exports)}"
            if tests:
                line += f"  tests: {', '.join(tests)}"
            if acs:
                line += f"  AC: {', '.join(acs)}"
            print(line)
            for q in quirks:
                print(f"[06]              ⚠ {q}")
        print()

    config_files = blueprint.get("config_files") or []
    if config_files:
        print(f"[06]   Config files ({len(config_files)}):")
        for cf in config_files:
            print(f"[06]     [CONFIG   ] {cf['path']}")
            if cf.get("note"):
                print(f"[06]              note: {cf['note']}")
        print()

    open_questions = blueprint.get("open_questions") or []
    if open_questions:
        print(f"[06]   Open questions ({len(open_questions)}):")
        for oq in open_questions:
            affects = oq.get("affects") or []
            print(f"[06]     {oq.get('id', '?')}: {oq.get('question', '')}")
            if affects:
                print(f"[06]       affects: {', '.join(affects)}")
            if oq.get("impact"):
                print(f"[06]       impact:  {oq['impact']}")
        print()

    summary = blueprint.get("summary", {})
    print(
        f"[06] Total files: {summary.get('total', 0)} "
        f"(source: {summary.get('source', 0)}, "
        f"test: {summary.get('test', 0)}, "
        f"config: {summary.get('config', 0)}, "
        f"migration: {summary.get('migration', 0)})"
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

        # Debug: show top-level keys returned by model
        print(
            f"[06][debug] blueprint top-level keys: {list(blueprint.keys())}",
            file=sys.stderr,
        )
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
