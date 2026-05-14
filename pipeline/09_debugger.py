"""
pipeline/09_debugger.py
=======================
Step 9 — Test / verify / iterate.

FULL mode:
  Clustered repair loop:
    - Run Vitest.
    - Parse failing clusters.
    - Apply static fixes.
    - Dispatch surface/component bugs to primary repair model (role: debugger).
    - Dispatch hook/data/type/util logic bugs to secondary repair model (role: debugger_secondary).
    - Write debugger_overwrite_test_summary.json.

MINI mode:
  Safe targeted verification:
    - Detect scope from execution/executor_overwrite_manifest.json.
    - Do not require the canonical spec.
    - Build context from clarificator_requirement_synthesis.md,
      planner_mini_execution_plan.json, planner_mini_impact_analysis.json,
      executor_overwrite_manifest.json.
    - Dispatch lightweight verifiers by target file extension:
        .py                  → py_compile
        .json                → json parse
        .yaml/.yml           → yaml parse if PyYAML is installed, else basic check
        .toml                → tomllib parse
        .ts/.tsx/.js/.jsx    → vitest if package.json exists, otherwise syntax skipped
    - For TS/Vitest projects, can still run the full Vitest repair loop.

Writes:
  artifacts_<slug>/execution/debugger_overwrite_test_summary.json
  artifacts_<slug>/src/**    repair loop patches, if needed
  artifacts_<slug>/tests/**  fragile-test repair, if auditor allows

Reads:
  artifacts_<slug>/specwright_spec_<slug>.md
  artifacts_<slug>/state/planner_full_execution_plan.json
  artifacts_<slug>/state/planner_mini_execution_plan.json
  artifacts_<slug>/state/planner_mini_impact_analysis.json
  artifacts_<slug>/state/clarificator_requirement_synthesis.md
  artifacts_<slug>/execution/enricher_overwrite_enriched_prompt.md
  artifacts_<slug>/execution/executor_overwrite_manifest.json
  artifacts_<slug>/knowledge/current/patcher_findings_snapshot.md
  artifacts_<slug>/knowledge/current/archivist_knowledge_log.md
  artifacts_<slug>/src/**
  artifacts_<slug>/tests/**

Direct execution:
  python 09_debugger.py --project my-app
  PIPELINE_PROJECT=my-app python 09_debugger.py

At the end of each run, prints:
  - artifacts/files read
  - artifacts/files created/updated/overwritten/appended

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# === WRITE AUTHORITY: debugger ===
# OWNS  : artifacts_<slug>/execution/debugger_overwrite_test_summary.json
#         artifacts_<slug>/src/**    repair patches, if needed
#         artifacts_<slug>/tests/**  fragile-test patches, if auditor allows
# READS : see module docstring

sys.path.insert(0, str(Path(__file__).parent.parent))
from artifacts.paths import (  # noqa: E402
    ARCHIVIST_KNOWLEDGE_LOG,
    CLARIFIED_REQ,
    DEBUGGER_OVERWRITE_TEST_SUMMARY,
    ENRICHER_OVERWRITE_PROMPT,
    EXECUTOR_OVERWRITE_MANIFEST,
    PATCHER_FINDINGS_SNAPSHOT,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_IMPACT,
    PLANNER_MINI_PLAN,
    SRC_DIR,
    TESTS_DIR,
    artifact_root,
    ensure_dirs,
    get_spec_path,
)
from artifacts.models import call_model, get_model, get_provider  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402

# Module-level model label constants — derived from models.py at import time.
# Used as owner labels in FailureCluster / ClusterRepairRecord so logs always
# show the actual model name, not an abstract role string.
_DEBUGGER_MODEL           = get_model("debugger")            # surface/component repair
_DEBUGGER_SECONDARY_MODEL = get_model("debugger_secondary")  # logic/hook/data repair

ROLE = "debugger"  # primary role — used for post_interactive next-step suggestion


_SRC_PREFIX = "src"

MINIMAX_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^src/hooks/"),
    re.compile(r"^src/data/"),
    re.compile(r"^src/types/"),
    re.compile(r"^src/utils/"),
]


# ════════════════════════════════════════════════════════════════════════════
# Artifact/file access tracking
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TestFailure:
    test_file: str
    test_name: str
    error_snippet: str


@dataclass
class FailureCluster:
    test_file: str
    src_file: str
    failures: list[TestFailure] = field(default_factory=list)

    attempt_count: int = 0
    last_fingerprint: str = ""
    escalated: bool = False
    is_transform_error: bool = False
    owner: str = ""  # set at runtime to get_model("debugger"); see _DEBUGGER_MODEL / _DEBUGGER_SECONDARY_MODEL

    @property
    def key(self) -> str:
        return self.test_file

    def error_block(self) -> str:
        return "\n\n".join(
            f"  x {failure.test_name}\n{failure.error_snippet}"
            for failure in self.failures
        )

    def fingerprint(self) -> str:
        return re.sub(r"\s+", " ", self.error_block()).strip()[:400]

    def is_minimax_scope(self) -> bool:
        return any(pattern.match(self.src_file) for pattern in MINIMAX_SCOPE_PATTERNS)


@dataclass
class ClusterRepairRecord:
    cluster: str
    src_file: str
    failures: int
    repaired: bool
    layer_used: str
    escalated: bool
    escalated_to: str
    owner: str
    note: str = ""
    consistency_verdict: str = ""


@dataclass
class IterationRecord:
    iteration: int
    passed: bool
    summary: str
    clusters_found: int
    clusters_repaired: int
    cluster_details: list[dict[str, Any]]
    log_snippet: str


# ════════════════════════════════════════════════════════════════════════════
# CLI / project setup
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="09_debugger.py",
        description="Run tests/verifiers and optionally repair failing clusters.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name for direct execution. Sets PIPELINE_PROJECT.",
    )
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--max-cluster-attempts", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--impl",
        default="primary",
        choices=["primary"],
        help="Kept for harness.py compatibility — internally dispatches via model roles debugger + debugger_secondary.",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Run verifier/test only. Do not call LLM repair loop.",
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
        "PIPELINE_PROJECT=<name> before running 09_debugger.py directly."
    )


# ════════════════════════════════════════════════════════════════════════════
# Scope / context loaders
# ════════════════════════════════════════════════════════════════════════════

def _read_json(path: Any, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        track_read(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_text_if_exists(path: Any, *, errors: str = "replace") -> str:
    if not path.exists():
        return ""
    track_read(path)
    return path.read_text(errors=errors)


def _load_impl_record() -> dict[str, Any]:
    data = _read_json(EXECUTOR_OVERWRITE_MANIFEST, {})
    return data if isinstance(data, dict) else {}


def _current_scope() -> str:
    rec = _load_impl_record()
    scope = rec.get("scope", "full")
    return scope if scope in {"full", "mini"} else "full"


def _load_spec_or_full_context() -> str:
    spec_path = get_spec_path()
    if spec_path.exists():
        track_read(spec_path)
        return spec_path.read_text(errors="replace").strip()

    return ""


def _load_mini_context() -> str:
    parts: list[str] = []

    if CLARIFIED_REQ.exists():
        parts.append(
            "## clarificator_requirement_synthesis.md\n\n"
            + _read_text_if_exists(CLARIFIED_REQ).strip()
        )

    if ENRICHER_OVERWRITE_PROMPT.exists():
        parts.append(
            "## enricher_overwrite_enriched_prompt.md\n\n"
            + _read_text_if_exists(ENRICHER_OVERWRITE_PROMPT).strip()
        )

    if PLANNER_MINI_PLAN.exists():
        parts.append(
            "## planner_mini_execution_plan.json\n\n```json\n"
            + _read_text_if_exists(PLANNER_MINI_PLAN).strip()
            + "\n```"
        )

    if PLANNER_MINI_IMPACT.exists():
        parts.append(
            "## planner_mini_impact_analysis.json\n\n```json\n"
            + _read_text_if_exists(PLANNER_MINI_IMPACT).strip()
            + "\n```"
        )

    if EXECUTOR_OVERWRITE_MANIFEST.exists():
        parts.append(
            "## executor_overwrite_manifest.json\n\n```json\n"
            + _read_text_if_exists(EXECUTOR_OVERWRITE_MANIFEST).strip()
            + "\n```"
        )

    if not parts:
        return "(mini context unavailable)"

    return "\n\n---\n\n".join(parts)


def _load_run_context() -> str:
    if _current_scope() == "mini":
        return _load_mini_context()

    ctx = _load_spec_or_full_context()
    return ctx or "(spec context unavailable)"


def _load_planner_global_notes() -> str:
    """
    Load global_notes from planner_full_execution_plan.json and augment with
    archivist_knowledge_log.md.

    Legacy state/plan_notes.json was removed and merged into
    archivist_knowledge_log.md.
    """
    parts: list[str] = []

    if PLANNER_FULL_PLAN.exists():
        try:
            plan = _read_json(PLANNER_FULL_PLAN, {})
            if isinstance(plan, dict):
                note = plan.get("global_notes", "")
                if note:
                    parts.append(str(note))
        except Exception:
            pass

    kb = _load_knowledge_base()
    if kb:
        parts.append("Accumulated knowledge:\n" + kb)

    return "\n\n---\n\n".join(parts)


def _load_judge_findings() -> str:
    """
    Load patcher_findings_snapshot.md and archivist_knowledge_log.md.

    Legacy findings.md is now patcher_findings_snapshot.md.
    Legacy findings_notes.md is now merged into archivist_knowledge_log.md.
    """
    parts: list[str] = []

    for fpath in (PATCHER_FINDINGS_SNAPSHOT, ARCHIVIST_KNOWLEDGE_LOG):
        if fpath.exists():
            try:
                text = _read_text_if_exists(fpath).strip()
                if text:
                    parts.append(text)
            except Exception:
                pass

    return "\n\n---\n\n".join(parts)


def _load_knowledge_base() -> str:
    if not ARCHIVIST_KNOWLEDGE_LOG.exists():
        return ""

    content = _read_text_if_exists(ARCHIVIST_KNOWLEDGE_LOG).strip()
    lines = content.splitlines()
    body_lines = [
        line
        for line in lines
        if not line.startswith("# ") and not line.startswith("_")
    ]
    return "\n".join(body_lines).strip()


# ════════════════════════════════════════════════════════════════════════════
# API helpers
# ════════════════════════════════════════════════════════════════════════════

def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())
    return text


def _parse_model_json(raw: str) -> dict[str, Any]:
    text = _strip_json_fences(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group())

    if not isinstance(parsed, dict):
        raise ValueError("model returned non-object JSON")

    return parsed


def _model_call(
    role: str,
    messages: list[dict[str, str]],
    max_tokens: int = 32768,
) -> str:
    """
    Thin wrapper around call_model() with retry, token logging, and empty-response guard.
    role must be registered in artifacts/models.py ROLES.
    """
    model_id = get_model(role)
    for attempt in range(2):
        resp = call_model(role, messages, max_tokens=max_tokens, temperature=0.1)
        usage = getattr(resp, "usage", None)
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=model_id, provider=get_provider(role))
            print_call(__file__, pt, ct, call_cost, label=f"[09] {model_id}")

        content = resp.choices[0].message.content
        if content and content.strip():
            return content

        if attempt == 0:
            print(
                f"    [warn] {model_id} returned empty response, retrying in 3s …",
                file=sys.stderr,
            )
            time.sleep(3)

    return ""


def call_qwen(messages: list[dict[str, str]]) -> str:
    """Surface/component repair — role: debugger. Owner label: _DEBUGGER_MODEL."""
    return _model_call("debugger", messages)


def call_minimax(messages: list[dict[str, str]]) -> str:
    """Logic/hook/data repair — role: debugger_secondary. Owner label: _DEBUGGER_SECONDARY_MODEL."""
    return _model_call("debugger_secondary", messages)


# ════════════════════════════════════════════════════════════════════════════
# Path helpers
# ════════════════════════════════════════════════════════════════════════════

def _safe_rel_path(raw: str) -> Path:
    normalized = raw.replace("\\", "/").strip()
    rel = Path(normalized)

    if not normalized:
        raise ValueError("empty path")

    if rel.is_absolute():
        raise ValueError(f"absolute path rejected: {raw}")

    if any(part == ".." for part in rel.parts):
        raise ValueError(f"path traversal rejected: {raw}")

    return rel


def _resolve_artifact_path(rel: str) -> Path:
    safe = _safe_rel_path(rel)
    raw = safe.as_posix()

    if raw.startswith("src/"):
        return SRC_DIR / raw[len("src/"):]

    if raw.startswith("tests/"):
        return TESTS_DIR / raw[len("tests/"):]

    return artifact_root() / safe


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _plan_mini_target_files() -> set[str]:
    plan = _read_json(PLANNER_MINI_PLAN, {})
    targets = plan.get("target_files", []) if isinstance(plan, dict) else []
    allowed: set[str] = set()

    if isinstance(targets, list):
        for entry in targets:
            if isinstance(entry, str):
                allowed.add(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
                allowed.add(entry["path"])

    return allowed


def _mini_allowed_to_write(rel_path: str) -> bool:
    if _current_scope() != "mini":
        return True

    allowed = _plan_mini_target_files()
    if not allowed:
        return False

    return rel_path in allowed


def _read_file_safe(path: Path) -> str:
    if path.exists():
        track_read(path)
        return path.read_text(errors="replace")
    return f"// FILE NOT FOUND: {path}\n"


# ════════════════════════════════════════════════════════════════════════════
# Verifier dispatcher for mini mode
# ════════════════════════════════════════════════════════════════════════════

def _implemented_files_from_record() -> list[str]:
    rec = _load_impl_record()
    files = rec.get("files", [])

    out: list[str] = []
    if isinstance(files, list):
        for item in files:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("path"), str):
                out.append(item["path"])

    return sorted(set(out))


def _target_files_for_mini() -> list[str]:
    files = set(_implemented_files_from_record())
    files.update(_plan_mini_target_files())
    return sorted(files)


def _package_json_exists() -> bool:
    return (artifact_root() / "package.json").exists()


def _verify_python(path: Path) -> tuple[bool, str]:
    try:
        track_read(path)
        py_compile.compile(str(path), doraise=True)
        return True, "py_compile OK"
    except Exception as exc:
        return False, f"py_compile failed: {exc}"


def _verify_json(path: Path) -> tuple[bool, str]:
    try:
        track_read(path)
        json.loads(path.read_text(errors="replace"))
        return True, "JSON parse OK"
    except Exception as exc:
        return False, f"JSON parse failed: {exc}"


def _verify_toml(path: Path) -> tuple[bool, str]:
    try:
        import tomllib
        track_read(path)
        tomllib.loads(path.read_text(errors="replace"))
        return True, "TOML parse OK"
    except Exception as exc:
        return False, f"TOML parse failed: {exc}"


def _verify_yaml(path: Path) -> tuple[bool, str]:
    track_read(path)

    try:
        import yaml  # type: ignore
    except Exception:
        text = path.read_text(errors="replace")
        if "\t" in text:
            return False, "YAML basic check failed: tabs found; install PyYAML for full parse"
        return True, "YAML basic check OK; PyYAML not installed"

    try:
        yaml.safe_load(path.read_text(errors="replace"))
        return True, "YAML parse OK"
    except Exception as exc:
        return False, f"YAML parse failed: {exc}"


def _run_mini_verifiers(verbose: bool = False) -> tuple[bool, dict[str, Any]]:
    files = _target_files_for_mini()

    if not files:
        return False, {
            "summary": "No mini target/implemented files found",
            "checks": [],
        }

    checks: list[dict[str, Any]] = []
    ts_like = False

    for rel in files:
        try:
            path = _resolve_artifact_path(rel)
        except Exception as exc:
            checks.append({
                "file": rel,
                "passed": False,
                "kind": "path",
                "message": str(exc),
            })
            continue

        ext = path.suffix.lower()
        kind = ext.lstrip(".") or "unknown"

        if not path.exists():
            checks.append({
                "file": rel,
                "passed": False,
                "kind": kind,
                "message": f"file not found: {path}",
            })
            continue

        if ext == ".py":
            ok, msg = _verify_python(path)
        elif ext == ".json":
            ok, msg = _verify_json(path)
        elif ext in {".yaml", ".yml"}:
            ok, msg = _verify_yaml(path)
        elif ext == ".toml":
            ok, msg = _verify_toml(path)
        elif ext in {".ts", ".tsx", ".js", ".jsx"}:
            ts_like = True
            track_read(path)
            ok, msg = True, "TS/JS file present; Vitest will run if package.json exists"
        elif ext in {".sql", ".md", ".txt", ".cfg", ".conf", ".ini"}:
            track_read(path)
            ok, msg = True, "basic existence check OK"
        else:
            track_read(path)
            ok, msg = True, "basic existence check OK"

        checks.append({
            "file": rel,
            "passed": ok,
            "kind": kind,
            "message": msg,
        })

        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"[09][mini] {status} {rel} — {msg}")

    all_basic_ok = all(c["passed"] for c in checks)

    if ts_like and _package_json_exists():
        print("[09][mini] TS/JS targets detected and package.json exists — running Vitest.")
        vitest_ok, vitest_output = run_vitest()
        checks.append({
            "file": "(vitest)",
            "passed": vitest_ok,
            "kind": "vitest",
            "message": _summarize_test_output(vitest_output),
            "log_snippet": vitest_output[-2000:],
        })
        all_basic_ok = all_basic_ok and vitest_ok
    elif ts_like:
        checks.append({
            "file": "(vitest)",
            "passed": True,
            "kind": "vitest",
            "message": "skipped: package.json not found",
        })

    return all_basic_ok, {
        "summary": "Mini verification complete",
        "checks": checks,
    }


# ════════════════════════════════════════════════════════════════════════════
# Phase B — run Vitest + parse failures
# ════════════════════════════════════════════════════════════════════════════

def run_vitest() -> tuple[bool, str]:
    result = subprocess.run(
        ["npx", "vitest", "run", "--reporter=verbose"],
        cwd=artifact_root(),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + "\n" + result.stderr


_RE_TEST_FILE = re.compile(
    r"^\s*(FAIL|PASS)\s+(tests/\S+\.test\.[tj]sx?)",
    re.MULTILINE,
)
_RE_FAIL_TEST = re.compile(r"^\s+[x\u00d7\u2717\u274c]\s+(.+)$", re.MULTILINE)
_RE_ERROR_BLOCK = re.compile(
    r"(AssertionError|Error|TypeError|ReferenceError)[^\n]*\n(?:[ \t]+[^\n]*\n)*",
    re.MULTILINE,
)
_RE_TRANSFORM_ERR = re.compile(
    r"(Transform failed|ERROR: Expected|SyntaxError.*esbuild|error TS\d+)",
    re.IGNORECASE,
)


def _summarize_test_output(output: str) -> str:
    return next(
        (
            line.strip()
            for line in output.splitlines()
            if ("passed" in line or "failed" in line) and "test" in line.lower()
        ),
        output.strip().splitlines()[-1] if output.strip() else "no output",
    )


def _infer_src_file(test_file: str) -> str:
    rel = test_file.replace("tests/", "", 1)
    rel = re.sub(r"\.test\.(tsx?)$", r".\1", rel)
    rel = re.sub(r"\.test\.(ts)$", r".\1", rel)
    return f"{_SRC_PREFIX}/{rel}"


def parse_failures(output: str) -> list[FailureCluster]:
    clusters: dict[str, FailureCluster] = {}
    file_matches = list(_RE_TEST_FILE.finditer(output))
    sections: list[tuple[str, str, str]] = []

    for index, match in enumerate(file_matches):
        start = match.end()
        end = file_matches[index + 1].start() if index + 1 < len(file_matches) else len(output)
        sections.append((match.group(1), match.group(2), output[start:end]))

    for status, test_file, section in sections:
        if status != "FAIL":
            continue

        src_file = _infer_src_file(test_file)
        cluster = clusters.setdefault(
            test_file,
            FailureCluster(test_file=test_file, src_file=src_file, owner=_DEBUGGER_MODEL),
        )

        if _RE_TRANSFORM_ERR.search(section):
            cluster.is_transform_error = True

        fail_names = _RE_FAIL_TEST.findall(section)
        error_matches = list(_RE_ERROR_BLOCK.finditer(section))
        errors = [m.group(0) for m in error_matches]

        for index, name in enumerate(fail_names):
            snippet = errors[index] if index < len(errors) else section[:500]
            cluster.failures.append(
                TestFailure(
                    test_file=test_file,
                    test_name=name.strip(),
                    error_snippet=snippet.strip()[:600],
                )
            )

        if not cluster.failures:
            cluster.failures.append(
                TestFailure(
                    test_file=test_file,
                    test_name="(parse fallback)",
                    error_snippet=section[:800].strip(),
                )
            )

    return list(clusters.values())


def merge_cluster_state(
    new_clusters: list[FailureCluster],
    prev_state: dict[str, FailureCluster],
) -> list[FailureCluster]:
    for cluster in new_clusters:
        if cluster.key in prev_state:
            prev = prev_state[cluster.key]
            cluster.attempt_count = prev.attempt_count
            cluster.last_fingerprint = prev.last_fingerprint
            cluster.escalated = prev.escalated
            cluster.owner = prev.owner
    return new_clusters


# ════════════════════════════════════════════════════════════════════════════
# Layer 0 — Static pre-pass
# ════════════════════════════════════════════════════════════════════════════

_RE_JSX_GENERIC = re.compile(
    r"(<\w[\w.]*)<(\w[\w,\s]*)>(\s*(?:events|data|items|props|value)\s*=)",
)
_RE_TEMPLATE_WIDTH = re.compile(r"(`\$\{)([^}]*\*\s*100)(\}%`)")
_RE_FLOAT_WIDTH = re.compile(r"(width:\s*)(\d+\.\d+)(%)")


def _static_fix_transform(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file not found"

    track_read(path)
    original = path.read_text(errors="replace")
    patched = _RE_JSX_GENERIC.sub(r"\1\3", original)

    if patched != original:
        path.write_text(patched)
        track_write(path)
        return True, "removed JSX generic type param causing esbuild parse error"

    return False, "no static transform pattern matched"


def _static_fix_src(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file not found"

    track_read(path)
    original = path.read_text(errors="replace")
    patched = _RE_TEMPLATE_WIDTH.sub(r"`${Math.round(\2)}\3", original)
    patched = _RE_FLOAT_WIDTH.sub(
        lambda m: f"{m.group(1)}{round(float(m.group(2)))}{m.group(3)}",
        patched,
    )

    if patched != original:
        path.write_text(patched)
        track_write(path)
        return True, "rounded floating-point percentage widths"

    return False, "no static src pattern matched"


def layer0_static_prepass(
    cluster: FailureCluster,
    verbose: bool,
) -> tuple[bool, str]:
    if verbose:
        print(f"    [L0] Static pre-pass for {cluster.test_file} …")

    if cluster.is_transform_error:
        test_path = TESTS_DIR / cluster.test_file.replace("tests/", "", 1)
        fixed, desc = _static_fix_transform(test_path)
        if fixed:
            print(f"    [L0] ✓ {desc}")
            return True, desc

    src_path = SRC_DIR / cluster.src_file.replace("src/", "", 1)
    fixed, desc = _static_fix_src(src_path)
    if fixed:
        print(f"    [L0] ✓ {desc}")
        return True, desc

    if verbose:
        print("    [L0] No static pattern matched.")

    return False, "no static fix applicable"


# ════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════

FIX_SYSTEM_QWEN = """\
You are a senior TypeScript/React developer doing a SURFACE-LEVEL fix for ONE failing cluster.

Your scope is LIMITED to surface bugs:
- Wrong DOM selector or missing test-id attribute
- Incorrect import path or missing export
- Tailwind class typo or wrong state class
- Floating-point precision in style values
- Text content or badge label mismatch
- Missing or wrong aria attribute

DO NOT:
- Rewrite hook logic, state machines, or data generation
- Change public interfaces
- Touch any file other than the one src file listed

If the bug is logic/algorithmic, return the file unchanged and set:
"explanation": "LOGIC_BUG — needs Minimax debugger"

Return raw JSON only:
{
  "file_path": "src/components/SummaryStickyBar.tsx",
  "code": "<full file content>",
  "explanation": "what was fixed, or LOGIC_BUG"
}
"""


def _build_qwen_system_with_findings(findings: str) -> str:
    if not findings:
        return FIX_SYSTEM_QWEN

    return (
        "## Previous judge/patcher/archivist findings — avoid regressions\n"
        f"{findings[:6000]}\n\n---\n\n"
        + FIX_SYSTEM_QWEN
    )


def _build_minimax_system(global_notes: str, judge_findings: str = "") -> str:
    notes_block = (
        f"\n## Planner global notes and accumulated knowledge\n{global_notes}\n"
        if global_notes else ""
    )

    findings_block = (
        f"\n## Judge/patcher findings from previous runs\n{judge_findings}\n"
        if judge_findings else ""
    )

    kb_content = _load_knowledge_base()
    kb_block = (
        "\n## Accumulated knowledge from human fixes\n"
        f"{kb_content}\n"
        if kb_content else ""
    )

    return f"""\
You are a senior TypeScript logic debugger specialising in hooks, data generation, types, and utilities.
{notes_block}{findings_block}{kb_block}
Fix the LOGIC — not UI styling.

SCOPE:
- You may ONLY write to src/hooks/, src/data/, src/types/, src/utils/
- Never touch src/components/
- Never change public interfaces unless they contradict run context + test

Return raw JSON only:
{{
  "file_path": "src/hooks/useReplay.ts",
  "code": "<full corrected file>",
  "root_cause": "one sentence",
  "explanation": "what changed and why"
}}
"""


def _build_state_timeline(test_code: str, max_entries: int = 12) -> str:
    timeline: list[str] = []

    for line in test_code.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("//") or stripped.startswith("import"):
            continue

        if stripped.startswith(("describe(", "it(", "test(")):
            timeline.append(f"TEST: {stripped[:100]}")
        elif any(token in stripped for token in ("render(", "fireEvent", "userEvent", "act(")):
            timeline.append(f"  ACTION: {stripped[:100]}")
        elif stripped.startswith("expect("):
            timeline.append(f"  ASSERT: {stripped[:100]}")

        if len(timeline) >= max_entries * 3:
            timeline.append("  … (truncated)")
            break

    return "\n".join(timeline) if timeline else "(could not extract timeline)"


# ════════════════════════════════════════════════════════════════════════════
# Consistency checker
# ════════════════════════════════════════════════════════════════════════════

CONSISTENCY_SYSTEM = """\
You are a test-vs-code consistency auditor. You do NOT fix code or tests.
Classify who is wrong.

Return raw JSON only:
{
  "verdict": "CODE_BUG" | "TEST_FRAGILE" | "SPEC_AMBIG" | "THRESHOLD_OK",
  "confidence": "high" | "medium" | "low",
  "reasoning": "one paragraph",
  "test_patch_allowed": true | false,
  "test_patch_rationale": "only if test_patch_allowed=true"
}

Never set test_patch_allowed=true for CODE_BUG.
"""

TEST_REPAIR_SYSTEM = """\
You are fixing a FRAGILE TEST or THRESHOLD.
Allowed changes only:
- DOM query selectors
- async patterns
- text-content matchers
- numeric threshold relaxation when justified

Not allowed:
- Changing intended behavior
- Removing assertions
- Touching src files

Return raw JSON only:
{
  "file_path": "tests/components/AnomalyFeed.test.tsx",
  "code": "<full corrected test file>",
  "changes_made": ["before → after"],
  "explanation": "one sentence"
}
"""


def check_consistency(
    cluster: FailureCluster,
    run_context: str,
    verbose: bool = False,
) -> dict[str, Any]:
    src_code = _read_file_safe(_resolve_artifact_path(cluster.src_file))
    test_code = _read_file_safe(_resolve_artifact_path(cluster.test_file))
    error_log = cluster.error_block()

    user_content = (
        f"### Run context\n\n{run_context}\n\n"
        f"### Test file: {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Source file: {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Failing assertions\n```\n{error_log}\n```"
    )

    messages = [
        {"role": "system", "content": CONSISTENCY_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    if verbose:
        print(f"    [P0] Consistency check → {_DEBUGGER_MODEL} ({cluster.test_file}) …")

    try:
        raw = call_qwen(messages)
        result = _parse_model_json(raw)
        verdict = result.get("verdict", "CODE_BUG")
        confidence = result.get("confidence", "low")
        print(f"    [P0] verdict={verdict} confidence={confidence}")
        return result
    except Exception as exc:
        print(
            f"    [P0] Check failed ({exc}), defaulting to CODE_BUG.",
            file=sys.stderr,
        )
        return {
            "verdict": "CODE_BUG",
            "confidence": "low",
            "test_patch_allowed": False,
            "reasoning": f"consistency check error: {exc}",
            "test_patch_rationale": "",
        }


def repair_test_file(
    cluster: FailureCluster,
    verdict: dict[str, Any],
    run_context: str,
    verbose: bool = False,
) -> bool:
    test_code = _read_file_safe(_resolve_artifact_path(cluster.test_file))
    src_code = _read_file_safe(_resolve_artifact_path(cluster.src_file))
    error_log = cluster.error_block()

    user_content = (
        f"### Run context\n\n{run_context}\n\n"
        f"### Source file (DO NOT MODIFY): {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Test file to fix: {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Failing assertions\n```\n{error_log}\n```\n\n"
        f"### Auditor rationale\n{verdict.get('test_patch_rationale', '')}\n\n"
        f"Verdict: {verdict.get('verdict')} — fix ONLY what rationale describes."
    )

    messages = [
        {"role": "system", "content": TEST_REPAIR_SYSTEM},
        {"role": "user", "content": user_content},
    ]

    if verbose:
        print(f"    [P0-fix] Rewriting test: {cluster.test_file} …")

    try:
        patch = _parse_model_json(call_qwen(messages))
    except Exception as exc:
        print(f"    [P0-fix] Parse error: {exc}", file=sys.stderr)
        return False

    out_rel = patch.get("file_path", cluster.test_file)
    out_path = _resolve_artifact_path(out_rel)

    if not _is_under(out_path, TESTS_DIR):
        print(
            f"    [P0-fix] Scope violation: tried to write {out_path}. Rejected.",
            file=sys.stderr,
        )
        return False

    code = patch.get("code")
    if not isinstance(code, str):
        print("    [P0-fix] Invalid patch: missing code string.", file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)
    track_write(out_path)

    print(f"    [P0-fix] ✓ Test updated — {patch.get('explanation', '(no explanation)')}")
    for change in patch.get("changes_made", []):
        print(f"      • {change}")

    return True


# ════════════════════════════════════════════════════════════════════════════
# Shared repair executor
# ════════════════════════════════════════════════════════════════════════════

def _call_repair(
    cluster: FailureCluster,
    call_api: Callable[[list[dict[str, str]]], str],
    system: str,
    run_context: str,
    extra_ctx: str = "",
    verbose: bool = False,
    layer_name: str = "L1",
    scope_check: bool = False,
) -> tuple[bool, str]:
    src_code = _read_file_safe(_resolve_artifact_path(cluster.src_file))
    test_code = _read_file_safe(_resolve_artifact_path(cluster.test_file))
    error_log = cluster.error_block()

    user_content = (
        f"### Run context\n\n{run_context}\n\n"
        f"### Test file (read-only): {cluster.test_file}\n"
        f"```typescript\n{test_code}\n```\n\n"
        f"### Source file to fix: {cluster.src_file}\n"
        f"```typescript\n{src_code}\n```\n\n"
        f"### Failing tests\n```\n{error_log}\n```"
        + (f"\n\n### Expected state timeline\n```\n{extra_ctx}\n```" if extra_ctx else "")
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    model_label = _DEBUGGER_MODEL if layer_name == "L1" else _DEBUGGER_SECONDARY_MODEL
    if verbose:
        print(
            f"    [{layer_name}] → {model_label} "
            f"(attempt #{cluster.attempt_count + 1}, "
            f"{len(cluster.failures)} failure(s)) …"
        )

    try:
        patch = _parse_model_json(call_api(messages))
    except Exception as exc:
        print(f"    [{layer_name}] Parse error: {exc}", file=sys.stderr)
        return False, f"parse error: {exc}"

    explanation = str(patch.get("explanation", ""))

    if layer_name == "L1" and "LOGIC_BUG" in explanation.upper():
        print(f"    [L1] {_DEBUGGER_MODEL} signalled LOGIC_BUG — deferring to {_DEBUGGER_SECONDARY_MODEL}.")
        return False, "LOGIC_BUG"

    out_rel = str(patch.get("file_path", cluster.src_file))

    if scope_check and not any(pattern.match(out_rel) for pattern in MINIMAX_SCOPE_PATTERNS):
        print(
            f"    [{layer_name}] Scope violation: tried to write {out_rel}. "
            "Allowed: src/hooks/, src/data/, src/types/, src/utils/. Patch rejected.",
            file=sys.stderr,
        )
        return False, f"scope violation: {out_rel}"

    if _current_scope() == "mini" and not _mini_allowed_to_write(out_rel):
        print(
            f"    [{layer_name}] Mini scope violation: tried to write {out_rel}; "
            "not in planner_mini_execution_plan.target_files. Patch rejected.",
            file=sys.stderr,
        )
        return False, f"mini scope violation: {out_rel}"

    out_path = _resolve_artifact_path(out_rel)
    code = patch.get("code")
    if not isinstance(code, str):
        return False, "invalid patch: missing code string"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)
    track_write(out_path)

    root_cause = patch.get("root_cause", "")
    summary = f"{root_cause} — {explanation}" if root_cause else explanation
    print(f"    [{layer_name}] ✓ Patched {out_rel} — {summary or '(no explanation)'}")
    return True, explanation


# ════════════════════════════════════════════════════════════════════════════
# Phase C — cluster dispatch
# ════════════════════════════════════════════════════════════════════════════

def repair_cluster(
    cluster: FailureCluster,
    global_notes: str,
    max_cluster_attempts: int,
    judge_findings: str = "",
    verbose: bool = False,
) -> ClusterRepairRecord:
    run_context = _load_run_context()

    if cluster.escalated:
        print(f"    [SKIP] {cluster.test_file} — ESCALATED→{cluster.owner}, skipping.")
        return ClusterRepairRecord(
            cluster=cluster.key,
            src_file=cluster.src_file,
            failures=len(cluster.failures),
            repaired=False,
            layer_used="skipped",
            escalated=True,
            escalated_to=cluster.owner,
            owner=cluster.owner,
            note="previously escalated",
        )

    if cluster.attempt_count == 0:
        cv = check_consistency(cluster, run_context, verbose=verbose)
        verdict = cv.get("verdict", "CODE_BUG")

        if verdict == "SPEC_AMBIG":
            cluster.escalated = True
            return ClusterRepairRecord(
                cluster=cluster.key,
                src_file=cluster.src_file,
                failures=len(cluster.failures),
                repaired=False,
                layer_used="skipped",
                escalated=True,
                escalated_to="human",
                owner=cluster.owner,
                note=f"spec ambiguous: {cv.get('reasoning', '')[:150]}",
                consistency_verdict=verdict,
            )

        if verdict in {"TEST_FRAGILE", "THRESHOLD_OK"} and cv.get("test_patch_allowed"):
            ok = repair_test_file(cluster, cv, run_context, verbose=verbose)
            cluster.attempt_count += 1
            cluster.last_fingerprint = cluster.fingerprint()
            return ClusterRepairRecord(
                cluster=cluster.key,
                src_file=cluster.src_file,
                failures=len(cluster.failures),
                repaired=ok,
                layer_used="test_rewrite",
                escalated=not ok,
                escalated_to="human" if not ok else "",
                owner=cluster.owner,
                note=cv.get("test_patch_rationale", "")[:150] if ok else "test rewrite failed",
                consistency_verdict=verdict,
            )

        consistency_verdict_label = verdict
    else:
        consistency_verdict_label = ""

    l0_fixed, l0_desc = layer0_static_prepass(cluster, verbose)
    if l0_fixed:
        cluster.attempt_count += 1
        cluster.last_fingerprint = cluster.fingerprint()
        return ClusterRepairRecord(
            cluster=cluster.key,
            src_file=cluster.src_file,
            failures=len(cluster.failures),
            repaired=True,
            layer_used="static",
            escalated=False,
            escalated_to="",
            owner=cluster.owner,
            note=l0_desc,
            consistency_verdict=consistency_verdict_label,
        )

    if cluster.attempt_count >= max_cluster_attempts:
        cluster.escalated = True
        print(
            f"    [L3] Gave up on {cluster.test_file} after "
            f"{cluster.attempt_count} attempt(s). ESCALATED→human."
        )
        return ClusterRepairRecord(
            cluster=cluster.key,
            src_file=cluster.src_file,
            failures=len(cluster.failures),
            repaired=False,
            layer_used="skipped",
            escalated=True,
            escalated_to="human",
            owner=cluster.owner,
            note=f"gave up after {cluster.attempt_count} attempts",
            consistency_verdict=consistency_verdict_label,
        )

    current_fp = cluster.fingerprint()
    is_stale = cluster.attempt_count > 0 and cluster.last_fingerprint == current_fp

    skip_qwen = (
        cluster.owner == _DEBUGGER_SECONDARY_MODEL
        or cluster.is_minimax_scope()
        or is_stale
    )

    if not skip_qwen:
        ok, note = _call_repair(
            cluster,
            call_qwen,
            system=_build_qwen_system_with_findings(judge_findings),
            run_context=run_context,
            verbose=verbose,
            layer_name="L1",
        )

        cluster.attempt_count += 1
        cluster.last_fingerprint = current_fp

        if ok:
            return ClusterRepairRecord(
                cluster=cluster.key,
                src_file=cluster.src_file,
                failures=len(cluster.failures),
                repaired=True,
                layer_used="qwen_targeted",
                escalated=False,
                escalated_to="",
                owner=_DEBUGGER_MODEL,
                consistency_verdict=consistency_verdict_label,
            )

        if cluster.is_minimax_scope():
            print(f"    [L1→L2] Transferring {cluster.test_file} to {_DEBUGGER_SECONDARY_MODEL}.")
            cluster.owner = _DEBUGGER_SECONDARY_MODEL
        else:
            cluster.escalated = True
            return ClusterRepairRecord(
                cluster=cluster.key,
                src_file=cluster.src_file,
                failures=len(cluster.failures),
                repaired=False,
                layer_used="qwen_targeted",
                escalated=True,
                escalated_to="human",
                owner=_DEBUGGER_MODEL,
                note=f"{_DEBUGGER_MODEL} failed on component outside {_DEBUGGER_SECONDARY_MODEL} scope: {note}",
                consistency_verdict=consistency_verdict_label,
            )

    cluster.owner = _DEBUGGER_SECONDARY_MODEL
    test_code = _read_file_safe(_resolve_artifact_path(cluster.test_file))
    timeline = _build_state_timeline(test_code)
    minimax_system = _build_minimax_system(global_notes, judge_findings)

    ok, note = _call_repair(
        cluster,
        call_minimax,
        system=minimax_system,
        run_context=run_context,
        extra_ctx=timeline,
        verbose=verbose,
        layer_name="L2",
        scope_check=True,
    )

    cluster.attempt_count += 1
    cluster.last_fingerprint = current_fp

    return ClusterRepairRecord(
        cluster=cluster.key,
        src_file=cluster.src_file,
        failures=len(cluster.failures),
        repaired=ok,
        layer_used="minimax_logic",
        escalated=False,
        escalated_to="",
        owner=_DEBUGGER_SECONDARY_MODEL,
        note=note,
        consistency_verdict=consistency_verdict_label,
    )


# ════════════════════════════════════════════════════════════════════════════
# Report writers
# ════════════════════════════════════════════════════════════════════════════

def _write_report(report: dict[str, Any]) -> None:
    DEBUGGER_OVERWRITE_TEST_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DEBUGGER_OVERWRITE_TEST_SUMMARY.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(DEBUGGER_OVERWRITE_TEST_SUMMARY)
    print(f"\n[09] Debugger test summary → {DEBUGGER_OVERWRITE_TEST_SUMMARY}")


def _write_mini_report(passed: bool, details: dict[str, Any]) -> None:
    report = {
        "impl": "mini-verifier",
        "scope": "mini",
        "final_status": "PASS" if passed else "FAIL",
        "total_iterations": 1,
        "mini_verification": details,
        "iterations": [
            {
                "iteration": 1,
                "passed": passed,
                "summary": details.get("summary", ""),
                "clusters_found": 0,
                "clusters_repaired": 0,
                "cluster_details": details.get("checks", []),
                "log_snippet": "",
            }
        ],
        "escalated": [],
    }
    _write_report(report)


# ════════════════════════════════════════════════════════════════════════════
# Full Vitest loop
# ════════════════════════════════════════════════════════════════════════════

def _run_full_vitest_loop(
    *,
    max_iter: int,
    max_cluster_attempts: int,
    verbose: bool,
    no_repair: bool,
) -> bool:
    global_notes = _load_planner_global_notes()
    if global_notes:
        print(
            f"[09] Planner/global knowledge loaded ({len(global_notes)} chars) "
            f"— will be injected into {_DEBUGGER_SECONDARY_MODEL} prompts"
        )

    judge_findings = _load_judge_findings()
    if judge_findings:
        print(
            f"[09] Patcher/archivist findings loaded ({len(judge_findings)} chars) "
            "— injected into repair prompts"
        )

    iteration_records: list[IterationRecord] = []
    cluster_state: dict[str, FailureCluster] = {}
    escalated_log: list[dict[str, Any]] = []

    for iteration in range(1, max_iter + 1):
        tag = f"[09][{iteration}/{max_iter}]"

        print(f"\n{tag} Phase B — running Vitest …")
        passed, output = run_vitest()
        summary_line = _summarize_test_output(output)
        print(f"{tag} {summary_line}")

        if passed:
            print(f"{tag} ✓ All tests passed.")
            iteration_records.append(
                IterationRecord(
                    iteration=iteration,
                    passed=True,
                    summary=summary_line,
                    clusters_found=0,
                    clusters_repaired=0,
                    cluster_details=[],
                    log_snippet="",
                )
            )
            break

        clusters = parse_failures(output)
        clusters = merge_cluster_state(clusters, cluster_state)

        print(f"{tag} {len(clusters)} failing cluster(s):")
        for cluster in clusters:
            markers: list[str] = []
            if cluster.attempt_count > 0 and cluster.last_fingerprint == cluster.fingerprint():
                markers.append("STALE")
            if cluster.escalated:
                markers.append("ESCALATED")
            if cluster.owner == _DEBUGGER_SECONDARY_MODEL:
                markers.append(_DEBUGGER_SECONDARY_MODEL.upper()[:12])

            scope_label = "[hook/data/logic]" if cluster.is_minimax_scope() else "[component]"
            marker_str = f"  [{', '.join(markers)}]" if markers else ""
            print(
                f"  * {scope_label} {cluster.test_file} "
                f"({len(cluster.failures)} failure(s)){marker_str}"
            )

        if not clusters:
            print(f"{tag} Could not parse clusters. Stopping.", file=sys.stderr)
            iteration_records.append(
                IterationRecord(
                    iteration=iteration,
                    passed=False,
                    summary=summary_line,
                    clusters_found=0,
                    clusters_repaired=0,
                    cluster_details=[],
                    log_snippet=output[-1200:],
                )
            )
            break

        if no_repair:
            print(f"{tag} --no-repair set; stopping after test run.")
            iteration_records.append(
                IterationRecord(
                    iteration=iteration,
                    passed=False,
                    summary=summary_line,
                    clusters_found=len(clusters),
                    clusters_repaired=0,
                    cluster_details=[
                        {
                            "cluster": c.key,
                            "src_file": c.src_file,
                            "failures": len(c.failures),
                            "owner": c.owner,
                            "attempts": c.attempt_count,
                        }
                        for c in clusters
                    ],
                    log_snippet=output[-1200:],
                )
            )
            break

        if iteration == max_iter:
            print(f"{tag} Reached max_iter — {len(clusters)} cluster(s) remaining.")
            iteration_records.append(
                IterationRecord(
                    iteration=iteration,
                    passed=False,
                    summary=summary_line,
                    clusters_found=len(clusters),
                    clusters_repaired=0,
                    cluster_details=[
                        {
                            "cluster": c.key,
                            "failures": len(c.failures),
                            "escalated": c.escalated,
                            "owner": c.owner,
                            "attempts": c.attempt_count,
                        }
                        for c in clusters
                    ],
                    log_snippet=output[-1200:],
                )
            )
            break

        print(f"{tag} Phase C — dispatching {len(clusters)} cluster(s) …")
        repaired = 0
        cluster_details: list[dict[str, Any]] = []

        for cluster in clusters:
            print(
                f"  -> {cluster.test_file} "
                f"(owner={cluster.owner}, attempt #{cluster.attempt_count + 1})"
            )

            record = repair_cluster(
                cluster,
                global_notes,
                max_cluster_attempts,
                judge_findings=judge_findings,
                verbose=verbose,
            )

            cluster_state[cluster.key] = cluster
            repaired += int(record.repaired)

            detail = {
                "cluster": record.cluster,
                "src_file": record.src_file,
                "failures": record.failures,
                "repaired": record.repaired,
                "layer_used": record.layer_used,
                "escalated": record.escalated,
                "escalated_to": record.escalated_to,
                "owner": record.owner,
                "note": record.note,
                "consistency_verdict": record.consistency_verdict,
            }
            cluster_details.append(detail)

            if record.escalated:
                escalated_log.append({"iteration": iteration, **detail})

        print(f"{tag} Phase C done — {repaired}/{len(clusters)} patched.")
        iteration_records.append(
            IterationRecord(
                iteration=iteration,
                passed=False,
                summary=summary_line,
                clusters_found=len(clusters),
                clusters_repaired=repaired,
                cluster_details=cluster_details,
                log_snippet=output[-1200:],
            )
        )

    final_passed = bool(iteration_records and iteration_records[-1].passed)

    report = {
        "impl": f"primary:{get_model('debugger')}+secondary:{get_model('debugger_secondary')}",
        "scope": _current_scope(),
        "max_iter": max_iter,
        "max_cluster_attempts": max_cluster_attempts,
        "total_iterations": len(iteration_records),
        "final_status": "PASS" if final_passed else "FAIL",
        "iterations": [asdict(record) for record in iteration_records],
        "escalated": escalated_log,
    }

    _write_report(report)

    if escalated_log:
        print(f"[09] ⚠ {len(escalated_log)} cluster(s) escalated")

    return final_passed


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)

    # Important: PIPELINE_PROJECT must be configured before ensure_dirs().
    ensure_dirs()

    exit_code = 0

    try:
        scope = _current_scope()
        print(f"[09] Scope detected: {scope}")

        if scope == "mini":
            passed, mini_details = _run_mini_verifiers(verbose=args.verbose)

            # If mini is TS/Vitest and failed, optionally allow the existing repair loop.
            checks = mini_details.get("checks", [])
            vitest_failed = any(
                c.get("kind") == "vitest" and not c.get("passed")
                for c in checks
                if isinstance(c, dict)
            )

            if vitest_failed and not args.no_repair:
                print("[09][mini] Vitest failed; entering TS repair loop with mini context.")
                final_passed = _run_full_vitest_loop(
                    max_iter=args.max_iter,
                    max_cluster_attempts=args.max_cluster_attempts,
                    verbose=args.verbose,
                    no_repair=args.no_repair,
                )
                if not final_passed:
                    exit_code = 1
            else:
                _write_mini_report(passed, mini_details)
                if not passed:
                    exit_code = 1

        else:
            final_passed = _run_full_vitest_loop(
                max_iter=args.max_iter,
                max_cluster_attempts=args.max_cluster_attempts,
                verbose=args.verbose,
                no_repair=args.no_repair,
            )

            if not final_passed:
                exit_code = 1

    except Exception as exc:
        print(f"[09] ERROR: {exc}", file=sys.stderr)

        # Best-effort failure report.
        try:
            _write_report(
                {
                    "impl": f"primary:{get_model('debugger')}+secondary:{get_model('debugger_secondary')}",
                    "scope": "unknown",
                    "final_status": "FAIL",
                    "total_iterations": 0,
                    "iterations": [],
                    "escalated": [],
                    "error": str(exc),
                }
            )
        except Exception:
            pass

        exit_code = 1

    finally:
        print_summary("[09]")
        print_artifact_summary("[09]")
        prompt_next_step(ROLE, prefix="[09]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
