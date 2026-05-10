"""
harness.py — Orchestrator for the LLM pipeline.

Canonical module pipeline:
    absorber      01_absorber.py
    clarificator  02_clarificator.py
    enricher      03_enricher.py
    specwright    04_specwright.py
    spectracker   05_spectracker.py
    scaffolder    06_scaffolder.py
    planner       07_planner.py
    executor      08_executor.py
    debugger      09_debugger.py
    reporter      10_reporter.py
    judge         11_judge.py
    patcher       12_patcher.py
    archivist     13_archivist.py

harness.py is the ONLY entrypoint for end-to-end pipeline runs.

New design:
    - Project-global artifacts live under artifacts_<slug>/
    - Session-local artifacts live under artifacts_<slug>/sessions/<NNN>/
    - harness owns orchestration metadata only:
        artifacts_<slug>/session_runs/session_<NNN>_runs.json
    - step scripts own their artifact writes.
    - harness may call spectracker.write_applied() during finalization only.

Session model:
    Session = logical unit of work.
    Run     = one harness.py invocation.

Artifact tracing is ON by default.
Disable:
    python harness.py --no-trace-artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re as _re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════════════════════════════════════
# Step registry
# ════════════════════════════════════════════════════════════════════════════

STEPS = [
    "absorber",
    "clarificator",
    "enricher",
    "specwright",
    "spectracker",
    "scaffolder",
    "planner",
    "executor",
    "debugger",
    "reporter",
    "judge",
    "patcher",
    "archivist",
]

STEP_SCRIPTS: dict[str, str] = {
    "absorber": "01_absorber.py",
    "clarificator": "02_clarificator.py",
    "enricher": "03_enricher.py",
    "specwright": "04_specwright.py",
    "spectracker": "05_spectracker.py",
    "scaffolder": "06_scaffolder.py",
    "planner": "07_planner.py",
    "executor": "08_executor.py",
    "debugger": "09_debugger.py",
    "reporter": "10_reporter.py",
    "judge": "11_judge.py",
    "patcher": "12_patcher.py",
    "archivist": "13_archivist.py",
}

STEP_ENV_KEYS: dict[str, list[str]] = {
    "scaffolder": ["GEMINI_API_KEY"],
    "planner": ["OPENROUTER_API_KEY"],
    "executor": ["OPENROUTER_API_KEY"],
    "debugger": ["OPENROUTER_API_KEY"],
    "judge": ["OPENROUTER_API_KEY"],
    "patcher": ["OPENROUTER_API_KEY"],
}

SCOPE_CHOICES = ("full", "mini")


# ════════════════════════════════════════════════════════════════════════════
# Artifact trace declaration registry — new naming/session layout
# ════════════════════════════════════════════════════════════════════════════

STEP_ARTIFACT_READS: dict[str, list[str]] = {
    "absorber": [],
    "clarificator": [
        "sessions/<NNN>/cache/absorber_overwrite_codebase_snapshot.json",
        "sessions/<NNN>/cache/absorber_overwrite_git_snapshot.json",
        "knowledge/current/clarificator_decision_log.md",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_config_map.json",
        "knowledge/current/absorber_blame_map.md",
    ],
    "enricher": [
        "sessions/<NNN>/state/clarificator_requirement_synthesis.md",
        "sessions/<NNN>/execution/clarificator_overwrite_raw.json",
        "sessions/<NNN>/cache/absorber_overwrite_codebase_snapshot.json",
        "sessions/<NNN>/cache/absorber_overwrite_git_snapshot.json",
        "knowledge/current/clarificator_decision_log.md",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_config_map.json",
        "knowledge/current/absorber_blame_map.md",
    ],
    "specwright": [
        "sessions/<NNN>/execution/enricher_overwrite_enriched_prompt.md",
        "sessions/<NNN>/state/clarificator_requirement_synthesis.md",
        "knowledge/current/archivist_spec_gaps.md",
    ],
    "spectracker": [
        "specwright_spec_<slug>.md",
        "state/spectracker_applied_version.json",
        "knowledge/history/<version>.md",
        "knowledge/history/spectracker_version_log.md",
    ],
    "scaffolder": [
        "specwright_spec_<slug>.md",
    ],
    "planner": [
        "specwright_spec_<slug>.md",
        "sessions/<NNN>/state/scaffolder_codebase_skeleton.json",
        "sessions/<NNN>/cache/scaffolder_compressed_spec.md",
        "sessions/<NNN>/cache/absorber_overwrite_codebase_snapshot.json",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_blame_map.md",
        "knowledge/current/archivist_knowledge_log.md",
        "sessions/<NNN>/execution/clarificator_overwrite_raw.json",
    ],
    "executor": [
        "specwright_spec_<slug>.md",
        "sessions/<NNN>/state/scaffolder_codebase_skeleton.json",
        "sessions/<NNN>/state/planner_full_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_impact_analysis.json",
        "sessions/<NNN>/cache/scaffolder_compressed_spec.md",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/archivist_knowledge_log.md",
    ],
    "debugger": [
        "sessions/<NNN>/state/planner_full_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_execution_plan.json",
        "sessions/<NNN>/execution/executor_overwrite_manifest.json",
        "knowledge/current/patcher_findings_snapshot.md",
        "knowledge/current/archivist_knowledge_log.md",
    ],
    "reporter": [
        "sessions/<NNN>/state/scaffolder_codebase_skeleton.json",
        "sessions/<NNN>/state/planner_full_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_execution_plan.json",
        "sessions/<NNN>/execution/executor_overwrite_manifest.json",
        "sessions/<NNN>/execution/debugger_overwrite_test_summary.json",
        "sessions/<NNN>/cache/spectracker_overwrite_version_delta.json",
    ],
    "judge": [
        "specwright_spec_<slug>.md",
        "sessions/<NNN>/state/scaffolder_codebase_skeleton.json",
        "sessions/<NNN>/state/planner_full_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_execution_plan.json",
        "sessions/<NNN>/execution/executor_overwrite_manifest.json",
        "sessions/<NNN>/execution/debugger_overwrite_test_summary.json",
        "sessions/<NNN>/reports/reporter_execution_summary.md",
        "knowledge/current/archivist_knowledge_log.md",
        "knowledge/current/archivist_spec_gaps.md",
    ],
    "patcher": [
        "sessions/<NNN>/execution/judge_overwrite_verdict_raw.json",
        "sessions/<NNN>/state/planner_full_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_execution_plan.json",
        "sessions/<NNN>/execution/executor_overwrite_manifest.json",
        "knowledge/current/archivist_knowledge_log.md",
        "sessions/<NNN>/cache/scaffolder_compressed_spec.md",
    ],
    "archivist": [
        "sessions/<NNN>/execution/debugger_overwrite_test_summary.json",
        "sessions/<NNN>/execution/judge_overwrite_verdict_raw.json",
        "knowledge/current/patcher_findings_snapshot.md",
        "knowledge/history/patcher_attempt_log.json",
    ],
}

STEP_ARTIFACT_WRITES: dict[str, list[str]] = {
    "absorber": [
        "sessions/<NNN>/cache/absorber_overwrite_codebase_snapshot.json",
        "sessions/<NNN>/cache/absorber_overwrite_git_snapshot.json",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_config_map.json",
        "knowledge/current/absorber_blame_map.md",
    ],
    "clarificator": [
        "sessions/<NNN>/execution/clarificator_overwrite_raw.json",
        "sessions/<NNN>/execution/clarificator_overwrite_questions.md",
        "sessions/<NNN>/state/clarificator_requirement_synthesis.md",
        "knowledge/current/clarificator_decision_log.md",
    ],
    "enricher": [
        "sessions/<NNN>/execution/enricher_overwrite_enriched_prompt.md",
    ],
    "specwright": [
        "specwright_spec_<slug>.md",
    ],
    "spectracker": [
        "sessions/<NNN>/cache/spectracker_overwrite_version_delta.json",
        "knowledge/history/spectracker_version_log.md",
        "knowledge/history/<version>.md",
        "knowledge/history/<version>.changelog.md",
        "state/spectracker_applied_version.json  (finalization via write_applied)",
    ],
    "scaffolder": [
        "sessions/<NNN>/state/scaffolder_codebase_skeleton.json",
        "sessions/<NNN>/cache/scaffolder_compressed_spec.md",
        "tests/",
    ],
    "planner": [
        "sessions/<NNN>/state/planner_full_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_execution_plan.json",
        "sessions/<NNN>/state/planner_mini_impact_analysis.json",
    ],
    "executor": [
        "sessions/<NNN>/execution/executor_overwrite_manifest.json",
        "src/",
    ],
    "debugger": [
        "sessions/<NNN>/execution/debugger_overwrite_test_summary.json",
        "src/",
    ],
    "reporter": [
        "sessions/<NNN>/reports/reporter_execution_summary.md",
    ],
    "judge": [
        "sessions/<NNN>/execution/judge_overwrite_verdict_raw.json",
        "sessions/<NNN>/reports/judge_verdict_summary.md",
    ],
    "patcher": [
        "sessions/<NNN>/execution/patcher_overwrite_fix_summary.md",
        "knowledge/current/patcher_findings_snapshot.md",
        "knowledge/history/patcher_attempt_log.json",
        "src/",
    ],
    "archivist": [
        "knowledge/current/archivist_knowledge_log.md",
        "knowledge/current/archivist_spec_gaps.md",
        "knowledge/history/archivist_curation_log.json",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# Core helpers
# ════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _die(msg: str) -> None:
    print(f"[harness] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def check_env(keys: list[str]) -> bool:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        print(f"[harness] Missing env vars: {', '.join(missing)}")
        print("          Set them in .env or export them before running.")
        return False
    return True


def _normalize_session_id(raw: str | int) -> str:
    return f"{int(raw):03d}"


def _pipeline_script(script: str) -> Path:
    return ROOT / "pipeline" / script


def run_step(label: str, script: str, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(_pipeline_script(script))] + (extra_args or [])

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
    print(f"  {status}  ({elapsed:.1f}s)")
    return result.returncode == 0


def skip_step(label: str, reason: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}  [SKIPPED — {reason}]")
    print(f"{'=' * 60}")


def _scope_args_for_script(script: str, scope: str) -> list[str]:
    if script in {"07_planner.py", "08_executor.py"}:
        return ["--scope", scope]
    return []


# ════════════════════════════════════════════════════════════════════════════
# Session selection + run metadata
# ════════════════════════════════════════════════════════════════════════════

def _existing_session_ids() -> list[str]:
    from artifacts.paths import artifact_root

    sessions_dir = artifact_root() / "sessions"
    if not sessions_dir.exists():
        return []

    ids: list[str] = []
    for path in sessions_dir.iterdir():
        if path.is_dir() and path.name.isdigit():
            ids.append(_normalize_session_id(path.name))
    return sorted(set(ids))


def _next_session_id() -> str:
    ids = _existing_session_ids()
    if not ids:
        return "001"
    return _normalize_session_id(int(ids[-1]) + 1)


def _resolve_session_for_run(args: argparse.Namespace) -> str:
    if args.session:
        return _normalize_session_id(args.session)

    if args.new_session:
        return _next_session_id()

    ids = _existing_session_ids()
    if ids:
        return ids[-1]

    return "001"


def _session_runs_path(session_id: str) -> Path:
    from artifacts.paths import get_session_runs_path

    return get_session_runs_path(session_id)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_session_runs(session_id: str) -> dict[str, Any]:
    path = _session_runs_path(session_id)

    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data.setdefault("session_id", session_id)
                data.setdefault("created_at", _now_iso())
                data.setdefault("runs", [])
                return data
        except Exception:
            pass

    return {
        "session_id": session_id,
        "created_at": _now_iso(),
        "runs": [],
    }


def _start_run_record(
    session_id: str,
    args: argparse.Namespace,
    from_step: str,
    until_step: str,
) -> str:
    data = _load_session_runs(session_id)
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    data["runs"].append(
        {
            "run_id": run_id,
            "started_at": _now_iso(),
            "completed_at": None,
            "status": "RUNNING",
            "scope": args.scope,
            "from_step": from_step,
            "until_step": until_step,
            "stopped_at_step": None,
            "resumed_from_run": data["runs"][-1]["run_id"] if data["runs"] else None,
            "steps": [],
            "spec_version": None,
        }
    )

    _atomic_write_json(_session_runs_path(session_id), data)
    return run_id


def _update_run_record(
    session_id: str,
    run_id: str,
    *,
    status: str | None = None,
    step: str | None = None,
    step_status: str | None = None,
    stopped_at_step: str | None = None,
    spec_version: str | None = None,
) -> None:
    data = _load_session_runs(session_id)

    for run in data.get("runs", []):
        if run.get("run_id") != run_id:
            continue

        if status is not None:
            run["status"] = status
            if status in {"PASS", "FAIL", "STOPPED"}:
                run["completed_at"] = _now_iso()

        if stopped_at_step is not None:
            run["stopped_at_step"] = stopped_at_step

        if spec_version is not None:
            run["spec_version"] = spec_version

        if step is not None and step_status is not None:
            run.setdefault("steps", []).append(
                {
                    "step": step,
                    "status": step_status,
                    "at": _now_iso(),
                }
            )

        break

    _atomic_write_json(_session_runs_path(session_id), data)


# ════════════════════════════════════════════════════════════════════════════
# Artifact tracing
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ArtifactFileState:
    size: int
    mtime_ns: int
    sha256: str
    content: bytes


ArtifactSnapshot = dict[str, ArtifactFileState]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_artifacts() -> ArtifactSnapshot:
    from artifacts.paths import artifact_root

    root = artifact_root()
    snapshot: ArtifactSnapshot = {}

    if not root.exists():
        return snapshot

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root).as_posix()

        # harness scratch space
        if rel.startswith("state/prev_src/"):
            continue

        try:
            content = path.read_bytes()
            stat = path.stat()
        except OSError:
            continue

        snapshot[rel] = ArtifactFileState(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=_hash_bytes(content),
            content=content,
        )

    return snapshot


def _classify_artifact_changes(
    before: ArtifactSnapshot,
    after: ArtifactSnapshot,
) -> dict[str, list[str]]:
    created: list[str] = []
    appended: list[str] = []
    overwritten: list[str] = []
    touched: list[str] = []
    deleted: list[str] = []

    before_keys = set(before)
    after_keys = set(after)

    for rel in sorted(after_keys - before_keys):
        created.append(rel)

    for rel in sorted(before_keys - after_keys):
        deleted.append(rel)

    for rel in sorted(before_keys & after_keys):
        old = before[rel]
        new = after[rel]

        if old.sha256 == new.sha256:
            if old.mtime_ns != new.mtime_ns:
                touched.append(rel)
            continue

        if new.size > old.size and new.content.startswith(old.content):
            appended.append(rel)
        else:
            overwritten.append(rel)

    return {
        "created": created,
        "appended": appended,
        "overwritten": overwritten,
        "touched": touched,
        "deleted": deleted,
    }


def _print_artifact_list(title: str, items: list[str], indent: str = "    ") -> None:
    print(f"{indent}{title}:")
    if not items:
        print(f"{indent}  - (none)")
        return

    sid = os.environ.get("PIPELINE_SESSION", "<NNN>")
    for item in items:
        print(f"{indent}  - {item.replace('<NNN>', sid)}")


def _print_artifact_trace(
    step: str,
    before: ArtifactSnapshot | None,
    after: ArtifactSnapshot | None,
) -> None:
    print(f"\n[harness] Artifact trace — {step}")

    _print_artifact_list("declared reads", STEP_ARTIFACT_READS.get(step, []))
    _print_artifact_list("declared writes", STEP_ARTIFACT_WRITES.get(step, []))

    if before is None or after is None:
        print("    actual changes:")
        print("      - (not captured)")
        return

    changes = _classify_artifact_changes(before, after)

    print("    actual changes:")
    _print_artifact_list("created", changes["created"], indent="      ")
    _print_artifact_list("appended", changes["appended"], indent="      ")
    _print_artifact_list("overwritten/updated", changes["overwritten"], indent="      ")
    _print_artifact_list("touched without content change", changes["touched"], indent="      ")
    _print_artifact_list("deleted", changes["deleted"], indent="      ")


def _run_step_with_trace(
    step: str,
    label: str,
    script: str,
    args: argparse.Namespace,
    extra_args: list[str] | None = None,
) -> bool:
    before = _snapshot_artifacts() if args.trace_artifacts else None
    ok = run_step(label, script, extra_args)
    after = _snapshot_artifacts() if args.trace_artifacts else None

    if args.trace_artifacts:
        _print_artifact_trace(step, before, after)

    return ok


# ════════════════════════════════════════════════════════════════════════════
# Project selection
# ════════════════════════════════════════════════════════════════════════════

def _interactive_project_select(root: Path) -> str:
    projects = sorted(
        p.name.removeprefix("artifacts_")
        for p in root.glob("artifacts_*")
        if p.is_dir()
    )

    if projects:
        print("\nExisting projects:")
        for i, project in enumerate(projects, 1):
            print(f"  {i}. {project}")
        print(f"  {len(projects) + 1}. New project\n")

        choice = input("Select project (number or name): ").strip()

        if choice.isdigit():
            idx = int(choice)

            if 1 <= idx <= len(projects):
                return projects[idx - 1]

            if idx == len(projects) + 1:
                name = input("New project name: ").strip()
                if name:
                    return name

        if choice and not choice.isdigit():
            return choice

    name = input("Project name: ").strip()
    if not name:
        _die("Project name is required.")

    return name


# ════════════════════════════════════════════════════════════════════════════
# Artifact/path helpers
# ════════════════════════════════════════════════════════════════════════════

def _artifact_rel(path: Any) -> str:
    try:
        from artifacts.paths import artifact_root

        return str(Path(path).relative_to(artifact_root()))
    except Exception:
        return str(path)


def _prev_src_dir() -> Path:
    from artifacts.paths import artifact_root

    return artifact_root() / "state" / "prev_src"


def _canonical_spec_path() -> Path:
    from artifacts.paths import get_spec_path

    return get_spec_path()


def _canonical_spec_exists() -> bool:
    return _canonical_spec_path().exists()


def load_delta() -> dict | None:
    from artifacts.paths import SPECTRACKER_VERSION_DELTA

    if not SPECTRACKER_VERSION_DELTA.exists():
        return None

    try:
        return json.loads(SPECTRACKER_VERSION_DELTA.read_text())
    except Exception:
        return None


def delta_requires(delta: dict | None, step: str) -> bool:
    if delta is None:
        return True

    rerun_steps = delta.get("rerun_steps", {})
    if not isinstance(rerun_steps, dict):
        return True

    if step in rerun_steps:
        return bool(rerun_steps.get(step))

    return True


def print_delta_summary(delta: dict) -> None:
    from_version = delta.get("from_version") or "(none)"
    to_version = delta.get("to_version", "?")
    baseline = delta.get("baseline_source")

    print(f"\n[harness] Spec: {from_version} → {to_version}")
    if baseline:
        print(f"[harness] Baseline: {baseline}")

    if delta.get("is_first_run"):
        print("[harness] First run — full pipeline.")
    else:
        changed = delta.get("changed_sections", [])
        new_sections = delta.get("new_sections", [])
        removed = delta.get("removed_sections", [])
        summaries = delta.get("section_summaries", {})

        print(f"[harness] Changed §: {changed or '(none)'}")
        print(f"[harness] New     §: {new_sections or '(none)'}")
        print(f"[harness] Removed §: {removed or '(none)'}")

        for section in changed:
            if section in summaries:
                print(f"    §{section}: {summaries[section]}")

    affected = delta.get("affected_files", [])
    rerun = [key for key, value in delta.get("rerun_steps", {}).items() if value]
    skip = [key for key, value in delta.get("rerun_steps", {}).items() if not value]

    print(f"[harness] Affected files  : {len(affected)}")
    print(f"[harness] Steps to re-run : {rerun or '(none)'}")
    print(f"[harness] Steps to skip   : {skip or '(none)'}")


# ════════════════════════════════════════════════════════════════════════════
# src/ snapshot + restore
# ════════════════════════════════════════════════════════════════════════════

def snapshot_src() -> None:
    from artifacts.paths import SRC_DIR

    src = Path(SRC_DIR)
    if not src.exists():
        # backward compat if executor still writes root-level src/
        src = ROOT / "src"

    if not src.exists():
        return

    prev_src = _prev_src_dir()
    if prev_src.exists():
        shutil.rmtree(prev_src)

    shutil.copytree(src, prev_src)
    print(f"[harness] src/ snapshot → {_artifact_rel(prev_src)}")


def restore_unaffected_files(delta: dict) -> int:
    from artifacts.paths import SRC_DIR

    unaffected = [
        file
        for file in delta.get("unaffected_files", [])
        if isinstance(file, str) and file.startswith("src/")
    ]

    prev_src = _prev_src_dir()
    if not unaffected or not prev_src.exists():
        return 0

    restored = 0
    src_root = Path(SRC_DIR)

    for rel in unaffected:
        prev = prev_src / rel[len("src/"):]
        dest = src_root / rel[len("src/"):]

        if prev.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prev, dest)
            restored += 1

    if restored:
        print(f"[harness] Restored {restored} unaffected file(s) from prev_src/")

    return restored


# ════════════════════════════════════════════════════════════════════════════
# Judge helpers
# ════════════════════════════════════════════════════════════════════════════

def _read_judge_verdict() -> str:
    from artifacts.paths import JUDGE_SESSION_VERDICT_RAW

    raw_path = JUDGE_SESSION_VERDICT_RAW
    if not raw_path.exists():
        return ""

    try:
        raw_data = json.loads(raw_path.read_text())
        response = raw_data.get("response", "")

        if isinstance(response, str):
            response = _re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", response.strip())
            response = _re.sub(r"\n?```$", "", response.strip())
            return json.loads(response).get("verdict", "")

        if isinstance(response, dict):
            return response.get("verdict", "")

        return ""
    except Exception:
        return ""


def _run_judge_fix_loop(args: argparse.Namespace, results: dict[str, bool]) -> None:
    from artifacts.paths import ARCHIVIST_CURATION_LOG

    max_rounds = args.max_judge_rounds

    for round_num in range(1, max_rounds + 1):
        round_sfx = f" (round {round_num}/{max_rounds})" if max_rounds > 1 else ""

        ok = _run_step_with_trace(
            "judge",
            f"judge{round_sfx}",
            STEP_SCRIPTS["judge"],
            args,
        )
        results[f"judge_r{round_num}"] = ok
        results["judge"] = ok

        verdict = _read_judge_verdict()
        print(f"\n[harness] Judge verdict: {verdict or '(unknown)'}")

        if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
            print(f"[harness] ✅ Judge {verdict} — pipeline complete.")
            results["judge"] = True
            return

        if round_num == max_rounds:
            print(f"[harness] ⚠ Reached max_judge_rounds ({max_rounds}).")
            print("[harness] Run archivist when ready: python pipeline/13_archivist.py")
            return

        if args.skip_fix:
            skip_step(
                f"patcher{round_sfx}",
                "--skip-fix set — review judge_verdict_summary.md manually",
            )
            return

        if verdict == "NEEDS_REVISION":
            patcher_args: list[str] = []
            if args.verbose:
                patcher_args.append("--verbose")
            if args.fix_non_blocking:
                patcher_args.append("--fix-non-blocking")

            patch_ok = _run_step_with_trace(
                "patcher",
                f"patcher{round_sfx}",
                STEP_SCRIPTS["patcher"],
                args,
                patcher_args or None,
            )
            results[f"judge_patcher_r{round_num}"] = patch_ok

            if not patch_ok:
                print("\n[harness] ⚠ Patcher failed; human review required.")
                print(f"[harness] See {ARCHIVIST_CURATION_LOG}")
                return

            print("\n[harness] Patch applied — re-running reporter + judge …")
            report_ok = _run_step_with_trace(
                "reporter",
                "reporter post-patch",
                STEP_SCRIPTS["reporter"],
                args,
            )
            results[f"reporter_post_patch_r{round_num}"] = report_ok

            if not report_ok:
                results["judge"] = False
                return

            continue

        print("[harness] Judge failed or returned non-actionable verdict — stopping.")
        return


def _run_fix_from_existing_judge(args: argparse.Namespace, results: dict[str, bool]) -> None:
    from artifacts.paths import JUDGE_SESSION_VERDICT_RAW, ARCHIVIST_CURATION_LOG

    raw_path = JUDGE_SESSION_VERDICT_RAW
    if not raw_path.exists():
        print("[harness] --repair-from-judge: judge_overwrite_verdict_raw.json not found.")
        results["patcher_from_judge"] = False
        return

    verdict = _read_judge_verdict()
    print(f"\n[harness] Existing judge verdict: {verdict or '(unknown)'}")

    if verdict in ("APPROVED", "APPROVED_WITH_NOTES"):
        print(f"[harness] ✅ Already {verdict} — nothing to patch.")
        results["patcher_from_judge"] = True
        return

    if verdict != "NEEDS_REVISION":
        print(f"[harness] ⚠ Unrecognised verdict '{verdict}' — stopping.")
        results["patcher_from_judge"] = False
        return

    if args.skip_fix:
        skip_step(
            "patcher from existing judge",
            "--skip-fix set — review judge_verdict_summary.md manually",
        )
        results["patcher_from_judge"] = True
        return

    patcher_args: list[str] = []
    if args.verbose:
        patcher_args.append("--verbose")
    if args.fix_non_blocking:
        patcher_args.append("--fix-non-blocking")

    patch_ok = _run_step_with_trace(
        "patcher",
        "patcher from existing judge",
        STEP_SCRIPTS["patcher"],
        args,
        patcher_args or None,
    )
    results["patcher_from_judge"] = patch_ok

    if not patch_ok:
        print("\n[harness] ⚠ Patcher failed; human review required.")
        print(f"[harness] See {ARCHIVIST_CURATION_LOG}")
        return

    print("\n[harness] Patch applied — refreshing reporter + re-judging …")

    report_ok = _run_step_with_trace(
        "reporter",
        "reporter post-patch",
        STEP_SCRIPTS["reporter"],
        args,
    )
    results["reporter_post_patch"] = report_ok

    if not report_ok:
        results["patcher_from_judge"] = False
        return

    if not check_env(["OPENROUTER_API_KEY"]):
        print("[harness] WARNING: cannot re-judge without OPENROUTER_API_KEY.")
        return

    judge_ok = _run_step_with_trace(
        "judge",
        "judge post-patch",
        STEP_SCRIPTS["judge"],
        args,
    )
    results["judge_post_patch"] = judge_ok

    final = _read_judge_verdict()
    if final:
        print(f"\n[harness] Post-patch verdict: {final}")


# ════════════════════════════════════════════════════════════════════════════
# Range + dry-run
# ════════════════════════════════════════════════════════════════════════════

def _validate_scope(scope: str) -> None:
    if scope not in SCOPE_CHOICES:
        _die(f"Invalid --scope {scope!r}. Expected one of: {', '.join(SCOPE_CHOICES)}")


def _selected_single_steps(args: argparse.Namespace) -> list[str]:
    return [step for step in STEPS if getattr(args, step, False)]


def _has_range_selection(args: argparse.Namespace) -> bool:
    return bool(args.from_step or args.until_step)


def _validate_execution_mode_conflicts(args: argparse.Namespace) -> None:
    single_steps = _selected_single_steps(args)

    if len(single_steps) > 1:
        _die(
            "Only one step shorthand allowed at a time, got: "
            + " ".join(f"--{step}" for step in single_steps)
        )

    active_modes: list[str] = []

    if _has_range_selection(args):
        active_modes.append("range mode (--from-*/--until-*)")

    if single_steps:
        active_modes.append(f"single-step mode (--{single_steps[0]})")

    if args.repair_from_judge:
        active_modes.append("repair flow (--repair-from-judge)")

    if len(active_modes) > 1:
        _die("Cannot mix execution modes: " + " + ".join(active_modes))


def _resolve_run_range(args: argparse.Namespace) -> tuple[str, str]:
    shorthands = _selected_single_steps(args)

    if shorthands:
        return shorthands[0], shorthands[0]

    if args.until_step and not args.from_step:
        _die(f"--until-{args.until_step} requires --from-<step>.")

    if not args.from_step and not args.until_step:
        return STEPS[0], STEPS[-1]

    if args.from_step not in STEPS:
        _die(f"Unknown --from step: {args.from_step}")

    if args.until_step and args.until_step not in STEPS:
        _die(f"Unknown --until step: {args.until_step}")

    from_step = args.from_step
    until_step = args.until_step or STEPS[-1]

    if STEPS.index(from_step) > STEPS.index(until_step):
        _die(
            f"--from-{from_step} comes after --until-{until_step}.\n"
            f"  Order: {' → '.join(STEPS)}"
        )

    return from_step, until_step


def _print_dry_run(from_step: str, until_step: str, args: argparse.Namespace) -> None:
    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)

    print("\n[harness] DRY RUN — nothing will be executed.")
    print(f"  Project         : {os.environ.get('PIPELINE_PROJECT', '?')}")
    print(f"  Session         : {os.environ.get('PIPELINE_SESSION', '?')}")
    print(f"  Scope           : {args.scope}")
    print(f"  Range           : {from_step} → {until_step}")
    print(f"  Artifact trace  : {'on' if args.trace_artifacts else 'off'}")
    print()

    for i, step in enumerate(STEPS):
        if not (from_idx <= i <= until_idx):
            print(f"  ⏭  {step:<14}  (skipped)")
            continue

        if args.scope == "mini" and step in {"specwright", "spectracker", "scaffolder"}:
            print(f"  ⏭  {step:<14}  (skipped: mini scope)")
            continue

        script = STEP_SCRIPTS[step]
        extra = _scope_args_for_script(script, args.scope)
        suffix = f" {' '.join(extra)}" if extra else ""
        print(f"  ▶  {step:<14}  {script}{suffix}")

    print()


# ════════════════════════════════════════════════════════════════════════════
# Executor retry helpers
# ════════════════════════════════════════════════════════════════════════════

def _retry_impl_args(executor_args: list[str]) -> list[str] | None:
    from artifacts.paths import EXECUTOR_SESSION_MANIFEST

    if not EXECUTOR_SESSION_MANIFEST.exists():
        print("[harness] --retry-impl: executor_overwrite_manifest.json not found.")
        return None

    try:
        record = json.loads(EXECUTOR_SESSION_MANIFEST.read_text())
        failed = record.get("failed_files", [])

        if not failed:
            print("[harness] --retry-impl: no failed_files — nothing to retry.")
            return None

        executor_args += ["--only-files", ",".join(failed)]
        print(f"[harness] --retry-impl: retrying {len(failed)} failed file(s).")
        return executor_args

    except Exception:
        print("[harness] --retry-impl: could not read executor_overwrite_manifest.json.")
        return None


def _print_impl_failed() -> None:
    from artifacts.paths import EXECUTOR_SESSION_MANIFEST

    if not EXECUTOR_SESSION_MANIFEST.exists():
        return

    try:
        record = json.loads(EXECUTOR_SESSION_MANIFEST.read_text())
        failed = record.get("failed_files", [])

        if failed:
            print(f"\n[harness] {len(failed)} file(s) failed to implement:")
            for file in failed:
                print(f"    {file}")
            print("\n[harness] Retry: python harness.py --retry-impl")

    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Step dispatch
# ════════════════════════════════════════════════════════════════════════════

def _run_step(
    step: str,
    args: argparse.Namespace,
    delta: dict | None,
    plan_available: bool,
    results: dict[str, bool],
    tests_passed: bool,
) -> bool:
    if step == "absorber":
        return _run_step_with_trace(step, "absorber", STEP_SCRIPTS[step], args)

    if step == "clarificator":
        clarify_args: list[str] = []
        if args.clarify_input:
            clarify_args += ["--input", args.clarify_input]

        return _run_step_with_trace(
            step,
            "clarificator",
            STEP_SCRIPTS[step],
            args,
            clarify_args or None,
        )

    if step == "enricher":
        return _run_step_with_trace(step, "enricher", STEP_SCRIPTS[step], args)

    if step == "specwright":
        if args.scope == "mini":
            skip_step("specwright", "mini scope does not update canonical spec")
            return True

        return _run_step_with_trace(step, "specwright", STEP_SCRIPTS[step], args)

    if step == "spectracker":
        if args.scope == "mini":
            skip_step("spectracker", "mini scope does not update spec delta")
            return True

        if not _canonical_spec_exists():
            skip_step(
                "spectracker",
                f"canonical spec not found — run specwright first: {_canonical_spec_path()}",
            )
            return True

        return _run_step_with_trace(step, "spectracker", STEP_SCRIPTS[step], args)

    if step == "scaffolder":
        if args.scope == "mini":
            skip_step("scaffolder", "mini scope uses targeted planner; no skeleton needed")
            return True

        if delta and not delta.get("is_first_run") and not delta_requires(delta, "scaffolder"):
            skip_step(
                "scaffolder",
                "delta: scaffold-relevant sections unchanged — reusing skeleton",
            )
            return True

        if not check_env(STEP_ENV_KEYS.get("scaffolder", [])):
            sys.exit(1)

        return _run_step_with_trace(step, "scaffolder", STEP_SCRIPTS[step], args)

    if step == "planner":
        if args.only_qwen:
            skip_step("planner", "--only-qwen")
            return True

        if args.scope == "full":
            if delta and not delta.get("is_first_run") and not delta_requires(delta, "planner"):
                skip_step(
                    "planner",
                    "delta: no affected files — reusing planner_full_execution_plan.json",
                )
                return True

        if not check_env(STEP_ENV_KEYS.get("planner", [])):
            sys.exit(1)

        planner_args = _scope_args_for_script(STEP_SCRIPTS[step], args.scope)
        ok = _run_step_with_trace(step, "planner", STEP_SCRIPTS[step], args, planner_args)

        if not ok:
            print("\n[harness] Planner failed.\n  Tip: --only-qwen to skip planning.")

        return ok

    if step == "executor":
        if not check_env(STEP_ENV_KEYS.get("executor", [])):
            sys.exit(1)

        executor_args = _scope_args_for_script(STEP_SCRIPTS[step], args.scope)

        if plan_available:
            executor_args.append("--use-glm-plan")

        if args.retry_impl:
            retry_args = _retry_impl_args(executor_args)
            if retry_args is None:
                return False
            executor_args = retry_args

        elif args.scope == "full" and delta and not delta.get("is_first_run"):
            restore_unaffected_files(delta)

            src_affected = [
                file
                for file in delta.get("affected_files", [])
                if isinstance(file, str) and file.startswith("src/")
            ]

            if src_affected:
                executor_args += ["--only-files", ",".join(src_affected)]
                print(f"[harness] executor: {len(src_affected)} affected file(s) only.")

        if args.scope == "mini":
            mode = "mini-targeted+plan" if plan_available else "mini-targeted"
        else:
            mode = "per-file+plan" if plan_available else "single-call"

        ok = _run_step_with_trace(
            step,
            f"executor ({mode})",
            STEP_SCRIPTS[step],
            args,
            executor_args,
        )

        if not ok:
            _print_impl_failed()

        return ok

    if step == "debugger":
        test_args = [
            "--impl",
            "qwen",
            "--max-iter",
            str(args.max_iter),
            "--max-cluster-attempts",
            str(args.max_cluster_attempts),
        ]

        if args.verbose:
            test_args.append("--verbose")

        return _run_step_with_trace(step, "debugger", STEP_SCRIPTS[step], args, test_args)

    if step == "reporter":
        return _run_step_with_trace(step, "reporter", STEP_SCRIPTS[step], args)

    if step == "judge":
        if not tests_passed:
            skip_step("judge", "debugger failed — fix tests before judge sign-off")
            return True

        if not check_env(STEP_ENV_KEYS.get("judge", [])):
            print("[harness] WARNING: cannot run judge without OPENROUTER_API_KEY.")
            return False

        _run_judge_fix_loop(args, results)
        return results.get("judge", False)

    if step == "patcher":
        if "judge" in results:
            skip_step("patcher", "already handled by judge loop")
            return True

        _run_fix_from_existing_judge(args, results)
        return results.get("patcher_from_judge", False)

    if step == "archivist":
        return _run_step_with_trace(step, "archivist", STEP_SCRIPTS[step], args)

    return True


# ════════════════════════════════════════════════════════════════════════════
# Summary + applied record
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(
    results: dict[str, bool],
    delta: dict | None,
    args: argparse.Namespace,
    tests_passed: bool,
) -> None:
    from artifacts.paths import (
        CLARIFICATOR_SESSION_QUESTIONS,
        JUDGE_VERDICT_SUMMARY,
        REPORTER_EXECUTION_SUMMARY,
    )

    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")

    print(f"  Project        : {os.environ.get('PIPELINE_PROJECT', '?')}")
    print(f"  Session        : {os.environ.get('PIPELINE_SESSION', '?')}")
    print(f"  Scope          : {args.scope}")
    print(f"  Artifact trace : {'on' if args.trace_artifacts else 'off'}")

    if delta:
        from_version = delta.get("from_version") or "(none)"
        to_version = delta.get("to_version", "?")
        affected_count = len(delta.get("affected_files", []))
        print(f"  Spec           : {from_version} → {to_version}  ({affected_count} file(s) affected)")

    for key, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {key}")

    all_ok = all(results.values()) if results else False
    print(f"\n  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")

    print("\n  Reports:")
    if CLARIFICATOR_SESSION_QUESTIONS.exists() and results.get("clarificator"):
        print(f"    Clarificator → {CLARIFICATOR_SESSION_QUESTIONS}")

    print(f"    Pipeline     → {REPORTER_EXECUTION_SUMMARY}")

    if results.get("judge") and tests_passed:
        print(f"    Judge        → {JUDGE_VERDICT_SUMMARY}")
        verdict = _read_judge_verdict()
        if verdict in ("APPROVED_WITH_NOTES", "NEEDS_REVISION"):
            print(f"\n  Judge verdict: {verdict}")
            print("  Run archivist when ready:")
            print("    python pipeline/13_archivist.py")


def _load_write_applied():
    path = _pipeline_script(STEP_SCRIPTS["spectracker"])

    if not path.exists():
        return None

    module_name = "pipeline_05_spectracker"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    write_applied = getattr(module, "write_applied", None)
    return write_applied if callable(write_applied) else None


def _write_apply_record(delta: dict, results: dict[str, bool]) -> None:
    from artifacts.paths import SPECTRACKER_APPLIED

    write_applied = _load_write_applied()
    if write_applied is None:
        print("[harness] WARNING: could not load spectracker.write_applied")
        return

    applied_steps = [key for key, value in results.items() if value]
    version = delta.get("to_version", "unknown")

    write_applied(
        version=version,
        steps=applied_steps,
        status="PASS",
    )

    print(
        f"\n  Apply record → {SPECTRACKER_APPLIED}  "
        f"(v{version} marked as applied)"
    )


def _should_mark_applied(
    args: argparse.Namespace,
    delta: dict | None,
    results: dict[str, bool],
    tests_passed: bool,
    steps_to_run: list[str],
) -> bool:
    if args.scope != "full":
        return False

    if delta is None:
        return False

    if not results:
        return False

    if not all(results.values()):
        return False

    if "executor" not in results:
        return False

    if "debugger" in steps_to_run and not tests_passed:
        return False

    if "judge" in steps_to_run:
        judge_actually_ran = any(k.startswith("judge_r") for k in results)
        if not judge_actually_ran:
            print("[harness] Apply record skipped: judge was in range but did not execute.")
            return False

        verdict = _read_judge_verdict()
        if verdict not in ("APPROVED", "APPROVED_WITH_NOTES"):
            print(f"[harness] Apply record skipped: judge verdict is {verdict or '(unknown)'}.")
            return False

    return True


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness.py",
        description="Pipeline orchestrator. Runs full end-to-end or a sub-range of steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py --project demo
  python harness.py --project demo --new-session
  python harness.py --project demo --session 001 --from-executor --until-debugger
  python harness.py --project demo --scope mini --from-clarificator --until-executor
  python harness.py --project demo --dry-run
  python harness.py --project demo --no-trace-artifacts
  python harness.py --project demo --repair-from-judge
""",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        metavar="NAME",
        help="Project name → artifacts_<slug>/. If omitted, interactive prompt lists projects.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        metavar="NNN",
        help="Use/resume a specific session id. Example: --session 001",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Create the next session id instead of resuming the latest session.",
    )
    parser.add_argument(
        "--scope",
        choices=list(SCOPE_CHOICES),
        default="full",
        help="Pipeline scope: full or mini.",
    )
    parser.add_argument(
        "--auto-continue",
        action="store_true",
        help="Non-interactive mode for harness-level prompts.",
    )

    for step in STEPS:
        parser.add_argument(
            f"--from-{step}",
            dest="from_step",
            action="store_const",
            const=step,
            help=f"Start pipeline from step '{step}'.",
        )
        parser.add_argument(
            f"--until-{step}",
            dest="until_step",
            action="store_const",
            const=step,
            help=f"Stop pipeline after step '{step}'.",
        )

    for step in STEPS:
        parser.add_argument(
            f"--{step}",
            dest=step,
            action="store_true",
            help=f"Run only step '{step}'.",
        )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-qwen", action="store_true")
    parser.add_argument("--retry-impl", action="store_true")
    parser.add_argument("--max-judge-rounds", type=int, default=2, metavar="N")
    parser.add_argument("--max-iter", type=int, default=3, metavar="N")
    parser.add_argument("--max-cluster-attempts", type=int, default=2, metavar="N")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-fix", action="store_true")
    parser.add_argument("--fix-non-blocking", dest="fix_non_blocking", action="store_true")
    parser.add_argument("--repair-from-judge", dest="repair_from_judge", action="store_true")
    parser.add_argument("--clarify-input", type=str, default=None, metavar="FILE")

    parser.add_argument(
        "--trace-artifacts",
        dest="trace_artifacts",
        action="store_true",
        default=True,
        help="Print artifact reads/writes and actual changes after each step. Default: enabled.",
    )
    parser.add_argument(
        "--no-trace-artifacts",
        dest="trace_artifacts",
        action="store_false",
        help="Disable artifact tracing.",
    )

    # Legacy mini mode.
    parser.add_argument(
        "--mini",
        type=str,
        default=None,
        metavar="PROMPT",
        help="Legacy mini mode delegated to pipeline/mini_mode.py. Prefer --scope mini.",
    )
    parser.add_argument("--files", nargs="+", default=None)
    parser.add_argument("--context-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--task-type", type=str, default=None)

    return parser


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    _validate_scope(args.scope)
    _validate_execution_mode_conflicts(args)

    if args.session and args.new_session:
        _die("Cannot combine --session and --new-session.")

    if not args.project:
        args.project = _interactive_project_select(ROOT)

    os.environ["PIPELINE_PROJECT"] = args.project

    # Import after PIPELINE_PROJECT is set.
    from artifacts.paths import (
        PLANNER_FULL_PLAN,
        PLANNER_MINI_PLAN,
        artifact_root,
        ensure_dirs,
        get_project_slug,
        session_root,
    )

    # Resolve and set session before ensure_dirs() so _SessLazyPath creates sessions/<NNN>/.
    session_id = _resolve_session_for_run(args)
    os.environ["PIPELINE_SESSION"] = session_id

    ensure_dirs()

    print(f"[harness] Project        : {args.project}")
    print(f"[harness] Project slug   : {get_project_slug()}")
    print(f"[harness] Workspace      : {artifact_root()}")
    print(f"[harness] Session        : {session_id}")
    print(f"[harness] Session root   : {session_root()}")
    print(f"[harness] Scope          : {args.scope}")
    print(f"[harness] Artifact trace : {'on' if args.trace_artifacts else 'off'}")

    # Legacy mini mode.
    if args.mini is not None:
        print(
            "[harness] WARNING: --mini is legacy and delegates to pipeline/mini_mode.py.\n"
            "          Prefer: python harness.py --scope mini ..."
        )

        from pipeline.mini_mode import run_mini  # type: ignore

        run_mini(
            prompt=args.mini,
            files=args.files,
            context_file=Path(args.context_file) if args.context_file else None,
            output_file=Path(args.output_file) if args.output_file else None,
            task_type_override=args.task_type,
            dry_run=args.dry_run,
        )
        return

    # Repair flow.
    if args.repair_from_judge:
        results: dict[str, bool] = {}

        if args.dry_run:
            print(
                "\n[harness] DRY RUN — would consume existing "
                "execution/judge_overwrite_verdict_raw.json, run patcher, "
                "refresh reporter, and re-run judge."
            )
            return

        run_id = _start_run_record(session_id, args, "patcher", "judge")

        try:
            _run_fix_from_existing_judge(args, results)
            _print_summary(results, delta=None, args=args, tests_passed=True)
            all_ok = all(results.values()) if results else False
            _update_run_record(session_id, run_id, status="PASS" if all_ok else "FAIL")
            sys.exit(0 if all_ok else 1)
        except Exception:
            _update_run_record(session_id, run_id, status="FAIL")
            raise

    from_step, until_step = _resolve_run_range(args)

    if args.dry_run:
        _print_dry_run(from_step, until_step, args)
        return

    run_id = _start_run_record(session_id, args, from_step, until_step)

    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)
    steps_to_run = STEPS[from_idx:until_idx + 1]

    print(f"[harness] Steps          : {from_step} → {until_step}\n")

    results: dict[str, bool] = {}

    delta: dict | None = None
    needs_delta = (
        args.scope == "full"
        and any(step in steps_to_run for step in ("scaffolder", "planner", "executor"))
    )

    try:
        # Spectracker preflight for mid-pipeline full runs.
        if needs_delta and "spectracker" not in steps_to_run:
            if _canonical_spec_exists():
                ok = _run_step_with_trace(
                    "spectracker",
                    "spectracker preflight",
                    STEP_SCRIPTS["spectracker"],
                    args,
                )
                _update_run_record(
                    session_id,
                    run_id,
                    step="spectracker_preflight",
                    step_status="PASS" if ok else "FAIL",
                )

                delta = load_delta()

                if delta:
                    print_delta_summary(delta)
                else:
                    print("[harness] spectracker preflight produced no readable delta.")

                if args.force:
                    print("[harness] --force: delta ignored — all steps will re-run.")
                    delta = None
            else:
                skip_step(
                    "spectracker preflight",
                    f"canonical spec not found — no delta available: {_canonical_spec_path()}",
                )
                delta = None

        elif args.scope == "mini":
            print("[harness] mini scope: skipping specwright/spectracker/scaffolder contracts.")

        tests_passed = True
        plan_available = PLANNER_MINI_PLAN.exists() if args.scope == "mini" else PLANNER_FULL_PLAN.exists()

        for step in steps_to_run:
            ok = _run_step(step, args, delta, plan_available, results, tests_passed)
            results[step] = ok

            _update_run_record(
                session_id,
                run_id,
                step=step,
                step_status="PASS" if ok else "FAIL",
            )

            if step == "spectracker" and ok and args.scope == "full":
                delta = load_delta()

                if delta:
                    print_delta_summary(delta)
                    _update_run_record(
                        session_id,
                        run_id,
                        spec_version=delta.get("to_version"),
                    )
                else:
                    print("[harness] spectracker produced no readable delta.")

                if args.force:
                    print("[harness] --force: delta ignored — all steps will re-run.")
                    delta = None

            if step == "planner" and ok:
                plan_available = True

            if step == "debugger":
                tests_passed = ok

            if step == "executor" and ok:
                snapshot_src()

            if not ok and step in ("scaffolder", "planner", "executor", "debugger"):
                print(f"\n[harness] {step} failed — stopping pipeline.")
                _print_summary(results, delta, args, tests_passed)
                _update_run_record(
                    session_id,
                    run_id,
                    status="FAIL",
                    stopped_at_step=step,
                )
                sys.exit(1)

        _print_summary(results, delta, args, tests_passed)

        all_ok = all(results.values()) if results else False

        if _should_mark_applied(
            args=args,
            delta=delta,
            results=results,
            tests_passed=tests_passed,
            steps_to_run=steps_to_run,
        ):
            _write_apply_record(delta, results)
        else:
            if args.scope == "full" and delta:
                print("[harness] Apply record skipped — finalization criteria not met.")

        _update_run_record(
            session_id,
            run_id,
            status="PASS" if all_ok else "FAIL",
            spec_version=delta.get("to_version") if delta else None,
        )

        sys.exit(0 if all_ok else 1)

    except KeyboardInterrupt:
        print("\n[harness] Interrupted.")
        _update_run_record(session_id, run_id, status="STOPPED")
        raise

    except Exception:
        _update_run_record(session_id, run_id, status="FAIL")
        raise


if __name__ == "__main__":
    main()
