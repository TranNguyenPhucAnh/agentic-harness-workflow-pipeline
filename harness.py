"""
harness.py — Orchestrator for the LLM pipeline.

Canonical module pipeline:
    spectracker   01_spectracker.py
    absorber      02_absorber.py
    clarificator  03_clarificator.py
    enricher      04_enricher.py
    specwright    05_specwright.py
    scaffolder    06_scaffolder.py
    planner       07_planner.py
    executor      08_executor.py
    debugger      09_debugger.py
    reporter      10_reporter.py
    judge         11_judge.py
    patcher       12_patcher.py
    archivist     13_archivist.py

harness.py is the ONLY entrypoint for end-to-end pipeline runs.
Step scripts are pure runners: execute one step, write owned artifacts, exit.

Artifact tracing is ON by default:
    after each step, harness prints declared reads/writes and actual artifact
    create/update/overwrite/append/delete changes under artifacts_<slug>/.

Disable tracing:
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent

# === WRITE AUTHORITY: harness ===
# OWNS  : orchestration only; optional legacy mini delegated to mini_mode.py
# READS : all artifacts
# NOTE  : step scripts own their respective artifact writes.

sys.path.insert(0, str(ROOT))

from artifacts.paths import (  # noqa: E402
    ARCHIVIST_CURATION_LOG,
    CLARIFICATOR_SESSION_QUESTIONS,
    EXECUTOR_SESSION_MANIFEST,
    JUDGE_SESSION_VERDICT_RAW,
    JUDGE_VERDICT_SUMMARY,
    PLANNER_FULL_PLAN,
    PLANNER_MINI_PLAN,
    REPORTER_EXECUTION_SUMMARY,
    SPECTRACKER_APPLIED,
    SPECTRACKER_VERSION_DELTA,
)


# ════════════════════════════════════════════════════════════════════════════
# Step registry
# ════════════════════════════════════════════════════════════════════════════

STEPS = [
    "spectracker",
    "absorber",
    "clarificator",
    "enricher",
    "specwright",
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
    "spectracker": "01_spectracker.py",
    "absorber": "02_absorber.py",
    "clarificator": "03_clarificator.py",
    "enricher": "04_enricher.py",
    "specwright": "05_specwright.py",
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

LEGACY_STEP_ALIASES: dict[str, str] = {
    "absorb": "absorber",
    "clarify": "clarificator",
    "scaffold": "scaffolder",
    "plan": "planner",
    "implement": "executor",
    "test": "debugger",
    "report": "reporter",
    "fix": "patcher",
}

CANONICAL_TO_LEGACY_STEP: dict[str, str] = {
    canonical: legacy for legacy, canonical in LEGACY_STEP_ALIASES.items()
}

SCOPE_CHOICES = ("full", "mini")


# ════════════════════════════════════════════════════════════════════════════
# Artifact tracing registry
# ════════════════════════════════════════════════════════════════════════════

STEP_ARTIFACT_READS: dict[str, list[str]] = {
    "spectracker": [
        "specwright_spec_<slug>.md",
        "state/spectracker_applied_version.json",
        "knowledge/history/<version>.md",
    ],
    "absorber": [],
    "clarificator": [
        "cache/absorber_session_codebase_snapshot.json",
        "cache/absorber_session_git_snapshot.json",
        "knowledge/current/clarificator_decision_log.md",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_config_map.json",
        "knowledge/current/absorber_blame_map.md",
    ],
    "enricher": [
        "state/clarificator_requirement_synthesis.md",
        "execution/clarificator_session_raw.json",
        "cache/absorber_session_codebase_snapshot.json",
        "cache/absorber_session_git_snapshot.json",
        "knowledge/current/clarificator_decision_log.md",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_config_map.json",
        "knowledge/current/absorber_blame_map.md",
    ],
    "specwright": [
        "execution/enricher_session_enriched_prompt.md",
        "state/clarificator_requirement_synthesis.md",
        "knowledge/current/archivist_spec_gaps.md",
    ],
    "scaffolder": [
        "specwright_spec_<slug>.md",
    ],
    "planner": [
        "specwright_spec_<slug>.md",
        "state/scaffolder_codebase_skeleton.json",
        "cache/scaffolder_compressed_spec.md",
        "cache/absorber_session_codebase_snapshot.json",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_blame_map.md",
        "knowledge/current/archivist_knowledge_log.md",
        "execution/clarificator_session_raw.json",
    ],
    "executor": [
        "specwright_spec_<slug>.md",
        "state/scaffolder_codebase_skeleton.json",
        "state/planner_full_execution_plan.json",
        "state/planner_mini_execution_plan.json",
        "state/planner_mini_impact_analysis.json",
        "cache/scaffolder_compressed_spec.md",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/archivist_knowledge_log.md",
    ],
    "debugger": [
        "state/planner_full_execution_plan.json",
        "state/planner_mini_execution_plan.json",
        "execution/executor_session_manifest.json",
        "knowledge/current/patcher_findings_snapshot.md",
        "knowledge/current/archivist_knowledge_log.md",
    ],
    "reporter": [
        "state/scaffolder_codebase_skeleton.json",
        "state/planner_full_execution_plan.json",
        "state/planner_mini_execution_plan.json",
        "execution/executor_session_manifest.json",
        "execution/debugger_session_test_summary.json",
        "cache/spectracker_session_version_delta.json",
    ],
    "judge": [
        "specwright_spec_<slug>.md",
        "state/scaffolder_codebase_skeleton.json",
        "state/planner_full_execution_plan.json",
        "state/planner_mini_execution_plan.json",
        "execution/executor_session_manifest.json",
        "execution/debugger_session_test_summary.json",
        "reports/reporter_execution_summary.md",
        "knowledge/current/archivist_knowledge_log.md",
        "knowledge/current/archivist_spec_gaps.md",
    ],
    "patcher": [
        "execution/judge_session_verdict_raw.json",
        "state/planner_full_execution_plan.json",
        "state/planner_mini_execution_plan.json",
        "execution/executor_session_manifest.json",
        "knowledge/current/archivist_knowledge_log.md",
        "cache/scaffolder_compressed_spec.md",
    ],
    "archivist": [
        "execution/debugger_session_test_summary.json",
        "execution/judge_session_verdict_raw.json",
        "knowledge/current/patcher_findings_snapshot.md",
        "knowledge/history/patcher_attempt_log.json",
    ],
}

STEP_ARTIFACT_WRITES: dict[str, list[str]] = {
    "spectracker": [
        "state/spectracker_applied_version.json",
        "cache/spectracker_session_version_delta.json",
        "knowledge/history/spectracker_version_log.md",
        "knowledge/history/<version>.md",
        "knowledge/history/<version>.changelog.md",
    ],
    "absorber": [
        "cache/absorber_session_codebase_snapshot.json",
        "cache/absorber_session_git_snapshot.json",
        "knowledge/current/absorber_codebase_map.md",
        "knowledge/current/absorber_config_map.json",
        "knowledge/current/absorber_blame_map.md",
    ],
    "clarificator": [
        "execution/clarificator_session_raw.json",
        "execution/clarificator_session_questions.md",
        "state/clarificator_requirement_synthesis.md",
        "knowledge/current/clarificator_decision_log.md",
    ],
    "enricher": [
        "execution/enricher_session_enriched_prompt.md",
    ],
    "specwright": [
        "specwright_spec_<slug>.md",
    ],
    "scaffolder": [
        "state/scaffolder_codebase_skeleton.json",
        "cache/scaffolder_compressed_spec.md",
        "tests/",
    ],
    "planner": [
        "state/planner_full_execution_plan.json",
        "state/planner_mini_execution_plan.json",
        "state/planner_mini_impact_analysis.json",
    ],
    "executor": [
        "execution/executor_session_manifest.json",
        "src/",
    ],
    "debugger": [
        "execution/debugger_session_test_summary.json",
        "src/",
    ],
    "reporter": [
        "reports/reporter_execution_summary.md",
    ],
    "judge": [
        "execution/judge_session_verdict_raw.json",
        "reports/judge_verdict_summary.md",
    ],
    "patcher": [
        "execution/patcher_session_fix_summary.md",
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
    """
    Snapshot files under artifacts_<slug>/.

    Note:
      - Traces pipeline artifacts only under artifacts_<slug>/.
      - Excludes harness scratch state/prev_src/.
      - Root-level src/ and tests/ are build outputs, not artifact files.
    """
    from artifacts.paths import artifact_root

    root = artifact_root()
    snapshot: ArtifactSnapshot = {}

    if not root.exists():
        return snapshot

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root).as_posix()
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
    for item in items:
        print(f"{indent}  - {item}")


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
# Core helpers
# ════════════════════════════════════════════════════════════════════════════

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


def run_step(label: str, script: str, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(ROOT / "pipeline" / script)] + (extra_args or [])

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


def check_env(keys: list[str]) -> bool:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        print(f"[harness] Missing env vars: {', '.join(missing)}")
        print("          Set them in .env or export them before running.")
        return False
    return True


def _artifact_rel(path: Any) -> str:
    try:
        from artifacts.paths import artifact_root

        return str(Path(path).relative_to(artifact_root()))
    except Exception:
        return str(path)


def _canonical_step(step: str | None) -> str | None:
    if step is None:
        return None
    return LEGACY_STEP_ALIASES.get(step, step)


def _prev_src_dir() -> Path:
    from artifacts.paths import artifact_root

    return artifact_root() / "state" / "prev_src"


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
# Delta helpers
# ════════════════════════════════════════════════════════════════════════════

def load_delta() -> dict | None:
    if not SPECTRACKER_VERSION_DELTA.exists():
        return None
    try:
        return json.loads(SPECTRACKER_VERSION_DELTA.read_text())
    except Exception:
        return None


def delta_requires(delta: dict | None, step: str) -> bool:
    """
    True if spectracker delta says step must re-run, or delta unavailable.

    Tolerates old delta keys during migration:
      scaffold/plan/implement
    """
    if delta is None:
        return True

    rerun_steps = delta.get("rerun_steps", {})
    if not isinstance(rerun_steps, dict):
        return True

    if step in rerun_steps:
        return bool(rerun_steps.get(step))

    legacy = CANONICAL_TO_LEGACY_STEP.get(step)
    if legacy and legacy in rerun_steps:
        return bool(rerun_steps.get(legacy))

    return True


def print_delta_summary(delta: dict) -> None:
    from_version = delta.get("from_version") or "(none)"
    to_version = delta.get("to_version", "?")

    print(f"\n[harness] Spec: {from_version} → {to_version}")

    if delta.get("is_first_run"):
        print("[harness] First run — full pipeline.")
        return

    changed = delta.get("changed_sections", [])
    affected = delta.get("affected_files", [])
    rerun = [key for key, value in delta.get("rerun_steps", {}).items() if value]
    skip = [key for key, value in delta.get("rerun_steps", {}).items() if not value]
    summaries = delta.get("section_summaries", {})

    if changed:
        print(f"[harness] Changed §: {changed}")
        for section in changed:
            if section in summaries:
                print(f"    §{section}: {summaries[section]}")

    print(f"[harness] Affected files  : {len(affected)}")
    print(f"[harness] Steps to re-run : {rerun or '(none)'}")
    print(f"[harness] Steps to skip   : {skip or '(none)'}")


# ════════════════════════════════════════════════════════════════════════════
# src/ snapshot + restore
# ════════════════════════════════════════════════════════════════════════════

def snapshot_src() -> None:
    src = ROOT / "src"
    if not src.exists():
        return

    prev_src = _prev_src_dir()
    if prev_src.exists():
        shutil.rmtree(prev_src)

    shutil.copytree(src, prev_src)
    print(f"[harness] src/ snapshot → {_artifact_rel(prev_src)}")


def restore_unaffected_files(delta: dict) -> int:
    unaffected = [
        file
        for file in delta.get("unaffected_files", [])
        if isinstance(file, str) and file.startswith("src/")
    ]

    prev_src = _prev_src_dir()
    if not unaffected or not prev_src.exists():
        return 0

    restored = 0
    for rel in unaffected:
        prev = prev_src / rel[len("src/"):]
        dest = ROOT / rel
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


def _scope_args_for_script(script: str, scope: str) -> list[str]:
    if script in {"07_planner.py", "08_executor.py"}:
        return ["--scope", scope]
    return []


def _run_judge_fix_loop(args: argparse.Namespace, results: dict[str, bool]) -> None:
    max_rounds = args.max_judge_rounds
    skip_fix = args.skip_fix

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
            break

        if round_num == max_rounds:
            print(
                f"[harness] ⚠ Reached max_judge_rounds ({max_rounds}) "
                f"with verdict {verdict or '(unknown)'}."
            )
            print("[harness] Run archivist when ready: python pipeline/13_archivist.py")
            break

        if skip_fix:
            skip_step(
                f"patcher{round_sfx}",
                "--skip-fix set — review judge_verdict_summary.md manually",
            )
            break

        if verdict == "NEEDS_REVISION":
            patcher_args: list[str] = []
            if args.verbose:
                patcher_args.append("--verbose")
            if getattr(args, "fix_non_blocking", False):
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
                break

            print("\n[harness] Patch applied successfully — re-running reporter + judge …")
            report_ok = _run_step_with_trace(
                "reporter",
                "reporter post-patch",
                STEP_SCRIPTS["reporter"],
                args,
            )
            results[f"reporter_post_patch_r{round_num}"] = report_ok
            if not report_ok:
                results["judge"] = False
                break
            continue

        print("[harness] Judge step failed or returned non-actionable verdict — stopping.")
        break


def _run_fix_from_existing_judge(args: argparse.Namespace, results: dict[str, bool]) -> None:
    raw_path = JUDGE_SESSION_VERDICT_RAW
    if not raw_path.exists():
        print("[harness] --patcher-only: execution/judge_session_verdict_raw.json not found.")
        print("          Run the full pipeline first to generate a judge report.")
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
    if getattr(args, "fix_non_blocking", False):
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


def _resolve_run_range(args: argparse.Namespace) -> tuple[str, str]:
    from_step = _canonical_step(getattr(args, "from_step", None))
    until_step = _canonical_step(getattr(args, "until_step", None))

    shorthands = [step for step in STEPS if getattr(args, step, False)]
    shorthands += [
        canonical
        for legacy, canonical in LEGACY_STEP_ALIASES.items()
        if getattr(args, legacy, False)
    ]

    if shorthands and (from_step or until_step):
        _die(f"Cannot mix --{shorthands[0]} shorthand with --from/--until flags.")

    unique_shorthands = list(dict.fromkeys(shorthands))
    if len(unique_shorthands) > 1:
        _die(
            "Only one step shorthand allowed at a time, got: "
            + " ".join(f"--{step}" for step in unique_shorthands)
        )

    if unique_shorthands:
        return unique_shorthands[0], unique_shorthands[0]

    if until_step and not from_step:
        _die(f"--until-{until_step} requires --from-<step>.")

    if not from_step and not until_step:
        return STEPS[0], STEPS[-1]

    if from_step not in STEPS:
        _die(f"Unknown --from step: {from_step}")

    if until_step and until_step not in STEPS:
        _die(f"Unknown --until step: {until_step}")

    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step) if until_step else len(STEPS) - 1
    until_step = until_step or STEPS[-1]

    if from_idx > until_idx:
        order = " → ".join(STEPS)
        _die(f"--from-{from_step} comes after --until-{until_step}.\n  Order: {order}")

    return from_step, until_step


def _print_dry_run(from_step: str, until_step: str, args: argparse.Namespace) -> None:
    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)

    print("\n[harness] DRY RUN — nothing will be executed.")
    print(f"  Project         : {os.environ.get('PIPELINE_PROJECT', '?')}")
    print(f"  Scope           : {args.scope}")
    print(f"  Range           : {from_step} → {until_step}")
    print(f"  Artifact trace  : {'on' if args.trace_artifacts else 'off'}")
    print()

    for i, step in enumerate(STEPS):
        in_range = from_idx <= i <= until_idx
        if not in_range:
            print(f"  ⏭  {step:<14}  (skipped)")
            continue

        if args.scope == "mini" and step in {"spectracker", "specwright", "scaffolder"}:
            print(f"  ⏭  {step:<14}  (skipped: mini scope)")
            continue

        script = STEP_SCRIPTS[step]
        extra = _scope_args_for_script(script, args.scope)
        suffix = f" {' '.join(extra)}" if extra else ""
        print(f"  ▶  {step:<14}  {script}{suffix}")

    print()

    if args.scope == "full" and args.force:
        print("  --force: spectracker delta checks will be bypassed.")

    if args.scope == "mini":
        print("  mini scope: spectracker/specwright/scaffolder are skipped.")

    if args.trace_artifacts:
        print("  artifact trace: will print after every executed step.")
    else:
        print("  artifact trace: disabled by --no-trace-artifacts.")

    print()


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
    if step == "spectracker":
        if args.scope == "mini":
            skip_step("spectracker", "mini scope does not update spec delta")
            return True
        return _run_step_with_trace(step, "spectracker", STEP_SCRIPTS[step], args)

    if step == "absorber":
        return _run_step_with_trace(step, "absorber", STEP_SCRIPTS[step], args)

    if step == "clarificator":
        clarify_args: list[str] = []
        if getattr(args, "clarify_input", None):
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

    if step == "scaffolder":
        if args.scope == "mini":
            skip_step("scaffolder", "mini scope uses targeted planner; no skeleton needed")
            return True

        if delta and not delta.get("is_first_run") and not delta_requires(delta, "scaffolder"):
            skip_step(
                "scaffolder",
                "delta: scaffold-relevant spec sections unchanged — "
                "reusing scaffolder_codebase_skeleton.json",
            )
            return True

        if not check_env(STEP_ENV_KEYS.get("scaffolder", [])):
            sys.exit(1)

        ok = _run_step_with_trace(step, "scaffolder", STEP_SCRIPTS[step], args)
        if not ok:
            print("\n[harness] Scaffolder failed — stopping.")
        return ok

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


def _retry_impl_args(executor_args: list[str]) -> list[str] | None:
    if not EXECUTOR_SESSION_MANIFEST.exists():
        print(
            "[harness] --retry-impl: executor_session_manifest.json not found "
            "— run executor first."
        )
        return None

    try:
        record = json.loads(EXECUTOR_SESSION_MANIFEST.read_text())
        failed = record.get("failed_files", [])

        if not failed:
            print(
                "[harness] --retry-impl: no failed_files in "
                "executor_session_manifest.json — nothing to retry."
            )
            return None

        executor_args += ["--only-files", ",".join(failed)]
        print(f"[harness] --retry-impl: retrying {len(failed)} failed file(s).")
        return executor_args

    except Exception:
        print("[harness] --retry-impl: could not read executor_session_manifest.json.")
        return None


def _print_impl_failed() -> None:
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
# Summary + apply record
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(
    results: dict[str, bool],
    delta: dict | None,
    args: argparse.Namespace,
    tests_passed: bool,
) -> None:
    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")

    print(f"  Scope          : {args.scope}")
    print(f"  Artifact trace : {'on' if args.trace_artifacts else 'off'}")

    if delta and not delta.get("is_first_run"):
        from_version = delta.get("from_version") or "?"
        to_version = delta.get("to_version", "?")
        affected_count = len(delta.get("affected_files", []))
        print(f"  Spec           : {from_version} → {to_version}  ({affected_count} file(s) affected)")

    for key, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {key}")

    all_ok = all(results.values())
    print(f"\n  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")

    print("\n  Reports:")
    if CLARIFICATOR_SESSION_QUESTIONS.exists() and results.get("clarificator"):
        print(f"    Clarificator → {CLARIFICATOR_SESSION_QUESTIONS}")

    print(f"    Pipeline     → {REPORTER_EXECUTION_SUMMARY}")

    if results.get("judge") and tests_passed:
        print(f"    Judge        → {JUDGE_VERDICT_SUMMARY}")
        judge_verdict = _read_judge_verdict()
        if judge_verdict in ("APPROVED_WITH_NOTES", "NEEDS_REVISION"):
            print(f"\n  Judge verdict: {judge_verdict}")
            print("  Run archivist when ready:")
            print("    python pipeline/13_archivist.py")


def _load_write_applied():
    candidates = [
        ROOT / "pipeline" / "01_spectracker.py",
        ROOT / "pipeline" / "spec_diff.py",
    ]

    for path in candidates:
        if not path.exists():
            continue

        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        write_applied = getattr(module, "write_applied", None)
        if callable(write_applied):
            return write_applied

    return None


def _write_apply_record(delta: dict, results: dict[str, bool]) -> None:
    write_applied = _load_write_applied()
    if write_applied is None:
        print("[harness] WARNING: could not load spectracker.write_applied")
        return

    applied_steps = [key for key, value in results.items() if value]
    write_applied(
        version=delta.get("to_version", "unknown"),
        steps=applied_steps,
        status="PASS",
    )

    print(
        f"\n  Apply record → {SPECTRACKER_APPLIED}  "
        f"(v{delta.get('to_version', '?')} marked as applied)"
    )


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
  python harness.py
  python harness.py --project demo
  python harness.py --project demo --scope full --from-clarificator
  python harness.py --project demo --scope full --from-executor --until-debugger
  python harness.py --project demo --scope mini --from-clarificator --until-executor
  python harness.py --project demo --dry-run
  python harness.py --project demo --no-trace-artifacts

Legacy aliases still work:
  python harness.py --from-implement --until-test --dry-run
  python harness.py --scaffold
""",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Project name → artifacts_<slug>/. "
            "If omitted, interactive prompt lists existing projects."
        ),
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

    # Hidden legacy aliases.
    for legacy, canonical in LEGACY_STEP_ALIASES.items():
        parser.add_argument(
            f"--from-{legacy}",
            dest="from_step",
            action="store_const",
            const=canonical,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            f"--until-{legacy}",
            dest="until_step",
            action="store_const",
            const=canonical,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            f"--{legacy}",
            dest=legacy,
            action="store_true",
            help=argparse.SUPPRESS,
        )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps that would run without executing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all full-scope steps even if spectracker delta says nothing changed.",
    )
    parser.add_argument(
        "--only-qwen",
        action="store_true",
        help="Skip planner step; executor runs in single-call mode.",
    )
    parser.add_argument(
        "--retry-impl",
        action="store_true",
        help=(
            "Re-implement only files listed as failed in "
            "execution/executor_session_manifest.json."
        ),
    )
    parser.add_argument(
        "--max-judge-rounds",
        type=int,
        default=2,
        metavar="N",
        help="Max judge→patcher→re-judge iterations.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        metavar="N",
        help="Max vitest→repair outer loops.",
    )
    parser.add_argument(
        "--max-cluster-attempts",
        type=int,
        default=2,
        metavar="N",
        help="Max LLM repair calls per failing cluster.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cluster debug output during debugger step.",
    )
    parser.add_argument(
        "--skip-fix",
        action="store_true",
        help="Run judge but skip auto-patcher step.",
    )
    parser.add_argument(
        "--patcher-only",
        action="store_true",
        help=(
            "Consume existing execution/judge_session_verdict_raw.json, "
            "then run patcher and one post-patch judge."
        ),
    )
    parser.add_argument(
        "--clarify-input",
        type=str,
        default=None,
        metavar="FILE",
        help="Pass file path as input to clarificator step.",
    )

    parser.add_argument(
        "--trace-artifacts",
        dest="trace_artifacts",
        action="store_true",
        default=True,
        help=(
            "Print artifact reads/writes and actual changes after each step. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-trace-artifacts",
        dest="trace_artifacts",
        action="store_false",
        help="Disable default artifact tracing.",
    )

    # Legacy mini mode.
    parser.add_argument(
        "--mini",
        type=str,
        default=None,
        metavar="PROMPT",
        help=(
            "Legacy mini mode delegated to pipeline/mini_mode.py. "
            "Prefer --scope mini."
        ),
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

    if not args.project:
        args.project = _interactive_project_select(ROOT)

    os.environ["PIPELINE_PROJECT"] = args.project

    from artifacts.paths import ensure_dirs, artifact_root

    ensure_dirs()

    print(f"[harness] Project        : {args.project}")
    print(f"[harness] Workspace      : {artifact_root()}")
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

    # --patcher-only special flow.
    if args.patcher_only:
        results: dict[str, bool] = {}

        if args.dry_run:
            print(
                "\n[harness] DRY RUN — would consume existing "
                "execution/judge_session_verdict_raw.json and run patcher."
            )
            return

        _run_fix_from_existing_judge(args, results)
        _print_summary(results, delta=None, args=args, tests_passed=True)
        sys.exit(0 if all(results.values()) else 1)

    from_step, until_step = _resolve_run_range(args)

    if args.dry_run:
        _print_dry_run(from_step, until_step, args)
        return

    from_idx = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)
    steps_to_run = STEPS[from_idx:until_idx + 1]

    print(f"[harness] Steps          : {from_step} → {until_step}\n")

    results: dict[str, bool] = {}

    # Spectracker delta preflight when starting mid-pipeline.
    delta: dict | None = None
    needs_delta = (
        args.scope == "full"
        and any(step in steps_to_run for step in ("scaffolder", "planner", "executor"))
    )

    if needs_delta and "spectracker" not in steps_to_run:
        _run_step_with_trace(
            "spectracker",
            "spectracker preflight",
            STEP_SCRIPTS["spectracker"],
            args,
        )
        delta = load_delta()

        if delta:
            print_delta_summary(delta)

        if args.force:
            print("[harness] --force: delta ignored — all steps will re-run.")
            delta = None

    elif args.scope == "mini":
        print("[harness] mini scope: skipping spectracker/specwright/scaffolder contracts.")

    tests_passed = True
    plan_available = (
        PLANNER_MINI_PLAN.exists()
        if args.scope == "mini"
        else PLANNER_FULL_PLAN.exists()
    )

    for step in steps_to_run:
        ok = _run_step(step, args, delta, plan_available, results, tests_passed)
        results[step] = ok

        if step == "spectracker" and ok and args.scope == "full":
            delta = load_delta()
            if delta:
                print_delta_summary(delta)
            if args.force:
                print("[harness] --force: delta ignored — all steps will re-run.")
                delta = None

        if step == "planner" and ok:
            plan_available = True

        if step == "debugger":
            tests_passed = ok

        if step == "executor" and ok:
            snapshot_src()

        if not ok and step in ("scaffolder", "executor"):
            print(f"\n[harness] {step} failed — stopping pipeline.")
            sys.exit(1)

    _print_summary(results, delta, args, tests_passed)

    all_ok = all(results.values())

    if args.scope == "full" and all_ok and delta and "executor" in results:
        _write_apply_record(delta, results)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
