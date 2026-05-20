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

Design:
    - Project-global artifacts live under artifacts_<slug>/
    - harness owns orchestration metadata only:
        harness_run_log.json (append-only, project root)
    - step scripts own their artifact writes.
    - harness calls spectracker.write_applied() during finalization only.

Run model:
    Run = one harness.py invocation.
    All run metadata is appended to harness_run_log.json.

Delta model (post spectracker refactor):
    spectracker_overwrite_version_delta.json contains section-level diff only.
    Fields: from_version, to_version, is_first_run, changed_sections,
            unchanged_sections, new_sections, removed_sections,
            section_summaries, baseline_source.
    There are NO affected_files, unaffected_files, or rerun_steps fields.
    Harness decides step execution policy from is_first_run + changed_sections.

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
    "absorber":     "01_absorber.py",
    "clarificator": "02_clarificator.py",
    "enricher":     "03_enricher.py",
    "specwright":   "04_specwright.py",
    "spectracker":  "05_spectracker.py",
    "scaffolder":   "06_scaffolder.py",
    "planner":      "07_planner.py",
    "executor":     "08_executor.py",
    "debugger":     "09_debugger.py",
    "reporter":     "10_reporter.py",
    "judge":        "11_judge.py",
    "patcher":      "12_patcher.py",
    "archivist":    "13_archivist.py",
}

from artifacts.models import PROVIDERS, ROLES, get_provider  # noqa: E402

# Derived from ROLES in artifacts/models.py — no manual sync needed.
_LLM_STEPS: frozenset[str] = frozenset(ROLES.keys())


def _env_key_for_step(step: str) -> list[str]:
    try:
        provider = get_provider(step)
        env_var  = PROVIDERS[provider]["api_key_env"]
        return [env_var]
    except (ValueError, KeyError):
        return []


STEP_ENV_KEYS: dict[str, list[str]] = {
    step: _env_key_for_step(step) for step in _LLM_STEPS
}

SCOPE_CHOICES = ("full", "mini")


# ════════════════════════════════════════════════════════════════════════════
# Artifact trace declaration registry (module-folder paths)
# ════════════════════════════════════════════════════════════════════════════

STEP_ARTIFACT_READS: dict[str, list[str]] = {
    "absorber": [],
    "clarificator": [
        "absorber/codebase_snapshot.json",
        "absorber/git_snapshot.json",
        "clarificator/decision_log.json",
        "absorber/codebase_map.md",
        "absorber/config_map.json",
        "absorber/blame_map.md",
    ],
    "enricher": [
        "clarificator/requirement_synthesis.md",
        "clarificator/raw.json",
        "absorber/codebase_snapshot.json",
        "absorber/git_snapshot.json",
        "clarificator/decision_log.json",
        "absorber/codebase_map.md",
        "absorber/config_map.json",
        "absorber/blame_map.md",
    ],
    "specwright": [
        "enricher/enriched_prompt.md",
        "clarificator/requirement_synthesis.md",
        "archivist/spec_gaps.md",
    ],
    "spectracker": [
        "spec/specwright_spec_<slug>.md",
        "spectracker/version_log.json",
    ],
    "scaffolder": [
        "spec/specwright_spec_<slug>.md",
    ],
    "planner": [
        "spec/specwright_spec_<slug>.md",
        "scaffolder/blueprint.json",
        "absorber/codebase_snapshot.json",
        "absorber/codebase_map.md",
        "absorber/blame_map.md",
        "archivist/knowledge_log.md",
        "clarificator/raw.json",
    ],
    "executor": [
        "spec/specwright_spec_<slug>.md",
        "scaffolder/blueprint.json",
        "planner/full_plan.json",
        "planner/mini_plan.json",
        "absorber/codebase_map.md",
        "archivist/knowledge_log.md",
    ],
    "debugger": [
        "planner/full_plan.json",
        "planner/mini_plan.json",
        "executor/manifest.json",
        "archivist/knowledge_log.md",
    ],
    "reporter": [
        "scaffolder/blueprint.json",
        "planner/full_plan.json",
        "planner/mini_plan.json",
        "executor/manifest.json",
        "debugger/test_summary.json",
        "spectracker/version_delta.json",
    ],
    "judge": [
        "spec/specwright_spec_<slug>.md",
        "scaffolder/blueprint.json",
        "planner/full_plan.json",
        "planner/mini_plan.json",
        "executor/manifest.json",
        "debugger/test_summary.json",
        "reporter/execution_summary.md",
        "archivist/knowledge_log.md",
        "archivist/spec_gaps.md",
    ],
    "patcher": [
        "judge/verdict_raw.json",
        "planner/full_plan.json",
        "planner/mini_plan.json",
        "executor/manifest.json",
        "archivist/knowledge_log.md",
    ],
    "archivist": [
        "debugger/test_summary.json",
        "judge/verdict_raw.json",
        "patcher/attempt_log.json",
    ],
}

STEP_ARTIFACT_WRITES: dict[str, list[str]] = {
    "absorber": [
        "absorber/codebase_snapshot.json",
        "absorber/git_snapshot.json",
        "absorber/codebase_map.md",
        "absorber/config_map.json",
        "absorber/blame_map.md",
        "absorber/codebase_log.json",
    ],
    "clarificator": [
        "clarificator/raw.json",
        "clarificator/questions.md",
        "clarificator/requirement_synthesis.md",
        "clarificator/decision_log.json",
    ],
    "enricher": [
        "enricher/enriched_prompt.md",
        "enricher/prompt_log.json",
    ],
    "specwright": [
        "spec/specwright_spec_<slug>.md",
    ],
    "spectracker": [
        "spectracker/version_delta.json",
        "spectracker/version_log.json",
    ],
    "scaffolder": [
        "scaffolder/blueprint.json",
        "scaffolder/skeleton_log.json",
        "output/tests/",
    ],
    "planner": [
        "planner/full_plan.json",
        "planner/mini_plan.json",
        "planner/plan_log.json",
    ],
    "executor": [
        "executor/manifest.json",
        "executor/manifest_log.json",
        "output/src/",
    ],
    "debugger": [
        "debugger/test_summary.json",
        "output/src/",
    ],
    "reporter": [
        "reporter/execution_summary.md",
    ],
    "judge": [
        "judge/verdict_raw.json",
        "judge/verdict_summary.md",
        "judge/verdict_log.json",
    ],
    "patcher": [
        "patcher/fix_summary.md",
        "patcher/attempt_log.json",
        "output/src/",
    ],
    "archivist": [
        "archivist/knowledge_log.md",
        "archivist/spec_gaps.md",
        "archivist/curation_log.json",
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


def _pipeline_script(script: str) -> Path:
    return ROOT / "pipeline" / script


def run_step(label: str, script: str, extra_args: list[str] | None = None) -> bool:
    cmd = [sys.executable, str(_pipeline_script(script))] + (extra_args or [])

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    t0      = time.time()
    result  = subprocess.run(cmd, cwd=ROOT)
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
# Run log — append-only harness_run_log.json
# ════════════════════════════════════════════════════════════════════════════

def _run_log_path() -> Path:
    from artifacts.paths import artifact_root
    return artifact_root() / "harness_run_log.json"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_run_log() -> dict[str, Any]:
    path = _run_log_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data.setdefault("runs", [])
                return data
        except Exception:
            pass
    return {"runs": []}


def _start_run_record(
    args: argparse.Namespace,
    from_step: str,
    until_step: str,
) -> str:
    data   = _load_run_log()
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    data["runs"].append(
        {
            "run_id":           run_id,
            "started_at":       _now_iso(),
            "completed_at":     None,
            "status":           "RUNNING",
            "scope":            args.scope,
            "from_step":        from_step,
            "until_step":       until_step,
            "stopped_at_step":  None,
            "steps":            [],
            "spec_version":     None,
        }
    )

    _atomic_write_json(_run_log_path(), data)
    return run_id


def _update_run_record(
    run_id: str,
    *,
    status: str | None = None,
    step: str | None = None,
    step_status: str | None = None,
    stopped_at_step: str | None = None,
    spec_version: str | None = None,
) -> None:
    data = _load_run_log()

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
                    "step":   step,
                    "status": step_status,
                    "at":     _now_iso(),
                }
            )

        break

    _atomic_write_json(_run_log_path(), data)


# ════════════════════════════════════════════════════════════════════════════
# Artifact tracing
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ArtifactFileState:
    size:     int
    mtime_ns: int
    sha256:   str
    content:  bytes


ArtifactSnapshot = dict[str, ArtifactFileState]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_artifacts() -> ArtifactSnapshot:
    from artifacts.paths import artifact_root

    root     = artifact_root()
    snapshot: ArtifactSnapshot = {}

    if not root.exists():
        return snapshot

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root).as_posix()

        if rel.startswith("output/prev_src/"):
            continue

        try:
            content = path.read_bytes()
            stat    = path.stat()
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
    created:     list[str] = []
    appended:    list[str] = []
    overwritten: list[str] = []
    touched:     list[str] = []
    deleted:     list[str] = []

    before_keys = set(before)
    after_keys  = set(after)

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
        "created":     created,
        "appended":    appended,
        "overwritten": overwritten,
        "touched":     touched,
        "deleted":     deleted,
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

    _print_artifact_list("declared reads",  STEP_ARTIFACT_READS.get(step, []))
    _print_artifact_list("declared writes", STEP_ARTIFACT_WRITES.get(step, []))

    if before is None or after is None:
        print("    actual changes:")
        print("      - (not captured)")
        return

    changes = _classify_artifact_changes(before, after)

    print("    actual changes:")
    _print_artifact_list("created",                        changes["created"],     indent="      ")
    _print_artifact_list("appended",                       changes["appended"],    indent="      ")
    _print_artifact_list("overwritten/updated",            changes["overwritten"], indent="      ")
    _print_artifact_list("touched without content change", changes["touched"],     indent="      ")
    _print_artifact_list("deleted",                        changes["deleted"],     indent="      ")


def _run_step_with_trace(
    step: str,
    label: str,
    script: str,
    args: argparse.Namespace,
    extra_args: list[str] | None = None,
) -> bool:
    before = _snapshot_artifacts() if args.trace_artifacts else None
    ok     = run_step(label, script, extra_args)
    after  = _snapshot_artifacts() if args.trace_artifacts else None

    if args.trace_artifacts:
        _print_artifact_trace(step, before, after)

    return ok


# ════════════════════════════════════════════════════════════════════════════
# Project directory cache  (.harness_projects.json)
# ════════════════════════════════════════════════════════════════════════════

_PROJECT_DIR_CACHE = ROOT / ".harness_projects.json"


def _load_project_dir_cache() -> dict[str, str]:
    if not _PROJECT_DIR_CACHE.exists():
        return {}
    try:
        data = json.loads(_PROJECT_DIR_CACHE.read_text())
        if not isinstance(data, dict):
            return {}
        # Lazy cleanup: prune stale entries whose directories no longer exist
        pruned = {k: v for k, v in data.items() if Path(v).exists()}
        if len(pruned) != len(data):
            _save_project_dir_cache(pruned)
        return pruned
    except Exception:
        return {}


def _save_project_dir_cache(cache: dict[str, str]) -> None:
    try:
        tmp = _PROJECT_DIR_CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(_PROJECT_DIR_CACHE)
    except Exception as exc:
        print(f"[harness] WARNING: could not save project dir cache: {exc}")


def _resolve_project_target_dir(slug: str) -> str | None:
    cache  = _load_project_dir_cache()
    cached = cache.get(slug)

    if cached:
        print(f"\n[harness] Absorber target directory (cached): {cached}")
        choice = input("  [1] Use this path  [2] Update path  [Enter=1]: ").strip()
        if choice != "2":
            return cached

    raw = input("  Target directory for absorber (Enter to use artifact workspace): ").strip()
    if not raw:
        if slug in cache:
            del cache[slug]
            _save_project_dir_cache(cache)
        return None

    path = Path(raw).expanduser().resolve()
    if not path.exists():
        print(f"[harness] WARNING: path does not exist: {path} — skipping cache.")
        return str(path)

    cache[slug] = str(path)
    _save_project_dir_cache(cache)
    print(f"[harness] Saved target directory → {_PROJECT_DIR_CACHE.name}")
    return str(path)


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

    return artifact_root() / "output" / "prev_src"


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


def print_delta_summary(delta: dict) -> None:
    from_version = delta.get("from_version") or "(none)"
    to_version   = delta.get("to_version", "?")
    baseline     = delta.get("baseline_source")

    print(f"\n[harness] Spec: {from_version} → {to_version}")
    if baseline:
        print(f"[harness] Baseline: {baseline}")

    if delta.get("is_first_run"):
        print("[harness] First run — full pipeline.")
    else:
        changed   = delta.get("changed_sections", [])
        new_secs  = delta.get("new_sections", [])
        removed   = delta.get("removed_sections", [])
        summaries = delta.get("section_summaries", {})

        print(f"[harness] Changed §: {changed or '(none)'}")
        print(f"[harness] New     §: {new_secs or '(none)'}")
        print(f"[harness] Removed §: {removed or '(none)'}")

        for section in changed:
            if section in summaries:
                print(f"    §{section}: {summaries[section]}")


# ════════════════════════════════════════════════════════════════════════════
# src/ snapshot
# ════════════════════════════════════════════════════════════════════════════

def snapshot_src() -> None:
    from artifacts.paths import SRC_DIR

    src = Path(SRC_DIR)
    if not src.exists():
        # Fallback: use output/src under artifact root
        from artifacts.paths import artifact_root
        src = artifact_root() / "output" / "src"

    if not src.exists():
        return

    prev_src = _prev_src_dir()
    if prev_src.exists():
        shutil.rmtree(prev_src)

    shutil.copytree(src, prev_src)
    print(f"[harness] src/ snapshot → {_artifact_rel(prev_src)}")


# ════════════════════════════════════════════════════════════════════════════
# Judge helpers
# ════════════════════════════════════════════════════════════════════════════

def _read_judge_verdict() -> str:
    from artifacts.paths import JUDGE_VERDICT_RAW

    raw_path = JUDGE_VERDICT_RAW
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
    from artifacts.paths import JUDGE_VERDICT_RAW, ARCHIVIST_CURATION_LOG

    raw_path = JUDGE_VERDICT_RAW
    if not raw_path.exists():
        print("[harness] --repair-from-judge: judge verdict_raw.json not found.")
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

    if not check_env(STEP_ENV_KEYS.get("judge", [])):
        print("[harness] WARNING: cannot re-judge without required API key.")
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

    from_step  = args.from_step
    until_step = args.until_step or STEPS[-1]

    if STEPS.index(from_step) > STEPS.index(until_step):
        _die(
            f"--from-{from_step} comes after --until-{until_step}.\n"
            f"  Order: {' → '.join(STEPS)}"
        )

    return from_step, until_step


def _print_dry_run(from_step: str, until_step: str, args: argparse.Namespace) -> None:
    from_idx  = STEPS.index(from_step)
    until_idx = STEPS.index(until_step)

    print("\n[harness] DRY RUN — nothing will be executed.")
    print(f"  Project         : {os.environ.get('PIPELINE_PROJECT', '?')}")
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
        extra  = _scope_args_for_script(script, args.scope)
        suffix = f" {' '.join(extra)}" if extra else ""
        print(f"  ▶  {step:<14}  {script}{suffix}")

    print()


# ════════════════════════════════════════════════════════════════════════════
# Executor retry helpers
# ════════════════════════════════════════════════════════════════════════════

def _retry_impl_args(executor_args: list[str]) -> list[str] | None:
    from artifacts.paths import EXECUTOR_OVERWRITE_MANIFEST

    if not EXECUTOR_OVERWRITE_MANIFEST.exists():
        print("[harness] --retry-impl: executor manifest.json not found.")
        return None

    try:
        record = json.loads(EXECUTOR_OVERWRITE_MANIFEST.read_text())
        failed = record.get("failed_files", [])

        if not failed:
            print("[harness] --retry-impl: no failed_files — nothing to retry.")
            return None

        executor_args += ["--only-files", ",".join(failed)]
        print(f"[harness] --retry-impl: retrying {len(failed)} failed file(s).")
        return executor_args

    except Exception:
        print("[harness] --retry-impl: could not read executor manifest.json.")
        return None


def _print_impl_failed() -> None:
    from artifacts.paths import EXECUTOR_OVERWRITE_MANIFEST

    if not EXECUTOR_OVERWRITE_MANIFEST.exists():
        return

    try:
        record = json.loads(EXECUTOR_OVERWRITE_MANIFEST.read_text())
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

        if not check_env(STEP_ENV_KEYS.get("scaffolder", [])):
            sys.exit(1)

        return _run_step_with_trace(step, "scaffolder", STEP_SCRIPTS[step], args)

    if step == "planner":
        if args.skip_planner:
            skip_step("planner", "--skip-planner")
            return True

        if not check_env(STEP_ENV_KEYS.get("planner", [])):
            sys.exit(1)

        planner_args = _scope_args_for_script(STEP_SCRIPTS[step], args.scope)
        ok = _run_step_with_trace(step, "planner", STEP_SCRIPTS[step], args, planner_args)

        if not ok:
            print("\n[harness] Planner failed.\n  Tip: --skip-planner to skip planning.")

        return ok

    if step == "executor":
        if not check_env(STEP_ENV_KEYS.get("executor", [])):
            sys.exit(1)

        executor_args = _scope_args_for_script(STEP_SCRIPTS[step], args.scope)

        if plan_available:
            executor_args.append("--use-planner-plan")

        if args.retry_impl:
            retry_args = _retry_impl_args(executor_args)
            if retry_args is None:
                return False
            executor_args = retry_args

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
            "--impl", "primary",
            "--max-iter", str(args.max_iter),
            "--max-cluster-attempts", str(args.max_cluster_attempts),
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
            print("[harness] WARNING: cannot run judge without required API key.")
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
        CLARIFICATOR_SESSION,
        JUDGE_VERDICT_SUMMARY,
        REPORTER_EXECUTION_SUMMARY,
    )

    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")

    print(f"  Project        : {os.environ.get('PIPELINE_PROJECT', '?')}")
    print(f"  Scope          : {args.scope}")
    print(f"  Artifact trace : {'on' if args.trace_artifacts else 'off'}")

    if delta:
        from_version = delta.get("from_version") or "(none)"
        to_version   = delta.get("to_version", "?")
        print(f"  Spec           : {from_version} → {to_version}")

    for key, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {key}")

    all_ok = all(results.values()) if results else False
    print(f"\n  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")

    print("\n  Reports:")
    if CLARIFICATOR_SESSION.exists() and results.get("clarificator"):
        print(f"    Clarificator → {CLARIFICATOR_SESSION}")

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
    spec        = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    write_applied = getattr(module, "write_applied", None)
    return write_applied if callable(write_applied) else None


def _write_apply_record(delta: dict, results: dict[str, bool]) -> None:
    write_applied = _load_write_applied()
    if write_applied is None:
        print("[harness] WARNING: could not load spectracker.write_applied")
        return

    version = delta.get("to_version", "unknown")
    write_applied(version=version, status="PASS")

    print(
        f"\n  Apply record written  "
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
  python harness.py --project demo --from-executor --until-debugger
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

    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument(
        "--skip-planner",
        dest="skip_planner",
        action="store_true",
        help="Skip the planner step entirely (use when planner fails or plan already exists).",
    )
    parser.add_argument(
        "--only-qwen",
        dest="skip_planner",
        action="store_true",
        help=argparse.SUPPRESS,  # backward-compat alias for --skip-planner
    )
    parser.add_argument("--retry-impl",            action="store_true")
    parser.add_argument("--max-judge-rounds",      type=int, default=2,  metavar="N")
    parser.add_argument("--max-iter",              type=int, default=3,  metavar="N")
    parser.add_argument("--max-cluster-attempts",  type=int, default=2,  metavar="N")
    parser.add_argument("--verbose",               action="store_true")
    parser.add_argument("--skip-fix",              action="store_true")
    parser.add_argument("--fix-non-blocking",      dest="fix_non_blocking", action="store_true")
    parser.add_argument("--repair-from-judge",     dest="repair_from_judge", action="store_true")
    parser.add_argument("--clarify-input",         type=str, default=None, metavar="FILE")

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
    parser.add_argument("--files",        nargs="+", default=None)
    parser.add_argument("--context-file", type=str,  default=None)
    parser.add_argument("--output-file",  type=str,  default=None)
    parser.add_argument("--task-type",    type=str,  default=None)

    return parser


def _maybe_resolve_absorber_target(args: argparse.Namespace) -> None:
    will_run_absorber = (
        not args.from_step
        or args.from_step == "absorber"
        or getattr(args, "absorber", False)
    )

    if not will_run_absorber:
        return

    if args.dry_run:
        return

    from artifacts.paths import get_project_slug
    slug   = get_project_slug()
    target = _resolve_project_target_dir(slug)
    if target:
        os.environ["PIPELINE_TARGET_DIR"] = target


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()

    parser = _build_parser()
    args   = parser.parse_args()

    _validate_scope(args.scope)
    _validate_execution_mode_conflicts(args)

    if not args.project:
        args.project = _interactive_project_select(ROOT)

    os.environ["PIPELINE_PROJECT"] = args.project

    _maybe_resolve_absorber_target(args)

    from artifacts.paths import (
        PLANNER_FULL_PLAN,
        PLANNER_MINI_PLAN,
        artifact_root,
        ensure_dirs,
        get_project_slug,
    )

    ensure_dirs()

    print(f"[harness] Project        : {args.project}")
    print(f"[harness] Project slug   : {get_project_slug()}")
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

    # Repair flow.
    if args.repair_from_judge:
        results: dict[str, bool] = {}

        if args.dry_run:
            print(
                "\n[harness] DRY RUN — would consume existing "
                "judge/verdict_raw.json, run patcher, "
                "refresh reporter, and re-run judge."
            )
            return

        run_id = _start_run_record(args, "patcher", "judge")

        try:
            _run_fix_from_existing_judge(args, results)
            _print_summary(results, delta=None, args=args, tests_passed=True)
            all_ok = all(results.values()) if results else False
            _update_run_record(run_id, status="PASS" if all_ok else "FAIL")
            sys.exit(0 if all_ok else 1)
        except Exception:
            _update_run_record(run_id, status="FAIL")
            raise

    from_step, until_step = _resolve_run_range(args)

    if args.dry_run:
        _print_dry_run(from_step, until_step, args)
        return

    run_id = _start_run_record(args, from_step, until_step)

    from_idx      = STEPS.index(from_step)
    until_idx     = STEPS.index(until_step)
    steps_to_run  = STEPS[from_idx:until_idx + 1]

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

        tests_passed   = True
        plan_available = PLANNER_MINI_PLAN.exists() if args.scope == "mini" else PLANNER_FULL_PLAN.exists()

        for step in steps_to_run:
            ok = _run_step(step, args, delta, plan_available, results, tests_passed)
            results[step] = ok

            _update_run_record(
                run_id,
                step=step,
                step_status="PASS" if ok else "FAIL",
            )

            if step == "spectracker" and ok and args.scope == "full":
                delta = load_delta()

                if delta:
                    print_delta_summary(delta)
                    _update_run_record(
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
            run_id,
            status="PASS" if all_ok else "FAIL",
            spec_version=delta.get("to_version") if delta else None,
        )

        sys.exit(0 if all_ok else 1)

    except KeyboardInterrupt:
        print("\n[harness] Interrupted.")
        _update_run_record(run_id, status="STOPPED")
        raise

    except Exception:
        _update_run_record(run_id, status="FAIL")
        raise


if __name__ == "__main__":
    main()
