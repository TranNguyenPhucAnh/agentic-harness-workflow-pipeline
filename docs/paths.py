"""
artifacts/paths.py
==================
SOURCE OF TRUTH cho tất cả artifact paths trong pipeline.

RULE: Không file nào được tự define artifact path — chỉ import từ đây.

Naming convention
─────────────────
  <owner>_<semantic 2-3 words>.<ext>

  .json = machine-readable
  .md   = human-readable

  _raw     = unprocessed output, consumed as-is by downstream
  _log     = append-only, tích lũy across sessions (long-term memory)
  _session = overwrite mỗi pipeline run, không tích lũy

  Suffixes human-readable (.md):
    _summary   = condensed overview, rút gọn từ nhiều nguồn
    _synthesis = rewrite/enrich từ nhiều nguồn thành document liền mạch
    _synopsis  = high-level narrative, không đi vào chi tiết

Module prefixes (owner):
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
  spectracker   01_spectracker.py

Project isolation
─────────────────
Mỗi pipeline run được gắn với một project name (bắt buộc).
harness.py nhận --project PROJECT_NAME và set env var PIPELINE_PROJECT trước khi
spawn child processes. Tất cả scripts đọc env var này qua _artifact_root() dưới đây.

Artifact root per-project:
    <repo_root>/artifacts_<project_slug>/
        state/
        cache/
        execution/
        knowledge/current/
        knowledge/history/
        reports/

Slug rule: lowercase, ký tự không phải alphanumeric thay bằng dấu gạch ngang,
dấu gạch ngang đầu/cuối bị trim.
    "My App 1"  → "my-app-1"
    "dashboard" → "dashboard"
    "IoT_MLOps" → "iot-mlops"

Usage trong mỗi pipeline script:
    from artifacts.paths import STATE_DIR, PLANNER_FULL_PLAN, ensure_dirs
    ensure_dirs()
    # paths tự động resolve theo PIPELINE_PROJECT env var

Ownership được ghi rõ trên mỗi path:
  - owner     = script DUY NHẤT được ghi file này
  - consumers = scripts chỉ được đọc (read-only)
  - lifecycle = session (overwrite) | persistent (append hoặc overwrite intentionally)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


# ── Project resolution ────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """
    Convert project name to a safe directory slug.
    "My App 1" → "my-app-1",  "IoT_MLOps" → "iot-mlops"
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Project name '{name}' produces an empty slug.")
    return slug


def _resolve_project() -> str:
    """
    Read project name from PIPELINE_PROJECT env var.
    Raises RuntimeError if not set — every run must specify a project.
    """
    name = os.environ.get("PIPELINE_PROJECT", "").strip()
    if not name:
        raise RuntimeError(
            "PIPELINE_PROJECT env var is not set.\n"
            "  Local:  python harness.py --project <name>\n"
            "  CI/CD:  add PIPELINE_PROJECT to your workflow env."
        )
    return name


def get_project_name() -> str:
    """Return the active project name (raw, not slugified)."""
    return _resolve_project()


def get_project_slug() -> str:
    """Return the active project slug used as directory suffix."""
    return _slugify(_resolve_project())


# ── Roots ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
ROOT = REPO_ROOT  # backward-compatible alias


def _artifact_root() -> Path:
    return REPO_ROOT / f"artifacts_{get_project_slug()}"


def artifact_root() -> Path:
    """Public alias for _artifact_root()."""
    return _artifact_root()


def project_info() -> dict[str, str]:
    name = get_project_name()
    return {
        "name": name,
        "slug": _slugify(name),
        "artifact_root": str(_artifact_root()),
    }


# ── LazyPath ──────────────────────────────────────────────────────────────────

class _LazyPath:
    """
    A Path-like object whose concrete location is resolved lazily at access time
    from the current PIPELINE_PROJECT env var.

    This lets all path constants be defined at module level (so `from
    artifacts.paths import PLANNER_FULL_PLAN` works) while still picking up the
    correct project-scoped root when the path is actually *used*.
    """
    __slots__ = ("_rel",)

    def __init__(self, rel: str) -> None:
        self._rel = rel

    def _resolve(self) -> Path:
        return _artifact_root() / self._rel

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __repr__(self) -> str:
        return f"LazyPath({self._rel!r} → {self._resolve()})"

    def __eq__(self, other: object) -> bool:
        return self._resolve() == Path(other) if other is not None else False

    def __hash__(self) -> int:
        return hash(self._resolve())

    def __truediv__(self, other: str | os.PathLike[str]) -> Path:
        return self._resolve() / other

    def __rtruediv__(self, other: str | os.PathLike[str]) -> Path:
        return Path(other) / self._resolve()

    @property
    def parent(self) -> Path:
        return self._resolve().parent

    @property
    def name(self) -> str:
        return self._resolve().name

    @property
    def stem(self) -> str:
        return self._resolve().stem

    @property
    def suffix(self) -> str:
        return self._resolve().suffix

    def read_text(self, **kwargs: Any) -> str:
        return self._resolve().read_text(**kwargs)

    def write_text(self, data: str, **kwargs: Any) -> int:
        return self._resolve().write_text(data, **kwargs)

    def read_bytes(self) -> bytes:
        return self._resolve().read_bytes()

    def write_bytes(self, data: bytes) -> int:
        return self._resolve().write_bytes(data)

    def open(self, *args: Any, **kwargs: Any):
        return self._resolve().open(*args, **kwargs)

    def exists(self) -> bool:
        return self._resolve().exists()

    def is_file(self) -> bool:
        return self._resolve().is_file()

    def is_dir(self) -> bool:
        return self._resolve().is_dir()

    def mkdir(self, **kwargs: Any) -> None:
        return self._resolve().mkdir(**kwargs)

    def stat(self):
        return self._resolve().stat()

    def unlink(self, **kwargs: Any) -> None:
        return self._resolve().unlink(**kwargs)

    def rename(self, target: str | os.PathLike[str]) -> Path:
        return self._resolve().rename(target)

    def relative_to(self, *args: Any) -> Path:
        return self._resolve().relative_to(*args)

    def with_name(self, name: str) -> Path:
        return self._resolve().with_name(name)

    def with_suffix(self, suffix: str) -> Path:
        return self._resolve().with_suffix(suffix)

    def glob(self, pattern: str):
        return self._resolve().glob(pattern)

    def rglob(self, pattern: str):
        return self._resolve().rglob(pattern)

    def iterdir(self):
        return self._resolve().iterdir()


# ── Directory constants ───────────────────────────────────────────────────────

STATE_DIR     = _LazyPath("state")
CACHE_DIR     = _LazyPath("cache")
EXECUTION_DIR = _LazyPath("execution")          # renamed from run/
KNOWLEDGE_DIR = _LazyPath("knowledge")
CURRENT_DIR   = _LazyPath("knowledge/current")
HISTORY_DIR   = _LazyPath("knowledge/history")
REPORTS_DIR   = _LazyPath("reports")
SRC_DIR       = _LazyPath("src")
TESTS_DIR     = _LazyPath("tests")

# Backward-compatible alias — remove after all scripts migrated
RUN_DIR = EXECUTION_DIR


# ── Root-level ────────────────────────────────────────────────────────────────

# owner: specwright (05_specwright.py)
# consumers: spectracker, scaffolder, planner, executor, judge, harness
# lifecycle: persistent — overwrite when specwright regenerates spec
# note: slug in filename allows cross-project spec extraction without renaming
SPEC_PATH = _LazyPath("specwright_spec_{slug}.md")  # resolved dynamically, see get_spec_path()


def get_spec_path() -> Path:
    """Return the spec path with slug resolved. Use instead of SPEC_PATH directly."""
    return _artifact_root() / f"specwright_spec_{get_project_slug()}.md"


# ── state/ ────────────────────────────────────────────────────────────────────

# owner: clarificator (03_clarificator.py)
# consumers: enricher, specwright (fallback)
# lifecycle: persistent — overwrite per clarification session
CLARIFIED_REQ = _LazyPath("state/clarificator_requirement_synthesis.md")

# owner: scaffolder (06_scaffolder.py)
# consumers: planner, executor, reporter, judge, harness
# lifecycle: persistent — overwrite when scaffolder reruns
SCAFFOLD_JSON = _LazyPath("state/scaffolder_file_scaffold.json")

# owner: planner (07_planner.py)
# consumers: executor, debugger, reporter, judge, patcher, harness
# lifecycle: persistent — overwrite per full-scope plan
PLANNER_FULL_PLAN = _LazyPath("state/planner_full_execution_plan.json")

# owner: planner (07_planner.py)
# consumers: executor, debugger, reporter, judge, patcher, harness
# lifecycle: persistent — overwrite per mini-scope plan
PLANNER_MINI_PLAN = _LazyPath("state/planner_mini_execution_plan.json")

# owner: planner (07_planner.py)
# consumers: executor
# lifecycle: persistent — overwrite per mini-scope run
# purpose: planner analysis of which files are impacted by mini task
PLANNER_MINI_IMPACT = _LazyPath("state/planner_mini_impact_analysis.json")

# owner: spectracker (01_spectracker.py)
# consumers: harness (to determine if spec was already applied)
# lifecycle: persistent — updated after each successful pipeline run
SPECTRACKER_APPLIED = _LazyPath("state/spectracker_applied_version.json")

# Backward-compatible aliases — remove after all scripts migrated
PLAN_JSON  = PLANNER_FULL_PLAN
PLAN_MINI  = PLANNER_MINI_PLAN
PLAN       = PLANNER_FULL_PLAN
SPEC_APPLIED = SPECTRACKER_APPLIED
CLARIFIED_REQUEST = CLARIFIED_REQ

# REMOVED: state/plan_notes.json (archivist_planner_injections)
# Rationale: planner reads archivist_knowledge_log.md directly from knowledge/current/
# No need for a separate injection file — merged into knowledge base.

# REMOVED: state/enriched_prompt.md
# Moved to execution/ as enricher_session_enriched_prompt.md (session artifact)


# ── cache/ ────────────────────────────────────────────────────────────────────

# owner: scaffolder (06_scaffolder.py)
# consumers: planner, executor, patcher
# lifecycle: persistent — overwrite when spec changes significantly
# purpose: compressed/summarized version of spec for prompt injection
SPECTRACKER_COMPRESSED_SPEC = _LazyPath("cache/spectracker_compressed_spec.md")

# owner: spectracker (01_spectracker.py)
# consumers: harness (decides which steps to rerun)
# lifecycle: session — overwrite each run
# purpose: structured diff between current and previous spec version
SPECTRACKER_VERSION_DELTA = _LazyPath("cache/spectracker_session_version_delta.json")

# owner: absorber (02_absorber.py)
# consumers: clarificator, enricher, planner
# lifecycle: session — overwrite each absorber run (point-in-time snapshot)
ABSORBER_CODEBASE_SNAPSHOT = _LazyPath("cache/absorber_codebase_snapshot.json")

# owner: absorber (02_absorber.py)
# consumers: clarificator, enricher
# lifecycle: session — overwrite each absorber run (point-in-time git snapshot)
# note: moved from knowledge/history — this is a snapshot, not a log
ABSORBER_GIT_SNAPSHOT = _LazyPath("cache/absorber_session_git_snapshot.json")

# Backward-compatible aliases
SPEC_COMPRESSED = SPECTRACKER_COMPRESSED_SPEC
SPEC_DELTA      = SPECTRACKER_VERSION_DELTA
ABSORBER_CACHE  = ABSORBER_CODEBASE_SNAPSHOT


# ── execution/ (renamed from run/) ───────────────────────────────────────────

# owner: clarificator (03_clarificator.py)
# consumers: enricher, planner (machine-readable session metadata)
# lifecycle: session — overwrite per clarification session
# purpose: structured session metadata: decisions array, tier counts, conflicts, unresolved
CLARIFICATOR_SESSION_RAW = _LazyPath("execution/clarificator_session_raw.json")

# owner: clarificator (03_clarificator.py)
# consumers: human review
# lifecycle: session — overwrite per clarification session
# purpose: human-readable questions generated for current session
CLARIFICATOR_SESSION_QUESTIONS = _LazyPath("execution/clarificator_session_questions.md")

# owner: enricher (04_enricher.py)
# consumers: specwright
# lifecycle: session — overwrite per enricher run
# purpose: structured prompt enriched with knowledge layer, sent to specwright
ENRICHER_SESSION_PROMPT = _LazyPath("execution/enricher_session_enriched_prompt.md")

# owner: executor (08_executor.py)
# consumers: reporter, judge, patcher, harness
# lifecycle: session — overwrite per executor run
# purpose: manifest of files implemented, status per file, scope
EXECUTOR_SESSION_MANIFEST = _LazyPath("execution/executor_session_manifest.json")

# owner: debugger (09_debugger.py)
# consumers: reporter, judge, archivist
# lifecycle: session — overwrite per debug run
# purpose: summarized test results: pass/fail counts, cluster info, escalations
DEBUGGER_SESSION_TEST_SUMMARY = _LazyPath("execution/debugger_session_test_summary.json")

# owner: judge (11_judge.py)
# consumers: patcher, archivist, harness
# lifecycle: session — overwrite per judge run
# purpose: raw judge verdict JSON as returned by model, unprocessed
JUDGE_SESSION_VERDICT_RAW = _LazyPath("execution/judge_session_verdict_raw.json")

# owner: patcher (12_patcher.py)
# consumers: human review
# lifecycle: session — overwrite per patcher run
# purpose: human-readable summary of fix attempts: patched files, escalations, confirm result
PATCHER_SESSION_FIX_SUMMARY = _LazyPath("execution/patcher_session_fix_summary.md")

# Backward-compatible aliases
IMPL_RECORD          = EXECUTOR_SESSION_MANIFEST
TEST_REPORT          = DEBUGGER_SESSION_TEST_SUMMARY
JUDGE_RAW            = JUDGE_SESSION_VERDICT_RAW
CLARIFICATION_REPORT = CLARIFICATOR_SESSION_RAW
CLARIFICATION_QUESTIONS = CLARIFICATOR_SESSION_QUESTIONS

# REMOVED: run/analysis_mini.json (ANALYSIS_MINI)
# Rationale: legacy artifact from mini_mode.py which is no longer used.
# planner_mini_impact_analysis.json in state/ serves this purpose in new flow.


# ── knowledge/current/ ───────────────────────────────────────────────────────

# owner: clarificator (03_clarificator.py)
# consumers: clarificator (next session, semantic dedup)
# lifecycle: append-only log across all sessions
# purpose: long-term Q&A memory — prevents re-asking semantically equivalent questions
CLARIFICATOR_DECISION_LOG = _LazyPath("knowledge/current/clarificator_decision_log.md")

# owner: absorber (02_absorber.py)
# consumers: clarificator, enricher, planner, executor
# lifecycle: persistent — overwrite per absorber run
ABSORBER_CODEBASE_MAP = _LazyPath("knowledge/current/absorber_codebase_map.md")

# owner: absorber (02_absorber.py)
# consumers: clarificator, enricher
# lifecycle: persistent — overwrite per absorber run
ABSORBER_CONFIG_MAP = _LazyPath("knowledge/current/absorber_config_map.json")

# owner: absorber (02_absorber.py)
# consumers: clarificator, enricher, planner
# lifecycle: persistent — overwrite per absorber run
ABSORBER_BLAME_MAP = _LazyPath("knowledge/current/absorber_blame_map.md")

# owner: patcher (12_patcher.py)
# consumers: debugger, archivist
# lifecycle: persistent — overwrite per patcher run
# purpose: patterns of bugs that were fixed — helps debugger and archivist
#          identify recurring failures across runs (regression = known bug resurfacing)
PATCHER_REGRESSION_LOG = _LazyPath("knowledge/current/patcher_regression_log.md")

# owner: archivist (13_archivist.py)
# consumers: specwright (next spec revision)
# lifecycle: persistent — append or overwrite as archivist curates
# purpose: spec gaps and edge cases surfaced by judge that spec.md doesn't yet cover
ARCHIVIST_SPEC_GAPS = _LazyPath("knowledge/current/archivist_spec_gaps.md")

# owner: archivist (13_archivist.py)
# consumers: planner, executor, debugger, patcher
# lifecycle: append-only log
# purpose: accumulated patterns and learnings from judge findings and human fixes,
#          injected into downstream prompts to prevent repeating past mistakes
ARCHIVIST_KNOWLEDGE_LOG = _LazyPath("knowledge/current/archivist_knowledge_log.md")

# Backward-compatible aliases
FINDINGS          = PATCHER_REGRESSION_LOG
FINDINGS_NOTES    = ARCHIVIST_KNOWLEDGE_LOG
SPEC_ADDENDUM     = ARCHIVIST_SPEC_GAPS
KNOWLEDGE_BASE    = ARCHIVIST_KNOWLEDGE_LOG
CODEBASE_MAP      = ABSORBER_CODEBASE_MAP
CONFIG_MAP        = ABSORBER_CONFIG_MAP
BLAME_MAP         = ABSORBER_BLAME_MAP
CLARIFICATION_LOG = CLARIFICATOR_DECISION_LOG


# ── knowledge/history/ ───────────────────────────────────────────────────────

# owner: archivist (13_archivist.py)
# consumers: human review
# lifecycle: append-only log
# purpose: audit trail of human curation decisions when reviewing judge findings
#          (which findings to apply, skip, escalate to spec)
ARCHIVIST_CURATION_LOG = _LazyPath("knowledge/history/archivist_curation_log.json")

# owner: patcher (12_patcher.py)
# consumers: human review, archivist
# lifecycle: append-only log
# purpose: longitudinal record of all patcher attempts across runs
PATCHER_ATTEMPT_LOG = _LazyPath("knowledge/history/patcher_attempt_log.json")

# owner: spectracker (01_spectracker.py)
# consumers: human review
# lifecycle: append-only log
# purpose: narrative history of all spec version changes over time
#          (distinct from version_delta which is per-session structured diff)
SPECTRACKER_VERSION_LOG = _LazyPath("knowledge/history/spectracker_version_log.md")

# Backward-compatible aliases
UPDATE_LOG     = ARCHIVIST_CURATION_LOG
FIX_LOG        = PATCHER_ATTEMPT_LOG
SPEC_CHANGELOG = SPECTRACKER_VERSION_LOG

# REMOVED: knowledge/history/git_history.json (GIT_HISTORY)
# Moved to cache/ as absorber_session_git_snapshot.json
# Rationale: this is a point-in-time snapshot, not a persistent log


# ── reports/ ─────────────────────────────────────────────────────────────────

# owner: reporter (10_reporter.py)
# consumers: human review
# lifecycle: persistent — overwrite per reporter run
# purpose: human-readable summary of the full pipeline execution
REPORTER_EXECUTION_SUMMARY = _LazyPath("reports/reporter_execution_summary.md")

# owner: judge (11_judge.py)
# consumers: human review, archivist
# lifecycle: persistent — overwrite per judge run
# purpose: human-readable verdict summary with scores, blocking issues, notes
JUDGE_VERDICT_SUMMARY = _LazyPath("reports/judge_verdict_summary.md")

# Backward-compatible aliases
SUMMARY      = REPORTER_EXECUTION_SUMMARY
JUDGE_REPORT = JUDGE_VERDICT_SUMMARY


# ── ensure_dirs ───────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """
    Tạo tất cả artifact directories cho project hiện tại.
    Gọi 1 lần ở đầu mỗi script (sau khi PIPELINE_PROJECT đã được set).
    Raises RuntimeError nếu PIPELINE_PROJECT chưa được set.
    """
    root = _artifact_root()
    for rel in (
        "src",
        "tests",
        "state",
        "cache",
        "execution",           # renamed from run/
        "knowledge/current",
        "knowledge/history",
        "reports",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
