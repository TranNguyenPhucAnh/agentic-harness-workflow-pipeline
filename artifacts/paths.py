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

  _raw        = unprocessed output, consumed as-is by downstream
  _log        = append-only, tích lũy across sessions (long-term memory)
  _overwrite_ = overwritten mỗi khi owning module chạy trong cùng session,
                không tích lũy; thay thế _session_ cũ để tránh trùng semantic
                với khái niệm Session trong session isolation model

  Suffixes .md:
    _summary   = condensed overview, rút gọn từ nhiều nguồn
    _synthesis = rewrite/enrich từ nhiều nguồn thành document liền mạch
    _synopsis  = high-level narrative, không đi vào chi tiết

Module prefixes (owner):
  absorber      01_absorber.py      scan codebase, build knowledge maps
  clarificator  02_clarificator.py  clarify requirements via Q&A
  enricher      03_enricher.py      enrich context into structured prompt
  specwright    04_specwright.py    generate/update spec
  spectracker   05_spectracker.py   track spec version changes
  scaffolder    06_scaffolder.py    generate stub + test files
  planner       07_planner.py       decompose work into execution plan
  executor      08_executor.py      implement src/ files
  debugger      09_debugger.py      test + repair loop
  reporter      10_reporter.py      aggregate pipeline summary
  judge         11_judge.py         qualitative review + verdict
  patcher       12_patcher.py       fix from judge verdict
  archivist     13_archivist.py     distill knowledge, long-term memory

Project isolation
─────────────────
Mỗi pipeline run được gắn với một project name (bắt buộc).
harness.py nhận --project PROJECT_NAME và set env var PIPELINE_PROJECT trước khi
spawn child processes. Tất cả scripts đọc env var này qua _artifact_root() dưới đây.

Slug rule: lowercase, ký tự không phải alphanumeric thay bằng dấu gạch ngang,
dấu gạch ngang đầu/cuối bị trim.
    "My App 1"  → "my-app-1"
    "dashboard" → "dashboard"
    "IoT_MLOps" → "iot-mlops"

Session isolation
─────────────────
Mỗi Session là một logical unit of work (implement một spec version đến judge APPROVED).
Một Session có thể gồm nhiều Runs — mỗi Run là một lần invoke harness.py.

harness.py set PIPELINE_SESSION (incremental int, zero-padded: "001", "002", ...)
trước khi spawn child processes. _session_root() resolve từ cả hai env vars.

Scope của từng artifact:
  session-local : state/, cache/, execution/, reports/
                  → nằm trong sessions/<NNN>/
                  → isolated per session, runs trong cùng session share artifacts
  project-global: knowledge/, session_runs/, specwright_spec_<slug>.md,
                  state/spectracker_applied_version.json, src/, tests/
                  → nằm trực tiếp dưới artifact_root()
                  → shared across all sessions

Artifact root layout:
    <repo_root>/artifacts_<project_slug>/
        specwright_spec_<slug>.md          ← project-global
        state/
            spectracker_applied_version.json  ← project-global exception (xem Special Notes)
        sessions/
            001/
                state/                     ← session-local
                cache/
                execution/
                reports/
            002/
                ...
        knowledge/
            current/
            history/
        session_runs/                      ← project-global run history
            session_001_runs.json
            session_002_runs.json
        src/                               ← build output, project-global
        tests/                             ← build output, project-global

Backward compat: nếu PIPELINE_SESSION chưa set, _session_root() fallback về
_artifact_root() để step scripts vẫn chạy đúng với old layout.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


# ── Project resolution ────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Project name '{name}' produces an empty slug.")
    return slug


def _resolve_project() -> str:
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


def _normalize_session_id(raw: str | int) -> str:
    """Zero-pad session id to 3 digits. Accepts int or string."""
    return f"{int(raw):03d}"


def _resolve_session() -> str | None:
    """Return normalized session id, or None if PIPELINE_SESSION not set."""
    raw = os.environ.get("PIPELINE_SESSION", "").strip()
    if not raw:
        return None
    return _normalize_session_id(raw)


def get_session_id() -> str | None:
    """Return the active session id (zero-padded), or None if not in session mode."""
    return _resolve_session()


def _session_root() -> Path:
    """
    Return the session-local artifact root.
    Falls back to _artifact_root() when PIPELINE_SESSION is not set,
    preserving backward compat with old single-session layout.
    """
    sid = _resolve_session()
    if sid is None:
        return _artifact_root()
    return _artifact_root() / "sessions" / sid


def session_root() -> Path:
    """Public alias for _session_root()."""
    return _session_root()


def project_info() -> dict[str, str]:
    name = get_project_name()
    return {
        "name": name,
        "slug": _slugify(name),
        "artifact_root": str(_artifact_root()),
    }


def get_spec_path() -> Path:
    """
    Return resolved spec path with project slug embedded.
    Always use this function — not SPEC_PATH — for actual file operations.
    Slug in filename enables cross-project spec extraction without renaming.
    owner: specwright (04_specwright.py)
    """
    return _artifact_root() / f"specwright_spec_{get_project_slug()}.md"


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


# ── Scoped LazyPath ───────────────────────────────────────────────────────────

class _SessLazyPath(_LazyPath):
    """
    Like _LazyPath but resolves under _session_root() instead of _artifact_root().
    Used for session-local artifacts: state/, cache/, execution/, reports/.
    Falls back to _artifact_root() when PIPELINE_SESSION is not set.
    """
    def _resolve(self) -> Path:
        return _session_root() / self._rel


# ── Directory constants ───────────────────────────────────────────────────────
# session-local dirs use _SessLazyPath; project-global dirs use _LazyPath.

STATE_DIR        = _SessLazyPath("state")
CACHE_DIR        = _SessLazyPath("cache")
EXECUTION_DIR    = _SessLazyPath("execution")      # renamed from run/
REPORTS_DIR      = _SessLazyPath("reports")
KNOWLEDGE_DIR    = _LazyPath("knowledge")           # project-global
CURRENT_DIR      = _LazyPath("knowledge/current")   # project-global
HISTORY_DIR      = _LazyPath("knowledge/history")   # project-global
SESSION_RUNS_DIR = _LazyPath("session_runs")        # project-global
SRC_DIR          = _LazyPath("src")                 # project-global (build output)
TESTS_DIR        = _LazyPath("tests")               # project-global (build output)

# Backward-compatible alias — remove after all scripts migrated off run/
RUN_DIR = EXECUTION_DIR


# ── state/ (session-local) ───────────────────────────────────────────────────

# owner:     clarificator (02_clarificator.py)
# consumers: enricher, specwright (fallback if enriched prompt absent)
# lifecycle: persistent within session — overwrite per clarification run
# purpose:   raw requirement rewritten inline with all clarification decisions resolved
# scope:     session-local
CLARIFIED_REQ = _SessLazyPath("state/clarificator_requirement_synthesis.md")

# owner:     scaffolder (06_scaffolder.py)
# consumers: planner, executor, reporter, judge, harness
# lifecycle: persistent within session — overwrite when scaffolder reruns
# purpose:   full stub file tree: signatures, interfaces, JSDoc, test skeletons.
#            Also carries implementation_instructions.for_executor read by executor.
# scope:     session-local
SCAFFOLD_JSON = _SessLazyPath("state/scaffolder_codebase_skeleton.json")

# owner:     planner (07_planner.py)
# consumers: executor, debugger, reporter, judge, patcher, harness
# lifecycle: persistent within session — overwrite per full-scope run
# purpose:   per-file implementation tasks, dependency order, gotchas, Tailwind hints.
#            Immutable after planner writes — patcher and debugger read only.
# scope:     session-local
PLANNER_FULL_PLAN = _SessLazyPath("state/planner_full_execution_plan.json")

# owner:     planner (07_planner.py)
# consumers: executor, debugger, reporter, judge, patcher, harness
# lifecycle: persistent within session — overwrite per mini-scope run
# scope:     session-local
PLANNER_MINI_PLAN = _SessLazyPath("state/planner_mini_execution_plan.json")

# owner:     planner (07_planner.py)
# consumers: executor
# lifecycle: persistent within session — overwrite per mini-scope run
# purpose:   planner analysis of which files are impacted by the mini task scope
# scope:     session-local
PLANNER_MINI_IMPACT = _SessLazyPath("state/planner_mini_impact_analysis.json")

# ── state/ (project-global exception) ────────────────────────────────────────

# owner:     spectracker (05_spectracker.py)
# consumers: spectracker (self-read for delta diff), harness
# lifecycle: hybrid — top-level fields overwrite each run;
#            embedded run_history[] array is append-only
# purpose:   tracks currently applied spec version + run history for delta computation
# scope:     PROJECT-GLOBAL — intentional exception; must persist across sessions so
#            spectracker can compute delta from the last successfully applied version.
#            If session-local, each new session would lose applied baseline → full rerun.
# note:      In full harness runs, updated at finalization time by harness
#            calling spectracker.write_applied(); ownership remains spectracker.
SPECTRACKER_APPLIED = _LazyPath("state/spectracker_applied_version.json")

# ── Backward-compatible aliases (state/) ─────────────────────────────────────
PLAN_JSON         = PLANNER_FULL_PLAN
PLAN_MINI         = PLANNER_MINI_PLAN
PLAN              = PLANNER_FULL_PLAN
SPEC_APPLIED      = SPECTRACKER_APPLIED
CLARIFIED_REQUEST = CLARIFIED_REQ
# REMOVED: state/plan_notes.json (PLAN_NOTES)
#   Rationale: merged into archivist_knowledge_log.md — planner reads knowledge log directly.
# REMOVED: state/enriched_prompt.md (ENRICHED_PROMPT)
#   Rationale: moved to execution/ as enricher_session_enriched_prompt.md (session artifact).


# ── cache/ (session-local) ───────────────────────────────────────────────────

# owner:     spectracker (05_spectracker.py)
# consumers: harness (decides which steps to rerun)
# lifecycle: overwrite — replaced each time spectracker runs within the session
# purpose:   structured diff between current and previous spec version:
#            changed_sections, affected_files, rerun_steps.
#            Exception: in cache/ but drives harness control flow.
# scope:     session-local
SPECTRACKER_VERSION_DELTA = _SessLazyPath("cache/spectracker_overwrite_version_delta.json")

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher, planner
# lifecycle: overwrite — point-in-time codebase snapshot, replaced each absorber run
# scope:     session-local
ABSORBER_CODEBASE_SNAPSHOT = _SessLazyPath("cache/absorber_overwrite_codebase_snapshot.json")

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher
# lifecycle: overwrite — point-in-time git state, replaced each absorber run
# note:      moved from knowledge/history/ — snapshot semantics, not a persistent log
# scope:     session-local
ABSORBER_GIT_SNAPSHOT = _SessLazyPath("cache/absorber_overwrite_git_snapshot.json")

# ── Backward-compatible aliases (cache/) ─────────────────────────────────────
SPEC_DELTA      = SPECTRACKER_VERSION_DELTA
ABSORBER_CACHE  = ABSORBER_CODEBASE_SNAPSHOT
# REMOVED: cache/absorber_cache.json → now ABSORBER_CODEBASE_SNAPSHOT
# REMOVED: knowledge/history/git_history.json (GIT_HISTORY)
#   Rationale: point-in-time snapshot, not a log → moved to cache/ as ABSORBER_GIT_SNAPSHOT


# ── execution/ (session-local, renamed from run/) ────────────────────────────

# owner:     clarificator (02_clarificator.py)
# consumers: enricher, planner
# lifecycle: overwrite — replaced each clarification run within the session
# purpose:   structured run metadata: decisions[], tier counts, conflicts detected,
#            unresolved findings list, requirement hash. Machine-readable.
# scope:     session-local
CLARIFICATOR_OVERWRITE_RAW = _SessLazyPath("execution/clarificator_overwrite_raw.json")

# owner:     clarificator (02_clarificator.py)
# consumers: human review
# lifecycle: overwrite — replaced each clarification run within the session
# purpose:   human-readable questions for current run, grouped by tier and priority
# scope:     session-local
CLARIFICATOR_OVERWRITE_QUESTIONS = _SessLazyPath("execution/clarificator_overwrite_questions.md")

# owner:     enricher (03_enricher.py)
# consumers: specwright
# lifecycle: overwrite — replaced each enricher run within the session
# purpose:   structured prompt enriched with knowledge layer, passed to specwright
# scope:     session-local
ENRICHER_OVERWRITE_PROMPT = _SessLazyPath("execution/enricher_overwrite_enriched_prompt.md")

# owner:     executor (08_executor.py)
# consumers: reporter, judge, patcher, harness
# lifecycle: overwrite — replaced each executor run within the session
# purpose:   manifest of files implemented this run: status per file
#            (written/skipped/failed), run mode (full/delta/mini), model used
# scope:     session-local
EXECUTOR_OVERWRITE_MANIFEST = _SessLazyPath("execution/executor_overwrite_manifest.json")

# owner:     debugger (09_debugger.py)
# consumers: reporter, judge, archivist
# lifecycle: overwrite — replaced each debugger run within the session
# purpose:   summarized test results: pass/fail counts, per-iteration breakdown,
#            cluster-level repair details, escalated clusters list
# scope:     session-local
DEBUGGER_OVERWRITE_TEST_SUMMARY = _SessLazyPath("execution/debugger_overwrite_test_summary.json")

# owner:     judge (11_judge.py)
# consumers: patcher, archivist, harness
# lifecycle: overwrite — replaced each judge run within the session
# purpose:   raw judge verdict JSON as returned by model, fully unprocessed.
#            Preserved so patcher and archivist parse independently;
#            failures debuggable without re-calling the API.
# scope:     session-local
JUDGE_OVERWRITE_VERDICT_RAW = _SessLazyPath("execution/judge_overwrite_verdict_raw.json")

# owner:     patcher (12_patcher.py)
# consumers: human review
# lifecycle: overwrite — replaced each patcher run within the session
# purpose:   human-readable summary: patched files, escalated findings,
#            scope rejections, confirm pass/fail result
# scope:     session-local
PATCHER_OVERWRITE_FIX_SUMMARY = _SessLazyPath("execution/patcher_overwrite_fix_summary.md")

# ── Backward-compatible aliases (execution/) ─────────────────────────────────
CLARIFICATOR_SESSION_RAW       = CLARIFICATOR_OVERWRITE_RAW
CLARIFICATOR_SESSION_QUESTIONS = CLARIFICATOR_OVERWRITE_QUESTIONS
ENRICHER_SESSION_PROMPT        = ENRICHER_OVERWRITE_PROMPT
EXECUTOR_SESSION_MANIFEST      = EXECUTOR_OVERWRITE_MANIFEST
DEBUGGER_SESSION_TEST_SUMMARY  = DEBUGGER_OVERWRITE_TEST_SUMMARY
JUDGE_SESSION_VERDICT_RAW      = JUDGE_OVERWRITE_VERDICT_RAW
PATCHER_SESSION_FIX_SUMMARY    = PATCHER_OVERWRITE_FIX_SUMMARY
IMPL_RECORD             = EXECUTOR_OVERWRITE_MANIFEST
TEST_REPORT             = DEBUGGER_OVERWRITE_TEST_SUMMARY
JUDGE_RAW               = JUDGE_OVERWRITE_VERDICT_RAW
CLARIFICATION_REPORT    = CLARIFICATOR_OVERWRITE_RAW
CLARIFICATION_QUESTIONS = CLARIFICATOR_OVERWRITE_QUESTIONS
# REMOVED: run/analysis_mini.json (ANALYSIS_MINI)
#   Rationale: legacy artifact from mini_mode.py (no longer used).
#   Replaced by PLANNER_MINI_IMPACT in state/.
# REMOVED: run/mini_log.json
#   Rationale: legacy mini_mode.py artifact, entire mode deprecated.


# ── knowledge/current/ ───────────────────────────────────────────────────────

# owner:     clarificator (02_clarificator.py)
# consumers: clarificator (next session — semantic dedup of already-answered questions)
# lifecycle: append-only log across all sessions
# purpose:   long-term Q&A memory — prevents re-asking semantically equivalent
#            questions across runs; also surfaced to enricher for context continuity
CLARIFICATOR_DECISION_LOG = _LazyPath("knowledge/current/clarificator_decision_log.md")

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher, planner, executor
# lifecycle: persistent — overwrite per absorber run
ABSORBER_CODEBASE_MAP = _LazyPath("knowledge/current/absorber_codebase_map.md")

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher
# lifecycle: persistent — overwrite per absorber run
ABSORBER_CONFIG_MAP = _LazyPath("knowledge/current/absorber_config_map.json")

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher, planner
# lifecycle: persistent — overwrite per absorber run
ABSORBER_BLAME_MAP = _LazyPath("knowledge/current/absorber_blame_map.md")

# owner:     patcher (12_patcher.py)
# consumers: debugger, archivist
# lifecycle: persistent — overwrite per patcher run
# purpose:   per-run snapshot of judge findings processed by patcher:
#            what was patched, escalated, confirm result.
#            NOT a persistent log (see patcher_attempt_log.json for that).
#            Injected into debugger prompts as regression prevention context.
PATCHER_FINDINGS_SNAPSHOT = _LazyPath("knowledge/current/patcher_findings_snapshot.md")

# owner:     archivist (13_archivist.py)
# consumers: specwright (next spec revision)
# lifecycle: persistent — append or curated overwrite by archivist
# purpose:   spec gaps and edge cases surfaced by judge, human-approved.
#            Injected into judge briefing so future runs are aware of known gaps.
#            Human-editable by design.
ARCHIVIST_SPEC_GAPS = _LazyPath("knowledge/current/archivist_spec_gaps.md")

# owner:     archivist (13_archivist.py)
# consumers: planner, executor, debugger, patcher, judge
# lifecycle: append-only log — human controls additions via interactive mode
# purpose:   accumulated architecture decisions, recurring bug patterns, lessons learned.
#            Consolidates: base.md + findings_notes.md + plan_notes.json.
#            Human-editable by design.
ARCHIVIST_KNOWLEDGE_LOG = _LazyPath("knowledge/current/archivist_knowledge_log.md")

# ── Backward-compatible aliases (knowledge/current/) ─────────────────────────
FINDINGS          = PATCHER_FINDINGS_SNAPSHOT
FINDINGS_NOTES    = ARCHIVIST_KNOWLEDGE_LOG
SPEC_ADDENDUM     = ARCHIVIST_SPEC_GAPS
KNOWLEDGE_BASE    = ARCHIVIST_KNOWLEDGE_LOG
CODEBASE_MAP      = ABSORBER_CODEBASE_MAP
CONFIG_MAP        = ABSORBER_CONFIG_MAP
BLAME_MAP         = ABSORBER_BLAME_MAP
CLARIFICATION_LOG = CLARIFICATOR_DECISION_LOG


# ── knowledge/history/ ───────────────────────────────────────────────────────

# owner:     archivist (13_archivist.py)
# consumers: human review
# lifecycle: append-only log
# purpose:   audit trail of human curation decisions when reviewing judge findings:
#            which findings were applied, skipped, or escalated to spec bump.
#            Distinct from patcher_attempt_log — archivist modifies knowledge artifacts,
#            patcher modifies src/ directly.
ARCHIVIST_CURATION_LOG = _LazyPath("knowledge/history/archivist_curation_log.json")

# owner:     patcher (12_patcher.py)
# consumers: human review, archivist
# lifecycle: append-only log
# purpose:   longitudinal record of all patcher attempts across runs:
#            files patched, judge findings that triggered each fix, outcome per file
PATCHER_ATTEMPT_LOG = _LazyPath("knowledge/history/patcher_attempt_log.json")

# owner:     spectracker (05_spectracker.py)
# consumers: human review only — no pipeline logic reads this
# lifecycle: append-only log
# purpose:   narrative history of all spec version changes over time.
#            Distinct from spectracker_overwrite_version_delta.json (per-run machine diff).
SPECTRACKER_VERSION_LOG = _LazyPath("knowledge/history/spectracker_version_log.md")

# NOTE: spectracker also writes dynamic per-version files at runtime:
#   knowledge/history/<version>.md           — raw spec snapshot at version apply time
#   knowledge/history/<version>.changelog.md — per-version changelog entry
# These are write-once, constructed by spectracker using the version string.
# Cannot be static LazyPath constants — spectracker builds paths at runtime.
# Consumed internally by spectracker._load_latest_snapshot() for future delta computation.

# ── Backward-compatible aliases (knowledge/history/) ─────────────────────────
UPDATE_LOG     = ARCHIVIST_CURATION_LOG
FIX_LOG        = PATCHER_ATTEMPT_LOG
SPEC_CHANGELOG = SPECTRACKER_VERSION_LOG


# ── reports/ (session-local) ─────────────────────────────────────────────────

# owner:     reporter (10_reporter.py)
# consumers: human review only — no pipeline script parses this
# lifecycle: overwrite — replaced each reporter run within the session
# purpose:   human-readable pipeline execution summary:
#            plan summary, test breakdown, scaffold stats, spec delta, impl record.
#            On GitHub Actions: piped to $GITHUB_STEP_SUMMARY, not committed to repo.
# scope:     session-local
REPORTER_EXECUTION_SUMMARY = _SessLazyPath("reports/reporter_execution_summary.md")

# owner:     judge (11_judge.py)
# consumers: human review only — no pipeline script parses this
# lifecycle: overwrite — replaced each judge run within the session
# purpose:   human-readable verdict: scores, blocking issues with suggested fixes,
#            non-blocking notes, spec gaps, sign-off status.
#            Primary artifact for human deciding whether to merge pipeline output.
# scope:     session-local
JUDGE_VERDICT_SUMMARY = _SessLazyPath("reports/judge_verdict_summary.md")

# ── Backward-compatible aliases (reports/) ───────────────────────────────────
SUMMARY      = REPORTER_EXECUTION_SUMMARY
JUDGE_REPORT = JUDGE_VERDICT_SUMMARY


# ── session_runs helpers ─────────────────────────────────────────────────────

def get_session_runs_path(session_id: str | int) -> Path:
    """
    Return path for a session's run history file.
    Owner: harness. Not a pipeline step artifact.
    Lifecycle: append-only at run-entry level; file is atomically rewritten on each update.
    """
    sid = _normalize_session_id(session_id)
    return _artifact_root() / "session_runs" / f"session_{sid}_runs.json"


# ── ensure_dirs ───────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """
    Tạo tất cả artifact directories cho project + session hiện tại.
    Gọi 1 lần ở đầu mỗi script (sau khi PIPELINE_PROJECT đã được set).
    Nếu PIPELINE_SESSION được set, tạo thêm session-local dirs.
    Raises RuntimeError nếu PIPELINE_PROJECT chưa được set.
    """
    project_root = _artifact_root()
    session_root = _session_root()

    # project-global dirs — always created
    for rel in (
        "src",
        "tests",
        "state",                # for SPECTRACKER_APPLIED (project-global exception)
        "knowledge/current",
        "knowledge/history",
        "session_runs",
    ):
        (project_root / rel).mkdir(parents=True, exist_ok=True)

    # session-local dirs — created under sessions/<NNN>/ when PIPELINE_SESSION is set,
    # or under project root when running without session mode (backward compat)
    for rel in (
        "state",
        "cache",
        "execution",
        "reports",
    ):
        (session_root / rel).mkdir(parents=True, exist_ok=True)
