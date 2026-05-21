"""
artifacts/paths.py
==================
SOURCE OF TRUTH cho tất cả artifact paths trong pipeline.

RULE: Không file nào được tự define artifact path — chỉ import từ đây.

Architecture (module-folder layout)
───────────────────────────────────
Mỗi module own folder riêng, write thẳng vào folder của mình. Không còn
session layer, không còn knowledge/ — audit trail thực hiện qua append-only
*_log.json trong folder của mỗi module.

Naming convention
─────────────────
  Folder = owner. Filename không cần module prefix nữa.

  Pair pattern (mỗi module, trừ vài exception):
    short-term  = overwrite mỗi run
                  .md nếu human/LLM target, .json nếu machine target
    long-term   = *_log.json, append-only, audit trail

  Cache (internal, không pair):
    <module>/cache/<name>.json

Module folders (owner):
  spec/         specwright (project-global spec file)
  absorber/     01_absorber.py      scan codebase, build knowledge maps
  clarificator/ 02_clarificator.py  clarify requirements via Q&A
  enricher/     03_enricher.py      enrich context into structured prompt
  spectracker/  05_spectracker.py   track spec version changes
  scaffolder/   06_scaffolder.py    generate stub + test files
  planner/      07_planner.py       decompose work into execution plan
  executor/     08_executor.py      implement src/ files
  debugger/     09_debugger.py      test + repair loop
  reporter/     10_reporter.py      aggregate pipeline summary
  judge/        11_judge.py         qualitative review + verdict
  patcher/      12_patcher.py       fix from judge verdict
  archivist/    13_archivist.py     distill knowledge, long-term memory
  output/       build artifacts (src/, tests/, dist/, coverage/)

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

Artifact root layout:
    <repo_root>/artifacts_<project_slug>/
        spec/
            specwright_spec_<slug>.md
        absorber/
            codebase_map.json        ← short-term
            codebase_log.json        ← long-term
            cache/
                codebase_snapshot.json
        clarificator/
            session.json             ← short-term (synthesis embedded as field)
            decision_log.json        ← long-term
        enricher/
            enriched_prompt.md       ← short-term
            prompt_log.json          ← long-term
        spectracker/
            version_delta.json       ← short-term (drives harness control flow)
            version_log.json         ← long-term (snapshot + applied state merged)
        scaffolder/
            blueprint.json           ← short-term
            skeleton_log.json        ← long-term
        planner/
            full_plan.json           ← short-term (full mode)
            mini_plan.json           ← short-term (mini mode, impact merged)
            plan_log.json            ← long-term
        executor/
            manifest.json
            manifest_log.json
        debugger/
            test_summary.json
            test_log.json
        reporter/
            execution_summary.md
            execution_log.json
        judge/
            verdict_raw.json         ← short-term (machine)
            verdict_summary.md       ← short-term (human)
            verdict_log.json         ← long-term
        patcher/
            fix_summary.md
            attempt_log.json
        archivist/
            knowledge_log.md         ← append, LLM prompt inject
            spec_gaps.md             ← append, LLM prompt inject
            curation_log.json        ← append, human audit
        output/
            src/                     ← build output (executor, debugger, patcher)
            tests/                   ← build output (scaffolder)
            dist/                    ← future
            coverage/                ← future
        harness_run_log.json         ← harness-owned, append-only run history
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
    return _artifact_root() / "spec" / f"specwright_spec_{get_project_slug()}.md"


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

    def resolve(self, **kwargs) -> Path:
        return self._resolve().resolve(**kwargs)

    def absolute(self) -> Path:
        return self._resolve().absolute()


# ── Top-level dirs ───────────────────────────────────────────────────────────

SPEC_DIR    = _LazyPath("spec")
OUTPUT_DIR  = _LazyPath("output")
SRC_DIR     = _LazyPath("output/src")     # build output (executor, debugger, patcher)
TESTS_DIR   = _LazyPath("output/tests")   # build output (scaffolder)


# ── absorber/ ────────────────────────────────────────────────────────────────

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher, planner, executor
# lifecycle: short-term — overwrite per absorber run
# purpose:   merged codebase + config + git/blame map (single source of truth
#            for codebase structure as exposed to downstream LLM steps)
ABSORBER_CODEBASE_MAP = _LazyPath("absorber/codebase_map.json")

# owner:     absorber (01_absorber.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only audit log
# purpose:   per-run history of codebase scans (counts, file deltas, scan time)
ABSORBER_CODEBASE_LOG = _LazyPath("absorber/codebase_log.json")

# owner:     absorber (01_absorber.py)
# consumers: clarificator, enricher, planner
# lifecycle: internal cache — overwrite per absorber run, no long-term pair
# purpose:   point-in-time codebase snapshot used by downstream prompts
ABSORBER_CODEBASE_SNAPSHOT = _LazyPath("absorber/cache/codebase_snapshot.json")


# ── clarificator/ ────────────────────────────────────────────────────────────

# owner:     clarificator (02_clarificator.py)
# consumers: enricher, planner, specwright, debugger, patcher
# lifecycle: short-term — overwrite per clarification run
# purpose:   structured run output: decisions[], conflicts[], unresolved[],
#            tier_counts, input_sources, req_hash, requirement_text.
#            requirement_synthesis lives in CLARIFICATOR_REQUIREMENT_SYNTHESIS.
CLARIFICATOR_SESSION = _LazyPath("clarificator/session.json")

# owner:     clarificator (02_clarificator.py)
# consumers: enricher, specwriter, planner — inject directly into LLM prompts
# lifecycle: short-term — overwrite per clarification run
# purpose:   LLM-generated clarified requirement document (markdown).
#            Tách khỏi session.json vì đây là document, không phải structured data.
CLARIFICATOR_REQUIREMENT_SYNTHESIS = _LazyPath("clarificator/requirement_synthesis.md")

# owner:     clarificator (02_clarificator.py)
# consumers: clarificator (next run — semantic dedup of already-answered Qs)
# lifecycle: long-term — append-only across all runs
# purpose:   long-term Q&A memory; prevents re-asking semantically equivalent
#            questions; surfaced to enricher for context continuity
CLARIFICATOR_DECISION_LOG = _LazyPath("clarificator/decision_log.json")


# ── enricher/ ────────────────────────────────────────────────────────────────

# owner:     enricher (03_enricher.py)
# consumers: specwright
# lifecycle: short-term — overwrite per enricher run
# purpose:   structured prompt enriched with knowledge layer, passed to specwright
ENRICHER_OVERWRITE_PROMPT = _LazyPath("enricher/enriched_prompt.md")

# owner:     enricher (03_enricher.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only
ENRICHER_PROMPT_LOG = _LazyPath("enricher/prompt_log.json")


# ── spectracker/ ─────────────────────────────────────────────────────────────

# owner:     spectracker (05_spectracker.py)
# consumers: harness (drives which steps to rerun)
# lifecycle: short-term — overwrite per spectracker run
# purpose:   structured diff between current and previous spec version:
#            changed_sections, affected_files, rerun_steps
SPECTRACKER_VERSION_DELTA = _LazyPath("spectracker/version_delta.json")

# owner:     spectracker (05_spectracker.py)
# consumers: spectracker (self-read for delta baseline), harness (applied check)
# lifecycle: long-term — hybrid: entries appended per version, applied flag
#            on existing entry updated when harness finalizes a run
# purpose:   version history merging snapshot + applied state. Each entry:
#              version, generated_at, applied (bool), applied_at,
#              changed_sections[], affected_files[], spec_content (full text)
#            Replaces the old triplet (version_log.md + applied.json + per-version snapshots).
SPECTRACKER_VERSION_LOG = _LazyPath("spectracker/version_log.json")


# ── scaffolder/ ──────────────────────────────────────────────────────────────

# owner:     scaffolder (06_scaffolder.py)
# consumers: planner, executor, reporter, judge, harness
# lifecycle: short-term — overwrite per scaffolder run
# purpose:   module-centric blueprint. Schema:
#              { generated_at, spec_version,
#                modules: [{ module, purpose,
#                            files: [{ path, kind }] }],
#                summary: { total, source, test } }
#            kind ∈ {"source","test","config","migration"}.
#            Replaces old file-centric is_test schema; field `code` removed.
SCAFFOLD_JSON = _LazyPath("scaffolder/blueprint.json")

# owner:     scaffolder (06_scaffolder.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only
SCAFFOLDER_SKELETON_LOG = _LazyPath("scaffolder/skeleton_log.json")


# ── planner/ ─────────────────────────────────────────────────────────────────

# owner:     planner (07_planner.py)
# consumers: executor, debugger, reporter, judge, patcher, harness
# lifecycle: short-term — overwrite per full-scope run
# purpose:   per-file implementation tasks, dependency order, gotchas.
#            Immutable after planner writes — patcher and debugger read only.
PLANNER_FULL_PLAN = _LazyPath("planner/full_plan.json")

# owner:     planner (07_planner.py)
# consumers: executor, debugger, reporter, judge, patcher, harness
# lifecycle: short-term — overwrite per mini-scope run
# purpose:   merged plan + impact analysis: { "plan": {...}, "impact": {...} }
PLANNER_MINI_PLAN = _LazyPath("planner/mini_plan.json")

# owner:     planner (07_planner.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only (full + mini share this log)
PLANNER_PLAN_LOG = _LazyPath("planner/plan_log.json")


# ── executor/ ────────────────────────────────────────────────────────────────

# owner:     executor (08_executor.py)
# consumers: reporter, judge, patcher, harness
# lifecycle: short-term — overwrite per executor run
# purpose:   manifest of files implemented this run: status per file
#            (written/skipped/failed), run mode (full/delta/mini), model used
EXECUTOR_OVERWRITE_MANIFEST = _LazyPath("executor/manifest.json")

# owner:     executor (08_executor.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only
EXECUTOR_MANIFEST_LOG = _LazyPath("executor/manifest_log.json")


# ── debugger/ ────────────────────────────────────────────────────────────────

# owner:     debugger (09_debugger.py)
# consumers: reporter, judge, archivist
# lifecycle: short-term — overwrite per debugger run
# purpose:   summarized test results: pass/fail counts, per-iteration breakdown,
#            cluster-level repair details, escalated clusters list
DEBUGGER_OVERWRITE_TEST_SUMMARY = _LazyPath("debugger/test_summary.json")

# owner:     debugger (09_debugger.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only with per-entry trim policy:
#              keep full: final_status, scope, total_iterations, max_iter, escalated[]
#              keep full: cluster_details for the LAST iteration only
#              trim: prior iterations → { iteration, passed, clusters_found,
#                                         clusters_repaired, summary }
#              drop: log_snippet (still in short-term test_summary.json)
DEBUGGER_TEST_LOG = _LazyPath("debugger/test_log.json")


# ── reporter/ ────────────────────────────────────────────────────────────────

# owner:     reporter (10_reporter.py)
# consumers: human review only — no pipeline script parses this
# lifecycle: short-term — overwrite per reporter run
# purpose:   human-readable pipeline execution summary.
#            On GitHub Actions: piped to $GITHUB_STEP_SUMMARY.
REPORTER_EXECUTION_SUMMARY = _LazyPath("reporter/execution_summary.md")

# owner:     reporter (10_reporter.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only
# entry:     { scope, final_status, iterations_used, max_iter,
#              files_implemented, failed_cluster_count, escalated_count,
#              spec_version, cost_total, generated_at }
REPORTER_EXECUTION_LOG = _LazyPath("reporter/execution_log.json")


# ── judge/ ───────────────────────────────────────────────────────────────────

# owner:     judge (11_judge.py)
# consumers: patcher, archivist, harness
# lifecycle: short-term — overwrite per judge run
# purpose:   raw judge verdict JSON as returned by model, fully unprocessed.
#            Preserved so patcher and archivist parse independently;
#            failures debuggable without re-calling the API.
JUDGE_OVERWRITE_VERDICT_RAW = _LazyPath("judge/verdict_raw.json")

# owner:     judge (11_judge.py)
# consumers: human review only — no pipeline script parses this
# lifecycle: short-term — overwrite per judge run
# purpose:   human-readable verdict: scores, blocking issues with suggested fixes,
#            non-blocking notes, spec gaps, sign-off status
JUDGE_VERDICT_SUMMARY = _LazyPath("judge/verdict_summary.md")

# owner:     judge (11_judge.py)
# consumers: human review, hypothesis/consultant queries
# lifecycle: long-term — append-only
JUDGE_VERDICT_LOG = _LazyPath("judge/verdict_log.json")


# ── patcher/ ─────────────────────────────────────────────────────────────────

# owner:     patcher (12_patcher.py)
# consumers: human review
# lifecycle: short-term — overwrite per patcher run
# purpose:   human-readable summary: patched files, escalated findings,
#            scope rejections, confirm pass/fail result
PATCHER_OVERWRITE_FIX_SUMMARY = _LazyPath("patcher/fix_summary.md")

# owner:     patcher (12_patcher.py)
# consumers: debugger, archivist, human review
# lifecycle: long-term — append-only
# purpose:   longitudinal record of all patcher attempts across runs.
#            Consumers needing the latest snapshot read the LAST entry
#            (replaces the removed PATCHER_FINDINGS_SNAPSHOT).
PATCHER_ATTEMPT_LOG = _LazyPath("patcher/attempt_log.json")


# ── archivist/ ───────────────────────────────────────────────────────────────

# owner:     archivist (13_archivist.py)
# consumers: planner, executor, debugger, patcher, judge
# lifecycle: append-only (human-curated)
# purpose:   accumulated architecture decisions, recurring bug patterns, lessons.
#            Human-editable. Injected into LLM prompts.
ARCHIVIST_KNOWLEDGE_LOG = _LazyPath("archivist/knowledge_log.md")

# owner:     archivist (13_archivist.py)
# consumers: specwright, judge briefing
# lifecycle: append-only / curated overwrite
# purpose:   spec gaps and edge cases surfaced by judge, human-approved.
#            Human-editable. Injected into LLM prompts.
ARCHIVIST_SPEC_GAPS = _LazyPath("archivist/spec_gaps.md")

# owner:     archivist (13_archivist.py)
# consumers: human review
# lifecycle: append-only audit log
# purpose:   audit trail of archivist's curation decisions when reviewing
#            judge findings: applied / skipped / escalated to spec bump.
ARCHIVIST_CURATION_LOG = _LazyPath("archivist/curation_log.json")


# ── harness/ (project root) ──────────────────────────────────────────────────

# owner:     harness.py
# consumers: harness (self), human review
# lifecycle: append-only run history at project scope.
#            Replaces the old session_runs/ directory.
HARNESS_RUN_LOG = _LazyPath("harness_run_log.json")


# ── KNOWLEDGE_SOURCES ────────────────────────────────────────────────────────
# Enumerates all long-term artifacts. hypothesis/consultant modules iterate
# this list to gather cross-module context without hardcoding paths.

KNOWLEDGE_SOURCES: list = [
    ABSORBER_CODEBASE_LOG,
    CLARIFICATOR_DECISION_LOG,
    ENRICHER_PROMPT_LOG,
    SPECTRACKER_VERSION_LOG,
    SCAFFOLDER_SKELETON_LOG,
    PLANNER_PLAN_LOG,
    EXECUTOR_MANIFEST_LOG,
    DEBUGGER_TEST_LOG,
    REPORTER_EXECUTION_LOG,
    JUDGE_VERDICT_LOG,
    PATCHER_ATTEMPT_LOG,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    ARCHIVIST_CURATION_LOG,
]


# ── Backward-compatible aliases ──────────────────────────────────────────────
# Kept so in-flight callers don't break during migration. New code should
# use the canonical names above.

# absorber
CODEBASE_MAP    = ABSORBER_CODEBASE_MAP
CONFIG_MAP      = ABSORBER_CODEBASE_MAP   # merged into codebase_map.json
BLAME_MAP       = ABSORBER_CODEBASE_MAP   # merged into codebase_map.json
ABSORBER_CACHE  = ABSORBER_CODEBASE_SNAPSHOT

# clarificator
CLARIFIED_REQ        = CLARIFICATOR_REQUIREMENT_SYNTHESIS   # LLM-synthesized requirement doc
CLARIFIED_REQUEST    = CLARIFICATOR_SESSION
CLARIFICATION_REPORT = CLARIFICATOR_SESSION
CLARIFICATION_LOG    = CLARIFICATOR_DECISION_LOG

# enricher
ENRICHER_SESSION_PROMPT = ENRICHER_OVERWRITE_PROMPT

# spectracker
SPEC_DELTA     = SPECTRACKER_VERSION_DELTA
SPEC_APPLIED   = SPECTRACKER_VERSION_LOG   # applied state merged into version_log
SPEC_CHANGELOG = SPECTRACKER_VERSION_LOG

# planner
PLAN      = PLANNER_FULL_PLAN
PLAN_JSON = PLANNER_FULL_PLAN
PLAN_MINI = PLANNER_MINI_PLAN

# executor
IMPL_RECORD               = EXECUTOR_OVERWRITE_MANIFEST
EXECUTOR_SESSION_MANIFEST = EXECUTOR_OVERWRITE_MANIFEST

# debugger
TEST_REPORT                   = DEBUGGER_OVERWRITE_TEST_SUMMARY
DEBUGGER_SESSION_TEST_SUMMARY = DEBUGGER_OVERWRITE_TEST_SUMMARY

# reporter
SUMMARY = REPORTER_EXECUTION_SUMMARY

# judge
JUDGE_RAW                 = JUDGE_OVERWRITE_VERDICT_RAW
JUDGE_SESSION_VERDICT_RAW = JUDGE_OVERWRITE_VERDICT_RAW
JUDGE_REPORT              = JUDGE_VERDICT_SUMMARY

# patcher
PATCHER_SESSION_FIX_SUMMARY = PATCHER_OVERWRITE_FIX_SUMMARY
FIX_LOG                     = PATCHER_ATTEMPT_LOG

# archivist
FINDINGS_NOTES = ARCHIVIST_KNOWLEDGE_LOG
KNOWLEDGE_BASE = ARCHIVIST_KNOWLEDGE_LOG
SPEC_ADDENDUM  = ARCHIVIST_SPEC_GAPS
UPDATE_LOG     = ARCHIVIST_CURATION_LOG


# ── ensure_dirs ───────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """
    Tạo tất cả artifact directories cho project hiện tại.
    Gọi 1 lần ở đầu mỗi script (sau khi PIPELINE_PROJECT đã được set).
    Raises RuntimeError nếu PIPELINE_PROJECT chưa được set.
    """
    root = _artifact_root()
    for rel in (
        "spec",
        "absorber/cache",
        "clarificator",
        "enricher",
        "spectracker",
        "scaffolder",
        "planner",
        "executor",
        "debugger",
        "reporter",
        "judge",
        "patcher",
        "archivist",
        "output/src",
        "output/tests",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
