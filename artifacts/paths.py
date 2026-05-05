"""
artifacts/paths.py
==================
SOURCE OF TRUTH cho tất cả artifact paths trong pipeline.

RULE: Không file nào được tự define artifact path — chỉ import từ đây.

Project isolation
─────────────────
Mỗi pipeline run được gắn với một project name (bắt buộc).
harness.py nhận --project PROJECT_NAME và set env var PIPELINE_PROJECT trước khi
spawn child processes.  Tất cả scripts đọc env var này qua _artifact_root() dưới đây.

Artifact root per-project:
    <repo_root>/artifacts_<project_slug>/
        state/
        cache/
        run/
        knowledge/current/
        knowledge/history/
        reports/

Slug rule: lowercase, ký tự không phải alphanumeric thay bằng dấu gạch ngang,
dấu gạch ngang đầu/cuối bị trim.
    "My App 1"  → "my-app-1"
    "dashboard" → "dashboard"
    "IoT_MLOps" → "iot-mlops"

Usage trong mỗi pipeline script (không thay đổi so với trước):
    from artifacts.paths import STATE_DIR, PLAN_JSON, ensure_dirs
    ensure_dirs()
    # paths tự động resolve theo PIPELINE_PROJECT env var

Ownership được ghi rõ trên mỗi path:
  - owner  = script DUY NHẤT được ghi file này
  - others = chỉ được đọc (read-only)
"""

from __future__ import annotations

import os
import re
from pathlib import Path


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

# REPO_ROOT = directory containing this file's parent (i.e. the repo root).
REPO_ROOT = Path(__file__).parent.parent

# ROOT = alias kept for backward compatibility (all scripts import ROOT).
ROOT = REPO_ROOT


def _artifact_root() -> Path:
    """
    Return the artifact root for the current project.
    e.g. /path/to/repo/artifacts_my-app
    Evaluated lazily so PIPELINE_PROJECT can be set after import time.
    """
    return REPO_ROOT / f"artifacts_{get_project_slug()}"


def artifact_root() -> Path:
    """Public alias for _artifact_root()."""
    return _artifact_root()


def project_info() -> dict:
    """Return project name, slug, and artifact root as a dict."""
    name = get_project_name()
    return {
        "name":          name,
        "slug":          _slugify(name),
        "artifact_root": str(_artifact_root()),
    }


# ── LazyPath ──────────────────────────────────────────────────────────────────

class _LazyPath:
    """
    A Path-like object whose concrete location is resolved lazily at access time
    from the current PIPELINE_PROJECT env var.

    This lets all path constants be defined at module level (so `from
    artifacts.paths import PLAN_JSON` works) while still picking up the correct
    project-scoped root when the path is actually *used*.

    Supports all common Path operations via delegation.
    """
    __slots__ = ("_rel",)

    def __init__(self, rel: str) -> None:
        self._rel = rel   # relative path inside the artifact root

    def _resolve(self) -> Path:
        return _artifact_root() / self._rel

    # ── os.fspath / str / repr ────────────────────────────────────────────────
    def __fspath__(self)  -> str:  return str(self._resolve())
    def __str__(self)     -> str:  return str(self._resolve())
    def __repr__(self)    -> str:  return f"LazyPath({self._rel!r} → {self._resolve()})"
    def __eq__(self, o)   -> bool: return self._resolve() == Path(o) if o is not None else False
    def __hash__(self)    -> int:  return hash(self._resolve())

    # ── Path arithmetic ───────────────────────────────────────────────────────
    def __truediv__(self, other):  return self._resolve() / other
    def __rtruediv__(self, other): return Path(other) / self._resolve()

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def parent(self) -> Path:      return self._resolve().parent
    @property
    def name(self)   -> str:       return self._resolve().name
    @property
    def stem(self)   -> str:       return self._resolve().stem
    @property
    def suffix(self) -> str:       return self._resolve().suffix

    # ── File I/O ─────────────────────────────────────────────────────────────
    def read_text(self, **kw):          return self._resolve().read_text(**kw)
    def write_text(self, data, **kw):   return self._resolve().write_text(data, **kw)
    def read_bytes(self):               return self._resolve().read_bytes()
    def write_bytes(self, data):        return self._resolve().write_bytes(data)
    def open(self, *a, **kw):           return self._resolve().open(*a, **kw)

    # ── Filesystem ops ────────────────────────────────────────────────────────
    def exists(self)         -> bool:   return self._resolve().exists()
    def is_file(self)        -> bool:   return self._resolve().is_file()
    def is_dir(self)         -> bool:   return self._resolve().is_dir()
    def mkdir(self, **kw):              return self._resolve().mkdir(**kw)
    def stat(self):                     return self._resolve().stat()
    def unlink(self, **kw):             return self._resolve().unlink(**kw)
    def rename(self, target):           return self._resolve().rename(target)

    # ── Path helpers ──────────────────────────────────────────────────────────
    def relative_to(self, *a):     return self._resolve().relative_to(*a)
    def with_name(self, name):     return self._resolve().with_name(name)
    def with_suffix(self, suf):    return self._resolve().with_suffix(suf)
    def glob(self, pattern):       return self._resolve().glob(pattern)
    def rglob(self, pattern):      return self._resolve().rglob(pattern)
    def iterdir(self):             return self._resolve().iterdir()


# ── Directory constants ───────────────────────────────────────────────────────

STATE_DIR     = _LazyPath("state")
CACHE_DIR     = _LazyPath("cache")
RUN_DIR       = _LazyPath("run")
KNOWLEDGE_DIR = _LazyPath("knowledge")
CURRENT_DIR   = _LazyPath("knowledge/current")
HISTORY_DIR   = _LazyPath("knowledge/history")
REPORTS_DIR   = _LazyPath("reports")


# ── Misc ──────────────────────────────────────────────────────────────────────

SPEC_PATH = ROOT / "spec.md"   # source — not project-scoped (lives at repo root)


# ── state/ ────────────────────────────────────────────────────────────────────

SCAFFOLD_JSON   = _LazyPath("state/scaffold.json")        # owner: 02_scaffold_gemini
SPEC_COMPRESSED = _LazyPath("cache/spec_compressed.md")   # owner: 02_scaffold_gemini
PLAN_JSON       = _LazyPath("state/plan.json")            # owner: 03b_implement_glm
SPEC_DELTA      = _LazyPath("cache/spec_delta.json")      # owner: spec_diff
SPEC_APPLIED    = _LazyPath("state/spec_applied.json")    # owner: spec_diff
PLAN_NOTES      = _LazyPath("state/plan_notes.json")      # owner: 07_update_knowledge


# ── run/ ─────────────────────────────────────────────────────────────────────

IMPL_RECORD     = _LazyPath("run/impl_record.json")       # owner: 03a_implement_qwen
TEST_REPORT     = _LazyPath("run/test_report.json")       # owner: 04_test_and_iterate
JUDGE_RAW       = _LazyPath("run/judge_raw.json")         # owner: 06_judge_deepseek


# ── knowledge/current/ ────────────────────────────────────────────────────────

FINDINGS        = _LazyPath("knowledge/current/findings.md")        # owner: 07_fix_from_judge
FINDINGS_NOTES  = _LazyPath("knowledge/current/findings_notes.md")  # owner: 07_update_knowledge
SPEC_ADDENDUM   = _LazyPath("knowledge/current/spec_addendum.md")   # owner: 06_judge_deepseek
KNOWLEDGE_BASE  = _LazyPath("knowledge/current/base.md")            # owner: 07_update_knowledge


# ── knowledge/history/ ───────────────────────────────────────────────────────

UPDATE_LOG      = _LazyPath("knowledge/history/update_log.json")    # owner: 07_update_knowledge
FIX_LOG         = _LazyPath("knowledge/history/fix_log.json")       # owner: 07_fix_from_judge
SPEC_CHANGELOG  = _LazyPath("knowledge/history/spec.changelog")     # owner: spec_diff


# ── reports/ ─────────────────────────────────────────────────────────────────

SUMMARY         = _LazyPath("reports/summary.md")         # owner: 05_report
JUDGE_REPORT    = _LazyPath("reports/judge_report.md")    # owner: 06_judge_deepseek


# ── absorber ─────────────────────────────────────────────────────────────────

CODEBASE_MAP   = _LazyPath("knowledge/current/codebase_map.md")     # owner: 01_absorber
CONFIG_MAP     = _LazyPath("knowledge/current/config_map.json")     # owner: 01_absorber
BLAME_MAP      = _LazyPath("knowledge/current/blame_map.md")        # owner: 01_absorber
GIT_HISTORY    = _LazyPath("knowledge/history/git_history.json")    # owner: 01_absorber
ABSORBER_CACHE = _LazyPath("cache/absorber_cache.json")             # owner: 01_absorber


# ── clarification ────────────────────────────────────────────────────────────

CLARIFICATION_REPORT    = _LazyPath("run/clarification_report.json")   # owner: 00_clarificator
CLARIFICATION_QUESTIONS = _LazyPath("run/clarification_questions.md")  # owner: 00_clarificator
CLARIFICATION_LOG       = _LazyPath("knowledge/current/clarification_log.md")  # owner: 00_clarificator
CLARIFIED_REQ           = _LazyPath("state/clarified_requirement.md")  # owner: 00_clarificator


# ── ensure_dirs ───────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """
    Tạo tất cả artifact directories cho project hiện tại.
    Gọi 1 lần ở đầu mỗi script (sau khi PIPELINE_PROJECT đã được set).
    Raises RuntimeError nếu PIPELINE_PROJECT chưa được set.
    """
    root = _artifact_root()
    for rel in (
        "state",
        "cache",
        "run",
        "knowledge/current",
        "knowledge/history",
        "reports",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
