"""
artifacts/paths.py
==================
SOURCE OF TRUTH cho tất cả artifact paths trong pipeline.

RULE: Không file nào được tự define artifact path — chỉ import từ đây.

Ownership được ghi rõ trên mỗi path:
  - owner  = script DUY NHẤT được ghi file này
  - others = chỉ được đọc (read-only)
"""

from pathlib import Path

# ROOT = project root (parent của artifacts/)
ROOT = Path(__file__).parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
STATE_DIR     = ROOT / "artifacts" / "state"
CACHE_DIR     = ROOT / "artifacts" / "cache"
RUN_DIR       = ROOT / "artifacts" / "run"
KNOWLEDGE_DIR = ROOT / "artifacts" / "knowledge"
CURRENT_DIR   = KNOWLEDGE_DIR / "current"
HISTORY_DIR   = KNOWLEDGE_DIR / "history"
REPORTS_DIR   = ROOT / "artifacts" / "reports"


def ensure_dirs() -> None:
    """Tạo tất cả artifact directories. Gọi 1 lần ở đầu mỗi script."""
    for d in (STATE_DIR, CACHE_DIR, RUN_DIR, CURRENT_DIR, HISTORY_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Misc ──────────────────────────────────────────────────────────────────────
SPEC_PATH = ROOT / "spec.md"                         # source, không ai owns


# ── state/ ────────────────────────────────────────────────────────────────────
SCAFFOLD_JSON   = STATE_DIR / "scaffold.json"        # owner: 02_scaffold_gemini
SPEC_COMPRESSED = CACHE_DIR / "spec_compressed.md"   # owner: 02_scaffold_gemini

PLAN_JSON       = STATE_DIR / "plan.json"            # owner: 03b_implement_glm

SPEC_DELTA      = CACHE_DIR / "spec_delta.json"      # owner: spec_diff
SPEC_APPLIED    = STATE_DIR / "spec_applied.json"    # owner: spec_diff
PLAN_NOTES      = STATE_DIR / "plan_notes.json"      # owner: 07_update_knowledge

# ── run/ ─────────────────────────────────────────────────────────────────────
IMPL_RECORD     = RUN_DIR / "impl_record.json"       # owner: 03a_implement_qwen
TEST_REPORT     = RUN_DIR / "test_report.json"       # owner: 04_test_and_iterate
JUDGE_RAW       = RUN_DIR / "judge_raw.json"         # owner: 06_judge_deepseek
MINI_LOG        = RUN_DIR / "mini_log.json"          # owner: mini_mode

# ── knowledge/current/ ────────────────────────────────────────────────────────
FINDINGS        = CURRENT_DIR / "findings.md"        # owner: 07_fix_from_judge
FINDINGS_NOTES  = CURRENT_DIR / "findings_notes.md"  # owner: 07_update_knowledge
SPEC_ADDENDUM   = CURRENT_DIR / "spec_addendum.md"   # owner: 06_judge_deepseek
KNOWLEDGE_BASE  = CURRENT_DIR / "base.md"            # owner: 07_update_knowledge

# ── knowledge/history/ ───────────────────────────────────────────────────────
UPDATE_LOG      = HISTORY_DIR / "update_log.json"    # owner: 07_update_knowledge
FIX_LOG         = HISTORY_DIR / "fix_log.json"        # owner: 07_fix_from_judge
SPEC_CHANGELOG  = HISTORY_DIR / "spec.changelog"     # owner: spec_diff

# ── reports/ ─────────────────────────────────────────────────────────────────
SUMMARY         = REPORTS_DIR / "summary.md"         # owner: 05_report
JUDGE_REPORT    = REPORTS_DIR / "judge_report.md"    # owner: 05_report

# ── clarification (run/ + state/ + knowledge/) ───────────────────────────────
CLARIFICATION_REPORT    = RUN_DIR     / "clarification_report.json"   # owner: 00_clarificator
CLARIFICATION_QUESTIONS = RUN_DIR     / "clarification_questions.md"  # owner: 00_clarificator
CLARIFICATION_LOG       = CURRENT_DIR / "clarification_log.md"        # owner: 00_clarificator
CLARIFIED_REQ           = STATE_DIR   / "clarified_requirement.md"    # owner: 00_clarificator (via 06)
