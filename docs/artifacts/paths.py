# artifacts/paths.py
# SOURCE OF TRUTH cho tất cả artifact paths.
# Không file nào được tự define path — chỉ import từ đây.

from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Directories ──────────────────────────────────────────
STATE_DIR        = ROOT / "artifacts" / "state"
CACHE_DIR        = ROOT / "artifacts" / "cache"
RUN_DIR          = ROOT / "artifacts" / "run"
KNOWLEDGE_DIR    = ROOT / "artifacts" / "knowledge"
CURRENT_DIR      = KNOWLEDGE_DIR / "current"
HISTORY_DIR      = KNOWLEDGE_DIR / "history"
REPORTS_DIR      = ROOT / "reports"

def ensure_dirs():
    for d in (STATE_DIR, CACHE_DIR, RUN_DIR, CURRENT_DIR, HISTORY_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

# ── state/ ────────────────────────────────────────────────
# owner: 02_scaffold_gemini
SCAFFOLD_JSON    = STATE_DIR / "scaffold.json"
SPEC_COMPRESSED  = CACHE_DIR / "spec_compressed.md"

# owner: 03b_implement_glm
PLAN_JSON        = STATE_DIR / "plan.json"

# owner: spec_diff
SPEC_DELTA       = CACHE_DIR / "spec_delta.json"
SPEC_APPLIED     = STATE_DIR / "spec_applied.json"

# ── run/ ─────────────────────────────────────────────────
# owner: 03a_implement_qwen
IMPL_RECORD      = RUN_DIR / "impl_record.json"

# owner: 04_test_and_iterate
TEST_REPORT      = RUN_DIR / "test_report.json"

# owner: 06_judge_deepseek
JUDGE_RAW        = RUN_DIR / "judge_raw.json"

# ── knowledge/ ───────────────────────────────────────────
# owner: 04_test_and_iterate
FINDINGS         = CURRENT_DIR / "findings.md"

# owner: 06_judge_deepseek
SPEC_ADDENDUM    = CURRENT_DIR / "spec_addendum.md"

# owner: 07_update_knowledge
KNOWLEDGE_BASE   = CURRENT_DIR / "base.md"
UPDATE_LOG       = HISTORY_DIR / "update_log.json"

# ── reports/ ─────────────────────────────────────────────
# owner: 05_report
SUMMARY          = REPORTS_DIR / "summary.md"
JUDGE_REPORT     = REPORTS_DIR / "judge_report.md"

# ── misc ─────────────────────────────────────────────────
SPEC_PATH        = ROOT / "spec.md"
SPEC_CHANGELOG   = HISTORY_DIR / "spec.changelog"
