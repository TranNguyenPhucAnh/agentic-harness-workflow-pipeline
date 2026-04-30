# 00_clarificator.py — Design Document

*Last updated: 2026-04-30*

---

## Position trong pipeline

```
[raw requirement: file | text]
         │
   [00_clarificator]          ← số 00, upstream nhất, trước Estimator
         │
    ┌────┴──────────────────────────────────────────────────────┐
    │  run/clarification_report.json          (machine)         │
    │  run/clarification_questions.md         (→ gửi client)    │
    │  knowledge/current/clarification_log_<slug>.md  (append)  │
    └───────────────────────────────────────────────────────────┘
         │
   [client answers via CLI — interactive loop]
         │
   [state/clarified_requirement.md]
         │
   [ESTIMATOR] → [harness / mini route]
```

---

## Artifacts — đăng ký vào paths.py

```python
# owner: 00_clarificator
CLARIFICATION_REPORT    = RUN_DIR     / "clarification_report.json"
CLARIFICATION_QUESTIONS = RUN_DIR     / "clarification_questions.md"
CLARIFIED_REQ           = STATE_DIR   / "clarified_requirement.md"

# Per-project log — NOT a static constant.
# Use _clarification_log_path(project_slug) at runtime.
# Pattern: knowledge/current/clarification_log_<slug>.md
```

### Lifecycle

| Artifact | Write mode | Consumer |
|---|---|---|
| `run/clarification_report.json` | overwrite per run | Estimator, harness |
| `run/clarification_questions.md` | overwrite per run | human (gửi client) |
| `knowledge/current/clarification_log_<slug>.md` | **append-only, per-project** | `00_clarificator` (dedup), human audit |
| `state/clarified_requirement.md` | overwrite per session | `01_estimator`, harness |

> `clarification_log_<slug>.md` là long-term memory quan trọng nhất — ghi lại toàn bộ Q→A→Impact xuyên suốt project lifetime. Mỗi project có file riêng, isolate hoàn toàn. Script đọc lại để semantic dedup trước khi generate findings.

---

## Input modes

```bash
python 00_clarificator.py --project "my-app" --input requirement.pdf
python 00_clarificator.py --project "my-app" --input spec_draft.md
python 00_clarificator.py --project "my-app" --text "Build a dashboard..."
python 00_clarificator.py --project "my-app"   # interactive multiline prompt
python 00_clarificator.py                      # prompts for project name
python 00_clarificator.py --list-projects      # list known workspaces
```

`--project` là required cho dedup hoạt động đúng. Nếu bỏ qua, script sẽ prompt chọn từ danh sách existing workspaces hoặc tạo mới. Knowledge context (`base.md`, log) auto-load nếu tồn tại; silent skip nếu không — standalone mode.

---

## Core logic — 4 phases

### Phase 1: Parse & Analyze

LLM call (DeepSeek V3) với full context:

```
input:
  - requirement text (không truncate)
  - base.md              (nếu có)
  - clarification_log_<slug>.md (nếu có) ← semantic dedup
  - ALREADY_ANSWERED_QA block: full Q/A text pairs (không truncate)

output:
  raw analysis JSON: { findings[], conflicts[], clarified_summary }
```

Cross-reference knowledge base theo **2 hướng**:
- **Conflict detection** — requirement mới có contradict decision cũ không?
- **Assumption surfacing** — requirement assume behavior chưa được implement.

**Semantic dedup** (quan trọng): LLM nhận full Q/A text từ log, không chỉ IDs. Nếu câu hỏi mới semantically equivalent với câu đã answered (dù worded khác), LLM skip — không generate lại.

---

### Phase 2: Classify & Prioritize — Rule Engine

LLM propose tier sơ bộ, nhưng **tier là rule, không phải suggestion**. Rule engine Python enforce lại sau khi LLM trả về:

```python
# Rules (first match wins):
# R1: suggestion + confidence ≥ 0.75 + citation present  → Tier 3
# R2: scenarios ≤ 5 + category in (technical/design/logic) → Tier 2
# R3: category == business OR priority == blocking
#     OR no scenarios                                     → Tier 1
# R4: Tier 3 missing citation OR confidence < 0.75       → demote to Tier 2
```

Finding object schema:

```json
{
  "id": "CLR-001",
  "text": "<collaborative question — see Tone Rules>",
  "tier": 1,
  "category": "business | logic | technical | design",
  "priority": "blocking | high | medium | low",
  "depends_on": ["CLR-003"],
  "scenarios": ["option A", "option B", "option C"],
  "suggestion": "...",
  "confidence": 0.87,
  "citation": "base.md §3, pattern X"
}
```

**Tier rules:**

| Tier | Answer space | Enforce rule | Ví dụ |
|---|---|---|---|
| **1** | Subjective / unbounded | category=business OR no scenarios | "Brand color?", "Onboarding flow?" |
| **2** | Bounded, ≤5 enumerable options | scenarios ≤ 5 + technical/design/logic | "Login: redirect hay modal?" |
| **3** | Near-deterministic | confidence ≥ 0.75 + citation mandatory | "Infinite scroll" với mobile+social+latency<200ms |

**Invariants enforced by rule engine:**
- Tier 1 và Tier 2: `scenarios[]` MUST be non-empty (fallback injected nếu LLM bỏ sót)
- Tier 3: `suggestion` MUST be present
- Tier 3 không có `citation` hoặc `confidence < 0.75` → demote to Tier 2

**Tone rules (enforced in prompt):** `finding.text` phải là câu hỏi collaboratively, không phán xét. Banned phrases: "The requirement does not specify", "It is unclear", "There is no mention of". Correct format: "Which X should be used?", "How should Y behave when Z?"

---

### Phase 3: Interactive answer loop

Sort order:

```
1. Tier 1 blocking  ← hỏi trước, unblock dependencies
2. Tier 1 high
3. Tier 2 (any priority)
4. Tier 3 (suggestions)
```

Tất cả câu hỏi — kể cả Tier 1 — đều hiển thị numbered options. User chọn số hoặc type custom answer tự do.

```
🔴 [1/4] CLR-001  tier=1  BLOCKING  [business]
  Which notification channel should be used when an order ships?

  Options:
    1. In-app notification only
    2. Email notification
    3. Both in-app and email

  → Choose 1–3 or type custom answer:
```

**Dependency resolution:**
- `known_ids` = tất cả IDs từng xuất hiện trong queue (kể cả delta-injected)
- Nếu `depends_on` ref đến ID không trong `known_ids` → dangling ref, treated as satisfied (LLM artifact)
- Nếu dep tồn tại nhưng chưa answered → defer một lần; lần thứ hai → `unresolved`

**Unresolved warning phân loại severity:**
- `category == business` hoặc `priority == blocking` → warn mạnh `⚠️`
- Còn lại → silent, chỉ ghi vào report

---

### Phase 4: Delta loop (sau mỗi Tier 1 blocking answer)

Sau mỗi Tier 1 blocking answer, gọi `_delta_analyze()` — targeted LLM call nhỏ:

```
input:
  - answered finding (ID, text, category, priority)
  - user answer
  - requirement context (tối đa _DELTA_REQ_CHARS = 4000 chars)
  - current pending queue IDs

output:
  - new_findings[]: câu hỏi mới được reveal bởi answer này
  - invalidated_ids[]: câu hỏi trong queue nay đã moot
```

Rule engine áp dụng lại cho `new_findings`. Content-hash dedup ngăn inject câu đã answered (dù ID khác). `known_ids` được update khi inject để deps không bị dangling.

Delta call là **non-fatal**: nếu fail, loop tiếp tục với queue hiện tại, không crash session.

---

### Post-loop: Impact derivation & Synthesis

Sau khi loop hoàn tất:

1. **`_batch_derive_impacts()`** — gọi LLM per decision để generate 1-line impact statement. Chạy sau loop, không blocking interactive Q&A.
2. **`_synthesize_requirement()`** — LLM synthesize `clarified_requirement.md` từ original requirement + toàn bộ decisions + conflicts.

---

## Output structure

### `run/clarification_report.json`

```json
{
  "requirement_hash": "sha256...",
  "session_id": "2026-04-30T...",
  "project_name": "Dashboard v2",
  "total_findings": 5,
  "tier1_answered": 2,
  "tier2_answered": 2,
  "tier3_accepted": 1,
  "tier3_rejected": 0,
  "conflicts_detected": 0,
  "unresolved": [],
  "decisions": [
    {
      "id": "CLR-001",
      "tier": 1,
      "category": "business",
      "priority": "blocking",
      "question": "Which notification channel should be used when an order ships?",
      "answer": "In-app notification only",
      "accepted": true,
      "impact": "Requires in-app notification system; no external service dependency."
    }
  ],
  "conflicts": []
}
```

### `run/clarification_questions.md`

```markdown
# Clarification Questions — Dashboard v2
Generated: 2026-04-30

## 🔴 Blocking
1. [CLR-001] Which notification channel should be used when an order ships?
   - In-app notification only
   - Email notification
   - Both in-app and email

## 🟡 Important
...

## 🟢 Suggestions (confirm nếu đồng ý)
- [CLR-005] **Context:** Pagination vs infinite scroll for order list
  **Suggestion:** Use infinite scroll
  Confidence: 85% | Reasoning: mobile-first + list UX pattern
  → Accept / Reject / Modify?
```

### `knowledge/current/clarification_log_<slug>.md`

```markdown
## 2026-04-30 | Project: Dashboard v2 | Session: 10:00:00

### CLR-001 [Tier 1]
**Q:** Which notification channel should be used when an order ships?
**A:** In-app notification only
**Impact:** Requires in-app notification system; no external service dependency.

### CLR-005 [Tier 3 / accepted]
**Q:** Pagination vs infinite scroll for order list?
**A:** Use infinite scroll
**Impact:** Implement virtual scroll component for performance.
```

Mỗi entry phân cách bằng blank line. Không truncate Q/A text. Full text cần thiết cho semantic dedup ở session sau.

---

## Workspace isolation

Mỗi project có log riêng: `clarification_log_<slug>.md`. Slug được derive từ project name (`_slugify()`). Không bao giờ đọc log của project khác.

```python
_clarification_log_path("Dashboard v2") → clarification_log_dashboard-v2.md
_clarification_log_path("my-app")       → clarification_log_my-app.md
```

Legacy `clarification_log.md` (global, không có slug) được detect và warn nếu tồn tại — không dùng cho dedup, phải migrate thủ công.

---

## Token / context limits

| Constant | Value | Rationale |
|---|---|---|
| `_MAX_TOKENS_ANALYZE` | 8192 | Phase 1+2: up to ~15 findings × ~200 tokens |
| `_MAX_TOKENS_DELTA` | 2048 | Delta output nhỏ theo design |
| `_MAX_TOKENS_SYNTHESIS` | 4096 | clarified_requirement.md có thể dài với AC phức tạp |
| `_MAX_TOKENS_IMPACT` | 128 | 1 sentence per decision |
| `_DELTA_REQ_CHARS` | 4000 | ~1000 tokens, đủ cho 1 AC block chi tiết |

Không có ceiling limit cho số findings per session — LLM generate as many as needed, rule engine và dedup filter down organically.

---

## Standalone vs integrated

```python
# Standalone — không có knowledge layer
if not KNOWLEDGE_BASE.exists() and not log_text:
    print("[00] Standalone mode — no knowledge context")
    knowledge_context = ""
    answered_qa_pairs = []
else:
    knowledge_context = _load_knowledge_context(project_slug)
    answered_qa_pairs = _extract_answered_qa_pairs(log_text)
```

Fully functional ở cả 2 modes. Standalone chỉ thiếu cross-reference với history.

---

## Model recommendation

| Task | Model | Lý do |
|---|---|---|
| Phase 1+2 Analyze & Classify | DeepSeek V3 | Cần reasoning sâu, doc có thể dài |
| Phase 4 Delta follow-up | DeepSeek V3 | Cần context awareness |
| Impact derivation | DeepSeek V3 (128 tokens) | 1 sentence, nhẹ |
| Synthesis | DeepSeek V3 | Full requirement rewrite |

---

## Corrections & design decisions ghi nhận

1. **Workspace isolation** — `clarification_log.md` global bị thay bằng per-project `clarification_log_<slug>.md`. CLR-IDs không stable across sessions nên dedup dựa trên ID là fragile. Fix: semantic dedup qua full Q/A text pairs.

2. **Tier là rule, không phải model output** — LLM propose, Python rule engine enforce. Đảm bảo deterministic tier assignment independent của model mood.

3. **`_derive_impact` không blocking** — không call LLM trong interactive loop. Batch sau khi loop kết thúc hoàn toàn.

4. **Dangling `depends_on` refs** — LLM đôi khi generate dep đến ID không tồn tại trong queue. Fix: `_dependencies_satisfied()` nhận `known_ids` set, skip dangling refs silently thay vì block finding.

5. **`--project` arg** — bắt buộc specify hoặc prompt chọn. Không để LLM infer project name rồi dùng làm key vì không stable.

6. **Delta loop scope** — chỉ trigger sau Tier 1 blocking answer. Tier 2/3 không trigger re-analyze. Trade-off: giảm LLM calls vs coverage đầy đủ hơn.

7. **Log format** — dùng `"\n\n".join(blocks)` thay vì `"\n".join(lines)`. Mỗi CLR entry là 1 block phân cách bằng blank line — đúng markdown convention, dễ parse bằng regex `re.split(r"\n(?=###\s+CLR-")`.

8. **Tone enforcement** — prompt liệt kê explicit banned phrases và positive examples. Negative examples cần thiết vì LLM default về "audit language" khi viết về gaps trong document.

---

## Checklist đăng ký artifact (theo rule trong artifacts.md)

Mỗi artifact mới phải có mặt ở **3 nơi**:

- [x] `artifacts/paths.py` — với owner comment
- [x] `artifacts/OWNERSHIP.md` — trong ownership table
- [x] `artifacts/artifacts.md` — directory overview + lifecycle table + special note

Artifacts đã đăng ký: `clarification_report.json`, `clarification_questions.md`, `clarification_log_<slug>.md`, `clarified_requirement.md`

---

*Maintained alongside `pipeline/00_clarificator.py`.*
