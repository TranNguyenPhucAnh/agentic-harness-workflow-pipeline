# 00_clarificator.py — Final Design Document

*Synthesized từ 2 design sessions. Last updated: 2026-04-27*

---

## Position trong pipeline

```
[raw requirement: file | text]
         │
   [00_clarificator]          ← số 00, upstream nhất, trước Estimator
         │
    ┌────┴──────────────────────────────────────────────┐
    │  run/clarification_report.json    (machine)       │
    │  run/clarification_questions.md   (→ gửi client)  │
    │  knowledge/current/clarification_log.md (append)  │
    └───────────────────────────────────────────────────┘
         │
   [client answers via CLI — interactive loop]
         │
   [state/clarified_requirement.md]
         │
   [ESTIMATOR] → [harness / mini route]
```

---

## Artifacts mới — đăng ký vào paths.py

```python
# owner: 00_clarificator
CLARIFICATION_REPORT    = RUN_DIR     / "clarification_report.json"
CLARIFICATION_QUESTIONS = RUN_DIR     / "clarification_questions.md"
CLARIFICATION_LOG       = CURRENT_DIR / "clarification_log.md"
CLARIFIED_REQ           = STATE_DIR   / "clarified_requirement.md"
```

### Lifecycle

| Artifact | Write mode | Consumer |
|---|---|---|
| `run/clarification_report.json` | overwrite per run | Estimator, harness |
| `run/clarification_questions.md` | overwrite per run | human (gửi client) |
| `knowledge/current/clarification_log.md` | **append-only** | `00_clarificator` (dedup), human audit |
| `state/clarified_requirement.md` | overwrite per session | `01_estimator`, harness |

> `clarification_log.md` là long-term memory quan trọng nhất — ghi lại toàn bộ Q→A→Decision xuyên suốt project lifetime. Script đọc lại để tránh hỏi lại câu đã answered.

---

## Input modes

```bash
python 00_clarificator.py --input requirement.pdf    # file upload
python 00_clarificator.py --input spec_draft.md
python 00_clarificator.py --text "Build a dashboard..."  # paste text
python 00_clarificator.py                            # interactive multiline prompt
```

Knowledge context (`base.md`, `clarification_log.md`) auto-load nếu tồn tại; silent skip nếu không — standalone mode.

---

## Core logic — 3 phases

### Phase 1: Parse & Analyze

LLM call (DeepSeek V3 / Gemini 2.5 Flash) với full context:

```
input:
  - requirement text
  - base.md              (nếu có)
  - clarification_log.md (nếu có) ← tránh hỏi lại
  - codebase_map.md      (nếu có)

output:
  raw analysis JSON: { findings[], conflicts[], assumptions[] }
```

Cross-reference knowledge base theo **2 hướng**:
- **Conflict detection** — requirement mới có contradict decision cũ không?  
  *VD: `base.md` ghi "UTC cho tất cả timestamps" nhưng requirement nói "hiển thị giờ local"*
- **Assumption surfacing** — requirement assume behavior chưa được implement.  
  *VD: "Add notification khi order ships" nhưng notification system chưa tồn tại*

---

### Phase 2: Classify & Prioritize

Mỗi finding được classify thành finding object:

```json
{
  "id": "CLR-001",
  "text": "Notification system chưa exist nhưng requirement assume nó có",
  "tier": 1,
  "category": "business | logic | technical | design",
  "priority": "blocking | high | medium | low",
  "depends_on": ["CLR-003"],
  "scenarios": [...],
  "suggestion": "...",
  "confidence": 0.87,
  "citation": "base.md §3, pattern X"
}
```

**Tier rules:**

| Tier | Answer space | Ví dụ |
|---|---|---|
| **1** | Unbounded / subjective | "Màu brand?", "Flow onboarding feel như thế nào?" |
| **2** | Bounded, enumerable (≤5 options) | "Chưa login → redirect hay modal?" |
| **3** | Near-deterministic, confidence ≥ 0.75 | "Infinite scroll" khi mobile + social feed + latency < 200ms |

**Dependency graph:** câu hỏi không độc lập. Answer của CLR-001 có thể eliminate hoặc generate CLR-002, CLR-003. Phải model dependency — tránh hỏi client những câu không cần thiết.

**Tier 3 phải có citation:**
```
SUGGEST: Dùng optimistic update cho like/unlike
CONFIDENCE: 87%
REASONING: Mobile context + social feed + latency < 200ms → user expects instant feedback
ACCEPT nếu: UX là priority
REJECT nếu: data consistency cao hơn UX
```

---

### Phase 3: Interactive answer loop

```
Hiển thị questions theo thứ tự:
  1. Blocking Tier 1  ← unblock dependencies
  2. High priority Tier 1
  3. Tier 2           ← show scenarios, user chọn
  4. Tier 3           ← show suggestion + citation, user accept/reject/modify

Sau mỗi answer:
  → re-evaluate dependency graph
  → nếu answer generates questions: inject vào queue
  → nếu answer eliminates questions: remove khỏi queue
```

**Clarificator chạy lại sau mỗi Tier 1 answer** — client's answer có thể generate questions mới hoặc upgrade Tier 3 thành Tier 1.

**Friction minimization:**
- Batch questions theo theme — đừng hỏi từng cái
- Show progress — *"3 questions còn lại, sau đó pipeline tự chạy"*
- Default to suggest (Tier 3) khi confidence > threshold — client chỉ approve/reject

---

## Output structure

### `run/clarification_report.json`

```json
{
  "requirement_hash": "sha256...",
  "session_id": "2026-04-27T...",
  "total_findings": 12,
  "tier1_answered": 4,
  "tier2_answered": 3,
  "tier3_accepted": 4,
  "tier3_rejected": 1,
  "unresolved": [],
  "decisions": [
    {
      "id": "CLR-001",
      "tier": 1,
      "question": "Notification system build mới hay integrate service ngoài?",
      "answer": "Integrate OneSignal",
      "impact": "Notification service cần build từ đầu với OneSignal SDK"
    }
  ]
}
```

### `run/clarification_questions.md`

```markdown
# Clarification Questions — [Project Name]
Generated: 2026-04-27

## 🔴 Blocking (cần trả lời trước khi estimate)
1. [CLR-001] Notification system: requirement đề cập "notify user khi..."
   nhưng hiện chưa có system này. Build mới hay integrate service ngoài (FCM, OneSignal)?

## 🟡 Important
...

## 🟢 Suggestions (confirm nếu đồng ý)
- [CLR-008] Dùng infinite scroll thay vì pagination cho feed
  Confidence: 87% | Reasoning: mobile-first + social pattern + latency < 200ms
  → Accept / Reject / Modify?
```

### `knowledge/current/clarification_log.md`

```markdown
## 2026-04-27 | Project: Dashboard v2 | Session: abc123

### CLR-001 [Tier 1 / blocking]
**Q:** Notification system build mới hay integrate?
**A:** Integrate OneSignal
**Decision:** Dùng OneSignal SDK, không build notification service
**Impact:** scope tăng ~3 ngày cho integration layer

### CLR-008 [Tier 3 / accepted]
**Suggest:** Infinite scroll
**Client:** Accepted
```

---

## Standalone vs integrated

```python
if not KNOWLEDGE_BASE.exists() and not CLARIFICATION_LOG.exists():
    print("[00] Standalone mode — no knowledge context")
    knowledge_context = ""
else:
    knowledge_context = _load_knowledge_context()
```

Fully functional ở cả 2 modes. Standalone chỉ thiếu cross-reference với history.

---

## Model recommendation

| Task | Model | Lý do |
|---|---|---|
| Phase 1 Parse & Analyze | DeepSeek V3 / Gemini 2.5 Flash | Cần reasoning sâu, doc có thể dài |
| Phase 2 Classify & Score | Same | Context đã có |
| Phase 3 Generate questions | Same | Cần nuance cho Tier 1/2 |
| Phase 3 Generate suggestions | Lighter model OK | Tier 3 near-deterministic |

---

## Corrections từ session 1

Những điểm trong draft dàn ý đã được adjust:

1. **`SPEC_ADDENDUM` owner conflict** — draft ban đầu assign owner là `06_judge_deepseek`, nhưng `paths.py` hiện tại không có entry này. Khi đăng ký artifacts mới vào `paths.py`, đảm bảo không duplicate key với entries đã có.

2. **`JUDGE_REPORT` owner** — `paths.py` ghi `owner: 05_report (via 06)` nhưng ownership table trong `artifacts__.md` ghi `06_judge_deepseek`. Cần align — recommended: `06_judge_deepseek` là sole writer, `05_report` không ghi file này.

3. **Clarificator chạy iteratively** — không phải một lần duy nhất. Design cần loop: answer → re-analyze → new questions → loop. Điều này chưa rõ trong draft gốc.

4. **`clarification_log.md` dedup check** — script phải đọc log trước khi generate questions để tránh hỏi lại. Draft gốc mention nhưng không nêu rõ là mandatory, không phải optional.

5. **Tier 3 citation là bắt buộc** — không phải optional field. Không có citation thì Tier 3 không có giá trị hơn guess.

---

## Checklist đăng ký artifact (theo rule trong artifacts__.md)

Mỗi artifact mới phải có mặt ở **3 nơi**:

- [ ] `artifacts/paths.py` — với owner comment
- [ ] `artifacts/OWNERSHIP.md` — trong ownership table  
- [ ] `artifacts/artifacts__.md` — directory overview + lifecycle table + special note

Artifacts cần thêm: `clarification_report.json`, `clarification_questions.md`, `clarification_log.md`, `clarified_requirement.md`

---

*Design document này là input cho implementation của `00_clarificator.py`.*
