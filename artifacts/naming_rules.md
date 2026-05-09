# Artifact Naming Rules

Rules này áp dụng cho tất cả artifacts trong `artifacts_<slug>/`.
Mọi artifact mới phải tuân theo trước khi được add vào `paths.py`.

---

## Rule 1 — Format cơ bản

```
<owner>_<semantic>.<ext>
```

- **`<owner>`** = module name duy nhất được ghi file này
- **`<semantic>`** = 2–3 từ mô tả nội dung, mục đích, hoặc lifecycle role
- **`<ext>`** = `.json` hoặc `.md` (xem Rule 3)

**Không có exception.** Không file nào tự define path ngoài `paths.py`.

---

## Rule 2 — Owner prefix = sole writer

Prefix phải phản ánh đúng module duy nhất có quyền ghi.

```
# ✅
clarificator_decision_log.md     ← chỉ clarificator ghi
archivist_knowledge_log.md       ← chỉ archivist ghi
judge_session_verdict_raw.json   ← chỉ judge ghi

# ❌
plan_notes.json                  ← "plan" không phải module name
findings.md                      ← không có owner prefix
spec_addendum.md                 ← spec không phải module name
```

**Module names chuẩn:**

| Prefix | Script |
|---|---|
| `spectracker` | `01_spectracker.py` |
| `absorber` | `02_absorber.py` |
| `clarificator` | `03_clarificator.py` |
| `enricher` | `04_enricher.py` |
| `specwright` | `05_specwright.py` |
| `scaffolder` | `06_scaffolder.py` |
| `planner` | `07_planner.py` |
| `executor` | `08_executor.py` |
| `debugger` | `09_debugger.py` |
| `reporter` | `10_reporter.py` |
| `judge` | `11_judge.py` |
| `patcher` | `12_patcher.py` |
| `archivist` | `13_archivist.py` |

---

## Rule 3 — Extension = readability contract

| Extension | Nghĩa | Dùng khi |
|---|---|---|
| `.json` | Machine-readable | Downstream scripts parse file này |
| `.md` | Human-readable | File chủ yếu để người đọc; pipeline có thể inject nhưng không parse |

Không dùng `.txt`, `.yaml`, `.csv` cho pipeline artifacts.

---

## Rule 4 — Lifecycle suffixes (bắt buộc nếu applicable)

### `_session_` — overwrite mỗi pipeline run

Dùng khi artifact bị overwrite hoàn toàn mỗi lần module chạy trong một pipeline session.

```
clarificator_session_raw.json
enricher_session_enriched_prompt.md
judge_session_verdict_raw.json
spectracker_session_version_delta.json
```

**Test:** nếu chạy pipeline 2 lần liên tiếp, file có nội dung run trước không? Nếu không → dùng `_session_`.

### `_log` — append-only, tích lũy across sessions

Dùng cho files chỉ được append, không bao giờ overwrite toàn bộ.

```
clarificator_decision_log.md
archivist_knowledge_log.md
archivist_curation_log.json
patcher_attempt_log.json
spectracker_version_log.md
```

**Test:** nếu xóa file rồi chạy lại, có mất thông tin không thể recover không? Nếu có → dùng `_log`.

### `_raw` — unprocessed model output

Dùng cho files chứa API response chưa qua parse hay transform.

```
judge_session_verdict_raw.json
clarificator_session_raw.json
```

**Invariant:** `_raw` luôn đi kèm `_session_` vì model output là per-run.

---

## Rule 5 — Semantic suffixes cho `.md`

Dùng khi cần phân biệt loại human-readable document:

| Suffix | Nghĩa | Ví dụ |
|---|---|---|
| `_summary` | Condensed overview, rút gọn từ nhiều nguồn | `reporter_execution_summary.md` |
| `_synthesis` | Rewrite/enrich từ nhiều nguồn thành document liền mạch, không rút gọn | `clarificator_requirement_synthesis.md` |
| `_synopsis` | High-level narrative, không đi vào chi tiết | (reserved) |

**Không dùng `_report`** — quá generic, dễ nhầm giữa machine và human-readable.
**Không dùng `_notes`, `_data`, `_info`, `_output`** — không encode semantic đủ mạnh.

---

## Rule 6 — Phân biệt full vs mini scope

Khi một module tạo artifact khác nhau cho 2 luồng, dùng infix `_full_` và `_mini_`:

```
planner_full_execution_plan.json
planner_mini_execution_plan.json
planner_mini_impact_analysis.json
```

Không dùng suffix `-mini` hay `-full` — infix giữ pattern `<owner>_<semantic>` nhất quán.

---

## Rule 7 — Không encode consumer vào tên

Tên chỉ encode owner. Consumer có thể thay đổi khi business logic thay đổi — tên file không nên phải đổi theo.

```
# ✅
archivist_knowledge_log.md       ← planner, executor, debugger, patcher đều đọc

# ❌
archivist_planner_knowledge.md   ← sẽ sai ngay khi executor cũng cần đọc
```

**Ngoại lệ được phép:** mapping 1-1 rõ ràng và stable, khi file được tạo ra *với mục đích duy nhất* phục vụ một consumer, và relationship đó không có khả năng thay đổi. Phải document lý do trong `paths.py` comment và TAXONOMY.md.

---

## Rule 8 — Không dùng abstract nouns

Tránh những từ không encode semantic cụ thể:

| ❌ Tránh | ✅ Dùng thay |
|---|---|
| `_notes` | `_directives`, `_gaps`, `_patterns` |
| `_log` (khi không phải append-only) | `_snapshot`, `_summary`, `_record` |
| `_data` | tên cụ thể theo content |
| `_info` | tên cụ thể theo content |
| `_output` | `_synthesis`, `_summary`, `_raw` |
| `_report` (cho human-readable) | `_summary`, `_synthesis` |
| `_record` (khi là snapshot session) | `_session_manifest`, `_session_snapshot` |

---

## Rule 9 — Dynamic paths

Khi filename phụ thuộc runtime data (ví dụ: version string, project slug), không define static `_LazyPath` constant. Thay vào đó:

1. Viết một function trả về `Path` trong `paths.py`
2. Document rõ trong TAXONOMY.md là dynamic path
3. Module owner tự construct path tại runtime

```python
# ✅ — slug resolved at runtime
def get_spec_path() -> Path:
    return _artifact_root() / f"specwright_spec_{get_project_slug()}.md"

# ❌ — slug không thể embed trong LazyPath static
SPEC_PATH = _LazyPath("specwright_spec_myapp.md")
```

---

## Rule 10 — Registration checklist

Mọi artifact mới phải được đăng ký đủ 3 nơi trước khi merge:

- [ ] `artifacts/paths.py` — `_LazyPath` constant với đầy đủ comments: owner, consumers, lifecycle, purpose
- [ ] `artifacts/OWNERSHIP.md` — thêm vào ownership table đúng section
- [ ] `artifacts/TAXONOMY.md` — thêm vào directory overview, ownership table, và Special Notes nếu có hybrid lifecycle hoặc exception

---

## Hybrid lifecycle

Khi một file có mixed write behavior (ví dụ: top-level fields overwrite, embedded array append-only), đây là **documented exception**. Phải:

1. Không dùng `_session_` hay `_log` suffix — cả hai đều sai một phần
2. Comment `# lifecycle: hybrid` trong `paths.py`
3. Giải thích rõ trong TAXONOMY.md Special Notes tại sao hybrid thay vì tách thành 2 artifacts

Ví dụ: `spectracker_applied_version.json` — top-level overwrite, `run_history[]` append-only.

---

## Quick reference

```
artifact mới → hỏi:

1. Module nào ghi?           → owner prefix
2. Machine hay human?        → .json hay .md
3. Overwrite mỗi run?        → thêm _session_
4. Append-only log?          → thêm _log
5. Unprocessed model output? → thêm _raw (+ _session_)
6. Full hay mini scope?      → thêm _full_ hay _mini_ infix
7. Content là gì?            → semantic 1-2 từ cụ thể
```
