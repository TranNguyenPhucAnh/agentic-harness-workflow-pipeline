# Artifact Ownership

> **RULE: 1 artifact = 1 script owner duy nhất được ghi.**
> Tất cả script khác chỉ được **READ**.
> Path được define tập trung tại `artifacts/paths.py`.

## Ownership Table

| Artifact | Owner (writer) | Consumers (readers) |
|---|---|---|
| `state/scaffold.json` | `02_scaffold_gemini` | `03a`, `03b`, `05`, `06`, `harness` |
| `cache/spec_compressed.md` | `02_scaffold_gemini` | `03a`, `03b`, `06`, `07_fix` |
| `state/plan.json` | `03b_implement_glm` | `03a`, `04`, `05`, `06`, `07_fix`, `07_update`, `harness` |
| `cache/spec_delta.json` | `spec_diff` | `05`, `06`, `harness` |
| `state/spec_applied.json` | `spec_diff` | — |
| `knowledge/history/spec.changelog` | `spec_diff` | — |
| `run/impl_record.json` | `03a_implement_qwen` | `05`, `06` |
| `run/test_report.json` | `04_test_and_iterate` | `05`, `06`, `07_update` |
| `knowledge/current/findings.md` | `04_test_and_iterate` | `07_update`, `07_fix` |
| `run/judge_raw.json` | `06_judge_deepseek` | `07_update`, `07_fix`, `harness` |
| `knowledge/current/spec_addendum.md` | `06_judge_deepseek` | `07_update` |
| `artifacts/reports/judge_report.md` | `06_judge_deepseek` | `05` |
| `knowledge/current/base.md` | `07_update_knowledge` | `04`, `06`, `07_fix` |
| `knowledge/history/update_log.json` | `07_update_knowledge` | `harness` |
| `knowledge/history/fix_log.json` | `07_fix_from_judge` | — |
| `artifacts/reports/summary.md` | `05_report` | — |

## Data Flow

```
spec.md
  │
  ├─[spec_diff]──────► spec_delta.json, spec_applied.json, spec.changelog
  │
  └─[02_scaffold]────► scaffold.json, spec_compressed.md
                              │
               ┌─────────────┘
               │
        [03b_glm]──────────► plan.json
               │
        [03a_qwen]─────────► impl_record.json  (reads: scaffold, plan)
               │
        [04_test]──────────► test_report.json, findings.md
               │
        [06_judge]─────────► judge_raw.json, spec_addendum.md, judge_report.md
               │
        [07_update]────────► base.md, knowledge/history/update_log.json
        [07_fix]───────────► fix_log.json  (fixes src/, reads: judge_raw, findings, base)
               │
        [05_report]────────► summary.md
```

## Per-Script Contract

Mỗi script có header:
```python
# === WRITE AUTHORITY: <script_name> ===
# OWNS  : <list of artifacts this script writes>
# READS : <list of artifacts this script reads>
```
