# Artifact Ownership

> RULE: 1 artifact = 1 script owner duy nhất được ghi.
> Các script khác chỉ được READ.

| Artifact                              | Owner                     | Consumers                              |
|---------------------------------------|---------------------------|----------------------------------------|
| `state/scaffold.json`                 | `02_scaffold_gemini`      | `03a`, `03b`, `05`, `06`               |
| `state/plan.json`                     | `03b_implement_glm`       | `03a`, `04`, `05`, `06`, `07_fix`      |
| `state/spec_applied.json`             | `spec_diff`               | —                                      |
| `cache/spec_compressed.md`            | `02_scaffold_gemini`      | `06`, `07_fix`                         |
| `cache/spec_delta.json`               | `spec_diff`               | `05`, `06`                             |
| `run/impl_record.json`                | `03a_implement_qwen`      | `05`, `06`                             |
| `run/test_report.json`                | `04_test_and_iterate`     | `05`, `06`, `07_update`                |
| `run/judge_raw.json`                  | `06_judge_deepseek`       | `07_update`, `07_fix`                  |
| `knowledge/current/findings.md`       | `04_test_and_iterate`     | `07_update`, `07_fix`                  |
| `knowledge/current/spec_addendum.md`  | `06_judge_deepseek`       | `07_update`                            |
| `knowledge/current/base.md`           | `07_update_knowledge`     | `04`, `06`, `07_fix`                   |
| `knowledge/history/update_log.json`   | `07_update_knowledge`     | `harness`                              |
| `knowledge/history/spec.changelog`    | `spec_diff`               | —                                      |
| `reports/summary.md`                  | `05_report`               | —                                      |
| `reports/judge_report.md`             | `05_report`               | —                                      |
