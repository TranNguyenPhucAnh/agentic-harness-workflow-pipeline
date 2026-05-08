# Artifact Ownership

> **RULE: 1 artifact = 1 script owner duy nhất được ghi.**
> Tất cả script khác chỉ được **READ**.
> Path được define tập trung tại `artifacts/paths.py`.

## Naming Convention

```
<owner>_<semantic 2-3 words>.<ext>

.json = machine-readable
.md   = human-readable

_raw      = unprocessed output, consumed as-is by downstream
_log      = append-only, tích lũy across sessions
_session_ = overwrite mỗi pipeline run, không tích lũy

Suffixes .md:
  _summary   = condensed overview
  _synthesis = rewrite/enrich từ nhiều nguồn thành document liền mạch
  _synopsis  = high-level narrative
```

## Module → Script Mapping

| Module | Script | Role |
|---|---|---|
| `spectracker` | `01_spectracker.py` | Track spec version changes, decide rerun steps |
| `absorber` | `02_absorber.py` | Scan codebase, build knowledge maps |
| `clarificator` | `03_clarificator.py` | Clarify requirements via Q&A |
| `enricher` | `04_enricher.py` | Enrich context into structured prompt |
| `specwright` | `05_specwright.py` | Generate/update spec.md |
| `scaffolder` | `06_scaffolder.py` | Generate stub + test files |
| `planner` | `07_planner.py` | Decompose work into execution plan |
| `executor` | `08_executor.py` | Implement src/ files |
| `debugger` | `09_debugger.py` | Test + repair loop |
| `reporter` | `10_reporter.py` | Aggregate pipeline summary |
| `judge` | `11_judge.py` | Qualitative review + verdict |
| `patcher` | `12_patcher.py` | Fix from judge verdict |
| `archivist` | `13_archivist.py` | Distill knowledge, long-term memory |

## Ownership Table

### Root

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `specwright_spec_<slug>.md` | `specwright` | spectracker, scaffolder, planner, executor, judge, harness | persistent |

### state/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `clarificator_requirement_synthesis.md` | `clarificator` | enricher, specwright (fallback) | persistent |
| `scaffolder_file_scaffold.json` | `scaffolder` | planner, executor, reporter, judge, harness | persistent |
| `planner_full_execution_plan.json` | `planner` | executor, debugger, reporter, judge, patcher, harness | persistent |
| `planner_mini_execution_plan.json` | `planner` | executor, debugger, reporter, judge, patcher, harness | persistent |
| `planner_mini_impact_analysis.json` | `planner` | executor | persistent |
| `spectracker_applied_version.json` | `spectracker` | harness | persistent |

### cache/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `spectracker_compressed_spec.md` | `spectracker` | planner, executor, patcher | persistent |
| `spectracker_session_version_delta.json` | `spectracker` | harness | session |
| `absorber_codebase_snapshot.json` | `absorber` | clarificator, enricher, planner | session |
| `absorber_session_git_snapshot.json` | `absorber` | clarificator, enricher | session |

### execution/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `clarificator_session_raw.json` | `clarificator` | enricher, planner | session |
| `clarificator_session_questions.md` | `clarificator` | human review | session |
| `enricher_session_enriched_prompt.md` | `enricher` | specwright | session |
| `executor_session_manifest.json` | `executor` | reporter, judge, patcher, harness | session |
| `debugger_session_test_summary.json` | `debugger` | reporter, judge, archivist | session |
| `judge_session_verdict_raw.json` | `judge` | patcher, archivist, harness | session |
| `patcher_session_fix_summary.md` | `patcher` | human review | session |

### knowledge/current/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `clarificator_decision_log.md` | `clarificator` | clarificator (next session dedup) | append-only log |
| `absorber_codebase_map.md` | `absorber` | clarificator, enricher, planner, executor | persistent |
| `absorber_config_map.json` | `absorber` | clarificator, enricher | persistent |
| `absorber_blame_map.md` | `absorber` | clarificator, enricher, planner | persistent |
| `patcher_regression_log.md` | `patcher` | debugger, archivist | persistent |
| `archivist_spec_gaps.md` | `archivist` | specwright (next revision) | persistent |
| `archivist_knowledge_log.md` | `archivist` | planner, executor, debugger, patcher | append-only log |

### knowledge/history/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `archivist_curation_log.json` | `archivist` | human review | append-only log |
| `patcher_attempt_log.json` | `patcher` | human review, archivist | append-only log |
| `spectracker_version_log.md` | `spectracker` | human review | append-only log |

### reports/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `reporter_execution_summary.md` | `reporter` | human review | persistent |
| `judge_verdict_summary.md` | `judge` | human review, archivist | persistent |

## Removed Artifacts

| Old name | Reason |
|---|---|
| `state/plan_notes.json` | Merged vào `archivist_knowledge_log.md` — planner đọc knowledge log trực tiếp |
| `run/analysis_mini.json` | Legacy từ `mini_mode.py` không còn dùng — replaced by `planner_mini_impact_analysis.json` |
| `knowledge/history/git_history.json` | Point-in-time snapshot, không phải log — moved to `cache/absorber_session_git_snapshot.json` |

## Data Flow

```
specwright_spec_<slug>.md
  │
  ├─[spectracker]────► spectracker_session_version_delta.json
  │                    spectracker_applied_version.json
  │                    spectracker_version_log.md
  │                    spectracker_compressed_spec.md
  │
  └─[scaffolder]─────► scaffolder_file_scaffold.json
                              │
  [absorber]─────────► absorber_codebase_map.md
                        absorber_config_map.json
                        absorber_blame_map.md
                        absorber_codebase_snapshot.json
                        absorber_session_git_snapshot.json
                              │
  [clarificator]─────► clarificator_session_raw.json
                        clarificator_session_questions.md
                        clarificator_requirement_synthesis.md
                        clarificator_decision_log.md (append)
                              │
  [enricher]─────────► enricher_session_enriched_prompt.md
                              │
  [specwright]───────► specwright_spec_<slug>.md
                              │
  [planner]──────────► planner_full_execution_plan.json
                        planner_mini_execution_plan.json
                        planner_mini_impact_analysis.json
                              │
  [executor]─────────► executor_session_manifest.json
                              │
  [debugger]─────────► debugger_session_test_summary.json
                              │
  [judge]────────────► judge_session_verdict_raw.json
                        judge_verdict_summary.md
                              │
  [patcher]──────────► patcher_session_fix_summary.md
                        patcher_regression_log.md
                        patcher_attempt_log.json (append)
                              │
  [reporter]─────────► reporter_execution_summary.md
                              │
  [archivist]────────► archivist_knowledge_log.md (append)
                        archivist_spec_gaps.md
                        archivist_curation_log.json (append)
```

## Per-Script Contract

Mỗi script có header:
```python
# === WRITE AUTHORITY: <module_name> ===
# OWNS  : <list of artifacts this script writes>
# READS : <list of artifacts this script reads>
```
