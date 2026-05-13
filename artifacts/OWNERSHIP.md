# Artifact Ownership

> **RULE: 1 artifact = 1 script owner duy nhất được ghi.**
> Tất cả script khác chỉ được **READ**.
> Paths được define tập trung tại `artifacts/paths.py`.

---

## Naming Convention

```
<owner>_<semantic 2-3 words>.<ext>

.json = machine-readable
.md   = human-readable

_raw      = unprocessed output, consumed as-is by downstream
_log      = append-only, tích lũy across sessions (long-term memory)
_session_ = overwrite mỗi pipeline run, không tích lũy

Suffixes .md:
  _summary   = condensed overview
  _synthesis = rewrite/enrich từ nhiều nguồn thành document liền mạch
  _synopsis  = high-level narrative
```

---

## Module → Script Mapping

| Module | Script | Role |
|---|---|---|
| `absorber` | `01_absorber.py` | Scan codebase, build knowledge maps |
| `clarificator` | `02_clarificator.py` | Clarify requirements via interactive Q&A |
| `enricher` | `03_enricher.py` | Enrich context into structured prompt |
| `specwright` | `04_specwright.py` | Generate/update spec |
| `spectracker` | `05_spectracker.py` | Track spec version changes, decide rerun steps |
| `scaffolder` | `06_scaffolder.py` | Generate stub + test files |
| `planner` | `07_planner.py` | Decompose work into execution plan |
| `executor` | `08_executor.py` | Implement src/ files |
| `debugger` | `09_debugger.py` | Test + repair loop |
| `reporter` | `10_reporter.py` | Aggregate pipeline summary |
| `judge` | `11_judge.py` | Qualitative review + verdict |
| `patcher` | `12_patcher.py` | Fix from judge verdict |
| `archivist` | `13_archivist.py` | Distill knowledge, long-term memory |

---

## Ownership Table

### Root

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `specwright_spec_<slug>.md` | `specwright` | spectracker, scaffolder, planner, executor, judge, harness | persistent — overwrite when specwright reruns |

> Use `get_spec_path()` from `artifacts/paths.py` — not a static constant.
> Slug in filename enables cross-project spec extraction without renaming.

---

### state/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `clarificator_requirement_synthesis.md` | `clarificator` | enricher, specwright (fallback) | persistent |
| `scaffolder_codebase_skeleton.json` | `scaffolder` | planner, executor, reporter, judge, harness | persistent |
| `planner_full_execution_plan.json` | `planner` | executor, debugger, reporter, judge, patcher, harness | persistent |
| `planner_mini_execution_plan.json` | `planner` | executor, debugger, reporter, judge, patcher, harness | persistent |
| `planner_mini_impact_analysis.json` | `planner` | executor | persistent |
| `spectracker_applied_version.json` | `spectracker` | spectracker (self), harness | hybrid† |

† `spectracker_applied_version.json`: top-level fields overwrite each run; embedded `run_history[]` is append-only.
In full harness runs, `write_applied()` is called by harness **at finalization time** only after the downstream pipeline succeeds — not during spectracker's normal run. This prevents a spec version from being marked applied before executor/debugger/judge completion. Ownership remains spectracker.

---

### cache/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `spectracker_session_version_delta.json` | `spectracker` | harness | session |
| `absorber_session_codebase_snapshot.json` | `absorber` | clarificator, enricher, planner | session |
| `absorber_session_git_snapshot.json` | `absorber` | clarificator, enricher | session |

> `spectracker_session_version_delta.json`: exception — in cache/ but drives harness control flow.

---

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

---

### knowledge/current/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `clarificator_decision_log.md` | `clarificator` | clarificator (next session), enricher | append-only log |
| `absorber_codebase_map.md` | `absorber` | clarificator, enricher, planner, executor | persistent |
| `absorber_config_map.json` | `absorber` | clarificator, enricher | persistent |
| `absorber_blame_map.md` | `absorber` | clarificator, enricher, planner | persistent |
| `patcher_findings_snapshot.md` | `patcher` | debugger, archivist | persistent |
| `archivist_spec_gaps.md` | `archivist` | specwright | persistent |
| `archivist_knowledge_log.md` | `archivist` | planner, executor, debugger, patcher, judge | append-only log |

> `patcher_findings_snapshot.md`: per-run snapshot only — for longitudinal history see `patcher_attempt_log.json`.
> `archivist_knowledge_log.md`: consolidates old `base.md` + `findings_notes.md` + `plan_notes.json`.

---

### knowledge/history/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `archivist_curation_log.json` | `archivist` | human review | append-only log |
| `patcher_attempt_log.json` | `patcher` | human review, archivist | append-only log |
| `spectracker_version_log.md` | `spectracker` | human review | append-only log |
| `spectracker_spec_snapshot_<version>.md` *(dynamic)* | `spectracker` | spectracker (delta computation) | write-once |

> Dynamic paths are constructed by spectracker at runtime using the version string.
> Cannot be static constants in paths.py.

---

### reports/

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `reporter_execution_summary.md` | `reporter` | human review only | persistent |
| `judge_verdict_summary.md` | `judge` | human review only | persistent |

---

## Removed Artifacts

| Old name | Removed from | Reason |
|---|---|---|
| `state/plan_notes.json` | state/ | Merged into `archivist_knowledge_log.md` — planner reads knowledge log directly |
| `state/enriched_prompt.md` | state/ | Moved to execution/ as session artifact `enricher_session_enriched_prompt.md` |
| `run/analysis_mini.json` | execution/ | Legacy from deprecated `mini_mode.py` — replaced by `planner_mini_impact_analysis.json` |
| `run/mini_log.json` | execution/ | Legacy from deprecated `mini_mode.py` |
| `knowledge/history/git_history.json` | history/ | Point-in-time snapshot, not a log — moved to `cache/absorber_session_git_snapshot.json` |

---

## Data Flow

> **Execution order note:**
> Spectracker runs **after** specwright has produced `specwright_spec_<slug>.md`.
> On first runs where no canonical spec exists yet, harness skips spectracker until specwright creates the spec.

```
[absorber]───────────────► absorber_codebase_map.md
                            absorber_config_map.json
                            absorber_blame_map.md
                            absorber_session_codebase_snapshot.json        (cache, session)
                            absorber_session_git_snapshot.json             (cache, session)

[clarificator]───────────► clarificator_session_raw.json           (execution, session)
                            clarificator_session_questions.md      (execution, session)
                            clarificator_requirement_synthesis.md  (state)
                            clarificator_decision_log.md (append)  (knowledge/current)

[enricher]───────────────► enricher_session_enriched_prompt.md     (execution, session)

[specwright]─────────────► specwright_spec_<slug>.md               (root)
                              │
                              ├─[spectracker]────► spectracker_session_version_delta.json (cache, session)
                              │                    spectracker_applied_version.json       (state, hybrid)
                              │                    spectracker_version_log.md (append)    (knowledge/history)
                              │                    spectracker_spec_snapshot_<version>.md (knowledge/history)
                              │
                              └─[scaffolder]─────► scaffolder_codebase_skeleton.json      (state)

[planner]────────────────► planner_full_execution_plan.json        (state)
                            planner_mini_execution_plan.json       (state)
                            planner_mini_impact_analysis.json      (state)

[executor]───────────────► executor_session_manifest.json          (execution, session)

[debugger]───────────────► debugger_session_test_summary.json      (execution, session)

[reporter]───────────────► reporter_execution_summary.md           (reports)

[judge]──────────────────► judge_session_verdict_raw.json          (execution, session)
                            judge_verdict_summary.md               (reports)

[patcher]────────────────► patcher_session_fix_summary.md          (execution, session)
                            patcher_findings_snapshot.md           (knowledge/current)
                            patcher_attempt_log.json (append)      (knowledge/history)

[archivist]──────────────► archivist_knowledge_log.md (append)     (knowledge/current)
                            archivist_spec_gaps.md                 (knowledge/current)
                            archivist_curation_log.json (append)   (knowledge/history)
```

---

## Per-Script Contract

Mỗi script có header:
```python
# === WRITE AUTHORITY: <module_name> ===
# OWNS  : <list of artifacts this script writes>
# READS : <list of artifacts this script reads>
```
