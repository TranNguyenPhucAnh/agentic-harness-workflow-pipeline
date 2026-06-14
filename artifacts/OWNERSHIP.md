# Artifact Ownership

> **RULE: 1 artifact = 1 script owner duy nhất được ghi.**
> Tất cả script khác chỉ được **READ**.
> Paths được define tập trung tại `artifacts/paths.py`.

---

## Naming Convention

```
<noun>.<ext>               ← short-term (overwrite each run)
<noun>_log.<ext>           ← long-term (append-only audit)

Folder encodes owner — no module prefix needed in filename.

.json = machine-readable
.md   = human/LLM target

Short-term .md suffixes (for reference):
  _summary   = condensed overview
  _synthesis = rewrite/enrich from multiple sources into coherent document

Long-term logs are always .json to facilitate structured append.
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
| `executor` | `08_executor.py` | Implement output/src/ files |
| `debugger` | `09_debugger.py` | Test + repair loop |
| `reporter` | `10_reporter.py` | Aggregate pipeline summary |
| `judge` | `11_judge.py` | Qualitative review + verdict |
| `patcher` | `12_patcher.py` | Fix from judge verdict |
| `archivist` | `13_archivist.py` | Distill knowledge, long-term memory |

---

## Ownership Table

### `spec/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `spec/specwright_spec_<slug>.md` | `specwright` | spectracker, scaffolder, planner, executor, judge, harness | overwrite when specwright reruns |

> Use `get_spec_path()` from `artifacts/paths.py` — not a static constant.
> Slug in filename enables cross-project spec extraction without renaming.

---

### `absorber/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `absorber/codebase_map.md` | `absorber` | clarificator, enricher, planner, executor | short-term, overwrite |
| `absorber/codebase_log.json` | `absorber` | human review | long-term, append |
| `absorber/cache/codebase_snapshot.json` | `absorber` | absorber (self) | internal cache, overwrite |

`codebase_map.md` merges three old files: config map, blame map, and codebase map. Sections: `## Codebase`, `## Config`, `## Git/Blame`.
`codebase_log.json` fields per entry: `generated_at`, `target`, `total_files`, `cached_files`, `extracted_files`, `modes`, `languages`, `map_size_bytes`, `cost`, `git_scope`, `hotspot_summary[]`.

---

### `clarificator/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `clarificator/session.json` | `clarificator` | enricher, planner, executor, debugger, patcher | short-term, overwrite |
| `clarificator/decision_log.json` | `clarificator` | clarificator (next run), enricher | long-term, append |

`session.json` fields: `decisions[]`, `conflicts[]`, `unresolved[]`, `tier_counts`, `input_sources`, `req_hash`, `requirement_synthesis`.
`decision_log.json` trims `impacts[]` to max 3 items per decision when `len(decisions) > 20` — prevents unbounded growth.

---

### `enricher/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `enricher/enriched_prompt.md` | `enricher` | specwright, planner | short-term, overwrite |
| `enricher/prompt_log.json` | `enricher` | human review | long-term, append |

`prompt_log.json` fields per entry: `generated_at`, `project`, `model`, `input_session_hash`, `decisions_count`, `extra_context`, `enriched_prompt_length`, `cost`.

---

### `spectracker/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `spectracker/version_delta.json` | `spectracker` | harness (control flow) | short-term, overwrite† |
| `spectracker/version_log.json` | `spectracker` | spectracker (self), harness | long-term, append‡ |

† `version_delta.json`: documented exception — derivable from spec + log, but placed here (not cache/) because it drives harness step-skipping logic.
‡ `version_log.json`: append-only at entry level; `write_applied()` mutates the matching entry's `applied` field in-place after downstream pipeline succeeds. This is the sole mutation exception. Replaces both `spectracker_applied_version.json` and `spectracker_version_log.md`.

`version_log.json` entry schema: `version`, `generated_at`, `applied`, `applied_at`, `changed_sections[]`, `affected_files[]`, `spec_content` (raw spec snapshot, truncated if > 50 000 chars).

---

### `scaffolder/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `scaffolder/blueprint.json` | `scaffolder` | planner, executor, reporter, judge, harness | short-term, overwrite |
| `scaffolder/skeleton_log.json` | `scaffolder` | human review | long-term, append |

`blueprint.json` schema: module-centric (`modules[].files[].kind`). `kind` values: `"source"` | `"test"` | `"config"` | `"migration"`. Fields `is_test` and `code` removed. Log uses `{"entries": [...]}` wrapper consistent with all other pipeline logs.

---

### `planner/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `planner/full_plan.json` | `planner` | executor, debugger, reporter, judge, patcher, harness | short-term, overwrite |
| `planner/mini_plan.json` | `planner` | executor, debugger, reporter, judge, patcher, harness | short-term, overwrite |
| `planner/plan_log.json` | `planner` | human review | long-term, append |

`mini_plan.json` schema: `{ "plan": {...}, "impact": {...} }` — merges two old separate files.
`plan_log.json` is shared by both full and mini scope runs. Entry fields: `scope`, `generated_at`, `model`, `cost`, `plan_version`, `task_count`, plus scope-specific fields.

---

### `executor/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `executor/manifest.json` | `executor` | reporter, judge, patcher, harness | short-term, overwrite |
| `executor/manifest_log.json` | `executor` | human review | long-term, append |

Build output written to `output/src/` (not `src/` directly).

---

### `debugger/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `debugger/test_summary.json` | `debugger` | reporter, judge, archivist | short-term, overwrite |
| `debugger/test_log.json` | `debugger` | human review | long-term, append |

`test_log.json` trim policy per entry: keep full `final_status`, `scope`, `total_iterations`, `max_iter`, `escalated[]`, and cluster_details for last iteration only. Trim earlier iterations to `{ iteration, passed, clusters_found, clusters_repaired, summary }`. Drop `log_snippet` (available in short-term file if needed).

---

### `reporter/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `reporter/execution_summary.md` | `reporter` | human review only | short-term, overwrite |
| `reporter/execution_log.json` | `reporter` | human review | long-term, append |

`execution_summary.md`: on GitHub Actions piped to `$GITHUB_STEP_SUMMARY`, not committed.
`execution_log.json` fields: `scope`, `final_status`, `iterations_used`, `max_iter`, `files_implemented`, `failed_cluster_count`, `escalated_count`, `spec_version`, `cost_total`, `generated_at`.

---

### `judge/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `judge/verdict_raw.json` | `judge` | patcher, archivist, harness | short-term, overwrite |
| `judge/verdict_summary.md` | `judge` | human review only | short-term, overwrite |
| `judge/verdict_log.json` | `judge` | human review | long-term, append |

Two short-term files retained: `.json` for machine consumers, `.md` for human reading without JSON parsing. `verdict_raw.json` uses `default=lambda o: vars(o) if hasattr(o, '__dict__') else str(o)` in `json.dumps()` to handle `CompletionTokensDetails`.

---

### `patcher/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `patcher/fix_summary.md` | `patcher` | human review | short-term, overwrite |
| `patcher/attempt_log.json` | `patcher` | debugger (last entry), archivist, planner (mini) | long-term, append |

`patcher_findings_snapshot.md` removed — consumers read last entry of `attempt_log.json` directly.
Build output written to `output/src/`.

---

### `archivist/`

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `archivist/knowledge_log.md` | `archivist` | planner, executor, debugger, patcher, judge | append, LLM prompt injection |
| `archivist/spec_gaps.md` | `archivist` | specwright, judge | append/curated, LLM prompt injection |
| `archivist/curation_log.json` | `archivist` | human review | append, audit |

No short-term overwrite — pure accumulation module. `.md` format because primary target is LLM prompt injection. Human-editable by design. Consolidates three retired artifacts: `base.md`, `findings_notes.md`, `plan_notes.json`.

---

### Project root

| Artifact | Owner | Consumers | Lifecycle |
|---|---|---|---|
| `harness_run_log.json` | `harness` | harness, human review | append-only |

Replaces `session_runs/session_<N>_runs.json`. Session layer removed entirely. Schema: `{ "entries": [{ "run_id", "started_at", "completed_at", "status", "scope", "from_step", "until_step", "stopped_at_step", "spec_version", "steps": [...] }] }`.

---

## Build Outputs (not pipeline artifacts)

| Path | Written by | Notes |
|---|---|---|
| `output/src/` | executor, debugger, patcher | Primary implementation output |
| `output/tests/` | scaffolder | Test skeleton output |
| `output/prev_src/` | harness (scratch) | Temp stash for delta runs — safe to delete |

---

## Data Flow

```
[absorber]───────────────► absorber/codebase_map.md
                            absorber/codebase_log.json        (append)
                            absorber/cache/codebase_snapshot.json

[clarificator]───────────► clarificator/session.json
                            clarificator/decision_log.json    (append)

[enricher]───────────────► enricher/enriched_prompt.md
                            enricher/prompt_log.json          (append)

[specwright]─────────────► spec/specwright_spec_<slug>.md
                              │
                              ├─[spectracker]────► spectracker/version_delta.json → harness
                              │                    spectracker/version_log.json    (append)
                              │
                              └─[scaffolder]─────► scaffolder/blueprint.json
                                                   scaffolder/skeleton_log.json   (append)

[planner]────────────────► planner/full_plan.json
                            planner/mini_plan.json            ({ "plan": {...}, "impact": {...} })
                            planner/plan_log.json             (append)

[executor]───────────────► executor/manifest.json
                            executor/manifest_log.json        (append)

[debugger]───────────────► debugger/test_summary.json
                            debugger/test_log.json            (append)

[reporter]───────────────► reporter/execution_summary.md
                            reporter/execution_log.json       (append)

[judge]──────────────────► judge/verdict_raw.json
                            judge/verdict_summary.md
                            judge/verdict_log.json            (append)

[patcher]────────────────► patcher/fix_summary.md
                            patcher/attempt_log.json          (append)

[archivist]──────────────► archivist/knowledge_log.md        (append)
                            archivist/spec_gaps.md            (append/curated)
                            archivist/curation_log.json       (append)
```

---

## Removed Artifacts

| Old name | Removed from | Reason |
|---|---|---|
| `sessions/<NNN>/` entire layer | everywhere | Session model removed |
| `session_runs/session_<N>_runs.json` | project root | Replaced by `harness_run_log.json` |
| `state/spectracker_applied_version.json` | state/ | Merged into `spectracker/version_log.json` |
| `knowledge/history/spectracker_spec_snapshot_<v>.md` | knowledge/history/ | Embedded in `spectracker/version_log.json` entries |
| `knowledge/current/absorber_config_map.json` | knowledge/current/ | Merged into `absorber/codebase_map.md` |
| `knowledge/current/absorber_blame_map.md` | knowledge/current/ | Merged into `absorber/codebase_map.md` |
| `cache/absorber_overwrite_git_snapshot.json` | cache/ | Git data inline in codebase_map.md |
| `state/clarificator_requirement_synthesis.md` | state/ | Field in `clarificator/session.json` |
| `execution/clarificator_overwrite_questions.md` | execution/ | Print terminal only |
| `knowledge/current/clarificator_decision_log.md` | knowledge/current/ | Converted to `clarificator/decision_log.json` |
| `knowledge/current/patcher_findings_snapshot.md` | knowledge/current/ | Consumers read last entry of `patcher/attempt_log.json` |
| `state/planner_mini_impact_analysis.json` | state/ | `impact` field in `planner/mini_plan.json` |
| `knowledge/current/base.md` | knowledge/current/ | Merged into `archivist/knowledge_log.md` |
| `knowledge/current/findings_notes.md` | knowledge/current/ | Merged into `archivist/knowledge_log.md` |
| `state/plan_notes.json` | state/ | Merged into `archivist/knowledge_log.md` |
| `run/analysis_mini.json` | execution/ | Legacy from deprecated `mini_mode.py` |
| `run/mini_log.json` | execution/ | Legacy from deprecated `mini_mode.py` |

---

## Per-Script Contract

Mỗi script có header:
```python
# === WRITE AUTHORITY: <module_name> ===
# OWNS  : <list of artifacts this script writes>
# READS : <list of artifacts this script reads>
```
