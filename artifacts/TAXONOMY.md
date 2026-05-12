# Artifact Taxonomy – LLM Pipeline

This document describes the purpose, lifecycle, and ownership of all artifacts produced by the LLM pipeline.
All paths are relative to `artifacts_<slug>/`.

> **RULE: 1 artifact = 1 script owner duy nhất được quyền ghi.**
> Tất cả script khác chỉ được READ. Paths được define tập trung tại `artifacts/paths.py`.

---

## Directory Overview

```
artifacts_<slug>/
│
├── specwright_spec_<slug>.md              ← canonical spec, project-global (specwright)
│
├── state/                                 ← project-global exception (spectracker only)
│   └── spectracker_applied_version.json
│
├── sessions/
│   └── <NNN>/                             ← session-local scope
│       ├── state/
│       │   ├── clarificator_requirement_synthesis.md
│       │   ├── scaffolder_codebase_skeleton.json
│       │   ├── planner_full_execution_plan.json
│       │   ├── planner_mini_execution_plan.json
│       │   └── planner_mini_impact_analysis.json
│       │
│       ├── cache/
│       │   ├── spectracker_overwrite_version_delta.json
│       │   ├── absorber_overwrite_codebase_snapshot.json
│       │   └── absorber_overwrite_git_snapshot.json
│       │
│       ├── execution/
│       │   ├── clarificator_overwrite_raw.json
│       │   ├── clarificator_overwrite_questions.md
│       │   ├── enricher_overwrite_enriched_prompt.md
│       │   ├── executor_overwrite_manifest.json
│       │   ├── debugger_overwrite_test_summary.json
│       │   ├── judge_overwrite_verdict_raw.json
│       │   └── patcher_overwrite_fix_summary.md
│       │
│       └── reports/
│           ├── reporter_execution_summary.md
│           └── judge_verdict_summary.md
│
├── knowledge/                             ← project-global, shared across sessions
│   ├── current/
│   │   ├── clarificator_decision_log.md
│   │   ├── absorber_codebase_map.md
│   │   ├── absorber_config_map.json
│   │   ├── absorber_blame_map.md
│   │   ├── patcher_findings_snapshot.md
│   │   ├── archivist_spec_gaps.md
│   │   └── archivist_knowledge_log.md
│   └── history/
│       ├── archivist_curation_log.json
│       ├── patcher_attempt_log.json
│       ├── spectracker_version_log.md
│       ├── <version>.md               ← write-once spec snapshots (spectracker)
│
├── session_runs/                          ← project-global run history (harness)
│   ├── session_001_runs.json
│   └── session_002_runs.json
│
├── src/                                   ← build output, project-global (executor)
└── tests/                                 ← build output, project-global (scaffolder)
```

> `src/` and `tests/` are **build outputs**, not pipeline artifacts. Project-global.
> `state/prev_src/` is **scratch space** used by harness for delta runs — not a pipeline artifact, safe to delete.
> `state/spectracker_applied_version.json` is a **project-global exception** — see Special Notes.

---

## Ownership & Lifecycle Table

**Scope legend:** `[P]` project-global · `[S]` session-local

| Artifact | Scope | Owner | Write mode | Consumers |
|---|---|---|---|---|
| `specwright_spec_<slug>.md` | P | `specwright` | overwrite | spectracker, scaffolder, planner, executor, judge, harness |
| `state/spectracker_applied_version.json` | P | `spectracker` | hybrid† | spectracker (self), harness |
| `sessions/<N>/state/clarificator_requirement_synthesis.md` | S | `clarificator` | overwrite | enricher, specwright |
| `sessions/<N>/state/scaffolder_codebase_skeleton.json` | S | `scaffolder` | overwrite | planner, executor, reporter, judge, harness |
| `sessions/<N>/state/planner_full_execution_plan.json` | S | `planner` | overwrite | executor, debugger, reporter, judge, patcher, harness |
| `sessions/<N>/state/planner_mini_execution_plan.json` | S | `planner` | overwrite | executor, debugger, reporter, judge, patcher, harness |
| `sessions/<N>/state/planner_mini_impact_analysis.json` | S | `planner` | overwrite | executor |
| `sessions/<N>/cache/spectracker_overwrite_version_delta.json` | S | `spectracker` | overwrite | harness |
| `sessions/<N>/cache/absorber_overwrite_codebase_snapshot.json` | S | `absorber` | overwrite | clarificator, enricher, planner |
| `sessions/<N>/cache/absorber_overwrite_git_snapshot.json` | S | `absorber` | overwrite | clarificator, enricher |
| `sessions/<N>/execution/clarificator_overwrite_raw.json` | S | `clarificator` | overwrite | enricher, planner |
| `sessions/<N>/execution/clarificator_overwrite_questions.md` | S | `clarificator` | overwrite | human review |
| `sessions/<N>/execution/enricher_overwrite_enriched_prompt.md` | S | `enricher` | overwrite | specwright |
| `sessions/<N>/execution/executor_overwrite_manifest.json` | S | `executor` | overwrite | reporter, judge, patcher, harness |
| `sessions/<N>/execution/debugger_overwrite_test_summary.json` | S | `debugger` | overwrite | reporter, judge, archivist |
| `sessions/<N>/execution/judge_overwrite_verdict_raw.json` | S | `judge` | overwrite | patcher, archivist, harness |
| `sessions/<N>/execution/patcher_overwrite_fix_summary.md` | S | `patcher` | overwrite | human review |
| `sessions/<N>/reports/reporter_execution_summary.md` | S | `reporter` | overwrite | human review only |
| `sessions/<N>/reports/judge_verdict_summary.md` | S | `judge` | overwrite | human review only |
| `knowledge/current/clarificator_decision_log.md` | P | `clarificator` | append | clarificator (next session), enricher |
| `knowledge/current/absorber_codebase_map.md` | P | `absorber` | overwrite | clarificator, enricher, planner, executor |
| `knowledge/current/absorber_config_map.json` | P | `absorber` | overwrite | clarificator, enricher |
| `knowledge/current/absorber_blame_map.md` | P | `absorber` | overwrite | clarificator, enricher, planner |
| `knowledge/current/patcher_findings_snapshot.md` | P | `patcher` | overwrite | debugger, archivist |
| `knowledge/current/archivist_spec_gaps.md` | P | `archivist` | append/curated | specwright |
| `knowledge/current/archivist_knowledge_log.md` | P | `archivist` | append | planner, executor, debugger, patcher, judge |
| `knowledge/history/archivist_curation_log.json` | P | `archivist` | append | human review |
| `knowledge/history/patcher_attempt_log.json` | P | `patcher` | append | human review, archivist |
| `knowledge/history/spectracker_version_log.md` | P | `spectracker` | append | human review |
| `knowledge/history/<version>.md` | P | `spectracker` | write-once | spectracker (delta computation) |
| `session_runs/session_<N>_runs.json` | P | `harness` | append‡ | harness, human review |

† `spectracker_applied_version.json`: top-level fields overwrite each run; embedded `run_history[]` is append-only.
‡ `session_<N>_runs.json`: append-only at run-entry level — run entries are never deleted. File is atomically rewritten on each update (read full JSON → append entry → write back). Not byte-level append.

---

## Data Flow

```
[absorber]───────────────► absorber_codebase_map.md
                            absorber_config_map.json
                            absorber_blame_map.md
                            absorber_overwrite_codebase_snapshot.json     (cache)
                            absorber_overwrite_git_snapshot.json          (cache)

[clarificator]───────────► clarificator_overwrite_raw.json
                            clarificator_overwrite_questions.md
                            clarificator_requirement_synthesis.md
                            clarificator_decision_log.md (append)

[enricher]───────────────► enricher_overwrite_enriched_prompt.md

[specwright]─────────────► specwright_spec_<slug>.md
                              │
                              ├─[spectracker]────► spectracker_overwrite_version_delta.json → harness
                              │                     spectracker_applied_version.json
                              │                     spectracker_version_log.md (append)
                              │                     <version>.md
                              │
                              └─[scaffolder]─────► scaffolder_codebase_skeleton.json

[planner]────────────────► planner_full_execution_plan.json
                            planner_mini_execution_plan.json
                            planner_mini_impact_analysis.json

[executor]───────────────► executor_overwrite_manifest.json

[debugger]───────────────► debugger_overwrite_test_summary.json

[reporter]───────────────► reporter_execution_summary.md

[judge]──────────────────► judge_overwrite_verdict_raw.json
                            judge_verdict_summary.md

[patcher]────────────────► patcher_overwrite_fix_summary.md
                            patcher_findings_snapshot.md
                            patcher_attempt_log.json (append)

[archivist]──────────────► archivist_knowledge_log.md (append)
                            archivist_spec_gaps.md
                            archivist_curation_log.json (append)
```

---

## Special Notes

### Session isolation model

Pipeline artifacts are split into two scopes:

**Session-local** (`[S]`) — `sessions/<NNN>/state/`, `cache/`, `execution/`, `reports/`:
Isolated per session. Runs within the same session share these artifacts (later runs may overwrite earlier runs' outputs within the session). Different sessions never interfere.

**Project-global** (`[P]`) — `knowledge/`, `session_runs/`, `specwright_spec_<slug>.md`, `state/spectracker_applied_version.json`, `src/`, `tests/`:
Shared across all sessions. Long-term memory and applied state live here.

**Session** = logical unit of work (implement one spec version through to judge APPROVED).
**Run** = one harness.py invocation. A session may contain multiple runs (stop → review → resume).

`session_runs/session_<N>_runs.json` tracks all runs within a session: which steps ran, pass/fail status, from/until step, resume chain. Owned by `harness`, not a pipeline step.

---

### Execution order note

Although spectracker owns the spec delta artifacts, it runs **after specwright** in the canonical full flow because it requires `specwright_spec_<slug>.md`.

Canonical full flow:

```
absorber → clarificator → enricher → specwright → spectracker → scaffolder → planner → executor → debugger → reporter → judge → patcher → archivist
```

If the canonical spec does not exist yet, harness skips spectracker preflight and waits until specwright creates the spec. Spectracker itself exits cleanly with a `SKIP` message when run against a missing spec (default non-strict mode); use `--strict` to treat missing spec as exit 1 in CI.

---

### `specwright_spec_<slug>.md`
Canonical spec per project. Slug embedded in filename enables cross-project extraction without renaming. Use `get_spec_path()` from `artifacts/paths.py` — not a static constant. Owner is specwright; all other modules read only.

### `state/clarificator_requirement_synthesis.md`
Raw requirement rewritten inline with all clarification decisions resolved. Not a summary — full document with decisions incorporated. Consumed by enricher as primary input; specwright falls back to this if enriched prompt absent.

### `state/scaffolder_codebase_skeleton.json`
Generated by scaffolder from spec. Contains full stub file tree: function signatures, interfaces, JSDoc, test skeletons — no implementation bodies. Also carries `implementation_instructions.for_executor` read by executor as briefing. Overwritten fully each run.

### `state/planner_full_execution_plan.json`
Output of planner for full-scope runs. Per-file implementation tasks, ordered sub-tasks, dependency order, gotchas, Tailwind hints. **Immutable after planner writes** — downstream scripts read only.

### `state/planner_mini_impact_analysis.json`
Planner analysis of which files are impacted by the mini task scope. Distinct from the execution plan — this is the impact analysis that informs what executor touches.

### `state/spectracker_applied_version.json`
Deliberate hybrid lifecycle: top-level fields (current version, last run metadata) overwrite each run; embedded `run_history[]` array is append-only across runs. Rationale: keeps history coupled with current state without an additional artifact. Used by spectracker to determine first-run vs delta, and to load the previous snapshot for diffing.

**Project-global exception:** this file lives at project root `state/`, not inside `sessions/<NNN>/state/`. Rationale: spectracker must know the last successfully applied version across all sessions to compute deltas correctly. If session-local, each new session would lose the applied baseline and force a full rerun. All other `state/` artifacts are session-local.

In full harness runs, `write_applied()` is called by harness **at finalization time** only after the downstream pipeline succeeds — not during spectracker's normal run. This prevents a spec version from being marked applied before executor/debugger/judge completion. Ownership remains spectracker. A manual CLI fallback (`--mark-applied`) is available for recovery.

### `cache/spectracker_overwrite_version_delta.json`
Computed by spectracker from diff between current spec and last applied snapshot. Describes changed sections, affected files, steps to skip. Documented exception: placed in cache/ because derivable from spec + applied_version, but drives harness control flow rather than being a passive cache.

### `cache/absorber_overwrite_git_snapshot.json`
Point-in-time git state captured by absorber: recent commits, blame data. Overwritten each run — not a persistent log. Moved from knowledge/history/ to cache/ to reflect snapshot semantics.

### `execution/clarificator_overwrite_raw.json`
Structured session metadata: decisions array with tier/category/impact, tier counts, conflicts detected, unresolved findings list, requirement hash. Machine-readable; consumed by enricher and planner. For long-term Q&A memory see `clarificator_decision_log.md`.

### `execution/judge_overwrite_verdict_raw.json`
Raw judge API response, fully unprocessed. Preserved so patcher and archivist can parse independently, and failures are debuggable without re-calling the API. Human-readable rendering in `judge_verdict_summary.md`.

### `knowledge/current/patcher_findings_snapshot.md`
Per-run snapshot of judge findings processed by patcher: what was patched, escalated, confirm result. Overwritten each run — not a persistent log. Injected into debugger prompts as regression prevention context. For longitudinal history see `patcher_attempt_log.json`.

### `knowledge/current/archivist_knowledge_log.md`
Core persistent knowledge base. Consolidates three retired artifacts: `base.md`, `findings_notes.md`, `plan_notes.json`. Accumulates architecture decisions, recurring bug patterns, lessons learned across all runs. Append-only; human controls additions via archivist interactive mode. Human-editable by design — one of the few artifacts where manual editing is intended.

### `knowledge/current/archivist_spec_gaps.md`
Edge cases and spec gaps surfaced by judge, human-approved via archivist interactive flow. Injected into judge briefing so future runs are aware of known gaps. Also feeds specwright on next spec revision. Human-editable.

### `knowledge/history/<version>.md`
Write-once files created by spectracker each time a new spec version is applied. `<version>.md` is the raw spec snapshot used internally by spectracker for future delta computation (`_load_latest_snapshot`). Dynamic path constructed at runtime — not static constants in paths.py.

### `knowledge/history/archivist_curation_log.json`
Audit trail of human decisions when reviewing judge findings: which findings were applied to knowledge base, skipped, or escalated to spec bump. Distinct from `patcher_attempt_log.json` — archivist modifies knowledge artifacts while patcher modifies src/ directly.

### `reports/reporter_execution_summary.md` and `reports/judge_verdict_summary.md`
Human-readable only — no pipeline script parses either file. `reporter_execution_summary.md` summarises the full pipeline execution; on GitHub Actions it is piped to `$GITHUB_STEP_SUMMARY` and not committed. `judge_verdict_summary.md` is the primary artifact for a human deciding whether pipeline output is acceptable before merging.

### `session_runs/session_<N>_runs.json`
Owned by `harness`, not a pipeline step. Records all runs within a session in chronological order. Schema: `{ session_id, created_at, runs: [{ run_id, started_at, completed_at, status, scope, from_step, until_step, stopped_at_step, resumed_from_run, steps: [...], spec_version }] }`.

"Append-only" means run entries are never deleted — it is the semantic guarantee. Mechanically, the file is atomically rewritten on each update (read full JSON → append/update current run entry → write to temp → rename). Not byte-level append.

### Scratch space: `state/prev_src/`
Used exclusively by harness to stash unaffected source files before a delta run and restore them after executor overwrites `src/`. Not a pipeline artifact — not tracked in paths.py, not versioned, safe to delete.

---

## Removed Artifacts

| Old name | Reason |
|---|---|
| `state/plan_notes.json` | Merged into `archivist_knowledge_log.md`. Planner reads knowledge log directly — no separate injection file needed. |
| `state/enriched_prompt.md` | Moved to `execution/enricher_overwrite_enriched_prompt.md`. Session artifact; wrong location in state/. |
| `run/analysis_mini.json` | Legacy from deprecated `mini_mode.py`. Replaced by `planner_mini_impact_analysis.json`. |
| `run/mini_log.json` | Legacy from deprecated `mini_mode.py`. Entire mode removed. |
| `knowledge/history/git_history.json` | Point-in-time snapshot semantics, not a persistent log. Moved to `cache/absorber_overwrite_git_snapshot.json`. |
| `knowledge/current/base.md` | Merged into `archivist_knowledge_log.md`. |
| `knowledge/current/findings_notes.md` | Merged into `archivist_knowledge_log.md`. |
| `knowledge/current/findings.md` | Renamed to `patcher_findings_snapshot.md` — clarifies owner and snapshot lifecycle. |
| `knowledge/current/spec_addendum.md` | Renamed to `archivist_spec_gaps.md` — corrects owner (archivist, not judge) and clarifies purpose. |

---

## Adding New Artifacts

First, decide scope:

- **Session-local** → goes into `sessions/<N>/state/`, `cache/`, `execution/`, or `reports/`; use `_SessLazyPath` in paths.py
- **Project-global** → goes into `knowledge/`, `session_runs/`, or project root; use `_LazyPath` in paths.py

Then pick the right directory:

1. **`sessions/<N>/state/`** — persistent within a session, enables step-skipping across runs in same session.
2. **`sessions/<N>/cache/`** — heavy intermediates regenerable from source. Use `_overwrite_` in filename. Document if used for control flow (exception).
3. **`sessions/<N>/execution/`** — per-run outputs consumed by downstream steps. Use `_overwrite_` in filename.
4. **`sessions/<N>/reports/`** — human-readable only. Never consumed by pipeline logic.
5. **`knowledge/current/`** — files read by pipeline steps across sessions. Append-only or curated overwrite.
6. **`knowledge/history/`** — append-only logs and write-once snapshots. Human audit only unless noted.
7. **Build outputs** (`src/`, `tests/`) — written only by scaffolder and executor. Project-global.
8. **Scratch space** — use `state/` subdirs; not versioned as artifacts.

Hybrid lifecycle (e.g., embedded append arrays inside overwritten file) must be documented as exception with rationale in Special Notes.

Every new artifact must be registered in all three places:
- `artifacts/paths.py` — with owner, consumers, lifecycle, purpose, **scope** comments; correct `_LazyPath` vs `_SessLazyPath`
- `artifacts/OWNERSHIP.md` — in ownership table with scope column
- This document — directory overview, lifecycle table, special note

---

*Last updated: 2026-05-09*
*Maintained alongside `pipeline/` and `harness.py`.*
