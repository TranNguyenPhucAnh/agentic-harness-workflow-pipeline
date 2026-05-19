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
├── spec/
│   └── specwright_spec_<slug>.md          ← canonical spec (specwright)
│
├── absorber/
│   ├── codebase_map.md                    ← short-term, overwrite (merged config + git/blame)
│   ├── codebase_log.json                  ← long-term, append
│   └── cache/
│       └── codebase_snapshot.json         ← internal cache, no long-term pair
│
├── clarificator/
│   ├── session.json                       ← short-term, overwrite (incl. requirement_synthesis)
│   └── decision_log.json                  ← long-term, append
│
├── enricher/
│   ├── enriched_prompt.md                 ← short-term, overwrite
│   └── prompt_log.json                    ← long-term, append
│
├── spectracker/
│   ├── version_delta.json                 ← short-term, overwrite (drives harness control flow)
│   └── version_log.json                   ← long-term, append (applied state + snapshot history)
│
├── scaffolder/
│   ├── blueprint.json                     ← short-term, overwrite
│   └── skeleton_log.json                  ← long-term, append
│
├── planner/
│   ├── full_plan.json                     ← short-term, overwrite (full scope)
│   ├── mini_plan.json                     ← short-term, overwrite (mini scope, plan+impact merged)
│   └── plan_log.json                      ← long-term, append (shared by full + mini)
│
├── executor/
│   ├── manifest.json                      ← short-term, overwrite
│   └── manifest_log.json                  ← long-term, append
│
├── debugger/
│   ├── test_summary.json                  ← short-term, overwrite
│   └── test_log.json                      ← long-term, append
│
├── reporter/
│   ├── execution_summary.md               ← short-term, overwrite (human)
│   └── execution_log.json                 ← long-term, append
│
├── judge/
│   ├── verdict_raw.json                   ← short-term, overwrite (machine)
│   ├── verdict_summary.md                 ← short-term, overwrite (human)
│   └── verdict_log.json                   ← long-term, append
│
├── patcher/
│   ├── fix_summary.md                     ← short-term, overwrite
│   └── attempt_log.json                   ← long-term, append
│
├── archivist/
│   ├── knowledge_log.md                   ← append, LLM prompt injection target
│   ├── spec_gaps.md                       ← append, LLM prompt injection target
│   └── curation_log.json                  ← append, human audit
│
├── output/
│   ├── src/                               ← build output (executor, debugger, patcher)
│   ├── tests/                             ← build output (scaffolder)
│   ├── dist/                              ← future
│   └── coverage/                          ← future
│
└── harness_run_log.json                   ← harness-owned, append-only (replaces session_runs/)
```

> `output/src/` and `output/tests/` are **build outputs**, not pipeline artifacts.
> `output/prev_src/` is **scratch space** used by harness for delta runs — not a pipeline artifact, safe to delete.
> `spectracker/version_delta.json` is a **documented control-flow exception** — see Special Notes.

---

## Lifecycle Pattern

Every module follows the same **short-term ↔ long-term pair** pattern:

| Type | Filename convention | Write mode | Purpose |
|---|---|---|---|
| Short-term | `<noun>.json` / `<noun>.md` | overwrite each run | consumed by downstream steps |
| Long-term | `<noun>_log.json` | append-only | audit trail across all runs |

`.md` short-term files target human/LLM reading. `.json` short-term files target machine consumers.
All long-term logs use `.json` to facilitate structured append.

**Exception:** `archivist/` has no short-term overwrite — it is a pure accumulation module by design.

---

## Ownership & Lifecycle Table

| Artifact | Owner | Write mode | Consumers |
|---|---|---|---|
| `spec/specwright_spec_<slug>.md` | `specwright` | overwrite | spectracker, scaffolder, planner, executor, judge, harness |
| `absorber/codebase_map.md` | `absorber` | overwrite | clarificator, enricher, planner, executor |
| `absorber/codebase_log.json` | `absorber` | append | human review |
| `absorber/cache/codebase_snapshot.json` | `absorber` | overwrite | absorber (self, incremental runs) |
| `clarificator/session.json` | `clarificator` | overwrite | enricher, planner, executor, debugger, patcher |
| `clarificator/decision_log.json` | `clarificator` | append | clarificator (next run), enricher |
| `enricher/enriched_prompt.md` | `enricher` | overwrite | specwright, planner |
| `enricher/prompt_log.json` | `enricher` | append | human review |
| `spectracker/version_delta.json` | `spectracker` | overwrite | harness (control flow)† |
| `spectracker/version_log.json` | `spectracker` | append‡ | spectracker (self), harness |
| `scaffolder/blueprint.json` | `scaffolder` | overwrite | planner, executor, reporter, judge, harness |
| `scaffolder/skeleton_log.json` | `scaffolder` | append | human review |
| `planner/full_plan.json` | `planner` | overwrite | executor, debugger, reporter, judge, patcher, harness |
| `planner/mini_plan.json` | `planner` | overwrite | executor, debugger, reporter, judge, patcher, harness |
| `planner/plan_log.json` | `planner` | append | human review |
| `executor/manifest.json` | `executor` | overwrite | reporter, judge, patcher, harness |
| `executor/manifest_log.json` | `executor` | append | human review |
| `debugger/test_summary.json` | `debugger` | overwrite | reporter, judge, archivist |
| `debugger/test_log.json` | `debugger` | append | human review |
| `reporter/execution_summary.md` | `reporter` | overwrite | human review only |
| `reporter/execution_log.json` | `reporter` | append | human review |
| `judge/verdict_raw.json` | `judge` | overwrite | patcher, archivist, harness |
| `judge/verdict_summary.md` | `judge` | overwrite | human review only |
| `judge/verdict_log.json` | `judge` | append | human review |
| `patcher/fix_summary.md` | `patcher` | overwrite | human review |
| `patcher/attempt_log.json` | `patcher` | append | debugger (last entry), archivist, planner (mini) |
| `archivist/knowledge_log.md` | `archivist` | append | planner, executor, debugger, patcher, judge |
| `archivist/spec_gaps.md` | `archivist` | append/curated | specwright, judge |
| `archivist/curation_log.json` | `archivist` | append | human review |
| `harness_run_log.json` | `harness` | append§ | harness, human review |

† `spectracker/version_delta.json`: documented exception — derivable from spec + log, but drives harness step-skipping control flow rather than being a passive cache.
‡ `spectracker/version_log.json`: append-only at entry level; `write_applied()` mutates the matching entry's `applied` field in-place after downstream pipeline succeeds.
§ `harness_run_log.json`: append-only at run level — entries never deleted. File is atomically rewritten (read → append → write to temp → rename). Not byte-level append.

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

## Special Notes

### Execution order

Canonical full flow:

```
absorber → clarificator → enricher → specwright → spectracker → scaffolder → planner → executor → debugger → reporter → judge → patcher → archivist
```

Spectracker runs **after** specwright because it requires `spec/specwright_spec_<slug>.md`. If the spec does not exist yet, harness skips spectracker until specwright creates it. Spectracker exits cleanly with a `SKIP` message when run against a missing spec (default non-strict mode); use `--strict` to treat missing spec as exit 1 in CI.

---

### `spec/specwright_spec_<slug>.md`

Canonical spec per project. Slug embedded in filename enables cross-project extraction without renaming. Use `get_spec_path()` from `artifacts/paths.py` — not a static constant. Owner is specwright; all other modules read only.

---

### `absorber/codebase_map.md`

Merged output of three old separate artifacts. Sections: `## Codebase`, `## Config`, `## Git/Blame`. Single file reduces consumer complexity. `codebase_log.json` records per-run metadata including `cost`, `git_scope`, and `hotspot_summary[]` (top-N files by change frequency).

---

### `clarificator/session.json`

Replaces three old artifacts (`clarificator_overwrite_raw.json`, `clarificator_requirement_synthesis.md`, `clarificator_session_questions.md`). Questions are now printed to terminal only — not persisted. `requirement_synthesis` is embedded as a text field inside `session.json`.

Fields: `decisions[]`, `conflicts[]`, `unresolved[]`, `tier_counts`, `input_sources`, `req_hash`, `requirement_synthesis`.

Consumers extract `session["requirement_synthesis"]` directly.

---

### `spectracker/version_delta.json`

Documented exception: derivable artifact placed in `spectracker/` (not `cache/`) because it is module-owned and drives harness step-skipping — not a passive cache. Harness reads this file to decide which pipeline steps to skip for delta runs.

---

### `spectracker/version_log.json`

Replaces two old artifacts (`spectracker_applied_version.json` and `spectracker_version_log.md`). Each entry embeds `spec_content` — eliminates separate per-version `.md` snapshot files. `write_applied()` patches `applied: true` on the matching entry after downstream pipeline succeeds. Harness calls `write_applied()` at finalization time only — prevents a version being marked applied before executor/debugger/judge complete.

Trim policy: if `len(spec_content) > 50_000` then truncate deterministically. No LLM summarization.

---

### `scaffolder/blueprint.json`

Module-centric schema replaces old flat-file `scaffolder_codebase_skeleton.json`:

```json
{
  "generated_at": "...",
  "spec_version": "v1.3",
  "modules": [
    {
      "module": "auth",
      "purpose": "authentication and session management",
      "files": [
        { "path": "src/auth/service.py", "kind": "source" },
        { "path": "tests/auth/test_service.py", "kind": "test" }
      ]
    }
  ],
  "summary": { "total": 12, "source": 8, "test": 4 }
}
```

`kind` values: `"source"` | `"test"` | `"config"` | `"migration"`. Fields `is_test` and `code` are removed. Consumers use `kind != "test"` instead of `is_test == false`.

---

### `planner/mini_plan.json`

Merges two old artifacts (`planner_mini_execution_plan.json` + `planner_mini_impact_analysis.json`) into one file:

```json
{ "plan": { ... }, "impact": { ... } }
```

Consumers extract `mini_plan["plan"]` and `mini_plan["impact"]` independently. `plan_log.json` is shared by both `full` and `mini` scope runs — entries carry a `"scope"` field, plus `"cost"` and `"model"` fields.

---

### `archivist/` — pure accumulation

No short-term overwrite file — intentional. `knowledge_log.md` and `spec_gaps.md` use `.md` because their primary target is LLM prompt injection. Human-editable by design — one of the few artifacts where manual editing is intended. `decision_log.json` entries are trimmed if `decisions > 20` per entry to prevent unbounded growth.

---

### `harness_run_log.json`

Replaces `session_runs/session_<N>_runs.json`. Session concept removed — there is no longer a `sessions/<NNN>/` layer. Each harness invocation appends one run entry:

```json
{
  "entries": [
    {
      "run_id": "run_1747123456_abc12345",
      "started_at": "2026-05-16T08:00:00Z",
      "completed_at": "2026-05-16T09:23:00Z",
      "status": "FAIL",
      "scope": "full",
      "from_step": "absorber",
      "until_step": "judge",
      "stopped_at_step": "judge",
      "spec_version": "v1.3",
      "steps": [
        { "step": "absorber", "status": "PASS", "at": "..." },
        { "step": "judge", "status": "FAIL", "at": "..." }
      ]
    }
  ]
}
```

---

### `output/prev_src/`

Scratch space used exclusively by harness to stash unaffected source files before a delta run and restore them after executor overwrites `output/src/`. Not a pipeline artifact — not in paths.py, not versioned, safe to delete.

---

### `KNOWLEDGE_SOURCES` in `paths.py`

Enumerates all long-term append logs so future hypothesis/consultant modules can query the full pipeline history:

```python
KNOWLEDGE_SOURCES: list = [
    ABSORBER_CODEBASE_LOG,
    CLARIFICATOR_DECISION_LOG,
    ENRICHER_PROMPT_LOG,
    SPECTRACKER_VERSION_LOG,
    SCAFFOLDER_SKELETON_LOG,
    PLANNER_PLAN_LOG,
    EXECUTOR_MANIFEST_LOG,
    DEBUGGER_TEST_LOG,
    REPORTER_EXECUTION_LOG,
    JUDGE_VERDICT_LOG,
    PATCHER_ATTEMPT_LOG,
    ARCHIVIST_KNOWLEDGE_LOG,
    ARCHIVIST_SPEC_GAPS,
    ARCHIVIST_CURATION_LOG,
]
```

---

## Removed Artifacts

| Old artifact | Reason |
|---|---|
| `sessions/<NNN>/` entire layer | Session model removed — audit trail via append-only logs per module |
| `session_runs/session_<N>_runs.json` | Replaced by `harness_run_log.json` at project root |
| `state/spectracker_applied_version.json` | Merged into `spectracker/version_log.json` entries (`applied` field) |
| `knowledge/history/spectracker_spec_snapshot_<v>.md` | Embedded as `spec_content` in `spectracker/version_log.json` entries |
| `knowledge/current/absorber_config_map.json` | Merged into `absorber/codebase_map.md` §§ Config |
| `knowledge/current/absorber_blame_map.md` | Merged into `absorber/codebase_map.md` §§ Git/Blame |
| `cache/absorber_overwrite_git_snapshot.json` | Git data now inline in `codebase_map.md` |
| `state/clarificator_requirement_synthesis.md` | Merged as `requirement_synthesis` field in `clarificator/session.json` |
| `execution/clarificator_overwrite_questions.md` | Print to terminal only — not persisted |
| `knowledge/current/clarificator_decision_log.md` | Converted to `clarificator/decision_log.json` |
| `knowledge/current/patcher_findings_snapshot.md` | Removed — consumers read last entry of `patcher/attempt_log.json` |
| `state/planner_mini_impact_analysis.json` | Merged as `impact` field in `planner/mini_plan.json` |
| `knowledge/current/base.md` | Merged into `archivist/knowledge_log.md` |
| `knowledge/current/findings_notes.md` | Merged into `archivist/knowledge_log.md` |
| `state/plan_notes.json` | Merged into `archivist/knowledge_log.md` |
| `run/analysis_mini.json` | Legacy from deprecated `mini_mode.py` |
| `run/mini_log.json` | Legacy from deprecated `mini_mode.py` |

---

## Adding New Artifacts

First decide lifecycle: short-term overwrite or long-term append?

- **Short-term** → `<module>/<noun>.json` or `<module>/<noun>.md` (`.md` if human/LLM target)
- **Long-term** → `<module>/<noun>_log.json` (always `.json`)
- **Internal cache** → `<module>/cache/<noun>.json`

Then register in all three places:

1. `artifacts/paths.py` — add `_LazyPath`, update `ensure_dirs()`, add to `KNOWLEDGE_SOURCES` if long-term log
2. `artifacts/OWNERSHIP.md` — add row to ownership table
3. This document — add to directory overview, lifecycle table, and a Special Note if non-obvious

Hybrid lifecycle (e.g., in-place field mutation within an append log) must be documented as an exception with rationale in Special Notes.

---

*Last updated: 2026-05-19*
*Maintained alongside `pipeline/` and `harness.py`.*
