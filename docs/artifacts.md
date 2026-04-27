# Artifact Taxonomy – LLM Pipeline

This document describes the purpose, lifecycle, and ownership of all artifacts produced by the LLM pipeline.  
All paths are relative to the project root.

> **RULE: 1 artifact = 1 script owner duy nhất được quyền ghi.**  
> Tất cả script khác chỉ được READ. Paths được define tập trung tại `artifacts/paths.py`.

---

## Directory Overview

```
artifacts/
├── state/
│   ├── scaffold.json
│   ├── plan.json
│   ├── plan_notes.json          ← accumulated architect notes (07_update)
│   └── spec_applied.json
├── cache/
│   ├── spec_compressed.md
│   └── spec_delta.json
├── knowledge/
│   ├── current/
│   │   ├── base.md
│   │   ├── findings.md          ← judge snapshot per run (07_fix)
│   │   ├── findings_notes.md    ← human regression notes, append-only (07_update)
│   │   └── spec_addendum.md
│   └── history/
│       ├── spec.changelog
│       ├── update_log.json      ← knowledge evolution log (07_update)
│       ├── fix_log.json         ← fix action log (07_fix)
│       ├── <version>.md
│       └── <version>.changelog.md
├── run/
│   ├── impl_record.json
│   ├── judge_raw.json
│   └── test_report.json
└── reports/
    ├── summary.md
    └── judge_report.md
```

> `src/` and `tests/` are **build outputs**, not pipeline artifacts. Written by `02_scaffold_gemini.py` and `03a_implement_qwen.py`.  
> `artifacts/state/prev_src/` is **scratch space** used by `harness.py` for delta runs — not a pipeline artifact, safe to delete.

---

## Ownership & Lifecycle Table

| Artifact | Owner (sole writer) | Write mode | Consumers (readers) |
|---|---|---|---|
| `state/scaffold.json` | `02_scaffold_gemini` | overwrite | `03a`, `03b`, `05`, `06`, `harness` |
| `cache/spec_compressed.md` | `02_scaffold_gemini` | overwrite | `03a`, `03b`, `06`, `07_fix` |
| `state/plan.json` | `03b_implement_glm` | overwrite | `03a`, `04`, `05`, `06`, `07_fix`, `07_update`, `harness` |
| `cache/spec_delta.json` | `spec_diff` | overwrite | `05`, `06`, `harness` |
| `state/spec_applied.json` | `spec_diff` | hybrid† | `spec_diff` (self-read) |
| `knowledge/history/spec.changelog` | `spec_diff` | append | — (human review) |
| `run/impl_record.json` | `03a_implement_qwen` | overwrite | `05`, `06` |
| `run/test_report.json` | `04_test_and_iterate` | overwrite | `05`, `06`, `07_update` |
| `knowledge/current/findings.md` | `07_fix_from_judge` | overwrite | `04`, `07_update` |
| `knowledge/history/fix_log.json` | `07_fix_from_judge` | append | — (human review) |
| `run/judge_raw.json` | `06_judge_deepseek` | overwrite | `07_update`, `07_fix`, `harness` |
| `knowledge/current/spec_addendum.md` | `06_judge_deepseek` | append | `07_update` |
| `reports/judge_report.md` | `06_judge_deepseek` | overwrite | — (human only) |
| `knowledge/current/base.md` | `07_update_knowledge` | append | `04`, `06`, `07_fix` |
| `knowledge/current/findings_notes.md` | `07_update_knowledge` | append | `04`, `07_fix` |
| `knowledge/history/update_log.json` | `07_update_knowledge` | append | — (human review) |
| `state/plan_notes.json` | `07_update_knowledge` | append | `04`, `07_fix` |
| `reports/summary.md` | `05_report` | overwrite | — (human only) |

† See special note on `spec_applied.json` below.

---

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
        [03a_qwen]─────────► impl_record.json        (reads: scaffold, plan)
               │
        [04_test]──────────► test_report.json         (reads: plan, plan_notes,
               │                                               findings, findings_notes, base)
        [06_judge]─────────► judge_raw.json, spec_addendum.md, judge_report.md
               │
        [07_fix]───────────► findings.md, fix_log.json
        [07_update]────────► base.md, findings_notes.md, update_log.json, plan_notes.json
               │
        [05_report]────────► summary.md
```

---

## Special Notes

### `state/plan.json` vs `state/plan_notes.json`
`plan.json` is **immutable after `03b`** — it is a snapshot of the planning step and must not be mutated by downstream scripts.

`plan_notes.json` is owned by `07_update_knowledge` and accumulates architect-level notes across runs (e.g., cross-cutting concerns discovered during judge/fix cycles). `04_test_and_iterate` merges both sources when building the Minimax system prompt.

### `knowledge/current/findings.md` vs `knowledge/current/findings_notes.md`
Two distinct semantics, two distinct owners:

- `findings.md` (owner: `07_fix_from_judge`) — a **per-run snapshot** of blocking/non-blocking findings from the automated judge. Overwritten each run.
- `findings_notes.md` (owner: `07_update_knowledge`) — **human-approved regression notes**, append-only across runs. Persists between runs as long-term memory.

`04_test_and_iterate` reads both and merges them when injecting context into Qwen/Minimax prompts.

### `state/spec_applied.json`
Stores the current applied version (overwritten each run) but also contains a `run_history` array that grows over time — a deliberate **hybrid**. The `run_history` is an append-only log embedded inside an otherwise overwritten file. Rationale: keeps history coupled with state without adding another artifact.

### `cache/spec_delta.json`
Used by `harness.py` for control-flow decisions (which steps to skip). Placed in `cache/` because it can be recomputed at any time from the current spec and last applied version, but its consumption by harness makes it more than a pure cache. Documented exception.

### `reports/summary.md` and `reports/judge_report.md`
Human-readable renders only. **No script parses these files.** On local runs, render on-demand. On GitHub Actions, `summary.md` is piped to `$GITHUB_STEP_SUMMARY` once and does not need to be committed to the repo.

### Scratch space: `state/prev_src/`
Used exclusively by `harness.py` to restore unaffected source files during delta runs. Not a pipeline artifact — can be deleted safely at any time. Not tracked in `paths.py`.

---

## Adding New Artifacts

When extending the pipeline, follow these rules:

1. **State** — records progress to enable step-skipping. Overwrite each run.
2. **Cache** — heavy intermediates regenerable from spec. Overwrite each run.
3. **Knowledge** — files appended across runs; may be read by pipeline steps.
4. **Run** — logs of a single run. Overwritten. May be consumed by later steps within the same run.
5. **Reports** — human readability only. Never consumed by pipeline logic.
6. **Build outputs** (`src/`, `tests/`) — written only by `02_scaffold_gemini.py` and `03a_implement_qwen.py`.
7. **Scratch space** — not versioned as artifacts; use `state/` subdirs for temporary data.

If a file has **hybrid behaviour** (e.g., embedded appended arrays inside an overwritten file), document the exception in the lifecycle table and add a rationale in Special Notes.

Every new artifact must be:
- Added to `artifacts/paths.py` with an owner comment
- Added to the Ownership table in `artifacts/OWNERSHIP.md`
- Added to this document

---

*Last updated: 2026-04-27*  
*Maintained alongside `pipeline/` and `harness.py`.*
