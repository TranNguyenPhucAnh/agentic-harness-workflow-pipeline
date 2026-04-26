# Artifact Taxonomy – LLM Pipeline

This document describes the purpose, lifecycle, and relationships of all artifacts produced by the LLM pipeline.  
All paths are relative to the project root.

## Directory Overview

artifacts/
├── state/
│   ├── scaffold.json
│   ├── plan.json
│   └── spec_applied.json
├── cache/
│   ├── spec_compressed.md
│   └── spec_delta.json
├── knowledge/
│   ├── current/
│   │   ├── base.md
│   │   ├── findings.md
│   │   └── spec_addendum.md
│   └── history/
│       ├── spec.changelog
│       ├── update_log.json
│       ├── <version>.md
│       └── <version>.changelog.md
├── run/
│   ├── judge_raw.json
│   ├── impl_record.json
│   └── test_report.json
└── reports/
    ├── summary.md
    └── judge_report.md

> The `src/` and `tests/` directories are **build outputs**, not part of `artifacts/`. They are written by `02_scaffold_gemini.py` and `03a_implement_qwen.py`.

## Lifecycle Table

| Artifact | Overwritten? | Appended? | Consumed by |
|----------|--------------|-----------|-------------|
| `state/scaffold.json` | Yes | No | `03a`, `03b`, `06` |
| `state/plan.json` | Yes | No | `03a`, `06` |
| `state/spec_applied.json` | *Hybrid* | Yes (embedded history) | `spec_diff.py` |
| `cache/spec_compressed.md` | Yes | No | `03a`, `03b`, `04`, `06`, `07` |
| `cache/spec_delta.json` | Yes | No | `harness`, `06` |
| `knowledge/current/base.md` | No | Yes | `04`, `07` |
| `knowledge/current/spec_addendum.md` | No | Yes | `06` |
| `knowledge/history/update_log.json` | No | Yes | (human review only) |
| `knowledge/history/spec.changelog` | No | Yes | (human review only) |
| `run/impl_record.json` | Yes | No | `harness` (retry) |
| `run/test_report.json` | Yes | No | `05`, `06`, `07_update_knowledge` |
| `run/judge_raw.json` | Yes | No | `07_fix_from_judge` |
| `reports/*.md` | Yes | No | (human only) |

## Special Notes

### `state/spec_applied.json`
This file stores the **current** applied version and metadata (overwritten each run), but also keeps a `run_history` array that grows over time. It is a deliberate **hybrid** – the `run_history` is effectively an append‑only log embedded inside an otherwise overwritten file.  
*Rationale:* Keeps the history coupled with the state for simplicity; does not break the pipeline logic.

### `cache/spec_delta.json`
Used by `harness.py` to decide which steps to skip. This is a **control‑flow** decision derived from the spec. It is placed in `cache/` because it can be recomputed at any time from the current spec and the last applied version, but its consumption by harness makes it more than a pure cache. This is an **exception** documented here.

### `knowledge/history/update_log.json`
Despite being under `knowledge/history/`, it is **appended by scripts** (`07_fix_from_judge.py`, `07_update_knowledge.py`) across runs. It belongs to `knowledge/` because it accumulates long‑term data; the `history/` subdirectory emphasises immutability.

### Scratch space: `scaffold/prev_src/`
Used exclusively by `harness.py` to restore unaffected source files during delta runs. It is **not a pipeline artifact** – it can be deleted safely at any time.

## Adding New Artifacts

When extending the pipeline, follow these rules:

1. **State** – for files that record progress to enable skipping steps. Overwrite each run.  
2. **Cache** – for heavy intermediates that can be regenerated from spec. Overwrite each run.  
3. **Knowledge** – for files that are appended across runs. They may be read by pipeline steps.  
4. **Run** – for logs of a single run. Overwritten. May be consumed by later steps within the same run.  
5. **Reports** – for human readability only. Never consumed by logic.  
6. **Build outputs** (`src/`, `tests/`) – must only be written by designated scripts (`02_scaffold_gemini.py`, `03a_implement_qwen.py`).  
7. **Scratch space** – not versioned as artifacts; use `scaffold/` for temporary data.

If a file has **hybrid behaviour** (e.g., embedded appended arrays inside an overwritten file), document the exception in the lifecycle table and add a rationale.

---

*Last updated: 2026-04-26*  
*Maintained with code in `pipeline/` and `harness.py`.*
