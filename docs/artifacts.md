# Artifact Taxonomy – LLM Pipeline

This document describes the purpose, lifecycle, and relationships of all artifacts produced by the LLM pipeline.  
All paths are relative to the project root.

## Directory Overview

artifacts/
├── state/ # Checkpoint: what has been generated/computed (overwritten each run)
├── cache/ # Derived intermediates: recomputable from spec (overwritten)
├── knowledge/ # Accumulated wisdom: appended over time (never overwritten)
│ ├── current/ # Latest version of evolving knowledge files
│ └── history/ # Immutable snapshots and changelogs
├── run/ # Per‑run records: logs of what happened this run (overwritten)
└── reports/ # Human‑readable summaries (no logic consumed, overwritten)

src/ # Final code (build output)
tests/ # Test files (stubs then final)

## 1. `artifacts/state/` – Checkpoint (Overwritten)

Artifacts that capture the current “state” of the pipeline. If a step is skipped, the corresponding file must exist and is reused.

| File | Purpose | Produced by |
|------|---------|-------------|
| `scaffold.json` | Gemini stub files (interfaces + signatures) | `02_scaffold_gemini.py` |
| `plan.json` | GLM decomposition + implementation order | `03b_implement_glm.py` |
| `spec_applied.json` | Last successfully applied spec version + run history | `spec_diff.py` (called from harness) |

## 2. `artifacts/cache/` – Derived Intermediates (Overwritten)

Artifacts that can be recomputed on‑the‑fly from the spec (or other inputs) whenever needed. They exist only to speed up execution.

| File | Purpose | Produced by |
|------|---------|-------------|
| `spec_compressed.md` | spec.md with §0 and §8 removed (token saving) | `02_scaffold_gemini.py` |
| `spec_delta.json` | Changes between current spec and last applied version | `spec_diff.py` |

## 3. `artifacts/knowledge/` – Accumulated Wisdom (Append‑only)

### 3.1 `current/` – Latest evolving knowledge files  

These files are appended to, never overwritten. They represent the current state of accumulated knowledge.

| File | Purpose | Produced by |
|------|---------|-------------|
| `findings.md` | Judge findings (blocking/non‑blocking) from previous runs | `07_fix_from_judge.py`, `07_update_knowledge.py` |
| `base.md` | Human‑fix patterns distilled from escalations | `07_update_knowledge.py` (both modes) |
| `spec_addendum.md` | Edge cases and clarifications not yet in spec.md | `07_update_knowledge.py` (judge‑driven) |

### 3.2 `history/` – Immutable historical log  

| File | Purpose | Produced by |
|------|---------|-------------|
| `<version>.md` | Raw snapshot of spec.md at that version | `spec_diff.py` |
| `<version>.changelog.md` | Human‑readable changelog entry for that version | `spec_diff.py` |
| `spec.changelog` | Aggregated changelog (git‑style) | `spec_diff.py` |

## 4. `artifacts/run/` – Per‑Run Records (Overwritten)

Records that document a single execution of the pipeline. They are overwritten each run.

| File | Purpose | Produced by |
|------|---------|-------------|
| `impl_record.json` | Which files Qwen implemented, mode, skipped delta, failed files | `03a_implement_qwen.py` |
| `test_report.json` | Vitest iteration history + escalated clusters (merged) | `04_test_and_iterate.py` |
| `update_log.json` | Unified log of fixes (judge‑driven & human‑capture) | `07_fix_from_judge.py`, `07_update_knowledge.py` |

## 5. `reports/` – Human‑Readable (Overwritten)

Display‑only files; not consumed by any pipeline step.

| File | Purpose | Produced by |
|------|---------|-------------|
| `summary.md` | Pipeline summary (used for GitHub step summary) | `05_report.py` |
| `judge_raw.json` | Full DeepSeek judge response + reasoning chain | `06_judge_deepseek.py` |
| `judge_report.md` | Rendered judge report (markdown) | `06_judge_deepseek.py` |

## 6. Build Output – `src/` and `tests/`

These are the actual code files that run in the application and tests.

| Directory | Content | Produced by |
|-----------|---------|-------------|
| `src/` | Final TypeScript source (stubs then implemented) | `02_scaffold_gemini.py` (stubs) → `03a_implement_qwen.py` (implemented) |
| `tests/` | Vitest test files (stubs only) | `02_scaffold_gemini.py` |

> **Note:** `scaffold/prev_src/` (not under `artifacts/`) is scratch space used by harness to restore unaffected files during delta runs. It is not considered a permanent artifact.

## Lifecycle Summary

| Artifact | Overwritten? | Appended? | Consumed by which steps? |
|----------|--------------|-----------|--------------------------|
| `state/scaffold.json` | Yes | No | `03a`, `03b`, `06` |
| `state/plan.json` | Yes | No | `03a`, `06` |
| `state/spec_applied.json` | Yes (but history kept inside) | No (history field grows) | `spec_diff.py` (next run) |
| `cache/spec_compressed.md` | Yes | No | `03a`, `03b`, `04`, `06`, `07` |
| `cache/spec_delta.json` | Yes | No | `harness.py`, `06` |
| `knowledge/current/findings.md` | No | Yes | `04`, `07` |
| `knowledge/current/base.md` | No | Yes | `04`, `07` |
| `knowledge/current/spec_addendum.md` | No | Yes | `06` |
| `knowledge/history/*` | No | Yes (new files) | (manual / future diff) |
| `run/impl_record.json` | Yes | No | `harness.py` (retry-impl) |
| `run/test_report.json` | Yes | No | `05`, `06`, `07_update_knowledge` |
| `run/update_log.json` | No | Yes | `07_fix`, `07_update` (for inspection) |
| `reports/*` | Yes | No | (human only) |

## Notes on Cross‑Run Reuse

- **Delta mode** uses `cache/spec_delta.json` to decide which steps to skip.  
- Unaffected source files are restored from `scaffold/prev_src/` (not an artifact).  
- `knowledge/current/` files are injected into prompts of subsequent runs to avoid repeating mistakes.  
- The `run/` directory is primarily for debugging and auditing; it does not affect pipeline decisions (except `impl_record.json` for `--retry-impl`).  

## Adding New Artifacts

When extending the pipeline, follow these guidelines:

1. **State** – for files that record “what has been done” to enable skipping.  
2. **Cache** – for heavy‑to‑compute intermediates that can be regenerated from spec.  
3. **Knowledge** – for append‑only files that accumulate wisdom across runs.  
4. **Run** – for logs that describe a single execution (overwritten on next run).  
5. **Reports** – for human eyes only.  

Never write to `src/` or `tests/` except through the designated pipeline steps (scaffold → implement → iterate).  
Never write to `knowledge/history/` except through `spec_diff.py` (which handles snapshots and changelogs).  

---

*Last updated: 2026-04-26*  
*Maintain with code changes in `pipeline/` and `harness.py`.*
