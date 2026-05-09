# LLM Pipeline — How to use

## Architecture

```
specwright_spec_<slug>.md
  └─ spectracker        →  version delta, step skip decisions
  └─ absorber           →  codebase map, blame map, git snapshot
  └─ clarificator       →  requirement Q&A → clarificator_requirement_synthesis.md
       └─ enricher      →  enriched prompt → specwright
            └─ specwright → specwright_spec_<slug>.md (update)
  └─ scaffolder         →  scaffolder_codebase_skeleton.json + src/ stubs + tests/
       └─ planner       →  planner_*_execution_plan.json
            └─ executor →  src/** (per-file)
                 └─ debugger   →  vitest + repair loop
                      └─ reporter    →  reporter_execution_summary.md
                           └─ judge  →  judge_verdict_summary.md (green only)
                                └─ patcher   →  auto-fix NEEDS_REVISION findings
                                     └─ archivist →  long-term knowledge update
```

### Model assignments

| Step | Script | Model | Role |
|---|---|---|---|
| 01 | `01_spectracker.py` | — | Spec version diff, step skip decisions |
| 02 | `02_absorber.py` | — | Scan codebase, build knowledge maps |
| 03 | `03_clarificator.py` | DeepSeek | Requirement Q&A, clarification synthesis |
| 04 | `04_enricher.py` | DeepSeek | Enrich context into structured prompt |
| 05 | `05_specwright.py` | Gemini 2.5 Flash | Generate/update spec |
| 06 | `06_scaffolder.py` | Gemini 2.5 Flash | Stubs + test files from spec |
| 07 | `07_planner.py` | GLM 5.1 | Decompose files into ordered sub-tasks |
| 08 | `08_executor.py` | Qwen 3.6 Plus | Implement src/ files guided by plan |
| 09 | `09_debugger.py` | vitest + Qwen | Test + cluster repair loop |
| 10 | `10_reporter.py` | — | Aggregate pipeline summary |
| 11 | `11_judge.py` | DeepSeek V3.2 | Qualitative review + verdict (green only) |
| 12 | `12_patcher.py` | Minimax / Qwen | Auto-fix NEEDS_REVISION findings |
| 13 | `13_archivist.py` | — | Distill judge findings into long-term memory |

---

## Secrets (GitHub → Settings → Secrets)

| Secret | Used by |
|---|---|
| `GEMINI_API_KEY` | scaffolder (06), specwright (05) |
| `OPENROUTER_API_KEY` | clarificator (03), enricher (04), planner (07), executor (08), debugger (09), judge (11), patcher (12) |

---

## Local dev setup

```bash
pip install httpx
npm install

# Keys — create .env in repo root:
GEMINI_API_KEY=<your key>
OPENROUTER_API_KEY=<your key>
```

---

## Typical workflows

### First run — full pipeline
```bash
python harness.py --project <name>
```
`spectracker → absorber → clarificator → enricher → specwright → scaffolder → planner → executor → debugger → reporter → judge`

---

### Spec changed — rerun full pipeline
```bash
# Edit specwright_spec_<slug>.md, then:
python harness.py --project <name>
```
spectracker computes delta automatically — unchanged steps are skipped.

---

### Resume from a specific step
```bash
python harness.py --project <name> --from-executor
python harness.py --project <name> --from-debugger --until-reporter
python harness.py --project <name> --scaffolder   # run only scaffolder
```

---

### Debug loop — tests failing, iterate without re-generating
```bash
# Fast loop: skip all generation, no judge
python harness.py --project <name> --from-debugger --until-reporter --verbose

# Once green, run with judge for final sign-off
python harness.py --project <name> --from-judge
```

---

### Re-run judge + auto-fix from existing verdict
```bash
# Consume existing judge_session_verdict_raw.json, run patcher + re-judge
python harness.py --project <name> --resume-judge
```

---

### Skip planner (executor single-call mode)
```bash
python harness.py --project <name> --from-executor --only-qwen
```
Use when: GLM quota exhausted, or faster cheaper first-pass needed.

---

### Mini scope — targeted change without full spec cycle
```bash
python harness.py --project <name> --scope mini --from-clarificator --until-executor
```
Mini scope skips spectracker delta check and scaffolder. Planner writes
`planner_mini_execution_plan.json` instead of full plan.

---

### Override iteration caps
```bash
python harness.py --project <name> --from-debugger --max-iter 5 --max-cluster-attempts 3
```

---

### Dry run — see what would execute
```bash
python harness.py --project <name> --dry-run
python harness.py --project <name> --from-executor --dry-run
```

---

### Decision tree

```
New project or new requirement?
  └─ Full pipeline:           python harness.py --project <name>

Spec changed, rerun everything?
  └─ Full pipeline:           python harness.py --project <name>
     (spectracker auto-skips unchanged steps)

Only implementation changed?
  └─ From executor:           python harness.py --project <name> --from-executor

Tests failing, no re-impl?
  └─ Debug loop:              python harness.py --project <name> --from-debugger --until-reporter --verbose
  └─ When green + judge:      python harness.py --project <name> --from-judge

Judge returned NEEDS_REVISION?
  └─ Auto-fix:                python harness.py --project <name> --resume-judge
  └─ Manual fix + capture:    python pipeline/13_archivist.py --capture-human-fix

GLM unavailable?
  └─ Skip planner:            python harness.py --project <name> --from-executor --only-qwen

Small targeted change?
  └─ Mini scope:              python harness.py --project <name> --scope mini --from-clarificator --until-executor
```

---

## Pipeline file map

```
specwright_spec_<slug>.md            ← canonical spec (specwright output)
harness.py                           ← orchestrator — single entrypoint
.github/workflows/llm-pipeline.yml

pipeline/
  01_spectracker.py                  ← spec version diff → step skip decisions
  02_absorber.py                     ← codebase scan → knowledge maps
  03_clarificator.py                 ← DeepSeek     → requirement Q&A
  04_enricher.py                     ← DeepSeek     → enriched prompt
  05_specwright.py                   ← Gemini        → spec generation/update
  06_scaffolder.py                   ← Gemini        → stubs + test files
  07_planner.py                      ← GLM 5.1       → execution plan
  08_executor.py                     ← Qwen 3.6+     → src/** implementation
  09_debugger.py                     ← vitest + Qwen → test + repair loop
  10_reporter.py                     ← pipeline summary
  11_judge.py                        ← DeepSeek V3.2 → verdict (green only)
  12_patcher.py                      ← Minimax/Qwen  → auto-fix from verdict
  13_archivist.py                    ← knowledge distillation

artifacts/
  paths.py                           ← SOURCE OF TRUTH for all artifact paths
  NAMING_RULES.md                    ← artifact naming convention
  OWNERSHIP.md                       ← ownership + data flow table
  TAXONOMY.md                        ← full artifact descriptions + lifecycle

artifacts_<slug>/
  specwright_spec_<slug>.md

  state/
    clarificator_requirement_synthesis.md
    scaffolder_codebase_skeleton.json
    planner_full_execution_plan.json
    planner_mini_execution_plan.json
    planner_mini_impact_analysis.json
    spectracker_applied_version.json

  cache/
    scaffolder_compressed_spec.md
    spectracker_session_version_delta.json
    absorber_session_codebase_snapshot.json
    absorber_session_git_snapshot.json

  execution/
    clarificator_session_raw.json
    clarificator_session_questions.md
    enricher_session_enriched_prompt.md
    executor_session_manifest.json
    debugger_session_test_summary.json
    judge_session_verdict_raw.json
    patcher_session_fix_summary.md

  knowledge/
    current/
      clarificator_decision_log.md
      absorber_codebase_map.md
      absorber_config_map.json
      absorber_blame_map.md
      patcher_findings_snapshot.md
      archivist_spec_gaps.md
      archivist_knowledge_log.md
    history/
      archivist_curation_log.json
      patcher_attempt_log.json
      spectracker_version_log.md
      <version>.md
      <version>.changelog.md

  reports/
    reporter_execution_summary.md
    judge_verdict_summary.md

  src/                               ← executor output (implementation)
  tests/                             ← scaffolder output (read-only after generation)
```

---

## Judge verdicts

| Verdict | Meaning | harness exit code |
|---|---|---|
| `APPROVED` | No blocking issues, avg score ≥ 3.5 | 0 |
| `APPROVED_WITH_NOTES` | No blocking issues, notable non-blocking notes | 0 |
| `NEEDS_REVISION` | Blocking issues found — see `reports/judge_verdict_summary.md` | 1 |

`NEEDS_REVISION` causes harness and GitHub Actions to exit non-zero.
Judge runs **only when vitest is fully green** — never reviews broken code.
After judge, patcher auto-applies fixes then re-runs judge (up to `--max-judge-rounds`).

---

## Knowledge update after judge

After human reviews `reports/judge_verdict_summary.md`:

```bash
# Interactive — approve/skip each finding
python pipeline/13_archivist.py

# Accept all suggested actions
python pipeline/13_archivist.py --accept-all

# After manual fix to code AI couldn't patch
python pipeline/13_archivist.py --capture-human-fix

# View accumulated knowledge base
python pipeline/13_archivist.py --show-knowledge
```

---

## Multi-project

Each project gets its own isolated artifact workspace:

```bash
python harness.py --project dashboard
python harness.py --project api-service
python harness.py --project mobile-app
```

Artifacts live under `artifacts_dashboard/`, `artifacts_api-service/`, etc.
`PIPELINE_PROJECT` env var can be used instead of `--project` in CI/CD.
