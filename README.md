# LLM Pipeline — How to use

## Architecture

```
spec.md
  └─ Gemini 2.5 Flash  →  scaffold/scaffold.json  (stubs + test files)
       └─ GLM 5.1       →  scaffold/glm_plan.json  (implementation plan)
            └─ Qwen 3.6+  →  src/**               (executor, per-file)
                 └─ vitest + Qwen  →  reports/     (test + targeted repair loop)
                      └─ DeepSeek V3.2  →  reports/judge_report.md  (judge, green only)
```

| Step | Model | Role | Output |
|---|---|---|---|
| 02 | Gemini 2.5 Flash | Scaffold: stubs + test files from spec | `scaffold/scaffold.json`, `src/` stubs, `tests/` |
| 03b | GLM 5.1 | Planner: decompose each file into ordered sub-tasks | `scaffold/glm_plan.json` |
| 03a | Qwen 3.6 Plus | Executor: implement each file guided by plan | `src/**` |
| 04+05 | vitest + Qwen | Test + cluster repair loop (max N iterations) | `reports/qwen_iterations.json`, `reports/summary.md` |
| 06 | DeepSeek V3.2 | Judge: qualitative review + sign-off (green only) | `reports/judge_report.md` |

**You** edit `spec.md` → push → GitHub Action runs automatically.
**Claude (Spec Agent)** normalises / generates `spec.md` on request — does NOT call any API.

---

## Secrets (GitHub → Settings → Secrets)

| Secret | Used by |
|---|---|
| `GEMINI_API_KEY` | Step 02 — Gemini scaffold |
| `OPENROUTER_API_KEY` | Step 03b GLM plan, Step 03a Qwen implement, Step 04 Qwen repair, Step 06 DeepSeek judge |

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
python harness.py
```
`Gemini scaffold → GLM plan → Qwen impl → vitest → report → DeepSeek judge`

---

### Spec changed — re-run from scratch
```bash
# Edit spec.md (or copy Claude's response over it), then:
python harness.py
```

### Spec changed — scaffold is still valid, only impl needs to rerun
```bash
python harness.py --skip-scaffold
# Reuse scaffold.json → GLM re-plan → Qwen re-impl → vitest → judge
```

### Spec changed — scaffold AND plan still valid
```bash
python harness.py --skip-scaffold --skip-plan
# Reuse scaffold + glm_plan → Qwen re-impl → vitest → judge
```

---

### Debug loop — tests failing, iterate without re-generating
```bash
# Fast loop: skip all generation, skip expensive judge call
python harness.py --test-only --skip-judge --verbose

# Once green, run with judge for final sign-off
python harness.py --test-only
```

`--test-only` automatically reuses `glm_plan.json` if it exists — no extra API call.

### Override iteration / attempt caps
```bash
python harness.py --test-only --skip-judge --max-iter 5 --max-cluster-attempts 3
```

---

### Skip GLM planning (Qwen single-call mode)
```bash
python harness.py --only-qwen --skip-judge
# Gemini scaffold → Qwen single-call impl → vitest → report (no judge)
```
Use when: GLM quota exhausted, or you want a faster cheaper first-pass.

---

### Decision tree

```
Changed spec.md?
  └─ Full rerun:         python harness.py

Scaffold still valid?
  └─ Skip scaffold:      python harness.py --skip-scaffold

Plan still valid too?
  └─ Skip both:          python harness.py --skip-scaffold --skip-plan

Tests failing, no re-impl?
  └─ Debug loop:         python harness.py --test-only --skip-judge --verbose
  └─ When green:         python harness.py --test-only

GLM unavailable?
  └─ Qwen only:          python harness.py --only-qwen --skip-judge
```

---

## Pipeline file map

```
spec.md                              ← YOU edit this (via Claude Spec Agent)
harness.py                           ← local runner — mirrors GitHub Actions exactly
.github/workflows/llm-pipeline.yml

pipeline/
  02_scaffold_gemini.py              ← Gemini 2.5 Flash  → scaffold JSON + stubs + tests
  03b_implement_glm.py               ← GLM 5.1           → glm_plan.json  [PLANNER]
  03a_implement_qwen.py              ← Qwen 3.6 Plus     → src/**          [EXECUTOR]
  04_test_and_iterate.py             ← vitest + L0/L1/L2 cluster repair loop
  05_report.py                       ← summary.md (pipeline report)
  06_judge_deepseek.py               ← DeepSeek V3.2     → judge_report.md [JUDGE]

scaffold/
  scaffold.json                      ← Gemini output (stubs + test files)
  glm_plan.json                      ← GLM planner output (consumed by 03a)
  instructions_qwen.txt              ← executor hints for Qwen (from scaffold)
  impl_qwen.json                     ← record of files written by Qwen

src/                                 ← Qwen-implemented source files
tests/                               ← Gemini-generated test files (read-only)

reports/
  qwen_iterations.json               ← per-iteration test + cluster repair log
  escalated_clusters.json            ← clusters that hit give-up threshold (if any)
  summary.md                         ← pipeline summary → GitHub Actions tab
  judge_report.md                    ← DeepSeek V3.2 final review + verdict
  judge_raw.json                     ← raw judge response + reasoning chain
```

---

## Prompting Claude as Spec Agent

Claude generates or updates `spec.md`. Copy the response, overwrite the file, commit, push.

```
"Add a <HeatmapPanel> component. Props: events: AnomalyEvent[], days: number.
 Update spec.md §4, §7, §8 and bump version to 1.2.0."

"The useSensorData hook needs a sensorId parameter for multi-sensor support.
 Update spec.md and bump version."

"Acceptance criterion AC-4 is wrong — useSensorData(7) should return 2016 points
 not 2880. Fix §10 and any affected sections."
```

---

## Judge verdicts

| Verdict | Meaning | harness exit code |
|---|---|---|
| `APPROVED` | No blocking issues, avg score ≥ 3.5 | 0 |
| `APPROVED_WITH_NOTES` | No blocking issues, notable non-blocking notes | 0 |
| `NEEDS_REVISION` | Blocking issues found — see `reports/judge_report.md` | 1 |

`NEEDS_REVISION` causes both `harness.py` and the GitHub Actions step to exit non-zero.
The judge runs **only when vitest is fully green** — it never reviews broken code.
