# LLM Pipeline — How to use

## Architecture

```
spec.md  →  Gemini 2.5 Flash  →  GLM 5.1 (planner)  →  Qwen 3.6+ (executor)  →  vitest  →  DeepSeek R1 (judge)
```

| Layer | Model | Role | Output |
|---|---|---|---|
| Scaffold | Gemini 2.5 Flash | Generate stub files + test files from spec | `scaffold/scaffold.json` |
| Plan | GLM 5.1 | Decompose scaffold into ordered per-file tasks | `scaffold/glm_plan.json` |
| Implement | Qwen 3.6 Plus | Implement source files per-file (guided by plan) | `src/**` |
| Test + Repair | vitest + Qwen | Run tests, targeted repair loop (max N iter) | `reports/qwen_iterations.json` |
| Judge | DeepSeek R1 | Qualitative review + sign-off (runs only on green) | `reports/judge_report.md` |

**You** edit `spec.md` → push → GitHub Action runs automatically.  
**Claude (Spec Agent)** normalises / generates `spec.md` on request.  
Claude does **not** call any API — you copy Claude's output, commit, push.

---

## Secrets required (GitHub → Settings → Secrets)

| Secret | Used by |
|---|---|
| `GEMINI_API_KEY` | Step 2 — Gemini scaffold |
| `OPENROUTER_API_KEY` | Step 3b GLM plan, Step 3a Qwen implement, Step 6 DeepSeek judge |

---

## Local dev

```bash
# Install deps
pip install httpx
npm install

# Set keys
cp .env.example .env   # fill in GEMINI_API_KEY and OPENROUTER_API_KEY

# Full pipeline (scaffold → plan → implement → test → report → judge)
python harness.py

# Skip GLM planning — Qwen single-call mode (judge still runs on green)
python harness.py --only-qwen

# Reuse existing scaffold.json (skip Gemini call)
python harness.py --skip-scaffold

# Reuse existing glm_plan.json (skip GLM call)
python harness.py --skip-scaffold --skip-plan

# Skip straight to test loop (reuse existing src/)
python harness.py --test-only

# Skip judge (faster debug loop — no DeepSeek call)
python harness.py --test-only --skip-judge

# Override iteration cap
python harness.py --max-iter 5

# Verbose cluster debug output during test loop
python harness.py --test-only --verbose
```

**Typical debug loop after a failed test:**
```bash
python harness.py --test-only --skip-judge --max-iter 3
# fix → repeat until green
python harness.py --test-only --max-iter 3   # final run WITH judge
```

---

## Updating the spec

1. Prompt Claude: *"Update spec.md to add component X with props Y"*
2. Copy Claude's `spec.md` response → overwrite `spec.md` in repo
3. `git add spec.md && git commit -m "spec: add X" && git push`
4. GitHub Action triggers automatically

---

## Pipeline file map

```
spec.md                              ← YOU edit this (via Claude)
harness.py                           ← local runner (mirrors GitHub Actions)
.github/workflows/llm-pipeline.yml

pipeline/
  02_scaffold_gemini.py              ← Gemini 2.5 Flash → scaffold JSON + stubs
  03b_implement_glm.py               ← GLM 5.1 → glm_plan.json  (PLANNER)
  03a_implement_qwen.py              ← Qwen 3.6 Plus → src/      (EXECUTOR)
  04_test_and_iterate.py             ← vitest + targeted repair loop
  05_report.py                       ← pipeline summary.md
  06_judge_deepseek.py               ← DeepSeek R1 → judge_report.md (JUDGE)

scaffold/
  scaffold.json                      ← Gemini output (stubs + test files)
  glm_plan.json                      ← GLM planner output
  instructions_qwen.txt              ← executor hints for Qwen
  instructions_glm.txt               ← planner hints for GLM
  impl_qwen.json                     ← record of files written by Qwen

src/                                 ← Qwen-implemented source files
tests/                               ← Gemini-generated test files

reports/
  qwen_iterations.json               ← per-iteration test + repair log
  summary.md                         ← pipeline summary (GitHub Actions tab)
  judge_report.md                    ← DeepSeek R1 final review + sign-off
  judge_raw.json                     ← raw judge response + metadata
```

---

## Prompting Claude as Spec Agent

Example prompts:

```
"Add a <HeatmapPanel> component showing anomaly severity by hour-of-day.
 Props: events: AnomalyEvent[], days: number.
 Update spec.md §4, §7, §8."
```

```
"The useSensorData hook needs to support a sensorId parameter for
 future multi-sensor support. Update spec.md and bump version to 1.1.0."
```

Claude will return a full updated `spec.md`. Copy-paste, commit, push.

---

## Judge verdicts

| Verdict | Meaning |
|---|---|
| `APPROVED` | No blocking issues, scores avg ≥ 3.5 |
| `APPROVED_WITH_NOTES` | No blocking issues but non-trivial notes |
| `NEEDS_REVISION` | Blocking issues found — review `reports/judge_report.md` |

`NEEDS_REVISION` causes `harness.py` and the GitHub Actions step to exit non-zero.
