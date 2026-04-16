# LLM Pipeline — How to use

## Role split

| Who | Does what |
|---|---|
| **You** | Edit `spec.md` → push → GitHub Action runs automatically |
| **Claude (Spec Agent)** | Normalise / generate `spec.md` on request |
| **GitHub Action** | Gemini scaffold → Qwen + GLM implement → vitest → iterate → report |

Claude does **not** call any API. You copy Claude's `spec.md` response, commit it, push.

---

## Secrets required (GitHub → Settings → Secrets)

| Secret | Value |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio key |
| `QWEN_API_KEY` | DashScope key (Alibaba Cloud) |
| `GLM_API_KEY` | Zhipu AI open platform key |

---

## Local dev

```bash
# Install deps
pip install httpx
npm install

# Set keys
cp .env.example .env   # fill in your keys

# Full pipeline
python harness.py

# Reuse existing scaffold (skip Gemini call)
python harness.py --skip-scaffold

# Only test one model
python harness.py --only qwen
python harness.py --only glm

# More iterations
python harness.py --max-iter 5
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
spec.md                          ← YOU edit this (via Claude)
harness.py                       ← local runner
.github/workflows/llm-pipeline.yml

pipeline/
  02_scaffold_gemini.py          ← Gemini 2.5 Flash → scaffold JSON
  03a_implement_qwen.py          ← Qwen 3.6 Plus → src/ impl
  03b_implement_glm.py           ← GLM 5.1 → src_glm/ impl
  04_test_and_iterate.py         ← vitest + fix loop (max 3 iter)
  05_report.py                   ← summary.md

scaffold/
  scaffold.json                  ← Gemini output
  instructions_qwen.txt
  instructions_glm.txt
  impl_qwen.json
  impl_glm.json

reports/
  qwen_iterations.json
  glm_iterations.json
  summary.md                     ← shown in GitHub Actions summary tab
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
