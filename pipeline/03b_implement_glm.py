"""
pipeline/03b_implement_glm.py
Step 3b — GLM 5.1 as PLANNER (reasoning-heavy, no code output).

Role change rationale:
    GLM 5.1 burns its 32k token budget on chain-of-thought reasoning before
    writing code, leaving too little room for actual output.  Instead of
    fighting the token cap, we lean into it: GLM reasons deeply to produce a
    *plan*, not code.  Qwen (03a) is the executor that turns the plan into src/.

What this script does:
    1. Read spec.md + scaffold/scaffold.json (stub files from Gemini).
    2. Call GLM 5.1 with reasoning ON — task: decompose each stub file into
       an ordered list of implementation tasks / sub-tasks.
    3. Write scaffold/glm_plan.json  ← consumed by 03a_implement_qwen.py
       when --use-glm-plan flag is passed.

Writes:
    scaffold/glm_plan.json

Does NOT write any src/ files.  03a_implement_qwen.py is the sole executor.
"""

import os
import json
import re
import sys
import httpx
from pathlib import Path

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "z-ai/glm-5.1"

ROOT          = Path(__file__).parent.parent
SPEC_PATH     = ROOT / "spec.md"
SCAFFOLD_JSON = ROOT / "scaffold" / "scaffold.json"
PLAN_OUT      = ROOT / "scaffold" / "glm_plan.json"
PIPELINE_CTX = ROOT / "scaffold" / "pipeline_context.json"


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior TypeScript/React architect acting as a PLANNER.
You will receive a spec and a scaffold JSON (stub files with signatures only).

Your job is NOT to write code.
Your job is to reason carefully and produce an implementation plan.

For each non-test stub file, output a task object describing:
- What the file does and its role in the system
- Ordered list of implementation sub-tasks (what to implement, in what order)
- Key types / interfaces this file depends on (with their source file)
- Gotchas or edge cases the implementer must handle
- Tailwind class hints for visual components (colours, layout, states)

Return a single JSON object — NO markdown fences, raw JSON only:
{
  "plan_version": "1.0.0",
  "tasks": [
    {
      "file_path": "src/hooks/useSensorData.ts",
      "role": "one-sentence role description",
      "depends_on": ["src/types/sensor.ts", "src/data/demoConstants.ts"],
      "sub_tasks": [
        "1. Generate base SensorPoint array using POINTS_PER_DAY constant ...",
        "2. Inject anomaly clusters at morning (07-09h) and evening (18-21h) ...",
        "3. ..."
      ],
      "gotchas": [
        "decisionScore must be negative for anomaly points (-0.05 to -0.45)",
        "..."
      ],
      "tailwind_hints": null
    }
  ],
  "implementation_order": [
    "src/types/sensor.ts",
    "src/data/demoConstants.ts",
    "src/hooks/useSensorData.ts",
    "src/hooks/useReplay.ts",
    "src/components/SummaryStickyBar.tsx",
    "src/components/ReplayControls.tsx",
    "src/components/AnomalyFeed.tsx",
    "src/components/ModelGates.tsx",
    "src/App.tsx",
    "src/main.tsx"
  ],
  "global_notes": "any cross-cutting concerns the implementer should know"
}

Rules:
- Reason as deeply as needed — this is your reasoning budget well spent.
- Be specific: reference exact constant names, prop names, type names from the spec.
- implementation_order must respect dependency order (types before hooks before components).
- tailwind_hints: include for visual components, null for hooks/types/data files.
- Output raw JSON only. Absolutely no markdown fences or preamble text.
"""


# ── API call ──────────────────────────────────────────────────────────────────

def _load_spec() -> str:
    """Dùng compressed spec nếu có, fallback về full spec."""
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()

def call_glm_planner(spec: str, stub_files: list) -> dict:
    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold stub files\n\n"
        f"{json.dumps(stub_files, indent=2)}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 32768,   # GLM spends this on reasoning → compact JSON output
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    print("[03b] Calling GLM 5.1 (planner role) …")
    with httpx.Client(timeout=240) as client:
        r = client.post(OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()

    #raw = r.json()["choices"][0]["message"]["content"].strip()
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    finish_reason = choice.get("finish_reason")
    
    if tool_calls:
        raise RuntimeError(f"Model returned tool_calls instead of text: {tool_calls}")
    
    if not content:
        raise RuntimeError(
            f"Model returned empty content. finish_reason={finish_reason}, message={msg}"
        )
    
    raw = content.strip()
    return _parse_json(raw, label="GLM planner response")

# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_json(raw: str, label: str) -> dict:
    """Extract JSON from model output robustly (handles accidental fences)."""
    # Strip markdown fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[03b] JSON parse failed for {label}: {e}", file=sys.stderr)
            print(f"[03b] Raw output (first 500 chars):\n{raw[:500]}", file=sys.stderr)
            sys.exit(1)

    print(f"[03b] No JSON object found in {label}.", file=sys.stderr)
    sys.exit(1)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_plan(plan: dict, stub_files: list) -> None:
    """Warn if any stub file is missing from the plan."""
    planned = {t["file_path"] for t in plan.get("tasks", [])}
    for f in stub_files:
        fp = f["file_path"]
        if fp not in planned:
            print(f"[03b] WARNING: stub file not covered by plan: {fp}")

    required_keys = {"plan_version", "tasks", "implementation_order"}
    missing = required_keys - set(plan.keys())
    if missing:
        print(f"[03b] WARNING: plan missing keys: {missing}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    spec     = _load_spec()                          # <-- đổi từ SPEC_PATH.read_text()
    scaffold = json.loads(SCAFFOLD_JSON.read_text())
 
    stub_files = [f for f in scaffold["files"] if not f.get("is_test")]
    print(f"[03b] Planning {len(stub_files)} stub files …")
 
    plan = call_glm_planner(spec, stub_files)
    validate_plan(plan, stub_files)
 
    PLAN_OUT.write_text(json.dumps(plan, indent=2))
    print(f"[03b] Plan written → {PLAN_OUT}")
    print(f"[03b] Tasks in plan: {len(plan.get('tasks', []))}")
    print(f"[03b] Implementation order: {plan.get('implementation_order', [])}")
 
    # ── NEW: append implementation_order to pipeline_context ──
    if PIPELINE_CTX.exists():
        ctx = json.loads(PIPELINE_CTX.read_text())
        ctx["implementation_order"] = plan.get("implementation_order", [])
        PIPELINE_CTX.write_text(json.dumps(ctx, indent=2))
        print("[03b] Updated pipeline_context.json with implementation_order")
 
    print("[03b] Done. Pass --use-glm-plan to 03a_implement_qwen.py to use this plan.")

if __name__ == "__main__":
    main()
