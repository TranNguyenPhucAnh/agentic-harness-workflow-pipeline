"""
pipeline/03b_implement_glm.py
Step 3b — Call GLM 5.1 (Zhipu AI) to implement scaffold stubs.
Reads:   spec.md, scaffold/scaffold.json
Writes:  src/**  (non-test files only)
         scaffold/impl_glm.json

NOTE: Runs in parallel with 03a via separate GitHub Actions steps.
      Each model writes to its OWN directory to avoid race conditions:
        Qwen → src/  (canonical for now; swapped by 04_test_and_iterate.py)
        GLM  → src_glm/  (isolated impl)
"""

import os
import json
import sys
import httpx
from pathlib import Path

GLM_API_KEY = os.environ["GLM_API_KEY"]
GLM_URL     = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL   = "glm-4-plus"   # GLM 5.1 / GLM-4-Plus

ROOT           = Path(__file__).parent.parent
SPEC_PATH      = ROOT / "spec.md"
SCAFFOLD_JSON  = ROOT / "scaffold" / "scaffold.json"
INSTRUCTIONS   = ROOT / "scaffold" / "instructions_glm.txt"
IMPL_RECORD    = ROOT / "scaffold" / "impl_glm.json"
OUT_DIR        = ROOT / "src_glm"   # isolated; test runner swaps this in


def build_system_prompt(instructions: str) -> str:
    return f"""You are a senior TypeScript/React developer.
You will receive:
1. A technical spec (spec.md)
2. A scaffold JSON with stub files (interfaces + signatures, no bodies)
3. Model-specific implementation instructions

Your task:
- Implement ONLY the function bodies in the non-test source files.
- Return a JSON object:
  {{
    "files": [
      {{
        "file_path": "src/hooks/useSensorData.ts",
        "code": "<full file content>"
      }}
    ]
  }}
- Do NOT modify test files.
- Do NOT add new files.
- TypeScript strict mode — no `any`.
- Tailwind only — no inline styles.
- Output raw JSON only. No markdown fences.

Model-specific instructions:
{instructions}"""


def call_glm(system: str, user_message: str) -> dict:
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.15,
        "max_tokens": 32768,
    }

    print("[03b] Calling GLM 5.1 …")
    with httpx.Client(timeout=180) as client:
        r = client.post(GLM_URL, headers=headers, json=payload)
        r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])

    return json.loads(text)


def main():
    spec     = SPEC_PATH.read_text()
    scaffold = json.loads(SCAFFOLD_JSON.read_text())
    instrs   = INSTRUCTIONS.read_text() if INSTRUCTIONS.exists() else ""

    stub_files = [f for f in scaffold["files"] if not f.get("is_test")]

    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold (stub files to implement)\n\n"
        f"{json.dumps(stub_files, indent=2)}"
    )

    result = call_glm(build_system_prompt(instrs), user_msg)

    written = []
    for entry in result["files"]:
        # Remap src/ → src_glm/ to isolate GLM's impl
        rel = entry["file_path"]
        if rel.startswith("src/"):
            rel = "src_glm/" + rel[len("src/"):]
        path = ROOT / rel

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["code"])
        written.append(rel)
        print(f"[03b] WROTE {rel}")

    IMPL_RECORD.write_text(json.dumps({"model": "glm", "files": written}, indent=2))
    print(f"[03b] Done — {len(written)} files written.")


if __name__ == "__main__":
    main()
