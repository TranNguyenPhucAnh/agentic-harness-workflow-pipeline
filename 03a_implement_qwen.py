"""
pipeline/03a_implement_qwen.py
Step 3a — Call Qwen 3.6 Plus to implement scaffold stubs.
Reads:   spec.md, scaffold/scaffold.json
Writes:  src/**  (non-test files only, overwrite stubs with real implementation)
         scaffold/impl_qwen.json  (record of what was written)
"""

import os
import json
import sys
import httpx
from pathlib import Path

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
QWEN_MODEL         = "qwen/qwen3.6-plus"

ROOT           = Path(__file__).parent.parent
SPEC_PATH      = ROOT / "spec.md"
SCAFFOLD_JSON  = ROOT / "scaffold" / "scaffold.json"
INSTRUCTIONS   = ROOT / "scaffold" / "instructions_qwen.txt"
IMPL_RECORD    = ROOT / "scaffold" / "impl_qwen.json"


def build_system_prompt(instructions: str) -> str:
    return f"""You are a senior TypeScript/React developer.
You will receive:
1. A technical spec (spec.md)
2. A scaffold JSON with stub files (interfaces + signatures, no bodies)
3. Model-specific implementation instructions

Your task:
- Implement ONLY the function bodies in the non-test source files.
- Return a JSON object with this exact schema:
  {{
    "files": [
      {{
        "file_path": "src/hooks/useSensorData.ts",
        "code": "<full file content>"
      }}
    ]
  }}
- Do NOT modify test files (is_test: true).
- Do NOT add new files beyond what is in the scaffold.
- TypeScript strict mode — no `any`.
- Tailwind only — no inline styles, no CSS modules.
- Output raw JSON only. No markdown fences.

Model-specific instructions:
{instructions}"""


def call_qwen(system: str, user_message: str) -> dict:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.15,
        "max_tokens": 32768,
    }

    print("[03a] Calling Qwen 3.6 Plus …")
    with httpx.Client(timeout=180) as client:
        r = client.post(OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])

    return json.loads(text)


def main():
    spec       = SPEC_PATH.read_text()
    scaffold   = json.loads(SCAFFOLD_JSON.read_text())
    instrs     = INSTRUCTIONS.read_text() if INSTRUCTIONS.exists() else ""

    # Only send stub (non-test) files to implement
    stub_files = [f for f in scaffold["files"] if not f.get("is_test")]

    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold (stub files to implement)\n\n"
        f"{json.dumps(stub_files, indent=2)}"
    )

    result = call_qwen(build_system_prompt(instrs), user_msg)

    written = []
    for entry in result["files"]:
        path = ROOT / entry["file_path"]
        # Safety: only allow writing to src/
        if not str(path).startswith(str(ROOT / "src")):
            print(f"[03a] SKIP (outside src/): {entry['file_path']}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["code"])
        written.append(entry["file_path"])
        print(f"[03a] WROTE {entry['file_path']}")

    IMPL_RECORD.write_text(json.dumps({"model": "qwen", "files": written}, indent=2))
    print(f"[03a] Done — {len(written)} files written.")


if __name__ == "__main__":
    main()
