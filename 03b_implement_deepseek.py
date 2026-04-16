"""
pipeline/03b_implement_deepseek.py
Step 3b — Call DeepSeek V3 to implement scaffold stubs.
Reads:   spec.md, scaffold/scaffold.json
Writes:  src_deepseek/**  (non-test files only)
         scaffold/impl_deepseek.json

NOTE: Runs in parallel with 03a via separate GitHub Actions steps.
      Each model writes to its OWN directory to avoid race conditions:
        Qwen     → src/           (canonical; swapped by 04_test_and_iterate.py)
        DeepSeek → src_deepseek/  (isolated impl)
"""

import os
import json
import sys
import httpx
from pathlib import Path

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_URL     = "https://api.deepseek.com"
DEEPSEEK_MODEL   = "deepseek-chat"

ROOT           = Path(__file__).parent.parent
SPEC_PATH      = ROOT / "spec.md"
SCAFFOLD_JSON  = ROOT / "scaffold" / "scaffold.json"
INSTRUCTIONS   = ROOT / "scaffold" / "instructions_deepseek.txt"
IMPL_RECORD    = ROOT / "scaffold" / "impl_deepseek.json"
OUT_DIR        = ROOT / "src_deepseek"   # isolated; test runner swaps this in


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


def call_deepseek(system: str, user_message: str) -> dict:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.15,
        "max_tokens": 32768,
    }

    print("[03b] Calling DeepSeek V3 …")
    with httpx.Client(timeout=180) as client:
        r = client.post(DEEPSEEK_URL, headers=headers, json=payload)
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

    result = call_deepseek(build_system_prompt(instrs), user_msg)

    written = []
    for entry in result["files"]:
        # Remap src/ → src_deepseek/ to isolate DeepSeek's impl
        rel = entry["file_path"]
        if rel.startswith("src/"):
            rel = "src_deepseek/" + rel[len("src/"):]
        path = ROOT / rel

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["code"])
        written.append(rel)
        print(f"[03b] WROTE {rel}")

    IMPL_RECORD.write_text(json.dumps({"model": "deepseek", "files": written}, indent=2))
    print(f"[03b] Done — {len(written)} files written.")


if __name__ == "__main__":
    main()
