"""
pipeline/02_scaffold_gemini.py
Step 2 — Call Gemini 2.5 Flash to generate scaffold JSON from spec.md
Writes: scaffold/scaffold.json
        + individual source files to src/ and tests/
"""

import os
import json
import sys
import textwrap
import httpx
from pathlib import Path

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = "gemini-2.5-flash-preview-05-20"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

ROOT       = Path(__file__).parent.parent
SPEC_PATH  = ROOT / "spec.md"
OUT_DIR    = ROOT / "scaffold"
OUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior TypeScript/React architect.
    You will receive a technical spec (spec.md) for a React + Vite + TypeScript project.

    Your task:
    1. Read the spec carefully, especially §7 (file tree) and §8 (output schema).
    2. Produce a SINGLE valid JSON object matching the schema in §8 EXACTLY.
    3. For non-test files: output interfaces + function signatures + JSDoc only.
       Use `throw new Error('not implemented')` for all function bodies.
    4. For test files: output complete, runnable vitest tests.
    5. Do NOT wrap your response in markdown fences. Output raw JSON only.
    6. Do NOT add files not listed in §7 of the spec.
""").strip()

def call_gemini(spec_content: str) -> dict:
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Here is spec.md:\n\n{spec_content}"}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 32768,
            "responseMimeType": "application/json"
        }
    }

    print("[02] Calling Gemini 2.5 Flash …")
    with httpx.Client(timeout=120) as client:
        r = client.post(GEMINI_URL, json=payload)
        r.raise_for_status()

    raw = r.json()
    text = raw["candidates"][0]["content"]["parts"][0]["text"]

    # Strip accidental markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])

    return json.loads(text)


def write_files(scaffold: dict) -> None:
    json_path = OUT_DIR / "scaffold.json"
    json_path.write_text(json.dumps(scaffold, indent=2))
    print(f"[02] Scaffold JSON written → {json_path}")

    for entry in scaffold["files"]:
        path = ROOT / entry["file_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["code"])
        tag = "TEST" if entry.get("is_test") else "SRC "
        print(f"[02] [{tag}] {entry['file_path']}")

    # Write implementation instructions for downstream models
    instructions = scaffold.get("implementation_instructions", {})
    (OUT_DIR / "instructions_qwen.txt").write_text(
        instructions.get("for_qwen", "No specific instructions.")
    )
    (OUT_DIR / "instructions_glm.txt").write_text(
        instructions.get("for_glm", "No specific instructions.")
    )
    print("[02] Done.")


def main():
    spec = SPEC_PATH.read_text()
    scaffold = call_gemini(spec)

    # Validate required keys
    required = {"scaffold_version", "files", "implementation_instructions"}
    missing = required - set(scaffold.keys())
    if missing:
        print(f"[02] ERROR: scaffold JSON missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    write_files(scaffold)


if __name__ == "__main__":
    main()
