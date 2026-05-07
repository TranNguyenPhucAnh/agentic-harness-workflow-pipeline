"""
pipeline/02_scaffold_gemini.py
Step 2 — Call Gemini 2.5 Flash to generate scaffold JSON from spec.md

Writes:
    artifacts_<slug>/state/scaffold.json       ← full scaffold with stubs + test files
    artifacts_<slug>/src/**                    ← individual stub source files
    artifacts_<slug>/tests/**                  ← individual test files
    artifacts_<slug>/cache/spec_compressed.md  ← compressed spec for downstream use

For taxonomy details see docs/artifacts.md
"""

import os
import json
import re
import sys
import textwrap
import httpx
from pathlib import Path
import random
import time

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

# === WRITE AUTHORITY: 02_scaffold_gemini ===
# OWNS  : artifacts_<slug>/state/scaffold.json
#         artifacts_<slug>/cache/spec_compressed.md
#         artifacts_<slug>/src/**
#         artifacts_<slug>/tests/**
# READS : artifacts_<slug>/spec.md

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from artifacts.paths import (
    SPEC_PATH,
    SCAFFOLD_JSON, SPEC_COMPRESSED,
    SRC_DIR, TESTS_DIR,
    ensure_dirs,
)
ensure_dirs()

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior TypeScript/React architect.
    You will receive a technical spec (spec.md) for a React + Vite + TypeScript project.

    Your task:
    1. Read the spec carefully, especially §7 (file tree) and §8 (output schema).
    2. Produce a SINGLE valid JSON object matching the schema in §8 EXACTLY.
    3. The JSON MUST be valid and parseable by JSON.parse / json.loads.
        Requirements:
        - Use double quotes " for all JSON strings.
        - Escape any internal " characters as \".
        - If you output TypeScript code in a "code" field, it MUST be a single JSON string value with all newlines as \n and all quotes properly escaped.
        - Do NOT use single quotes ' for JSON string delimiters.
        - Do NOT include comments or trailing commas in the JSON.
    4. For non-test files: output interfaces + function signatures + JSDoc only.
       Use `throw new Error('not implemented')` for all function bodies.
    5. For test files: output complete, runnable vitest tests.
    6. Do NOT wrap your response in markdown fences. Output raw JSON only.
    7. Do NOT add files not listed in §7 of the spec.
""").strip()


# ── API call ──────────────────────────────────────────────────────────────────

def _extract_gemini_text(raw: dict) -> str:
    try:
        parts = raw["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {raw}") from exc

    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()

    if not text:
        raise ValueError(f"Gemini returned no text parts: {raw}")

    return text

def call_gemini(spec_content: str, max_retries: int = 5) -> dict:
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
            "responseMimeType": "application/json",
        }
    }

    print("[02] Calling Gemini 2.5 Flash …")

    timeout = httpx.Timeout(120.0, connect=30.0)

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = client.post(GEMINI_URL, json=payload)
                r.raise_for_status()

                raw = r.json()
                text = _extract_gemini_text(raw)
                return _parse_json(text)

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None

                if status == 503 and attempt < max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(f"[02] Gemini 503 overloaded, retry {attempt}/{max_retries} in {wait:.1f}s …")
                    time.sleep(wait)
                    continue

                raise

    raise RuntimeError("Gemini call failed after retries")

# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Robust JSON extraction — handles accidental markdown fences."""
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f" Primary JSON parse failed: {e}", file=sys.stderr)

    # Fallback: find outermost { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f" JSON parse failed even after extracting outer {{...}} block: {e}", file=sys.stderr)
            # Heuristic hint
            if "'LOW' | 'MED' | 'HIGH'" in candidate or "export type" in candidate:
                print(" Hint: Gemini likely emitted raw TypeScript with single quotes "
                      "inside the JSON string. Tighten SYSTEM_PROMPT to require valid JSON "
                      "and escaped code strings.", file=sys.stderr)
            print(f" Raw output (first 500 chars):\n{cleaned[:500]}", file=sys.stderr)
            sys.exit(1)

    print(" No JSON object found in Gemini response.", file=sys.stderr)
    print(f" Raw output (first 500 chars):\n{cleaned[:500]}", file=sys.stderr)
    sys.exit(1)
    
def _compress_spec(spec: str) -> str:
    """
    Create compressed version of spec.md for downstream models.
    Removes §0 (meta/pipeline instructions for Gemini) and §8 (Gemini output schema).
    Keeps §1-7, §9-11 (component specs, types, AC).
    Saves ~35% tokens on every downstream call.
    """
    lines = spec.splitlines()
    out: list[str] = []
    skip = False
    SKIP_HEADERS  = ("## 0.", "## 8.")
    RESUME_PREFIX = "## "
    for line in lines:
        if any(line.startswith(h) for h in SKIP_HEADERS):
            skip = True
        elif skip and line.startswith(RESUME_PREFIX) and not any(line.startswith(h) for h in SKIP_HEADERS):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)

# ── File writer ───────────────────────────────────────────────────────────────

def write_files(scaffold: dict, spec: str) -> None:
    # Write scaffold.json to state/
    SCAFFOLD_JSON.write_text(json.dumps(scaffold, indent=2))
    print(f"[02] Scaffold JSON → {SCAFFOLD_JSON}")

    # Write individual source and test stubs into artifacts_<slug>/src/ and tests/
    for entry in scaffold["files"]:
        file_path = entry["file_path"]           # e.g. "src/App.tsx" or "tests/App.test.tsx"
        is_test   = entry.get("is_test", False)

        if is_test:
            # strip leading "tests/" prefix if present, resolve under TESTS_DIR
            rel = file_path[len("tests/"):] if file_path.startswith("tests/") else file_path
            dest = TESTS_DIR / rel
        else:
            # strip leading "src/" prefix if present, resolve under SRC_DIR
            rel = file_path[len("src/"):] if file_path.startswith("src/") else file_path
            dest = SRC_DIR / rel

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(entry["code"])
        tag = "TEST" if is_test else "SRC "
        print(f"[02] [{tag}] {dest}")

    # Write compressed spec to cache/
    compressed = _compress_spec(spec)
    SPEC_COMPRESSED.parent.mkdir(parents=True, exist_ok=True)
    SPEC_COMPRESSED.write_text(compressed)
    savings = round((1 - len(compressed) / len(spec)) * 100)
    print(f"[02] Compressed spec → {SPEC_COMPRESSED}  ({savings}% smaller)")

    # NOTE: instructions_qwen.txt and pipeline_context.json are removed.
    # Qwen will read from scaffold.json["implementation_instructions"]["for_qwen"].

    print("[02] Done.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spec     = SPEC_PATH.read_text()
    scaffold = call_gemini(spec)

    required = {"scaffold_version", "files", "implementation_instructions"}
    missing  = required - set(scaffold.keys())
    if missing:
        print(f"[02] ERROR: scaffold JSON missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    write_files(scaffold, spec)

if __name__ == "__main__":
    main()
