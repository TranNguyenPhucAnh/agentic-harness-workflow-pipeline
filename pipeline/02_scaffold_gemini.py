"""
pipeline/02_scaffold_gemini.py
Step 2 — Call Gemini 2.5 Flash to generate scaffold JSON from spec.md

Writes:
    scaffold/scaffold.json          ← full scaffold with stubs + test files
    src/**                          ← individual stub source files
    tests/**                        ← individual test files
    scaffold/instructions_qwen.txt  ← executor hints for Qwen (consumed by 03a)
    NOTE: GLM 5.1 is now a PLANNER (03b) with its own hardcoded system prompt.
          No instructions file is written for GLM.
"""

import os
import json
import re
import sys
import textwrap
import httpx
from pathlib import Path

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

ROOT      = Path(__file__).parent.parent
SPEC_PATH = ROOT / "spec.md"
OUT_DIR   = ROOT / "scaffold"
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


# ── API call ──────────────────────────────────────────────────────────────────

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
            "responseMimeType": "application/json",
        }
    }

    print("[02] Calling Gemini 2.5 Flash …")
    with httpx.Client(timeout=120) as client:
        r = client.post(GEMINI_URL, json=payload)
        r.raise_for_status()

    raw  = r.json()
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json(text)


# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Robust JSON extraction — handles accidental markdown fences."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
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
            print(f"[02] JSON parse failed: {e}", file=sys.stderr)
            print(f"[02] Raw output (first 500 chars):\n{raw[:500]}", file=sys.stderr)
            sys.exit(1)
    print("[02] No JSON object found in Gemini response.", file=sys.stderr)
    sys.exit(1)

def _compress_spec(spec: str) -> str:
    """
    Tạo bản rút gọn của spec.md để downstream models dùng thay full spec.
    Bỏ §0 (meta/pipeline instructions dành cho Gemini) và §8 (Gemini output schema)
    vì các bước sau không cần. Giữ §1-7, §9-11 (component specs, types, AC).
    Tiết kiệm ~35% tokens trên mọi call downstream.
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

def write_files(scaffold: dict, spec: str) -> None:   # <-- thêm param spec
    # scaffold.json
    json_path = OUT_DIR / "scaffold.json"
    json_path.write_text(json.dumps(scaffold, indent=2))
    print(f"[02] Scaffold JSON → {json_path}")
 
    # Individual source + test stubs
    for entry in scaffold["files"]:
        path = ROOT / entry["file_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["code"])
        tag = "TEST" if entry.get("is_test") else "SRC "
        print(f"[02] [{tag}] {entry['file_path']}")
 
    # Downstream model instructions (Qwen only)
    instructions = scaffold.get("implementation_instructions", {})
    (OUT_DIR / "instructions_qwen.txt").write_text(
        instructions.get("for_qwen", "No specific instructions.")
    )
 
    # ── NEW: compressed spec cho downstream models ────────────
    compressed = _compress_spec(spec)
    (OUT_DIR / "spec_compressed.md").write_text(compressed)
    savings = round((1 - len(compressed) / len(spec)) * 100)
    print(f"[02] Compressed spec → scaffold/spec_compressed.md  ({savings}% smaller)")
 
    # ── NEW: pipeline_context.json — shared state file ────────
    # 03b append implementation_order ke file ini setelah plan.
    context = {
        "spec_compressed_path": "scaffold/spec_compressed.md",
        "scaffold_version":     scaffold["scaffold_version"],
        "file_tree":            [f["file_path"] for f in scaffold["files"]],
        "stub_map":             {                              # stub code keyed by path
            f["file_path"]: f["code"]
            for f in scaffold["files"] if not f.get("is_test")
        },
        "instructions_qwen":    instructions.get("for_qwen", ""),
        "implementation_order": [],    # filled by 03b
        "spec_hash": hashlib.sha256(spec.encode()).hexdigest()[:16],
        "spec_version": _extract_version(spec),   # parse "# Version: 1.1.0" từ header
        
        # NEW: section → file mapping (Gemini tự generate khi scaffold)
        # scaffold JSON nên có thêm field "owned_by_section": "§4.2"
        "section_file_map": {
            "§4.1": ["src/components/SummaryStickyBar.tsx"],
            "§4.5": ["src/hooks/useSensorData.ts"],
            # etc — Gemini điền vào khi scaffold
        },
        
        # NEW: file dependency graph từ GLM plan
        "file_deps": {},   # filled by 03b: {"src/hooks/useReplay.ts": ["src/types/sensor.ts"]}
    }
    (OUT_DIR / "pipeline_context.json").write_text(json.dumps(context, indent=2))
    print("[02] Pipeline context → scaffold/pipeline_context.json")
 
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
 
    write_files(scaffold, spec)    # <-- thêm spec

if __name__ == "__main__":
    main()
