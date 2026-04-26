"""
pipeline/02_scaffold.py
Step 2 — scaffold_agent: generate scaffold JSON (stubs + test files) from spec.md.

The agent model and techstack are read from pipeline/pipeline_config.json.
Supported providers: gemini, openrouter, openai, anthropic.

Writes:
    scaffold/scaffold.json          ← full scaffold with stubs + test files
    src/**                          ← individual stub source files
    tests/**                        ← individual test files
    scaffold/instructions_executor.txt  ← executor hints (consumed by 03a)
    scaffold/spec_compressed.md     ← spec with scaffold-only sections stripped
    scaffold/pipeline_context.json  ← shared pipeline state, updated by later steps
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

ROOT      = Path(__file__).parent.parent
SPEC_PATH = ROOT / "spec.md"
OUT_DIR   = ROOT / "scaffold"
OUT_DIR.mkdir(exist_ok=True)

TAG = "[02]"


# ════════════════════════════════════════════════════════════════════════════
# Config loader
# ════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    cfg_path = ROOT / "pipeline" / "pipeline_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def get_agent_cfg(cfg: dict, role: str = "scaffold") -> dict:
    return cfg.get("agents", {}).get(role, {})


def get_techstack(cfg: dict) -> str:
    return cfg.get("techstack", "unknown")


# ════════════════════════════════════════════════════════════════════════════
# System prompt (techstack-aware, model-agnostic)
# ════════════════════════════════════════════════════════════════════════════

# Maps techstack id → plain-language description used inside the prompt.
# Add new techstacks here without touching the prompt template.
_TECHSTACK_DESCRIPTIONS: dict[str, str] = {
    "react-vite-ts":   "React + Vite + TypeScript",
    "nextjs-ts":       "Next.js + TypeScript",
    "vue-vite-ts":     "Vue 3 + Vite + TypeScript",
    "svelte-ts":       "SvelteKit + TypeScript",
    "python-fastapi":  "Python + FastAPI",
    "python-flask":    "Python + Flask",
    "node-express-ts": "Node.js + Express + TypeScript",
    "django":          "Python + Django",
}

# Maps techstack id → stub body convention (language-specific placeholder).
_STUB_PLACEHOLDER: dict[str, str] = {
    "react-vite-ts":   "throw new Error('not implemented')",
    "nextjs-ts":       "throw new Error('not implemented')",
    "vue-vite-ts":     "throw new Error('not implemented')",
    "svelte-ts":       "throw new Error('not implemented')",
    "python-fastapi":  "raise NotImplementedError('not implemented')",
    "python-flask":    "raise NotImplementedError('not implemented')",
    "node-express-ts": "throw new Error('not implemented')",
    "django":          "raise NotImplementedError('not implemented')",
}

# Maps techstack id → test framework name (for the prompt).
_TEST_FRAMEWORK: dict[str, str] = {
    "react-vite-ts":   "vitest",
    "nextjs-ts":       "jest / vitest",
    "vue-vite-ts":     "vitest",
    "svelte-ts":       "vitest",
    "python-fastapi":  "pytest",
    "python-flask":    "pytest",
    "node-express-ts": "jest",
    "django":          "pytest-django",
}


def build_system_prompt(techstack: str) -> str:
    description  = _TECHSTACK_DESCRIPTIONS.get(techstack, techstack)
    placeholder  = _STUB_PLACEHOLDER.get(techstack, "throw new Error('not implemented')")
    test_fw      = _TEST_FRAMEWORK.get(techstack, "the project's test framework")

    return f"""\
You are a senior software architect specialising in {description}.
You will receive a technical spec (spec.md) for a {description} project.

Your task:
1. Read the spec carefully, especially §7 (file tree) and §8 (output schema).
2. Produce a SINGLE valid JSON object matching the schema in §8 EXACTLY.
3. The JSON MUST be valid and parseable by JSON.parse / json.loads.
   Requirements:
   - Use double quotes for all JSON strings.
   - Escape any internal double-quotes as \\".
   - Code in "code" fields must be a single JSON string: newlines as \\n, quotes escaped.
   - No single quotes as JSON delimiters. No comments. No trailing commas.
4. For non-test files: output type definitions + function/class signatures + docstrings only.
   Use `{placeholder}` for all function bodies.
5. For test files: output complete, runnable {test_fw} tests.
6. Do NOT wrap your response in markdown fences. Output raw JSON only.
7. Do NOT add files not listed in §7 of the spec.
"""


# ════════════════════════════════════════════════════════════════════════════
# Provider adapters
# ════════════════════════════════════════════════════════════════════════════

# ── Gemini (native REST) ──────────────────────────────────────────────────

def _call_gemini(model: str, system_prompt: str, user_msg: str,
                 max_retries: int = 5) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 32768,
            "responseMimeType": "application/json",
        },
    }

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = client.post(url, json=payload)
                r.raise_for_status()
                raw = r.json()
                parts = raw["candidates"][0]["content"]["parts"]
                text = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                ).strip()
                if not text:
                    raise ValueError(f"Gemini returned no text. Response: {raw}")
                return text

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                if status == 503 and attempt < max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(f"{TAG} Gemini 503, retry {attempt}/{max_retries} in {wait:.1f}s …")
                    time.sleep(wait)
                    continue
                raise

    raise RuntimeError("Gemini call failed after retries")


# ── OpenAI-compatible (OpenRouter, OpenAI, Mistral, …) ───────────────────

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai":     "https://api.openai.com/v1/chat/completions",
    "mistral":    "https://api.mistral.ai/v1/chat/completions",
    "together":   "https://api.together.xyz/v1/chat/completions",
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "together":   "TOGETHER_API_KEY",
}


def _call_openai_compat(provider: str, model: str, system_prompt: str,
                         user_msg: str, max_retries: int = 3) -> str:
    env_key = _PROVIDER_ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
    api_key = os.environ.get(env_key, "")
    if not api_key:
        raise RuntimeError(f"{env_key} not set")

    url = _PROVIDER_BASE_URLS.get(provider)
    if not url:
        # Allow arbitrary base URL via env, e.g. MYPROVIDER_BASE_URL
        url = os.environ.get(f"{provider.upper()}_BASE_URL")
    if not url:
        raise RuntimeError(
            f"Unknown provider '{provider}'. "
            f"Set {provider.upper()}_BASE_URL or add it to _PROVIDER_BASE_URLS."
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        "temperature":  0.2,
        "max_tokens":   32768,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data    = r.json()
                content = data["choices"][0]["message"].get("content", "").strip()
                if not content:
                    raise ValueError(f"Empty response from {provider}/{model}: {data}")
                usage = data.get("usage", {})
                print(f"{TAG} Tokens: prompt={usage.get('prompt_tokens','?')}, "
                      f"completion={usage.get('completion_tokens','?')}")
                return content

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                if status in (429, 503) and attempt < max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(f"{TAG} {provider} {status}, retry {attempt}/{max_retries} "
                          f"in {wait:.1f}s …")
                    time.sleep(wait)
                    continue
                raise

    raise RuntimeError(f"{provider} call failed after retries")


# ── Anthropic ──────────────────────────────────────────────────────────────

def _call_anthropic(model: str, system_prompt: str, user_msg: str,
                    max_retries: int = 3) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model":      model,
        "max_tokens": 32768,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_msg}],
    }
    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json",
    }

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        for attempt in range(1, max_retries + 1):
            try:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data    = r.json()
                content = "".join(
                    b.get("text", "") for b in data.get("content", [])
                    if b.get("type") == "text"
                ).strip()
                if not content:
                    raise ValueError(f"Empty Anthropic response: {data}")
                return content

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                if status in (429, 529) and attempt < max_retries:
                    wait = (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(f"{TAG} Anthropic {status}, retry {attempt}/{max_retries} "
                          f"in {wait:.1f}s …")
                    time.sleep(wait)
                    continue
                raise

    raise RuntimeError("Anthropic call failed after retries")


# ════════════════════════════════════════════════════════════════════════════
# Dispatch
# ════════════════════════════════════════════════════════════════════════════

def call_scaffold_agent(agent_cfg: dict, system_prompt: str, spec: str) -> str:
    provider = agent_cfg.get("provider", "openrouter")
    model    = agent_cfg.get("model", "")
    user_msg = f"Here is spec.md:\n\n{spec}"

    print(f"{TAG} scaffold_agent  provider={provider}  model={model}")

    if provider == "gemini":
        return _call_gemini(model, system_prompt, user_msg)
    elif provider == "anthropic":
        return _call_anthropic(model, system_prompt, user_msg)
    else:
        # openrouter, openai, mistral, together, …
        return _call_openai_compat(provider, model, system_prompt, user_msg)


# ════════════════════════════════════════════════════════════════════════════
# JSON extraction (robust, provider-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str) -> dict:
    """Strip accidental markdown fences, then parse JSON robustly."""
    cleaned = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$",        "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"{TAG} Primary JSON parse failed: {e}", file=sys.stderr)

    # Fallback: find outermost { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f"{TAG} JSON parse failed after extracting {{...}} block: {e}",
                  file=sys.stderr)
            print(f"{TAG} Raw output (first 500 chars):\n{cleaned[:500]}",
                  file=sys.stderr)
            sys.exit(1)

    print(f"{TAG} No JSON object found in scaffold_agent response.", file=sys.stderr)
    print(f"{TAG} Raw output (first 500 chars):\n{cleaned[:500]}", file=sys.stderr)
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Spec compression
# ════════════════════════════════════════════════════════════════════════════

def _compress_spec(spec: str) -> str:
    """
    Strip scaffold-only sections from spec.md to save tokens on downstream calls.
    §0 (pipeline meta) and §8 (output schema) are scaffold-agent-only — later
    steps (planner, executor, judge) don't need them.
    Keeps §1-7, §9-11 (component specs, types, acceptance criteria).
    Saves ~35% tokens on every downstream call.
    """
    lines  = spec.splitlines()
    out:   list[str] = []
    skip   = False
    SKIP_HEADERS  = ("## 0.", "## 8.")
    RESUME_PREFIX = "## "
    for line in lines:
        if any(line.startswith(h) for h in SKIP_HEADERS):
            skip = True
        elif skip and line.startswith(RESUME_PREFIX) and not any(
            line.startswith(h) for h in SKIP_HEADERS
        ):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# File writer
# ════════════════════════════════════════════════════════════════════════════

def write_files(scaffold: dict, spec: str) -> None:
    # ── scaffold.json ──────────────────────────────────────────────────────
    json_path = OUT_DIR / "scaffold.json"
    json_path.write_text(json.dumps(scaffold, indent=2))
    print(f"{TAG} Scaffold JSON → {json_path}")

    # ── Individual source + test stubs ─────────────────────────────────────
    for entry in scaffold["files"]:
        path = ROOT / entry["file_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(entry["code"])
        tag = "TEST" if entry.get("is_test") else "SRC "
        print(f"{TAG} [{tag}] {entry['file_path']}")

    # ── Executor instructions (role-neutral name) ───────────────────────────
    instructions = scaffold.get("implementation_instructions", {})
    # Accept either "for_executor" (new generic key) or "for_qwen" (legacy)
    executor_hints = (
        instructions.get("for_executor")
        or instructions.get("for_qwen")
        or "No specific instructions."
    )
    (OUT_DIR / "instructions_executor.txt").write_text(executor_hints)
    print(f"{TAG} Executor instructions → scaffold/instructions_executor.txt")

    # ── Compressed spec for downstream models ──────────────────────────────
    compressed = _compress_spec(spec)
    (OUT_DIR / "spec_compressed.md").write_text(compressed)
    savings = round((1 - len(compressed) / len(spec)) * 100)
    print(f"{TAG} Compressed spec → scaffold/spec_compressed.md  ({savings}% smaller)")

    # ── pipeline_context.json — shared pipeline state ──────────────────────
    # Step 03b (planner) will append implementation_order to this file.
    context = {
        "spec_compressed_path":  "scaffold/spec_compressed.md",
        "scaffold_version":      scaffold["scaffold_version"],
        "file_tree":             [f["file_path"] for f in scaffold["files"]],
        "stub_map":              {
            f["file_path"]: f["code"]
            for f in scaffold["files"] if not f.get("is_test")
        },
        "instructions_executor": executor_hints,
        "implementation_order":  [],    # filled by 03b_plan.py
    }
    (OUT_DIR / "pipeline_context.json").write_text(json.dumps(context, indent=2))
    print(f"{TAG} Pipeline context → scaffold/pipeline_context.json")

    print(f"{TAG} Done.")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg        = load_config()
    agent_cfg  = get_agent_cfg(cfg, "scaffold")
    techstack  = get_techstack(cfg)

    print(f"{TAG} Techstack : {techstack}")
    print(f"{TAG} Agent     : {agent_cfg.get('model','?')} "
          f"({agent_cfg.get('provider','?')})")

    system_prompt = build_system_prompt(techstack)
    spec          = SPEC_PATH.read_text()

    raw_text = call_scaffold_agent(agent_cfg, system_prompt, spec)
    scaffold = _parse_json(raw_text)

    required = {"scaffold_version", "files", "implementation_instructions"}
    missing  = required - set(scaffold.keys())
    if missing:
        print(f"{TAG} ERROR: scaffold JSON missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    write_files(scaffold, spec)


if __name__ == "__main__":
    main()
