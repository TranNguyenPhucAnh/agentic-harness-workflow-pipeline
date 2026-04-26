"""
pipeline/03b_plan.py
Step 3b — planner_agent: reason over scaffold stubs and produce an ordered
implementation plan consumed by 03a_implement.py (--use-plan flag).

The agent model, provider, and techstack are read from pipeline/pipeline_config.json.
Supported providers: openrouter, openai, mistral, together, anthropic, gemini.

Design rationale:
    Many reasoning-heavy models burn their token budget on chain-of-thought before
    writing code, leaving too little room for actual output.  Rather than fighting
    that behaviour, we lean into it: the planner_agent reasons deeply to produce a
    *plan*, not code.  The executor_agent (03a_implement.py) turns the plan into src/.

What this script does:
    1. Read spec.md (compressed if available) + scaffold/scaffold.json.
    2. Call planner_agent — task: decompose each stub file into an ordered list of
       implementation tasks / sub-tasks.
    3. Write scaffold/plan.json  ← consumed by 03a_implement.py (--use-plan).
    4. Append implementation_order to scaffold/pipeline_context.json.

Writes:
    scaffold/plan.json

Does NOT write any src/ files — 03a_implement.py is the sole executor.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import httpx

ROOT          = Path(__file__).parent.parent
SCAFFOLD_JSON = ROOT / "scaffold" / "scaffold.json"
PLAN_OUT      = ROOT / "scaffold" / "plan.json"
PIPELINE_CTX  = ROOT / "scaffold" / "pipeline_context.json"

TAG = "[03b]"


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    cfg_path = ROOT / "pipeline" / "pipeline_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def get_agent_cfg(cfg: dict, role: str = "planner") -> dict:
    return cfg.get("agents", {}).get(role, {})


def get_techstack(cfg: dict) -> str:
    return cfg.get("techstack", "unknown")


# ════════════════════════════════════════════════════════════════════════════
# Spec loader
# ════════════════════════════════════════════════════════════════════════════

def _load_spec() -> str:
    """Use compressed spec if available — saves ~35% tokens on planner call."""
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()


# ════════════════════════════════════════════════════════════════════════════
# System prompt (techstack-aware)
# ════════════════════════════════════════════════════════════════════════════

# Maps techstack → (architect_label, style_hint_label, ordering_hint)
_TECHSTACK_META: dict[str, tuple[str, str, str]] = {
    "react-vite-ts": (
        "TypeScript/React",
        "Tailwind class hints for visual components (colours, layout, interactive states)",
        "types → hooks → components → pages → entry point",
    ),
    "nextjs-ts": (
        "TypeScript/Next.js",
        "Tailwind class hints for visual components (colours, layout, interactive states)",
        "types → lib/utils → hooks → components → pages/app → entry point",
    ),
    "vue-vite-ts": (
        "TypeScript/Vue 3",
        "Tailwind class hints for visual components (colours, layout, interactive states)",
        "types → composables → components → views → router → entry point",
    ),
    "svelte-ts": (
        "TypeScript/SvelteKit",
        "Tailwind class hints for visual components (colours, layout, interactive states)",
        "types → stores → components → routes → entry point",
    ),
    "python-fastapi": (
        "Python/FastAPI",
        "Notes on response shapes, status codes, or dependency injection patterns",
        "models/schemas → database → repositories → services → routers → main",
    ),
    "python-flask": (
        "Python/Flask",
        "Notes on response shapes or blueprint patterns",
        "models → database → services → blueprints/routes → app factory",
    ),
    "node-express-ts": (
        "TypeScript/Node.js + Express",
        "Notes on middleware patterns or response conventions",
        "types → models → repositories → services → routes → app → entry point",
    ),
    "django": (
        "Python/Django",
        "Notes on queryset patterns, serializer fields, or permission classes",
        "models → migrations → serializers → views → urls → settings",
    ),
}

_DEFAULT_META = (
    "software",
    "Any style, layout, or presentation hints relevant to visual files",
    "shared types/models → utilities → business logic → presentation → entry point",
)


def build_system_prompt(techstack: str) -> str:
    meta = _TECHSTACK_META.get(techstack, _DEFAULT_META)
    architect_label, style_hint_label, ordering_hint = meta

    return f"""\
You are a senior {architect_label} architect acting as a PLANNER.
You will receive a technical spec and a scaffold JSON (stub files with signatures only).

Your job is NOT to write code.
Your job is to reason carefully and produce a detailed implementation plan.

For each non-test stub file, output a task object describing:
- What the file does and its role in the system
- Ordered list of implementation sub-tasks (what to implement, in what order)
- Key types / interfaces / modules this file depends on (with their source file)
- Gotchas or edge cases the implementer must handle
- {style_hint_label}

Return a single JSON object — NO markdown fences, raw JSON only:
{{
  "plan_version": "1.0.0",
  "tasks": [
    {{
      "file_path": "src/example/file.ts",
      "role": "one-sentence role description",
      "depends_on": ["src/types/example.ts", "src/data/constants.ts"],
      "sub_tasks": [
        "1. First thing to implement ...",
        "2. Second thing to implement ...",
        "3. ..."
      ],
      "gotchas": [
        "Specific edge case the implementer must not miss",
        "..."
      ],
      "style_hints": null
    }}
  ],
  "implementation_order": [
    "src/types/example.ts",
    "src/data/constants.ts",
    "src/hooks/useExample.ts",
    "src/components/ExampleComponent.tsx",
    "src/App.tsx"
  ],
  "global_notes": "Any cross-cutting concerns the implementer should know"
}}

Rules:
- Reason as deeply as needed — this is your reasoning budget well spent.
- Be specific: reference exact constant names, prop names, type names from the spec.
- implementation_order MUST respect dependency order ({ordering_hint}).
- style_hints: include for visual/presentation files; set to null for logic/type/data files.
- Output raw JSON only. Absolutely no markdown fences or preamble text.
"""


# ════════════════════════════════════════════════════════════════════════════
# Provider adapters  (same pattern as 02_scaffold.py / 03a_implement.py)
# ════════════════════════════════════════════════════════════════════════════

_PROVIDER_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "openai":     "https://api.openai.com/v1/chat/completions",
    "mistral":    "https://api.mistral.ai/v1/chat/completions",
    "together":   "https://api.together.xyz/v1/chat/completions",
}

_PROVIDER_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "gemini":     "GEMINI_API_KEY",
}


def _call_openai_compat(provider: str, model: str,
                         system: str, user_msg: str,
                         max_retries: int = 2) -> str:
    env_key = _PROVIDER_ENV.get(provider, f"{provider.upper()}_API_KEY")
    api_key = os.environ.get(env_key, "")
    if not api_key:
        raise RuntimeError(f"{env_key} not set")

    url = _PROVIDER_URLS.get(provider) or os.environ.get(f"{provider.upper()}_BASE_URL")
    if not url:
        raise RuntimeError(
            f"Unknown provider '{provider}'. "
            f"Set {provider.upper()}_BASE_URL or add it to _PROVIDER_URLS."
        )

    payload = {
        "model":       model,
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens":  32768,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=240) as client:
        for attempt in range(max_retries):
            try:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()

                try:
                    data = r.json()
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"{provider} returned non-JSON: {e}\nBody: {r.text[:1000]}"
                    ) from e

                usage = data.get("usage", {})
                print(f"{TAG} Tokens: prompt={usage.get('prompt_tokens','?')}, "
                      f"completion={usage.get('completion_tokens','?')}")

                choice  = data["choices"][0]
                msg     = choice["message"]
                content = msg.get("content", "").strip()

                if msg.get("tool_calls"):
                    raise RuntimeError(
                        f"{provider} returned tool_calls instead of text: "
                        f"{msg['tool_calls']}"
                    )
                if not content:
                    raise RuntimeError(
                        f"{provider} returned empty content. "
                        f"finish_reason={choice.get('finish_reason')}, msg={msg}"
                    )
                return content

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                last_error = RuntimeError(
                    f"HTTP {status} from {provider}: {e}\n"
                    f"Body: {e.response.text[:1000] if e.response else ''}"
                )
                print(f"{TAG} {last_error}", file=sys.stderr)

            except (httpx.HTTPError, RuntimeError) as e:
                last_error = e
                print(f"{TAG} {e}", file=sys.stderr)

            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"{TAG} Retrying in {wait:.1f}s …", file=sys.stderr)
                time.sleep(wait)

    raise RuntimeError(
        f"{provider} planner failed after {max_retries} attempts: {last_error}"
    )


def _call_anthropic(model: str, system: str, user_msg: str,
                    max_retries: int = 2) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model":      model,
        "max_tokens": 32768,
        "system":     system,
        "messages":   [{"role": "user", "content": user_msg}],
    }
    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type":      "application/json",
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=240) as client:
        for attempt in range(max_retries):
            try:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data    = r.json()
                content = "".join(
                    b.get("text", "") for b in data.get("content", [])
                    if b.get("type") == "text"
                ).strip()
                if not content:
                    raise RuntimeError(f"Empty Anthropic response: {data}")
                return content

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                last_error = RuntimeError(f"HTTP {status} from Anthropic: {e}")
                print(f"{TAG} {last_error}", file=sys.stderr)

            except (httpx.HTTPError, RuntimeError) as e:
                last_error = e
                print(f"{TAG} {e}", file=sys.stderr)

            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"{TAG} Retrying in {wait:.1f}s …", file=sys.stderr)
                time.sleep(wait)

    raise RuntimeError(
        f"Anthropic planner failed after {max_retries} attempts: {last_error}"
    )


def _call_gemini(model: str, system: str, user_msg: str,
                 max_retries: int = 2) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents":           [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig":   {"temperature": 0.2, "maxOutputTokens": 32768},
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=240) as client:
        for attempt in range(max_retries):
            try:
                r = client.post(url, json=payload)
                r.raise_for_status()
                raw   = r.json()
                parts = raw["candidates"][0]["content"]["parts"]
                text  = "\n".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                ).strip()
                if not text:
                    raise ValueError(f"Gemini returned no text: {raw}")
                return text

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else None
                if status == 503 and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"{TAG} Gemini 503, retry in {wait:.1f}s …")
                    time.sleep(wait)
                    continue
                last_error = e
                print(f"{TAG} {e}", file=sys.stderr)

            except (httpx.HTTPError, ValueError, RuntimeError) as e:
                last_error = e
                print(f"{TAG} {e}", file=sys.stderr)

            if attempt < max_retries - 1:
                time.sleep(3)

    raise RuntimeError(
        f"Gemini planner failed after {max_retries} attempts: {last_error}"
    )


def call_planner(agent_cfg: dict, system: str, user_msg: str) -> str:
    """Dispatch to the correct provider based on agent config."""
    provider = agent_cfg.get("provider", "openrouter")
    model    = agent_cfg.get("model", "")

    print(f"{TAG} planner_agent  provider={provider}  model={model}")

    if provider == "anthropic":
        return _call_anthropic(model, system, user_msg)
    elif provider == "gemini":
        return _call_gemini(model, system, user_msg)
    else:
        return _call_openai_compat(provider, model, system, user_msg)


# ════════════════════════════════════════════════════════════════════════════
# JSON extraction
# ════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str, label: str) -> dict:
    """Robust JSON extraction — strips accidental markdown fences."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$",        "", raw.strip())

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"{TAG} JSON parse failed for {label}: {e}", file=sys.stderr)
            print(f"{TAG} Raw output (first 500 chars):\n{raw[:500]}", file=sys.stderr)
            sys.exit(1)

    print(f"{TAG} No JSON object found in {label}.", file=sys.stderr)
    print(f"{TAG} Raw output (first 500 chars):\n{raw[:500]}", file=sys.stderr)
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════

def validate_plan(plan: dict, stub_files: list) -> None:
    """Warn if any stub file is missing from the plan or required keys are absent."""
    planned = {t["file_path"] for t in plan.get("tasks", [])}
    for f in stub_files:
        fp = f["file_path"]
        if fp not in planned:
            print(f"{TAG} WARNING: stub file not covered by plan: {fp}")

    required_keys = {"plan_version", "tasks", "implementation_order"}
    missing = required_keys - set(plan.keys())
    if missing:
        print(f"{TAG} WARNING: plan missing keys: {missing}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg       = load_config()
    agent_cfg = get_agent_cfg(cfg, "planner")
    techstack = get_techstack(cfg)

    print(f"{TAG} Techstack : {techstack}")
    print(f"{TAG} Agent     : {agent_cfg.get('model','?')} "
          f"({agent_cfg.get('provider','?')})")

    spec     = _load_spec()
    scaffold = json.loads(SCAFFOLD_JSON.read_text())

    stub_files = [f for f in scaffold["files"] if not f.get("is_test")]
    print(f"{TAG} Planning {len(stub_files)} stub file(s) …")

    system   = build_system_prompt(techstack)
    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold stub files\n\n"
        f"{json.dumps(stub_files, indent=2)}"
    )

    raw_text = call_planner(agent_cfg, system, user_msg)
    plan     = _parse_json(raw_text, label="planner response")

    validate_plan(plan, stub_files)

    # ── Write plan.json ───────────────────────────────────────────────────────
    PLAN_OUT.write_text(json.dumps(plan, indent=2))
    print(f"{TAG} Plan written → {PLAN_OUT}")
    print(f"{TAG} Tasks in plan       : {len(plan.get('tasks', []))}")
    print(f"{TAG} Implementation order: {plan.get('implementation_order', [])}")

    # ── Update pipeline_context.json ─────────────────────────────────────────
    if PIPELINE_CTX.exists():
        ctx = json.loads(PIPELINE_CTX.read_text())
        ctx["implementation_order"] = plan.get("implementation_order", [])
        PIPELINE_CTX.write_text(json.dumps(ctx, indent=2))
        print(f"{TAG} Updated pipeline_context.json with implementation_order")

    print(f"{TAG} Done. Pass --use-plan to 03a_implement.py to use this plan.")


if __name__ == "__main__":
    main()
