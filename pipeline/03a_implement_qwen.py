"""
pipeline/03a_implement.py
Step 3a — executor_agent: implement src/ files from scaffold stubs.

The agent model, provider, and techstack are read from pipeline/pipeline_config.json.
Supported providers: openrouter, openai, mistral, together, anthropic, gemini.

Modes:
    Default (no plan):
        One API call per stub file → src/<file>  (per-file mode)
        Falls back to single-call if plan is absent.

    With planner plan (--use-plan):
        Reads scaffold/plan.json produced by 03b_plan.py.
        For each file, injects the matching task (sub_tasks, gotchas,
        style_hints, depends_on) into the prompt before generating.
        Files are generated in implementation_order from the plan.

Writes:
    src/**                       (non-test files only)
    scaffold/impl_executor.json  (implementation record)
"""

from __future__ import annotations

import argparse
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
INSTRUCTIONS  = ROOT / "scaffold" / "instructions_executor.txt"
PLAN_JSON     = ROOT / "scaffold" / "plan.json"
IMPL_RECORD   = ROOT / "scaffold" / "impl_executor.json"
PIPELINE_CTX  = ROOT / "scaffold" / "pipeline_context.json"

TAG = "[03a]"


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    cfg_path = ROOT / "pipeline" / "pipeline_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def get_agent_cfg(cfg: dict, role: str) -> dict:
    return cfg.get("agents", {}).get(role, {})


def get_techstack(cfg: dict) -> str:
    return cfg.get("techstack", "unknown")


# ════════════════════════════════════════════════════════════════════════════
# Spec loader
# ════════════════════════════════════════════════════════════════════════════

def _load_spec() -> str:
    """Use compressed spec if available (saves ~35% tokens)."""
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()


# ════════════════════════════════════════════════════════════════════════════
# System prompts (techstack-aware)
# ════════════════════════════════════════════════════════════════════════════

# Maps techstack → (language_label, style_rules, stub_placeholder)
_TECHSTACK_META: dict[str, tuple[str, str, str]] = {
    "react-vite-ts": (
        "TypeScript/React",
        "- TypeScript strict mode — no `any`.\n- Tailwind only — no inline styles, no CSS modules.",
        "throw new Error('not implemented')",
    ),
    "nextjs-ts": (
        "TypeScript/Next.js",
        "- TypeScript strict mode — no `any`.\n- Tailwind only — no inline styles.",
        "throw new Error('not implemented')",
    ),
    "vue-vite-ts": (
        "TypeScript/Vue 3",
        "- TypeScript strict mode — no `any`.\n- Tailwind only — no inline styles.",
        "throw new Error('not implemented')",
    ),
    "svelte-ts": (
        "TypeScript/SvelteKit",
        "- TypeScript strict mode — no `any`.\n- Tailwind only — no inline styles.",
        "throw new Error('not implemented')",
    ),
    "python-fastapi": (
        "Python/FastAPI",
        "- Type hints required on all functions.\n- Follow PEP 8.",
        "raise NotImplementedError('not implemented')",
    ),
    "python-flask": (
        "Python/Flask",
        "- Type hints required on all functions.\n- Follow PEP 8.",
        "raise NotImplementedError('not implemented')",
    ),
    "node-express-ts": (
        "TypeScript/Node.js + Express",
        "- TypeScript strict mode — no `any`.",
        "throw new Error('not implemented')",
    ),
    "django": (
        "Python/Django",
        "- Type hints required on all functions.\n- Follow PEP 8.",
        "raise NotImplementedError('not implemented')",
    ),
}

_DEFAULT_META = ("the project's language", "", "raise NotImplementedError('not implemented')")


def _get_techstack_meta(techstack: str) -> tuple[str, str, str]:
    return _TECHSTACK_META.get(techstack, _DEFAULT_META)


def build_system_prompt_single(techstack: str, instructions: str) -> str:
    """Single-call prompt — all files in one request (no plan)."""
    lang_label, style_rules, _ = _get_techstack_meta(techstack)
    return f"""\
You are a senior {lang_label} developer.
You will receive:
1. A technical spec (spec.md)
2. A scaffold JSON with stub files (interfaces + signatures, no bodies)
3. Executor-specific implementation instructions

Your task:
- Implement ONLY the function/class bodies in the non-test source files.
- Return a JSON object with this exact schema:
  {{
    "files": [
      {{
        "file_path": "src/example.ts",
        "code": "<full file content>"
      }}
    ]
  }}
- Do NOT modify test files (is_test: true).
- Do NOT add new files beyond what is in the scaffold.
{style_rules}
- Output raw JSON only. No markdown fences.

Executor instructions:
{instructions}"""


def build_system_prompt_per_file(techstack: str) -> str:
    """Per-file generation prompt — one file per API call."""
    lang_label, style_rules, _ = _get_techstack_meta(techstack)
    return f"""\
You are a senior {lang_label} developer implementing ONE source file.
You will receive:
1. The technical spec (spec.md) for full context
2. The stub for the SINGLE file you must implement
3. (Optional) A task plan from a senior architect — follow it carefully

Your task:
- Implement the function/class body for this ONE file only.
- Return a JSON object with this EXACT schema:
  {{
    "file_path": "src/example.ts",
    "code": "<full file content — complete, runnable>"
  }}
{style_rules}
- The code field must be the COMPLETE file (imports + types + implementation).
- Output raw JSON only. Absolutely no markdown fences or explanation text.
"""


# ════════════════════════════════════════════════════════════════════════════
# Provider adapters  (same pattern as 02_scaffold.py)
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
        "temperature": 0.15,
        "max_tokens":  32768,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=180) as client:
        for attempt in range(max_retries):
            try:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()

                try:
                    data = r.json()
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"{provider} returned non-JSON: {e}\n"
                        f"Body: {r.text[:1000]}"
                    ) from e

                usage = data.get("usage", {})
                print(f"{TAG} Tokens: prompt={usage.get('prompt_tokens','?')}, "
                      f"completion={usage.get('completion_tokens','?')}")

                choice  = data["choices"][0]
                msg     = choice["message"]
                content = msg.get("content", "").strip()

                if msg.get("tool_calls"):
                    raise RuntimeError(f"{provider} returned tool_calls instead of text.")
                if not content:
                    raise RuntimeError(
                        f"{provider} returned empty content. "
                        f"finish_reason={choice.get('finish_reason')}"
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

    raise RuntimeError(f"{provider} call failed after {max_retries} attempts: {last_error}")


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
    with httpx.Client(timeout=180) as client:
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

    raise RuntimeError(f"Anthropic call failed after {max_retries} attempts: {last_error}")


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
        "generationConfig":   {"temperature": 0.15, "maxOutputTokens": 32768},
    }

    last_error: Exception | None = None
    with httpx.Client(timeout=180) as client:
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

            except (httpx.HTTPError, RuntimeError, ValueError) as e:
                last_error = e
                print(f"{TAG} {e}", file=sys.stderr)

            if attempt < max_retries - 1:
                time.sleep(3)

    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_error}")


def call_executor(agent_cfg: dict, system: str, user_msg: str) -> str:
    """Dispatch to the correct provider based on agent config."""
    provider = agent_cfg.get("provider", "openrouter")
    model    = agent_cfg.get("model", "")

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
            raise RuntimeError(
                f"JSON parse failed for {label}: {e}\nRaw (first 500):\n{raw[:500]}"
            )
    raise RuntimeError(f"No JSON found in response for {label}.")


# ════════════════════════════════════════════════════════════════════════════
# Plan task block builder (planner-output-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def _build_task_block(task: dict | None) -> str:
    """Format a planner task as a prompt section (works with any planner output)."""
    if not task:
        return ""
    lines = ["### Implementation plan from architect\n"]
    if task.get("role"):
        lines.append(f"**Role:** {task['role']}\n")
    deps = task.get("depends_on", [])
    if deps:
        lines.append(f"**Depends on:** {', '.join(deps)}\n")
    sub_tasks = task.get("sub_tasks", [])
    if sub_tasks:
        lines.append("**Sub-tasks (implement in this order):**")
        for st in sub_tasks:
            lines.append(f"  {st}")
        lines.append("")
    gotchas = task.get("gotchas", [])
    if gotchas:
        lines.append("**Gotchas / edge cases:**")
        for g in gotchas:
            lines.append(f"  - {g}")
        lines.append("")
    # Generic style hints key (was "tailwind_hints", now "style_hints"; accept both)
    hints = task.get("style_hints") or task.get("tailwind_hints")
    if hints:
        lines.append(f"**Style hints:** {hints}\n")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Context injection helpers
# ════════════════════════════════════════════════════════════════════════════

# Files whose prompts should only receive signature-level context to avoid
# exceeding token limits. Paths are relative to ROOT (src/ prefix).
# Extend this set in pipeline_config.json under "signature_only_files" if needed.
_DEFAULT_SIGNATURE_ONLY: set[str] = {
    "src/App.tsx", "src/main.tsx",        # React/Vite
    "src/App.vue",                         # Vue
    "src/routes/+page.svelte",             # SvelteKit
    "src/main.py", "app/main.py",          # FastAPI / Flask entry points
}


def _load_signature_only_set(cfg: dict) -> set[str]:
    extra = cfg.get("signature_only_files", [])
    return _DEFAULT_SIGNATURE_ONLY | set(extra)


def _build_context_block(
    file_path: str,
    task: dict | None,
    already_written: dict[str, str],
    signature_only_set: set[str],
) -> str:
    """Inject relevant already-written files as import context."""
    deps     = set(task.get("depends_on", [])) if task else set()
    relevant = {fp: code for fp, code in already_written.items() if fp in deps}

    if not relevant and already_written:
        # Fallback: types + data/constants dirs — almost always needed
        relevant = {
            fp: code for fp, code in already_written.items()
            if "types/" in fp or "data/" in fp or "constants" in fp
        }

    # Signature-only mode: truncate each dep to first ~1500 chars of non-blank lines
    if file_path in signature_only_set:
        relevant = {
            fp: "\n".join(
                l for l in code.splitlines()
                if l.strip() and not l.strip().startswith("//")
                   and not l.strip().startswith("#")
            )[:1500]
            for fp, code in already_written.items()
            if "types/" in fp or "data/" in fp or "hooks/" in fp
               or "constants" in fp or "utils/" in fp
        }

    if not relevant:
        return ""

    if file_path in signature_only_set:
        label = "API reference (signatures only — full implementations omitted)"
    elif deps:
        label = "Dependencies (already implemented — for import reference)"
    else:
        label = "Shared types & constants (for import reference)"

    block = f"### {label}\n"
    for fp, code in relevant.items():
        block += f"\n#### {fp}\n```\n{code}\n```\n"
    return block


# ════════════════════════════════════════════════════════════════════════════
# Per-file implementation
# ════════════════════════════════════════════════════════════════════════════

def implement_file(
    agent_cfg:          dict,
    techstack:          str,
    spec:               str,
    stub:               dict,
    task:               dict | None,
    already_written:    dict[str, str],
    signature_only_set: set[str],
) -> dict:
    file_path  = stub["file_path"]
    task_block = _build_task_block(task)
    ctx_block  = _build_context_block(
        file_path, task, already_written, signature_only_set
    )

    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"{ctx_block}\n"
        f"{task_block}\n"
        f"### Stub file to implement: {file_path}\n"
        f"```\n{stub['code']}\n```"
    )

    approx_tokens = len(user_msg) // 4
    if approx_tokens > 28000:
        print(
            f"{TAG} ⚠ Large prompt for {file_path}: ~{approx_tokens:,} tokens "
            f"(limit ~32k). Response may be truncated.",
            file=sys.stderr,
        )

    print(f"{TAG}   → Implementing {file_path} …")
    system = build_system_prompt_per_file(techstack)
    raw    = call_executor(agent_cfg, system, user_msg)
    result = _parse_json(raw, file_path)

    # Normalise: some models return {"files": [...]} even in per-file mode
    if "files" in result and isinstance(result["files"], list):
        for entry in result["files"]:
            if entry.get("file_path") == file_path:
                return entry
        return result["files"][0]
    return result


# ════════════════════════════════════════════════════════════════════════════
# Single-call fallback
# ════════════════════════════════════════════════════════════════════════════

def implement_all_single_call(
    agent_cfg:   dict,
    techstack:   str,
    spec:        str,
    stub_files:  list,
    instructions: str,
) -> list[dict]:
    """All files in one request (no plan). Faster but lower quality for large specs."""
    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold (stub files to implement)\n\n"
        f"{json.dumps(stub_files, indent=2)}"
    )
    model = agent_cfg.get("model", "?")
    print(f"{TAG} Single-call mode [{model}] — {len(stub_files)} file(s) …")
    system = build_system_prompt_single(techstack, instructions)
    raw    = call_executor(agent_cfg, system, user_msg)
    result = _parse_json(raw, "single-call")
    return result.get("files", [])


# ════════════════════════════════════════════════════════════════════════════
# Ordering
# ════════════════════════════════════════════════════════════════════════════

def order_stubs(stub_files: list[dict], plan: dict | None) -> list[dict]:
    """Sort stub files by implementation_order from plan, or keep scaffold order."""
    if not plan:
        return stub_files
    order     = plan.get("implementation_order", [])
    order_map = {fp: i for i, fp in enumerate(order)}
    return sorted(stub_files, key=lambda f: order_map.get(f["file_path"], 999))


# ════════════════════════════════════════════════════════════════════════════
# Delta helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_restored_files(only_set: set[str], cfg: dict) -> dict[str, str]:
    """
    Read already-restored src/ files into memory so they can be used as import
    context for the executor in delta mode.
    Loads only files NOT in only_set (i.e. unaffected/restored ones).
    Scans extensions based on techstack.
    """
    techstack = get_techstack(cfg)
    _EXT_MAP: dict[str, list[str]] = {
        "react-vite-ts":   ["*.ts", "*.tsx"],
        "nextjs-ts":       ["*.ts", "*.tsx"],
        "vue-vite-ts":     ["*.ts", "*.vue"],
        "svelte-ts":       ["*.ts", "*.svelte"],
        "python-fastapi":  ["*.py"],
        "python-flask":    ["*.py"],
        "node-express-ts": ["*.ts", "*.js"],
        "django":          ["*.py"],
    }
    patterns = _EXT_MAP.get(techstack, ["*.ts", "*.tsx", "*.py", "*.js"])

    restored: dict[str, str] = {}
    src_dir = ROOT / "src"
    if not src_dir.exists():
        return restored

    for pattern in patterns:
        for p in sorted(src_dir.rglob(pattern)):
            rel = str(p.relative_to(ROOT))
            if rel not in only_set:
                try:
                    restored[rel] = p.read_text()
                except Exception:
                    pass
    return restored


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="executor_agent: implement src/ files from scaffold stubs"
    )
    parser.add_argument(
        "--use-plan", action="store_true",
        help="Inject scaffold/plan.json as per-file implementation guidance",
    )
    parser.add_argument(
        "--only-files", default="",
        help="Comma-separated src/ paths to implement (delta mode). "
             "All other stubs are skipped — assumed already restored by harness.",
    )
    args = parser.parse_args()

    cfg        = load_config()
    agent_cfg  = get_agent_cfg(cfg, "executor")
    techstack  = get_techstack(cfg)
    sig_only   = _load_signature_only_set(cfg)

    print(f"{TAG} Techstack : {techstack}")
    print(f"{TAG} Agent     : {agent_cfg.get('model','?')} "
          f"({agent_cfg.get('provider','?')})")

    spec      = _load_spec()
    scaffold  = json.loads(SCAFFOLD_JSON.read_text())
    instrs    = INSTRUCTIONS.read_text() if INSTRUCTIONS.exists() else ""

    all_stubs = [f for f in scaffold["files"] if not f.get("is_test")]

    # ── Delta filtering ───────────────────────────────────────────────────────
    only_set: set[str] = set()
    if args.only_files.strip():
        only_set   = {fp.strip() for fp in args.only_files.split(",") if fp.strip()}
        stub_files = [f for f in all_stubs if f["file_path"] in only_set]
        skipped    = [f["file_path"] for f in all_stubs if f["file_path"] not in only_set]
        print(f"{TAG} Delta mode — {len(stub_files)} file(s) to implement, "
              f"{len(skipped)} unaffected (skipped).")
        for fp in skipped:
            print(f"{TAG}   SKIP (unaffected): {fp}")
    else:
        stub_files = all_stubs

    # ── Load plan if requested ────────────────────────────────────────────────
    plan:       dict | None      = None
    task_index: dict[str, dict]  = {}

    if args.use_plan:
        if not PLAN_JSON.exists():
            print(f"{TAG} ERROR: --use-plan set but scaffold/plan.json not found.")
            print(f"{TAG}        Run 03b_plan.py first.")
            sys.exit(1)
        plan       = json.loads(PLAN_JSON.read_text())
        task_index = {t["file_path"]: t for t in plan.get("tasks", [])}
        print(f"{TAG} Plan loaded — {len(task_index)} tasks, "
              f"order: {plan.get('implementation_order', [])}")
    else:
        print(f"{TAG} No plan — using single-call mode.")

    # ── Execute ───────────────────────────────────────────────────────────────
    written:      list[str] = []
    failed_files: list[str] = []

    if plan:
        # Per-file generation in plan order
        ordered = order_stubs(stub_files, plan)

        # Seed already_written with restored (unaffected) files in delta mode
        already_written: dict[str, str] = (
            _load_restored_files(only_set, cfg) if only_set else {}
        )
        if already_written:
            print(f"{TAG} Import context seeded with {len(already_written)} "
                  f"restored file(s).")

        for stub in ordered:
            fp   = stub["file_path"]
            task = task_index.get(fp)
            try:
                entry = implement_file(
                    agent_cfg, techstack, spec, stub,
                    task, already_written, sig_only,
                )
            except Exception as e:
                print(f"{TAG} FAILED to implement {fp}: {e}", file=sys.stderr)
                failed_files.append(fp)
                continue

            out_path = ROOT / fp
            if not str(out_path.resolve()).startswith(str((ROOT / "src").resolve())):
                print(f"{TAG} SKIP (outside src/): {fp}")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(entry["code"])
            already_written[fp] = entry["code"]
            written.append(fp)
            print(f"{TAG} WROTE {fp}")

    else:
        # Single-call: send only affected stubs
        try:
            entries = implement_all_single_call(
                agent_cfg, techstack, spec, stub_files, instrs
            )
        except Exception as e:
            print(f"{TAG} FAILED single-call generation: {e}", file=sys.stderr)
            failed_files.extend([f["file_path"] for f in stub_files])
            entries = []

        for entry in entries:
            fp       = entry["file_path"]
            out_path = ROOT / fp
            if not str(out_path.resolve()).startswith(str((ROOT / "src").resolve())):
                print(f"{TAG} SKIP (outside src/): {fp}")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(entry["code"])
            written.append(fp)
            print(f"{TAG} WROTE {fp}")

    # ── Write implementation record ───────────────────────────────────────────
    mode = "per-file-with-plan" if plan else "single-call"
    if only_set:
        mode += "-delta"

    skipped_delta = sorted(
        {f["file_path"] for f in all_stubs} - only_set
    ) if only_set else []

    IMPL_RECORD.write_text(json.dumps({
        "agent":         agent_cfg.get("model", "unknown"),
        "provider":      agent_cfg.get("provider", "unknown"),
        "techstack":     techstack,
        "mode":          mode,
        "files":         written,
        "skipped_delta": skipped_delta,
        "failed_files":  failed_files,
    }, indent=2))

    if failed_files:
        print(
            f"{TAG} Done with {len(written)} file(s) written, "
            f"{len(failed_files)} failed: {failed_files}",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"{TAG} Done — {len(written)} file(s) written.")


if __name__ == "__main__":
    main()
