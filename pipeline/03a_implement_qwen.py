"""
pipeline/03a_implement_qwen.py
Step 3a — Qwen 3.6 Plus as EXECUTOR (per-file generation).

Modes:
    Default (--only-qwen, no plan):
        One API call per stub file → src/<file>
        Falls back to original single-call behaviour if only 1 file.

    With GLM plan (--use-glm-plan):
        Reads scaffold/glm_plan.json produced by 03b_implement_glm.py.
        For each file, injects the matching GLM task (sub_tasks, gotchas,
        tailwind_hints, depends_on) into the prompt before generating.
        Files are generated in implementation_order from the plan.

Writes:
    src/**  (non-test files only)
    scaffold/impl_qwen.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import httpx
from pathlib import Path

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "qwen/qwen3.6-plus"

ROOT          = Path(__file__).parent.parent
SPEC_PATH     = ROOT / "spec.md"
SCAFFOLD_JSON = ROOT / "scaffold" / "scaffold.json"
INSTRUCTIONS  = ROOT / "scaffold" / "instructions_qwen.txt"
GLM_PLAN      = ROOT / "scaffold" / "glm_plan.json"
IMPL_RECORD   = ROOT / "scaffold" / "impl_qwen.json"
PIPELINE_CTX  = ROOT / "scaffold" / "pipeline_context.json"

# ── System prompts ────────────────────────────────────────────────────────────

def _load_spec() -> str:
    compressed = ROOT / "scaffold" / "spec_compressed.md"
    return compressed.read_text() if compressed.exists() else (ROOT / "spec.md").read_text()

def build_system_prompt_single(instructions: str) -> str:
    """Original single-call prompt — used when no GLM plan is available."""
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


def build_system_prompt_per_file() -> str:
    """Per-file generation prompt — one file per API call."""
    return """\
You are a senior TypeScript/React developer implementing ONE source file.
You will receive:
1. The technical spec (spec.md) for full context
2. The stub for the SINGLE file you must implement
3. (Optional) A task plan produced by a senior architect — follow it carefully

Your task:
- Implement the function body / component body for this ONE file only.
- Return a JSON object with this EXACT schema:
  {
    "file_path": "src/hooks/useSensorData.ts",
    "code": "<full file content — complete, runnable TypeScript>"
  }
- TypeScript strict mode — no `any`.
- Tailwind only — no inline styles, no CSS modules.
- The code field must be the COMPLETE file (imports + types + implementation).
- Output raw JSON only. Absolutely no markdown fences or explanation text.
"""


# ── API call ──────────────────────────────────────────────────────────────────

def _extract_chat_text_response(data: dict, label: str) -> str:
    choice = data["choices"][0]
    msg = choice["message"]

    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    finish_reason = choice.get("finish_reason")

    if tool_calls:
        raise RuntimeError(
            f"{label} returned tool_calls instead of text: {tool_calls}"
        )

    if not content or not content.strip():
        raise RuntimeError(
            f"{label} returned empty content. "
            f"finish_reason={finish_reason}, message={msg}"
        )

    return content.strip()


def _call_qwen(system: str, user_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.15,
        "max_tokens": 32768,
    }

    last_error = None

    with httpx.Client(timeout=180) as client:
        for attempt in range(2):
            try:
                r = client.post(OPENROUTER_URL, headers=headers, json=payload)
                r.raise_for_status()

                try:
                    data = r.json()
                except json.JSONDecodeError as e:
                    body_preview = r.text[:1000] if r.text else "<empty body>"
                    raise RuntimeError(
                        f"OpenRouter returned non-JSON response: {e}\n"
                        f"Response body (first 1000 chars):\n{body_preview}"
                    ) from e

                usage = data.get("usage", {})
                prompt_t = usage.get("prompt_tokens", "?")
                completion_t = usage.get("completion_tokens", "?")
                print(f"[qwen] Tokens: prompt={prompt_t}, completion={completion_t}")

                return _extract_chat_text_response(data, label="Qwen")

            except httpx.HTTPStatusError as e:
                body_preview = e.response.text[:1000] if e.response is not None and e.response.text else "<empty body>"
                last_error = RuntimeError(
                    f"HTTP error from OpenRouter: {e}\n"
                    f"Response body (first 1000 chars):\n{body_preview}"
                )
                print(f"[qwen] {last_error}", file=sys.stderr)

            except (httpx.HTTPError, RuntimeError) as e:
                last_error = e
                print(f"[qwen] {e}", file=sys.stderr)

            if attempt == 0:
                print("[qwen] Retrying in 3s …", file=sys.stderr)
                time.sleep(3)

    raise RuntimeError(f"Qwen call failed after retries: {last_error}")

# ── JSON extraction ───────────────────────────────────────────────────────────

def _parse_json(raw: str, label: str) -> dict:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[03a] JSON parse failed for {label}: {e}", file=sys.stderr)
            print(f"[03a] Raw (first 500):\n{raw[:500]}", file=sys.stderr)
            sys.exit(1)
    print(f"[03a] No JSON found in response for {label}.", file=sys.stderr)
    sys.exit(1)


# ── Per-file generation ───────────────────────────────────────────────────────

def _build_task_block(task: dict | None) -> str:
    """Format GLM plan task as a prompt section."""
    if not task:
        return ""
    lines = ["### Implementation plan from architect (GLM 5.1)\n"]
    lines.append(f"**Role:** {task.get('role', '')}\n")
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
    hints = task.get("tailwind_hints")
    if hints:
        lines.append(f"**Tailwind hints:** {hints}\n")
    return "\n".join(lines)

def implement_file(
    spec: str,
    stub: dict,
    task: dict | None,
    already_written: dict[str, str],
) -> dict:
    file_path  = stub["file_path"]
    task_block = _build_task_block(task)

    # ── Context injection: deps only, not all already_written ────────────────
    deps     = set(task.get("depends_on", [])) if task else set()
    relevant = {fp: code for fp, code in already_written.items() if fp in deps}

    if not relevant and already_written:
        # Fallback: types + constants only (small, always needed)
        relevant = {fp: code for fp, code in already_written.items()
                    if "types/" in fp or "data/" in fp}

    # ── Signature-only for App.tsx / main.tsx ─────────────────────────────────
    # These files only need to know hook/component public APIs to import them.
    # Injecting full implementations (200+ lines each) pushes prompt past 32k
    # tokens and causes OpenRouter to return a truncated (non-JSON) response.
    _SIGNATURE_ONLY_FOR = {"src/App.tsx", "src/main.tsx"}
    if file_path in _SIGNATURE_ONLY_FOR:
        relevant = {
            fp: "\n".join(
                l for l in code.splitlines()
                if l.strip() and not l.strip().startswith("//")
            )[:1500]   # cap at ~1500 chars per file
            for fp, code in already_written.items()
            if "types/" in fp or "data/" in fp or "hooks/" in fp
        }

    context_block = ""
    if relevant:
        if file_path in _SIGNATURE_ONLY_FOR:
            label = "API reference (signatures only — full implementations omitted)"
        elif deps:
            label = "Dependencies (already implemented — for import reference)"
        else:
            label = "Shared types & constants (for import reference)"
        context_block = f"### {label}\n"
        for fp, code in relevant.items():
            context_block += f"\n#### {fp}\n```typescript\n{code}\n```\n"

    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"{context_block}\n"
        f"{task_block}\n"
        f"### Stub file to implement: {file_path}\n"
        f"```typescript\n{stub['code']}\n```"
    )

    # ── Prompt size guard ─────────────────────────────────────────────────────
    approx_tokens = len(user_msg) // 4
    if approx_tokens > 28000:
        print(f"[03a] ⚠ Large prompt for {file_path}: ~{approx_tokens:,} tokens "
              f"(limit ~32k). Response may be truncated.", file=sys.stderr)
 
    print(f"[03a]   → Implementing {file_path} …")
    raw    = _call_qwen(build_system_prompt_per_file(), user_msg)
    result = _parse_json(raw, file_path)
 
    if "files" in result and isinstance(result["files"], list):
        for entry in result["files"]:
            if entry.get("file_path") == file_path:
                return entry
        return result["files"][0]
    return result

# ── Ordering helpers ──────────────────────────────────────────────────────────

def order_stubs(stub_files: list[dict], plan: dict | None) -> list[dict]:
    """Sort stub files by implementation_order from plan, or keep scaffold order."""
    if not plan:
        return stub_files
    order = plan.get("implementation_order", [])
    order_map = {fp: i for i, fp in enumerate(order)}
    return sorted(stub_files, key=lambda f: order_map.get(f["file_path"], 999))


# ── Single-call fallback (original behaviour) ─────────────────────────────────

def implement_all_single_call(spec: str, stub_files: list, instructions: str) -> list[dict]:
    """Original single-call mode — all files in one request."""
    user_msg = (
        f"### spec.md\n\n{spec}\n\n"
        f"### scaffold (stub files to implement)\n\n"
        f"{json.dumps(stub_files, indent=2)}"
    )
    print("[03a] Calling Qwen 3.6 Plus (single-call mode) …")
    raw = _call_qwen(build_system_prompt_single(instructions), user_msg)
    result = _parse_json(raw, "single-call")
    return result.get("files", [])


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_restored_files(only_set: set[str]) -> dict[str, str]:
    """
    Read already-restored src/ files (copied by harness from prev_src/) into
    memory so they can be used as import context for Qwen in delta mode.
    Loads only files NOT in only_set (i.e. the unaffected/restored ones).
    """
    restored: dict[str, str] = {}
    src_dir = ROOT / "src"
    if not src_dir.exists():
        return restored
    all_src = sorted(src_dir.rglob("*.ts")) + sorted(src_dir.rglob("*.tsx"))
    for p in all_src:
        rel = str(p.relative_to(ROOT))
        if rel not in only_set:
            try:
                restored[rel] = p.read_text()
            except Exception:
                pass
    return restored

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-glm-plan", action="store_true",
        help="Inject scaffold/glm_plan.json as per-file implementation guidance",
    )
    
    parser.add_argument(
    "--only-files",
    default="",
    help="Comma-separated src/ paths to implement (delta mode). "
         "All other stubs are skipped — assumed already restored by harness.",
    )

    args = parser.parse_args()

    spec      = _load_spec()
    scaffold  = json.loads(SCAFFOLD_JSON.read_text())
    instrs    = INSTRUCTIONS.read_text() if INSTRUCTIONS.exists() else ""

    all_stubs = [f for f in scaffold["files"] if not f.get("is_test")]

    # ── Delta filtering ───────────────────────────────────────────────────────
    only_set: set[str] = set()
    if args.only_files.strip():
        only_set  = {fp.strip() for fp in args.only_files.split(",") if fp.strip()}
        stub_files = [f for f in all_stubs if f["file_path"] in only_set]
        skipped    = [f["file_path"] for f in all_stubs if f["file_path"] not in only_set]
        print(f"[03a] Delta mode — {len(stub_files)} file(s) to implement, "
              f"{len(skipped)} unaffected (skipped).")
        for fp in skipped:
            print(f"[03a]   SKIP (unaffected): {fp}")
    else:
        stub_files = all_stubs

    # ── Load GLM plan if requested ────────────────────────────────────────────
    plan: dict | None = None
    task_index: dict[str, dict] = {}

    if args.use_glm_plan:
        if not GLM_PLAN.exists():
            print("[03a] ERROR: --use-glm-plan set but scaffold/glm_plan.json not found.")
            print("             Run 03b_implement_glm.py first.")
            sys.exit(1)
        plan = json.loads(GLM_PLAN.read_text())
        task_index = {t["file_path"]: t for t in plan.get("tasks", [])}
        print(f"[03a] GLM plan loaded — {len(task_index)} tasks, "
              f"order: {plan.get('implementation_order', [])}")
    else:
        print("[03a] No GLM plan — using single-call mode.")

    # ── Choose execution mode ─────────────────────────────────────────────────
    written: list[str] = []

    if plan:
        # Per-file generation in plan order
        ordered = order_stubs(stub_files, plan)
        
        # In delta mode, seed already_written with restored (unaffected) files
        # so Qwen has full import context without re-implementing them.
        already_written: dict[str, str] = (
            _load_restored_files(only_set) if only_set else {}
        )
        if already_written:
            print(f"[03a] Import context seeded with {len(already_written)} "
                  f"restored file(s).")

        for stub in ordered:
            fp   = stub["file_path"]
            task = task_index.get(fp)
            try:
                entry = implement_file(spec, stub, task, already_written)
            except SystemExit:
                print(f"[03a] FAILED to implement {fp}, continuing.",
                      file=sys.stderr)
                continue

            out_path = ROOT / fp
            if not str(out_path).startswith(str(ROOT / "src")):
                print(f"[03a] SKIP (outside src/): {fp}")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(entry["code"])
            already_written[fp] = entry["code"]
            written.append(fp)
            print(f"[03a] WROTE {fp}")

    else:
        # Single-call: only send affected stubs, Qwen doesn't need the rest
        entries = implement_all_single_call(spec, stub_files, instrs)
        for entry in entries:
            fp = entry["file_path"]
            out_path = ROOT / fp
            if not str(out_path).startswith(str(ROOT / "src")):
                print(f"[03a] SKIP (outside src/): {fp}")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(entry["code"])
            written.append(fp)
            print(f"[03a] WROTE {fp}")

    mode = "per-file-with-glm-plan" if plan else "single-call"
    if only_set:
        mode += "-delta"

    # skipped_delta: stubs that were in scaffold but not re-implemented this run
    skipped_delta = sorted(
        {f["file_path"] for f in all_stubs} - only_set
    ) if only_set else []
    
    IMPL_RECORD.write_text(json.dumps({
        "model":         "qwen",
        "mode":          mode,
        "files":         written,
        "skipped_delta": skipped_delta,
    }, indent=2))
    print(f"[03a] Done — {len(written)} files written.")


if __name__ == "__main__":
    main()
