"""
toolkits/devops_mlops/system_explorer.py
=========================================
Persistent system knowledge accumulator for unfamiliar internal platforms.

When you join a project with no codebase, no docs, and only credentials to
internal platforms (Grafana, Tableau, FortiPAM, Redshift, AWS...),
system_explorer helps you build a structured understanding incrementally
across multiple sessions.

Each session: you provide text + files/screenshots → LLM asks questions
(never answers) → knowledge accumulates → system map grows over time.

────────────────────────────────────────────────────────────────
How it works
────────────────────────────────────────────────────────────────

  Session N-1: (previous run, already ended)
    system_map.md exists — LLM reads it as starting context.

  Session N: (this run)
    1. Load system_map.md from previous session (if exists)
    2. You provide text + files (screenshots, exports, configs)
    3. LLM acknowledges what it learned, asks 3-5 targeted questions
    4. You answer, provide more files → repeat
    5. When you type /end: session ends automatically
       → system_map.md is updated (OVERWRITE with new version)
       → session_log.json is appended
    6. Next session starts fresh with the updated system_map.md

────────────────────────────────────────────────────────────────
LLM behavior (enforced by system prompt)
────────────────────────────────────────────────────────────────

  The LLM does NOT diagnose, explain, or solve problems.
  It ONLY:
    - Acknowledges what it learned from your input (1-2 lines)
    - Asks 3-5 targeted questions to fill gaps in its understanding
    - Suggests export when it has enough to write a useful system map

  Focus areas for questions:
    - Data lineage (who writes what, who reads what)
    - Service connections (what calls what, via what protocol)
    - Platform quirks (non-obvious behaviors of internal systems)
    - Missing links in known flows

────────────────────────────────────────────────────────────────
Input pattern (same as ckey.py)
────────────────────────────────────────────────────────────────

  Text first, files after. Two ways to attach files:
    1. Type/paste text → /done → drag files into terminal
    2. Drag files directly into text prompt (auto-detected as paths)

  /done   — end text input, proceed to file attachment prompt
  /end    — end entire session, write system_map.md and exit
  /export — write system_map.md now, then continue session

────────────────────────────────────────────────────────────────
Artifacts
────────────────────────────────────────────────────────────────

  <artifact_root>/system_explorer/<client>/<project>/
    system_map.md        (short-term, OVERWRITE on session end)
                         Human-readable system map. Grows each session.
                         LLM reads this at start of next session.

    session_log.json     (long-term, APPEND)
                         One entry per session: timestamp, turn count,
                         input summary, version number.

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python system_explorer.py --client client-a --project icp-alert
  python system_explorer.py --client client-a --project icp-alert --model claude-opus-4-7

  During session:
    /done    end text input for this turn, go to file prompt
    /end     end session, write system_map.md, exit
    /export  write system_map.md now, continue session
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
MULTILINE_SENTINEL = "/done"
END_SESSION_CMD    = "/end"
EXPORT_CMD         = "/export"
DEFAULT_MODEL      = os.environ.get("EXPLORER_MODEL", "claude-opus-4-7")
MAX_TOKENS         = int(os.environ.get("EXPLORER_MAX_TOKENS", "4096"))
PREVIEW_CHARS      = 600

MAX_TEXT_FILE_BYTES   = 200_000
MAX_BINARY_FILE_BYTES = 10 * 1024 * 1024

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".txt", ".md", ".json", ".yaml", ".yml",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
    ".sh", ".toml", ".ini", ".csv", ".xml", ".log",
    ".env", ".sql", ".conf", ".rst", ".cfg",
})


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    base     = Path(override) if override else _REPO_ROOT.parent / "outputs" / "devops_mlops"
    return base


def _explorer_dir(client: str, project: str) -> Path:
    return _devops_artifact_root() / "system_explorer" / client / project


def _map_path(client: str, project: str) -> Path:
    return _explorer_dir(client, project) / "system_map.md"


def _log_path(client: str, project: str) -> Path:
    return _explorer_dir(client, project) / "session_log.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic client (same as ckey.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> Any:
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError:
        print("[system_explorer][error] anthropic not installed — pip install anthropic")
        sys.exit(2)

    api_key = (
        os.environ.get("CKEY_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        print("[system_explorer][error] Set CKEY_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(2)

    base_url = "https://ckey.vn"
    return Anthropic(api_key=api_key, base_url=base_url)


# ─────────────────────────────────────────────────────────────────────────────
# File → message block (same pattern as ckey.py)
# ─────────────────────────────────────────────────────────────────────────────

def _file_to_block(file_path: str) -> dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext       = path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(path.name)

    # Text file
    if ext in TEXT_EXTENSIONS or (mime_type and mime_type.startswith("text/")):
        size = path.stat().st_size
        if size > MAX_TEXT_FILE_BYTES:
            raise ValueError(f"File too large: {path.name} ({size} bytes)")
        text = path.read_text(encoding="utf-8", errors="replace")
        return {"type": "text", "text": f"File: {path.name}\n\n{text}"}

    # Image
    if mime_type and mime_type.startswith("image/"):
        size = path.stat().st_size
        if size > MAX_BINARY_FILE_BYTES:
            raise ValueError(f"Image too large: {path.name} ({size} bytes)")
        data = base64.b64encode(path.read_bytes()).decode()
        return {
            "type":   "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data},
        }

    # PDF
    if mime_type == "application/pdf":
        size = path.stat().st_size
        if size > MAX_BINARY_FILE_BYTES:
            raise ValueError(f"PDF too large: {path.name} ({size} bytes)")
        data = base64.b64encode(path.read_bytes()).decode()
        return {
            "type":   "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        }

    raise ValueError(f"Unsupported file type: {path.name} ({mime_type or 'unknown'})")


def _build_user_message(text: str, file_paths: list[str]) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    if text.strip():
        blocks.append({"type": "text", "text": text.strip()})
    elif file_paths:
        blocks.append({"type": "text", "text": "Please review the attached files."})

    for fp in file_paths:
        try:
            blocks.append(_file_to_block(fp))
            print(f"  Attached: {Path(fp).name}")
        except Exception as exc:
            print(f"  [warn] Could not attach {fp}: {exc}")

    if not blocks:
        raise ValueError("No content — provide text or files.")

    return {"role": "user", "content": blocks}


def _build_assistant_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


# ─────────────────────────────────────────────────────────────────────────────
# File path detection (same as ckey.py)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_file_paths(raw: str) -> list[str] | None:
    """
    If raw text looks like drag-dropped file paths (all tokens are valid files),
    return them. Otherwise return None so raw is treated as text prompt.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if not tokens:
        return None

    valid:   list[str] = []
    invalid: list[str] = []
    for t in tokens:
        p = Path(t.strip()).expanduser()
        if p.exists() and p.is_file():
            valid.append(str(p.resolve()))
        else:
            invalid.append(t)

    if valid and not invalid:
        return valid
    return None


def _prompt_for_files(prompt_text: str = "Files (drag-drop or paths, Enter to skip): ") -> list[str]:
    while True:
        try:
            raw = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            return []
        if not raw:
            return []
        try:
            tokens = shlex.split(raw)
        except ValueError:
            print("  Could not parse paths — try again.")
            continue

        valid:   list[str] = []
        invalid: list[str] = []
        for t in tokens:
            p = Path(t.strip()).expanduser()
            if p.exists() and p.is_file():
                valid.append(str(p.resolve()))
            else:
                invalid.append(t)

        if valid and not invalid:
            return valid
        if not valid and invalid:
            print("  No valid file paths found — treated as text? Skip files.")
            return []
        if invalid:
            print(f"  Invalid: {invalid} — try again or Enter to skip.")


# ─────────────────────────────────────────────────────────────────────────────
# Multiline prompt (same as ckey.py)
# ─────────────────────────────────────────────────────────────────────────────

def _prompt_multiline(intro: str, allow_empty_exit: bool = False) -> str:
    print(intro)
    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if allow_empty_exit and not lines and not line.strip():
            return ""
        stripped = line.strip()
        if stripped == MULTILINE_SENTINEL:
            break
        if stripped in (END_SESSION_CMD, EXPORT_CMD):
            # Pass sentinel through so caller can detect
            return stripped
        lines.append(line)
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system(client: str, project: str) -> str:
    return f"""\
You are a system knowledge accumulator for the project "{project}" (client: "{client}").

Your ONLY job is to help the user build a structured understanding of an unfamiliar
internal infrastructure system. You do NOT diagnose problems, suggest fixes, or explain
root causes. You ask questions.

────────────────────────────────────────────────────────────────
Every response MUST have exactly these three parts:
────────────────────────────────────────────────────────────────

1. LEARNED (1-3 bullet points)
   What new facts did you extract from the user's input this turn?
   Be specific: service names, table names, URLs, quirks, flow directions.
   If nothing new was learned, say so briefly.

2. QUESTIONS (3-5 numbered questions)
   Ask targeted questions to fill the most important gaps.
   Priority order:
     a) Data lineage gaps — who writes to what, who reads from what
     b) Service connections — what calls what, via what API/protocol
     c) Platform quirks — non-obvious behaviors (e.g. Grafana writes silenced
        alerts to annotation table without a silenced flag)
     d) Missing links — a known service with no known input or output yet

   Rules for questions:
   - One specific question per line (not compound questions)
   - Reference specific things you already know when possible
     e.g. "You mentioned Lambda writes to annotation_updated —
           what triggers this Lambda? EventBridge schedule, API call, or other?"
   - Do NOT ask generic questions like "How does the system work?"

3. EXPORT READINESS
   Assess whether you have enough to write a useful system map.
   Format: "EXPORT: READY — <reason>" or "EXPORT: NOT YET — missing: <what>"

   Suggest export when ALL of these are true:
   - At least 2 systems/services identified with names
   - At least 1 data flow with source → transform → sink identified
   - At least 1 platform quirk or non-obvious behavior documented

────────────────────────────────────────────────────────────────
When you receive a system_map.md at the start:
────────────────────────────────────────────────────────────────
This is your accumulated knowledge from previous sessions.
Treat it as ground truth for what you already know.
Do NOT re-ask questions already answered there.
Focus your questions on gaps and ambiguities in the existing map.

────────────────────────────────────────────────────────────────
IMPORTANT:
────────────────────────────────────────────────────────────────
- Never diagnose. Never explain root causes. Never suggest fixes.
- Never say "I think the problem is..." or "This could be caused by..."
- If the user shares an error or incident, acknowledge it factually
  and ask questions about the surrounding system, not about the error itself.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Export prompt — generate system_map.md
# ─────────────────────────────────────────────────────────────────────────────

_EXPORT_SYSTEM = """\
You are writing a structured system map document based on accumulated knowledge
from an ongoing investigation of an internal infrastructure system.

Write a clear, factual markdown document. Include only what is known or
reasonably inferred from the conversation. Clearly mark uncertain items.

Structure:
  # System Map: <project name>
  _Version N — <date>_

  ## Systems & Services
  For each identified system: name, URL if known, purpose, known version/type.

  ## Data Flows
  Directed flows in format: Source → [Transform/Service] → Sink
  Include schedule/trigger if known.

  ## Platform Quirks
  Non-obvious behaviors that affect how the system works.
  These are the most valuable findings — be specific.

  ## Service Connections
  Which services call which, via what protocol/API.

  ## Known Tables / Schemas
  Database tables, views, schemas identified. Include what writes/reads each.

  ## Open Questions
  Gaps that remain unresolved. Number them.

  ## Investigation Notes
  Anything that doesn't fit above but is useful context.

Be concise. Use bullet points. Do not pad with generic observations.
Mark uncertain items with _(unconfirmed)_.
"""


def _generate_system_map(
    history:  list[dict[str, Any]],
    client:   Any,
    model:    str,
    project:  str,
    version:  int,
) -> str:
    """Call LLM to synthesize conversation history into system_map.md content."""
    export_user = (
        f"Based on everything discussed, write the system map for project '{project}'. "
        f"This is version {version}. Today: {_now_display()}."
    )
    export_messages = history + [{"role": "user", "content": [
        {"type": "text", "text": export_user}
    ]}]

    print("\n  Generating system map …")
    try:
        raw = client.messages.with_raw_response.create(
            model      = model,
            max_tokens = 4096,
            system     = _EXPORT_SYSTEM,
            messages   = export_messages,
        )
        body = json.loads(raw.text)
        # Extract text from Anthropic response
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        return body.get("content", [{}])[0].get("text", "")
    except Exception as exc:
        print(f"  [warn] Export generation failed: {exc}")
        return f"# System Map: {project}\n\n_Export failed: {exc}_\n"


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _ask(
    client:  Any,
    model:   str,
    system:  str,
    history: list[dict[str, Any]],
) -> str:
    try:
        raw = client.messages.with_raw_response.create(
            model      = model,
            max_tokens = MAX_TOKENS,
            system     = system,
            messages   = history,
        )
        body = json.loads(raw.text)
        for block in body.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        # OpenAI-style fallback
        choices = body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return msg.get("content", "")
        return f"[no text in response] raw: {raw.text[:300]}"
    except Exception as exc:
        return f"[LLM error: {exc}]"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_map(client_name: str, project: str, content: str, version: int) -> Path:
    path = _map_path(client_name, project)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"<!-- system_map: {project} | client: {client_name} "
        f"| version: {version} | updated: {_now_display()} -->\n\n"
    )
    path.write_text(header + content.strip() + "\n", encoding="utf-8")
    return path


def _append_session_log(
    client_name: str,
    project:     str,
    version:     int,
    turn_count:  int,
    input_summary: str,
    exported:    bool,
) -> None:
    log = _log_path(client_name, project)
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            data     = json.loads(log.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    entry: dict[str, Any] = {
        "session_at":    _now_iso(),
        "client":        client_name,
        "project":       project,
        "version":       version,
        "turn_count":    turn_count,
        "exported":      exported,
        "input_summary": input_summary[:200],
    }
    existing.append(entry)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"entries": existing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _get_current_version(client_name: str, project: str) -> int:
    """Read version from most recent session log entry."""
    log = _log_path(client_name, project)
    if not log.exists():
        return 0
    try:
        data    = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
        if entries:
            return entries[-1].get("version", 0)
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main session loop
# ─────────────────────────────────────────────────────────────────────────────

def run_session(client_name: str, project: str, model: str) -> None:
    anth_client = _get_client()
    system      = _build_system(client_name, project)
    history:    list[dict[str, Any]] = []
    turn_count  = 0
    all_inputs: list[str] = []
    exported    = False
    version     = _get_current_version(client_name, project) + 1

    print()
    print("=" * 60)
    print(f"  SYSTEM EXPLORER")
    print(f"  Client:  {client_name}")
    print(f"  Project: {project}")
    print(f"  Model:   {model}")
    print(f"  Version: {version}")
    print("=" * 60)
    print()
    print(f"  Commands:")
    print(f"    {MULTILINE_SENTINEL}  — end text input, go to file prompt")
    print(f"    {END_SESSION_CMD}   — end session, write system_map.md, exit")
    print(f"    {EXPORT_CMD} — write system_map.md now, then continue")
    print()

    # ── Load existing system map into first user message ──────────────────────
    map_p = _map_path(client_name, project)
    if map_p.exists():
        existing_map = map_p.read_text(encoding="utf-8")
        print(f"  Loaded existing system map ({len(existing_map):,} chars)")
        print(f"  LLM will continue from where previous session left off.")
        print()
        # Inject as first user message so LLM has full context
        history.append(_build_user_message(
            f"Here is the system map built so far from previous sessions:\n\n{existing_map}\n\n"
            f"Continue building on this. I will now provide new information.",
            [],
        ))
        # Get LLM acknowledgment of existing map
        print("  Resuming — LLM reviewing previous session …")
        reply = _ask(anth_client, model, system, history)
        history.append(_build_assistant_message(reply))
        print()
        print(reply[:PREVIEW_CHARS] + ("…" if len(reply) > PREVIEW_CHARS else ""))
        print()
    else:
        print(f"  No existing map — starting fresh.")
        print()

    # ── Main turn loop ────────────────────────────────────────────────────────
    while True:
        intro = (
            f"\nYou (turn {turn_count + 1}) — paste text, drag files, or commands "
            f"({MULTILINE_SENTINEL} / {END_SESSION_CMD} / {EXPORT_CMD}).\n"
            f"Enter rỗng ngay dòng đầu để thoát:"
        )
        raw_text = _prompt_multiline(intro, allow_empty_exit=True)

        # Empty → exit
        if not raw_text:
            print("  Empty input — ending session.")
            break

        # /end command
        if raw_text.strip() == END_SESSION_CMD:
            print("  Ending session …")
            break

        # /export command
        do_export_and_continue = raw_text.strip() == EXPORT_CMD
        if do_export_and_continue:
            map_content = _generate_system_map(
                history, anth_client, model, project, version
            )
            out = _write_map(client_name, project, map_content, version)
            print(f"  Written: {out}")
            exported = True
            version += 1
            print(f"  Continuing session (next export will be v{version}) …")
            continue

        # Detect drag-dropped files in text
        inline_files = _detect_file_paths(raw_text)
        if inline_files is not None:
            text       = ""
            file_paths = inline_files
            print(f"  Detected {len(inline_files)} file(s) — skipping separate file prompt.")
        else:
            text       = raw_text
            file_paths = _prompt_for_files(
                f"  Files for this turn (drag-drop, Enter to skip): "
            )

        all_inputs.append(text[:100] if text else f"[{len(file_paths)} file(s)]")

        # Build message and call LLM
        try:
            user_msg = _build_user_message(text, file_paths)
            history.append(user_msg)
            turn_count += 1
        except Exception as exc:
            print(f"  [error] Could not build message: {exc}")
            if history and history[-1].get("role") == "user":
                history.pop()
            continue

        print(f"\n  Thinking …")
        reply = _ask(anth_client, model, system, history)
        history.append(_build_assistant_message(reply))

        print()
        print("─" * 60)
        print(reply)
        print("─" * 60)

        # Check if LLM suggests export
        if "EXPORT: READY" in reply and not exported:
            print()
            print("  LLM suggests the system map has enough info to export.")
            try:
                ans = input("  Export now? [y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans == "y":
                map_content = _generate_system_map(
                    history, anth_client, model, project, version
                )
                out = _write_map(client_name, project, map_content, version)
                print(f"  Written: {out}")
                exported = True
                version += 1

    # ── Session end — always write system_map.md ──────────────────────────────
    if turn_count == 0 and not history:
        print("  No turns taken — nothing to write.")
        return

    # Only export on end if we haven't just exported
    if turn_count > 0:
        print()
        print("  Writing system map …")
        map_content = _generate_system_map(
            history, anth_client, model, project, version
        )
        out = _write_map(client_name, project, map_content, version)
        print(f"  Written: {out}")
        exported = True

    # Append session log
    input_summary = " | ".join(all_inputs[:5])
    _append_session_log(
        client_name   = client_name,
        project       = project,
        version       = version,
        turn_count    = turn_count,
        input_summary = input_summary,
        exported      = exported,
    )
    print(f"  Appended: {_log_path(client_name, project)}")
    print()
    print(f"  Session complete. Version {version} written.")
    print(f"  Next session will continue from this map.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="system_explorer.py",
        description="Persistent system knowledge accumulator for unfamiliar platforms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python system_explorer.py --client client-a --project icp-alert
              python system_explorer.py --client client-a --project icp-alert --model claude-opus-4-7
        """) if False else "",
    )
    p.add_argument("--client",  required=True,
                   help="Client identifier (used as folder name, no spaces).")
    p.add_argument("--project", required=True,
                   help="Project identifier within the client.")
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help=f"Model to use (default: {DEFAULT_MODEL}).")
    return p


def main() -> None:
    import textwrap  # noqa: F401 — used in epilog
    parser = _build_parser()
    args   = parser.parse_args()

    try:
        run_session(
            client_name = args.client,
            project     = args.project,
            model       = args.model,
        )
    except KeyboardInterrupt:
        print("\n\n  Interrupted — session data may be incomplete.")
        sys.exit(130)


if __name__ == "__main__":
    main()
