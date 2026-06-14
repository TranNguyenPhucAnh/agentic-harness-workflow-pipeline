"""
pipeline/9e_error_fixer.py
===========================
MLOps / DevOps interactive error Q&A loop.

Mục đích: Chat loop hỏi-đáp với LLM để debug lỗi infrastructure/ops:
  - Terraform syntax/plan/apply errors
  - Docker build/runtime errors
  - Kubernetes manifest, Helm chart, YAML issues
  - Jenkinsfile, GitHub Actions, CI/CD errors
  - AWS service errors (ECS, Lambda, S3, IAM, CloudFormation…)
  - Airflow DAG errors, MLflow, Kubeflow
  - Bất kỳ config/log error nào liên quan DevOps/MLOps stack

Flow mỗi round:
  1. User nhập câu hỏi / paste error / drag-drop file (log, yaml, tf, Jenkinsfile…)
  2. LLM phân tích, suggest fix cụ thể với commands
  3. User thử → feedback → loop tiếp
  4. Gõ "done"/"q" → clean exit → LLM summarize session → ghi artifacts

Artifacts:
  error_fixer/error_qa.md       — short-term overwrite: Q&A transcript session hiện tại
  error_fixer/error_qa_log.md   — long-term append: history qua các session

Knowledge injection (start mỗi session):
  error_fixer/error_qa_log.md   — previous session summaries (context cache)
  absorber/codebase_map.md      — codebase context nếu có

Usage:
  python 9e_error_fixer.py --project my-app
  python 9e_error_fixer.py --project my-app --no-codebase   # skip absorber context
  python 9e_error_fixer.py --project my-app --max-rounds 20
  PIPELINE_PROJECT=my-app python 9e_error_fixer.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

from artifacts.paths import (          # noqa: E402
    ABSORBER_CODEBASE_MD,
    artifact_root,
    ensure_dirs,
    _LazyPath,
)
from modules.artifact_tracking import (  # noqa: E402
    track_read,
    track_write,
    print_summary as print_artifact_summary,
)
from modules.call_llm import call_llm_messages  # noqa: E402
from modules.cost import print_summary as print_cost_summary  # noqa: E402
from modules.drag_and_drop import gather_text_file_bundle  # noqa: E402
from modules.post_interactive import prompt_next_step       # noqa: E402
from artifacts.models import get_model                      # noqa: E402

# ── Artifact paths ────────────────────────────────────────────────────────────
ERROR_FIXER_QA     = _LazyPath("error_fixer/error_qa.md")      # short-term overwrite
ERROR_FIXER_QA_LOG = _LazyPath("error_fixer/error_qa_log.md")  # long-term append

ROLE            = "error_fixer"
ROLE_SUMMARIZER = "summarizer"

MIN_ROUNDS_FOR_SUMMARY = 2   # chỉ gọi summarizer nếu session có ít nhất N rounds


# ════════════════════════════════════════════════════════════════════════════
# System prompts
# ════════════════════════════════════════════════════════════════════════════

_SYSTEM_ERROR_FIXER = """\
You are a senior MLOps/DevOps engineer and troubleshooting expert.

Your role: help the user diagnose and fix infrastructure, configuration,
and operational errors across the DevOps/MLOps stack.

Stack coverage:
- Terraform (HCL syntax, plan/apply errors, state issues, provider config)
- Docker (Dockerfile, build errors, runtime crashes, compose issues)
- Kubernetes (manifest YAML, Helm charts, kubectl errors, pod/service/ingress)
- CI/CD (Jenkinsfile, GitHub Actions, GitLab CI, ArgoCD, Tekton)
- AWS services (ECS, Lambda, S3, IAM policies, CloudFormation, CDK, ECR, RDS)
- Airflow (DAG errors, operator config, XCom, connection issues)
- MLflow, Kubeflow, Ray, Seldon (model serving, experiment tracking errors)
- General: YAML/JSON config, environment variables, networking, TLS/certs

Response style:
- Lead with the most likely root cause in 1-2 sentences.
- Give concrete commands or config snippets — no vague advice.
- If multiple causes are possible, rank them and address the most likely first.
- When suggesting a fix, explain WHY it works (brief, 1 sentence).
- If you need more info to diagnose (logs, config, Terraform version…), ask
  for exactly that — do not guess when a specific detail would change your answer.
- Use code blocks for all commands and config snippets.

Format:
- Use markdown headers for multi-part answers.
- Keep prose tight — developers prefer signal over explanation.
"""

_SYSTEM_SUMMARIZER = """\
You are a technical knowledge distiller.

Given a Q&A transcript from a DevOps/MLOps debugging session, write a concise
session summary for future reference.

Output format (markdown, no fences):

## Summary — {ISO_TIMESTAMP}

**Stack**: comma-separated list of technologies touched (e.g. Terraform, ECS, Docker)

**Problems solved**:
- one line per issue resolved, with the root cause and fix applied

**Unresolved**:
- one line per issue not fully resolved (or "none")

**Key commands / snippets**:
- include only commands that are reusable in similar situations

**Lessons**:
- 1-3 concise takeaways for future debugging in this stack

Keep the total under 400 words. No preamble, no postamble — output the markdown directly.
"""


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="error_fixer.py",
        description="MLOps/DevOps interactive error Q&A loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--project",
        default=os.environ.get("PIPELINE_PROJECT"),
        help="Project name (sets PIPELINE_PROJECT).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=30,
        help="Max Q&A rounds per session (default: 30).",
    )
    p.add_argument(
        "--no-codebase",
        action="store_true",
        help="Skip loading absorber/codebase_map.md as context.",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable TTY prompts.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if os.environ.get("PIPELINE_PROJECT"):
        return
    parser.error("PIPELINE_PROJECT is not set. Use --project <name>.")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flush_stdin() -> None:
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def _read_text(path: Any) -> str:
    p = Path(str(path))
    if not p.exists():
        return ""
    track_read(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _read_input_file(path: Path) -> str:
    track_read(path)
    return path.read_text(encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════════════════
# Context loader (start of session)
# ════════════════════════════════════════════════════════════════════════════

def _load_session_context(no_codebase: bool) -> str:
    """
    Build context string injected into system prompt at session start.

    Sources (in order):
      1. error_qa_log.md — summaries of previous sessions (most recent 3000 chars)
      2. absorber/codebase_map.md — codebase structure (truncated to 2000 chars)
    """
    parts: list[str] = []

    # Previous session summaries
    log_text = _read_text(ERROR_FIXER_QA_LOG)
    if log_text.strip():
        # Take last 3000 chars (most recent sessions)
        snippet = log_text[-3000:].strip()
        parts.append(
            "## Previous session summaries (knowledge cache)\n\n"
            f"{snippet}"
        )

    # Codebase map
    if not no_codebase:
        codebase_md = _read_text(ABSORBER_CODEBASE_MD)
        if codebase_md.strip():
            snippet = codebase_md[:2000].strip()
            parts.append(
                "## Codebase context\n\n"
                f"{snippet}"
                + ("\n\n_(truncated)_" if len(codebase_md) > 2000 else "")
            )

    if not parts:
        return ""

    return (
        "\n\n---\n\n".join(parts)
    )


# ════════════════════════════════════════════════════════════════════════════
# LLM calls
# ════════════════════════════════════════════════════════════════════════════

def _call_error_fixer(
    messages: list[dict],
    context: str,
) -> str:
    """
    Call error_fixer LLM. Prepend context to system prompt if present.
    """
    system = _SYSTEM_ERROR_FIXER
    if context:
        system = f"{_SYSTEM_ERROR_FIXER}\n\n---\n\n{context}"

    full_messages = [{"role": "system", "content": system}] + messages

    content, _ = call_llm_messages(
        ROLE,
        full_messages,
        retries=2,
        backoff=True,
        caller_file=__file__,
        label=f"[9e] {get_model(ROLE)}",
        max_tokens=4096,
    )
    return content


def _call_summarizer(transcript: str) -> str:
    """
    Call summarizer LLM to distill the session transcript.
    Returns markdown summary string.
    """
    ts = _now_iso()
    prompt = (
        f"Session timestamp: {ts}\n\n"
        f"Please summarize the following Q&A session transcript:\n\n"
        f"---\n\n{transcript}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_SUMMARIZER},
        {"role": "user",   "content": prompt},
    ]

    content, _ = call_llm_messages(
        ROLE_SUMMARIZER,
        messages,
        retries=1,
        backoff=False,
        caller_file=__file__,
        label=f"[9e] {get_model(ROLE_SUMMARIZER)}",
        max_tokens=1024,
    )
    return content


# ════════════════════════════════════════════════════════════════════════════
# Transcript builder
# ════════════════════════════════════════════════════════════════════════════

def _build_transcript(rounds: list[dict]) -> str:
    """
    Build human-readable transcript from rounds list.
    Each round: { "round": int, "question": str, "answer": str, "ts": str }
    """
    lines: list[str] = []
    for r in rounds:
        lines.append(f"### Round {r['round']} — {r['ts']}")
        lines.append("")
        lines.append(f"**User:**\n{r['question']}")
        lines.append("")
        lines.append(f"**Assistant:**\n{r['answer']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Artifact writers
# ════════════════════════════════════════════════════════════════════════════

def _write_error_qa(
    session_start: str,
    rounds: list[dict],
    status: str,
    dry_run: bool = False,
) -> None:
    """Overwrite error_fixer/error_qa.md — transcript of current session."""
    if dry_run:
        return

    transcript = _build_transcript(rounds)
    content = (
        f"<!-- error_qa.md — session started {session_start} | status: {status} -->\n\n"
        f"# Error Q&A Session — {session_start}\n\n"
        f"**Status**: {status}  \n"
        f"**Rounds**: {len(rounds)}\n\n"
        f"---\n\n"
        f"{transcript}"
    )

    path = Path(str(ERROR_FIXER_QA))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    track_write(ERROR_FIXER_QA)


def _append_qa_log(
    session_start: str,
    rounds: list[dict],
    summary: str,
    dry_run: bool = False,
) -> None:
    """
    Append one session block to error_fixer/error_qa_log.md.
    Block = summary (from LLM) + brief round count + timestamp.
    If summary is empty (session too short), write minimal entry.
    """
    if dry_run:
        return

    ts = _now_iso()

    if summary.strip():
        block = (
            f"{summary.strip()}\n\n"
            f"_Session: {session_start} → {ts} | {len(rounds)} round(s)_\n\n"
            f"---\n\n"
        )
    else:
        # Short session — no summarizer call, write minimal entry
        block = (
            f"## Session — {session_start}\n\n"
            f"_Short session ({len(rounds)} round(s)), no summary generated._\n\n"
            f"---\n\n"
        )

    path = Path(str(ERROR_FIXER_QA_LOG))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(block)
    track_write(ERROR_FIXER_QA_LOG)


# ════════════════════════════════════════════════════════════════════════════
# Interactive loop
# ════════════════════════════════════════════════════════════════════════════

def run_session(
    max_rounds:     int,
    no_codebase:    bool,
    no_interactive: bool,
    verbose:        bool,
) -> tuple[list[dict], str]:
    """
    Main Q&A loop. Returns (rounds, status).
    status: "done" | "interrupted" | "max_rounds"
    """
    session_start = _now_display()
    print(f"\n{'═' * 68}")
    print(f"  ERROR FIXER — Session started {session_start}")
    print(f"{'═' * 68}")
    print(f"  Stack: Terraform · Docker · Kubernetes · AWS · Airflow · CI/CD")
    print(f"  Gõ câu hỏi, paste error, hoặc drag-drop file.")
    print(f'  Gõ "done" hoặc "q" để kết thúc session.\n')

    # Load knowledge context once at session start
    context = _load_session_context(no_codebase)
    if context and verbose:
        print(f"[9e] Context loaded ({len(context)} chars)")

    # LLM conversation history (không bao gồm system — injected per call)
    conversation: list[dict] = []

    # Session rounds for transcript + log
    rounds: list[dict] = []
    status = "done"

    for round_num in range(1, max_rounds + 1):
        print(f"\n{'─' * 68}")
        print(f"  Round {round_num}/{max_rounds}")
        print(f"{'─' * 68}\n")

        # ── Gather input ─────────────────────────────────────────────────────
        if no_interactive or not sys.stdin.isatty():
            print("[9e] Non-interactive mode — stopping.")
            status = "interrupted"
            break

        _flush_stdin()

        try:
            bundle = gather_text_file_bundle(
                cli_text=None,
                cli_files=[],
                read_file_fn=_read_input_file,
                prompt_title=f"[Round {round_num}] Question / error",
                prompt_body=(
                    "Paste error message, describe the issue, or drag-drop a file\n"
                    "(log, yaml, tf, Jenkinsfile, docker-compose.yml…)\n"
                    'Press Enter twice to submit. Gõ "done" để kết thúc.'
                ),
                attachment_prompt="Attach file(s) if relevant",
                default_attachment_only_prompt="Analyze the attached file(s) for errors.",
                allow_interactive=True,
                ask_for_attachments_after_text=True,
            )
        except RuntimeError as exc:
            msg = str(exc)
            # User gõ "done" sẽ raise hoặc empty input
            if "No input" in msg or not msg:
                print("[9e] No input — ending session.")
                status = "done"
                break
            print(f"[9e] Input error: {exc}")
            continue
        except (EOFError, KeyboardInterrupt):
            print("\n[9e] Interrupted.")
            status = "interrupted"
            break

        question_text = bundle.text.strip()
        if not question_text:
            print("[9e] Empty input — skipping.")
            continue

        # Check for exit commands embedded in input
        if question_text.lower() in ("done", "q", "quit", "exit", "/done"):
            print("[9e] Session ended by user.")
            status = "done"
            break

        # ── Call LLM ─────────────────────────────────────────────────────────
        print(f"\n[9e] Thinking …\n")
        conversation.append({"role": "user", "content": question_text})

        try:
            answer = _call_error_fixer(conversation, context)
        except Exception as exc:
            print(f"[9e] LLM error: {exc}")
            conversation.pop()  # remove failed user message
            continue

        conversation.append({"role": "assistant", "content": answer})

        # ── Print answer ─────────────────────────────────────────────────────
        print("─" * 68)
        print(answer)
        print("─" * 68)

        # Record round
        rounds.append({
            "round":    round_num,
            "ts":       _now_display(),
            "question": question_text,
            "answer":   answer,
        })

        # Update short-term transcript after each round (in case of crash)
        _write_error_qa(
            session_start=session_start,
            rounds=rounds,
            status="in_progress",
        )

        if round_num >= max_rounds:
            print(f"\n[9e] Reached max rounds ({max_rounds}).")
            status = "max_rounds"
            break

    return rounds, status


# ════════════════════════════════════════════════════════════════════════════
# Post-session: summarize + write artifacts
# ════════════════════════════════════════════════════════════════════════════

def finalize_session(
    rounds: list[dict],
    status: str,
    dry_run: bool,
    verbose:  bool,
) -> None:
    """
    Called on clean exit (done / Ctrl+C / max_rounds).
    1. Summarize if rounds >= MIN_ROUNDS_FOR_SUMMARY
    2. Overwrite error_qa.md (final)
    3. Append to error_qa_log.md
    """
    session_start = rounds[0]["ts"] if rounds else _now_display()

    print(f"\n[9e] Session ended — {len(rounds)} round(s), status: {status}")

    # ── Summarize ─────────────────────────────────────────────────────────────
    summary = ""
    if len(rounds) >= MIN_ROUNDS_FOR_SUMMARY:
        print(f"[9e] Generating session summary …")
        transcript = _build_transcript(rounds)
        try:
            summary = _call_summarizer(transcript)
            if verbose:
                print("\n[9e] Summary:\n")
                print(summary)
                print()
        except Exception as exc:
            print(f"[9e][warn] Summarizer failed: {exc} — skipping summary.")

    # ── Write artifacts ────────────────────────────────────────────────────────
    _write_error_qa(
        session_start=session_start,
        rounds=rounds,
        status=status,
        dry_run=dry_run,
    )

    _append_qa_log(
        session_start=session_start,
        rounds=rounds,
        summary=summary,
        dry_run=dry_run,
    )

    if not dry_run:
        print(f"[9e] error_qa.md      → {ERROR_FIXER_QA}")
        print(f"[9e] error_qa_log.md  → {ERROR_FIXER_QA_LOG}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_project(args.project, parser)
    ensure_dirs()

    # Ensure error_fixer/ dir
    ef_dir = artifact_root() / "error_fixer"
    ef_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    rounds: list[dict] = []
    status = "done"

    try:
        rounds, status = run_session(
            max_rounds=args.max_rounds,
            no_codebase=args.no_codebase,
            no_interactive=args.no_interactive,
            verbose=args.verbose,
        )

    except KeyboardInterrupt:
        print("\n[9e] Ctrl+C — finalizing session …")
        status = "interrupted"

    except Exception as exc:
        print(f"[9e] ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        status = "error"
        exit_code = 1

    finally:
        # Always finalize — write whatever rounds were collected
        if rounds:
            try:
                finalize_session(
                    rounds=rounds,
                    status=status,
                    dry_run=False,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print(f"[9e][warn] finalize_session failed: {exc}", file=sys.stderr)

        print_cost_summary("[9e]")
        print_artifact_summary("[9e]")
        prompt_next_step(ROLE, prefix="[9e]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()