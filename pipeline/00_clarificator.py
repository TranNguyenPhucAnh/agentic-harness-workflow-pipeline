#!/usr/bin/env python3
"""
00_clarificator.py
==================
Clarificator agent — upstream nhất trong pipeline, chạy trước Estimator.

Nhận raw requirement (file hoặc text), phân tích holes/conflicts/assumptions,
tổ chức Q&A với user theo 3-tier system, và output clarified_requirement.md
cùng với structured report cho downstream steps.

Usage:
    python 00_clarificator.py --input requirement.pdf
    python 00_clarificator.py --input spec_draft.md
    python 00_clarificator.py --text "Build a dashboard that shows..."
    python 00_clarificator.py                    # interactive multiline prompt

Artifacts produced (owner: 00_clarificator):
    run/clarification_report.json
    run/clarification_questions.md
    knowledge/current/clarification_log.md   ← append-only
    state/clarified_requirement.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ── Attempt to import from sibling paths.py ───────────────────────────────────
try:
    # 00_clarificator.py lives in pipeline/ — project root is one level up
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from artifacts.paths import (  # type: ignore
        CURRENT_DIR,
        KNOWLEDGE_BASE,
        RUN_DIR,
        STATE_DIR,
        ensure_dirs,
    )
    # Note: CLARIFICATION_* paths are defined at module level below,
    # using the imported RUN_DIR/STATE_DIR/CURRENT_DIR from paths.py
except ImportError:
    # Standalone mode: derive paths relative to this script
    _SCRIPT_DIR = Path(__file__).parent
    _ART = _SCRIPT_DIR / "artifacts"
    CURRENT_DIR   = _ART / "knowledge" / "current"
    KNOWLEDGE_BASE = CURRENT_DIR / "base.md"
    RUN_DIR        = _ART / "run"
    STATE_DIR      = _ART / "state"

    def ensure_dirs() -> None:  # type: ignore[misc]
        for d in (CURRENT_DIR, RUN_DIR, STATE_DIR):
            d.mkdir(parents=True, exist_ok=True)

# ── New artifact paths (owned by this script) ─────────────────────────────────
CLARIFICATION_REPORT    = RUN_DIR     / "clarification_report.json"
CLARIFICATION_QUESTIONS = RUN_DIR     / "clarification_questions.md"
CLARIFICATION_LOG       = CURRENT_DIR / "clarification_log.md"
CLARIFIED_REQ           = STATE_DIR   / "clarified_requirement.md"

# ── Model config ──────────────────────────────────────────────────────────────
_ANALYZE_MODEL   = "deepseek/deepseek-chat"          # Phase 1+2: reasoning heavy
_SUGGEST_MODEL   = "deepseek/deepseek-chat"          # Phase 3 Tier3: lighter OK but same default
_MAX_TOKENS      = 8192
_TIER3_MIN_CONF  = 0.75                              # below this → promote to Tier 2

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _print_banner(msg: str) -> None:
    width = min(80, len(msg) + 4)
    print("\n" + "─" * width)
    print(f"  {msg}")
    print("─" * width)


def _wrap(text: str, indent: int = 0) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=80, initial_indent=prefix, subsequent_indent=prefix)


def _load_text_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _read_pdf(path: Path) -> str:
    """Extract text from PDF via pdftotext (poppler) or fallback."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # fallback: try python-based extraction
    try:
        import importlib
        pypdf = importlib.import_module("pypdf")
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _read_input_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
        if not text.strip():
            print(f"[warn] PDF extraction returned empty text from {path.name}.")
        return text
    # .md, .txt, .rst, anything text-based
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _load_clarification_log() -> str:
    return _load_text_file(CLARIFICATION_LOG)


def _extract_answered_ids(log_text: str) -> set[str]:
    """Pull all CLR-XXX ids that already appear in the log."""
    return set(re.findall(r"CLR-\d{3}", log_text))


def _load_knowledge_context() -> str:
    parts: list[str] = []
    if KNOWLEDGE_BASE.exists():
        parts.append(f"=== base.md ===\n{KNOWLEDGE_BASE.read_text(encoding='utf-8')}")
    if CLARIFICATION_LOG.exists():
        parts.append(f"=== clarification_log.md ===\n{CLARIFICATION_LOG.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (model-agnostic thin wrapper — swap backend as needed)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, model: str = _ANALYZE_MODEL) -> str:
    """
    Call an LLM via OpenAI-compatible API.
    Reads OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY from env.
    Falls back to a simple stdin mock when running offline.
    """
    import os

    # Detect which provider to use based on model prefix
    if model.startswith("deepseek"):
        api_key  = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = "https://api.deepseek.com/v1"
        model_id = model.split("/")[-1]  # "deepseek-chat"
    elif model.startswith("gemini"):
        # Use OpenAI-compat endpoint for Gemini via Google AI SDK shim
        api_key  = os.environ.get("GOOGLE_API_KEY", "")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        model_id = model.split("/")[-1]
    elif "/" in model:  # OpenRouter format: "provider/model-name"
        api_key  = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        model_id = model  # OpenRouter expects full "provider/model" string
    else:
        api_key  = os.environ.get("OPENAI_API_KEY", "")
        base_url = "https://api.openai.com/v1"
        model_id = model

    if not api_key:
        # Offline mock for development — prompts user to paste JSON manually
        print("\n[00][offline] No API key found. Paste LLM response JSON then EOF (Ctrl-D):")
        return sys.stdin.read()

    try:
        payload = {
            "model": model_id,
            "max_tokens": _MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[00][error] LLM call failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 + 2: Analyze & Classify
# ─────────────────────────────────────────────────────────────────────────────

_ANALYZE_SYSTEM = """
You are a senior software architect acting as a Requirements Clarificator.
Your job is to deeply read a requirement document and identify EVERY ambiguity,
assumption, conflict, and gap that would block or risk the implementation.

You have access to the project's knowledge base and past clarification history.
Use them to:
1. AVOID asking questions that have already been answered (ids in clarification_log).
2. DETECT conflicts between new requirements and past decisions.
3. SURFACE assumptions the requirement makes about existing systems.

Output ONLY a valid JSON object — no markdown fences, no preamble.
Schema:
{
  "project_name": "<inferred from requirement, or 'Unknown'>",
  "findings": [
    {
      "id": "CLR-001",
      "text": "<clear description of the hole/conflict/assumption>",
      "tier": 1 | 2 | 3,
      "category": "business" | "logic" | "technical" | "design",
      "priority": "blocking" | "high" | "medium" | "low",
      "depends_on": ["CLR-XXX"],
      "scenarios": ["<option A>", "<option B>"],
      "suggestion": "<recommended approach>",
      "confidence": 0.0,
      "citation": "<source: base.md §X, pattern Y, etc.>"
    }
  ],
  "conflicts": [
    {
      "id": "CON-001",
      "description": "<what conflicts with what>",
      "source_a": "<new requirement text>",
      "source_b": "<existing decision from knowledge base>"
    }
  ],
  "clarified_summary": "<one paragraph: what the requirement IS asking for, stated confidently>"
}

TIER RULES (strict):
- Tier 1: answer space unbounded OR subjective (business goals, brand, user emotions, priorities).
           → Must ask client. No suggestion.
- Tier 2: answer space bounded and enumerable (≤5 concrete options).
           → Present scenarios for client to choose. No suggestion needed.
- Tier 3: answer is near-deterministic from context (tech stack, patterns, stated constraints).
           → confidence ≥ 0.75 required. MUST include citation explaining why.
           → confidence < 0.75 → downgrade to Tier 2.

DEPENDENCY RULES:
- If finding B only makes sense after finding A is answered, put A's id in B's depends_on.
- Findings with depends_on should have priority "low" or "medium" initially.

PRIORITY RULES:
- "blocking": estimate or implementation cannot proceed without this answer.
- "high": significant scope/architecture impact.
- "medium": affects one module or UX flow.
- "low": nice-to-have clarification.

ID FORMAT: CLR-001, CLR-002, ... (3-digit zero-padded, sequential)
"""


def _analyze(requirement_text: str, knowledge_context: str, answered_ids: set[str]) -> dict:
    """Run Phase 1+2: call LLM, parse JSON, filter already-answered findings."""
    already_answered_note = ""
    if answered_ids:
        already_answered_note = (
            f"\n\nALREADY ANSWERED (skip these, do not re-generate):\n"
            + "\n".join(sorted(answered_ids))
        )

    user_msg = f"""KNOWLEDGE CONTEXT:
{knowledge_context if knowledge_context else "(none — standalone mode)"}
{already_answered_note}

REQUIREMENT DOCUMENT:
{requirement_text}

Analyze thoroughly. Output only the JSON object."""

    raw = _call_llm(_ANALYZE_SYSTEM, user_msg)

    # Strip markdown fences if model wrapped output anyway
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as exc:
        print(f"[00][error] Failed to parse LLM analysis JSON: {exc}")
        print("Raw output:\n", raw[:500])
        raise

    # Enforce Tier 3 confidence threshold — downgrade if below
    for f in result.get("findings", []):
        if f.get("tier") == 3 and f.get("confidence", 0) < _TIER3_MIN_CONF:
            f["tier"] = 2
            f["confidence"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Interactive answer loop
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY_ORDER = {"blocking": 0, "high": 1, "medium": 2, "low": 3}
_TIER_LABEL     = {1: "🔴", 2: "🟡", 3: "🟢"}


def _sort_findings(findings: list[dict]) -> list[dict]:
    """Sort: Tier 1 blocking first, then by tier, then priority."""
    return sorted(
        findings,
        key=lambda f: (
            f.get("tier", 9),
            _PRIORITY_ORDER.get(f.get("priority", "low"), 9),
        )
    )


def _print_finding(f: dict, index: int, total: int) -> None:
    tid = f["id"]
    tier = f.get("tier", 1)
    icon = _TIER_LABEL.get(tier, "⚪")
    cat  = f.get("category", "?")
    pri  = f.get("priority", "?")

    print(f"\n{icon} [{index}/{total}] {tid}  tier={tier}  {pri.upper()}  [{cat}]")
    print(_wrap(f["text"], indent=2))

    if f.get("depends_on"):
        print(f"  ↳ depends on: {', '.join(f['depends_on'])}")

    if tier == 2 and f.get("scenarios"):
        print("\n  Options:")
        for i, s in enumerate(f["scenarios"], 1):
            print(f"    {i}. {s}")

    if tier == 3:
        print(f"\n  💡 Suggestion: {f.get('suggestion', '')}")
        conf = f.get("confidence")
        if conf:
            print(f"  Confidence: {int(conf * 100)}%")
        if f.get("citation"):
            print(f"  Citation: {f['citation']}")


def _ask_tier1(f: dict) -> str:
    print()
    raw = input("  → Your answer: ").strip()
    return raw or "(no answer provided)"


def _ask_tier2(f: dict) -> str:
    scenarios = f.get("scenarios", [])
    while True:
        print()
        choice = input(f"  → Choose 1–{len(scenarios)} or type custom answer: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scenarios):
                return scenarios[idx]
        if choice:
            return choice
        print("  Please enter a choice.")


def _ask_tier3(f: dict) -> tuple[str, bool]:
    """Returns (answer_text, accepted)."""
    print()
    choice = input("  → [A]ccept / [R]eject / [M]odify: ").strip().upper()
    if choice.startswith("A") or choice == "":
        return f.get("suggestion", "accepted"), True
    if choice.startswith("R"):
        reason = input("  → Rejection reason (optional): ").strip()
        return reason or "rejected", False
    # Modify
    modified = input("  → Modified answer: ").strip()
    return modified or f.get("suggestion", ""), True


def _dependencies_satisfied(f: dict, answered: dict[str, str]) -> bool:
    for dep in f.get("depends_on", []):
        if dep not in answered:
            return False
    return True


def _derive_impact(question: str, answer: str, category: str) -> str:
    """
    Produce a one-line impact statement from a Q&A pair.
    Uses a lightweight heuristic first; falls back to LLM for ambiguous cases.
    """
    # Short answers that are simple acceptances don't need LLM
    if answer.lower() in ("accepted", "yes", "no", "rejected", "(no answer provided)"):
        return ""
    # For substantive answers, derive impact via a fast LLM call
    try:
        system = (
            "You are a technical analyst. Given a clarification Q&A pair, "
            "output ONE concise sentence (max 20 words) describing the "
            "implementation impact of this decision. No preamble."
        )
        user = f"Question: {question}\nAnswer: {answer}\nCategory: {category}"
        return _call_llm(system, user).strip().splitlines()[0][:120]
    except Exception:
        return ""


def _run_interactive_loop(
    findings: list[dict],
    project_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Drive the Q&A loop. Returns (decisions, unresolved).
    decisions: list of {id, tier, question, answer, accepted, impact}
    """
    decisions:   list[dict] = []
    unresolved:  list[dict] = []
    answered:    dict[str, str] = {}   # id → answer text
    queue = _sort_findings([f for f in findings])

    total = len(queue)
    processed = 0

    _print_banner(f"Clarification session — {project_name}")
    print(f"  {total} findings to process.\n")

    i = 0
    while i < len(queue):
        f = queue[i]
        i += 1

        # Skip if dependency not yet answered (push to back)
        if not _dependencies_satisfied(f, answered):
            queue.append(f)
            # Safety: if we loop without progress, break
            if len(queue) - i > total * 2:
                unresolved.append(f)
                break  # circular dependency — cannot resolve
            continue

        processed += 1
        remaining = len([x for x in queue[i:] if _dependencies_satisfied(x, answered)]) + 1
        _print_finding(f, processed, processed + remaining - 1)

        tier = f.get("tier", 1)
        accepted = True

        if tier == 1:
            answer = _ask_tier1(f)
        elif tier == 2:
            answer = _ask_tier2(f)
        else:
            answer, accepted = _ask_tier3(f)

        answered[f["id"]] = answer
        # Derive a concise impact statement from the answer
        impact = _derive_impact(f["text"], answer, f.get("category", ""))
        decisions.append({
            "id":       f["id"],
            "tier":     tier,
            "category": f.get("category", ""),
            "priority": f.get("priority", ""),
            "question": f["text"],
            "answer":   answer,
            "accepted": accepted,
            "impact":   impact,
        })

        # Check if this answer unlocks any previously-deferred findings
        # (already handled by re-checking depends_on in next iteration)
        # NOTE: v1 limitation — new questions generated from answers are not
        # injected mid-session. Queue is fixed after Phase 1 analysis.
        # Re-run clarificator with updated knowledge context for follow-ups.

    return decisions, unresolved


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis: produce clarified requirement
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """
You are a technical writer. Given a raw requirement document and a set of
clarification Q&A decisions, produce a clean, unambiguous
"Clarified Requirement" document in markdown.

Rules:
- Incorporate all decisions into the narrative naturally.
- Preserve the original structure but resolve every ambiguity.
- Add a "## Decisions Log" section at the end listing each CLR-XXX with
  one-line summary of the answer.
- Be concise. No preamble. Output only the markdown document.
"""


def _synthesize_requirement(
    original: str,
    decisions: list[dict],
    conflicts: list[dict],
    summary: str,
) -> str:
    decisions_text = "\n".join(
        f"- {d['id']}: {d['question'][:80]}... → {d['answer']}"
        for d in decisions
    )
    conflicts_text = "\n".join(
        f"- {c['id']}: {c['description']}"
        for c in conflicts
    ) or "None detected."

    user_msg = f"""ORIGINAL REQUIREMENT:
{original}

CLARIFIED SUMMARY (from analysis):
{summary}

DECISIONS ({len(decisions)} total):
{decisions_text}

CONFLICTS DETECTED:
{conflicts_text}

Produce the clarified requirement document now."""

    return _call_llm(_SYNTHESIS_SYSTEM, user_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(
    session_id: str,
    req_hash: str,
    project_name: str,
    decisions: list[dict],
    unresolved: list[dict],
    conflicts: list[dict],
    findings: list[dict],
) -> None:
    tier_counts = {1: 0, 2: 0, 3: 0}
    tier3_accepted = tier3_rejected = 0
    for d in decisions:
        tier_counts[d["tier"]] = tier_counts.get(d["tier"], 0) + 1
        if d["tier"] == 3:
            if d["accepted"]:
                tier3_accepted += 1
            else:
                tier3_rejected += 1

    report = {
        "requirement_hash": req_hash,
        "session_id":       session_id,
        "project_name":     project_name,
        "total_findings":   len(findings),
        "tier1_answered":   tier_counts.get(1, 0),
        "tier2_answered":   tier_counts.get(2, 0),
        "tier3_accepted":   tier3_accepted,
        "tier3_rejected":   tier3_rejected,
        "conflicts_detected": len(conflicts),
        "unresolved":       [u["id"] for u in unresolved],
        "decisions":        decisions,
        "conflicts":        conflicts,
    }
    CLARIFICATION_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[00] ✓ Report → {CLARIFICATION_REPORT}")


def _write_questions_md(
    project_name: str,
    session_id: str,
    findings: list[dict],
    conflicts: list[dict],
) -> None:
    lines: list[str] = [
        f"# Clarification Questions — {project_name}",
        f"Generated: {session_id[:10]}",
        "",
    ]

    blocking    = [f for f in findings if f.get("tier") in (1, 2) and f.get("priority") == "blocking"]
    high        = [f for f in findings if f.get("tier") in (1, 2) and f.get("priority") == "high"]
    medium_low  = [f for f in findings if f.get("tier") in (1, 2) and f.get("priority") not in ("blocking", "high")]
    suggestions = [f for f in findings if f.get("tier") == 3]

    def _render_q(f: dict, numbered: int) -> list[str]:
        out = [f"{numbered}. [{f['id']}] {f['text']}"]
        if f.get("scenarios"):
            for s in f["scenarios"]:
                out.append(f"   - {s}")
        return out

    if blocking:
        lines += ["## 🔴 Blocking (cần trả lời trước khi estimate)", ""]
        for n, f in enumerate(blocking, 1):
            lines += _render_q(f, n)
            lines.append("")

    if high:
        lines += ["## 🟡 Important", ""]
        for n, f in enumerate(high, 1):
            lines += _render_q(f, n)
            lines.append("")

    if medium_low:
        lines += ["## ⚪ Other Questions", ""]
        for n, f in enumerate(medium_low, 1):
            lines += _render_q(f, n)
            lines.append("")

    if suggestions:
        lines += ["## 🟢 Suggestions (confirm nếu đồng ý)", ""]
        for f in suggestions:
            conf = f.get("confidence", 0)
            conf_str = f"{int(conf * 100)}%" if conf else "?"
            lines.append(f"- [{f['id']}] **Context:** {f['text']}")
            lines.append(f"  **Suggestion:** {f.get('suggestion', '')}")
            lines.append(f"  Confidence: {conf_str} | Reasoning: {f.get('citation', 'N/A')}")
            lines.append("  → Accept / Reject / Modify?")
            lines.append("")

    if conflicts:
        lines += ["---", "## ⚠️ Conflicts Detected", ""]
        for c in conflicts:
            lines.append(f"- [{c['id']}] {c['description']}")
            lines.append(f"  New: _{c.get('source_a', '')[:60]}_")
            lines.append(f"  Existing: _{c.get('source_b', '')[:60]}_")
            lines.append("")

    CLARIFICATION_QUESTIONS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[00] ✓ Questions → {CLARIFICATION_QUESTIONS}")


def _append_to_log(
    session_id: str,
    project_name: str,
    decisions: list[dict],
    conflicts: list[dict],
) -> None:
    """Append this session's decisions to the permanent clarification_log.md."""
    lines: list[str] = [
        f"\n## {session_id[:10]} | Project: {project_name} | Session: {session_id[11:19]}",
        "",
    ]
    for d in decisions:
        tier_label = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}.get(d["tier"], "?")
        accepted_label = "" if d["tier"] != 3 else (" / accepted" if d["accepted"] else " / rejected")
        lines.append(f"### {d['id']} [{tier_label}{accepted_label}]")
        lines.append(f"**Q:** {d['question']}")
        lines.append(f"**A:** {d['answer']}")
        if d.get("impact"):
            lines.append(f"**Impact:** {d['impact']}")
        lines.append("")

    if conflicts:
        lines.append("### Conflicts resolved this session")
        for c in conflicts:
            lines.append(f"- [{c['id']}] {c['description']}")
        lines.append("")

    with CLARIFICATION_LOG.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[00] ✓ Log appended → {CLARIFICATION_LOG}")


def _write_clarified_req(content: str) -> None:
    CLARIFIED_REQ.write_text(content, encoding="utf-8")
    print(f"[00] ✓ Clarified requirement → {CLARIFIED_REQ}")


# ─────────────────────────────────────────────────────────────────────────────
# Input gathering
# ─────────────────────────────────────────────────────────────────────────────

def _gather_requirement(args: argparse.Namespace) -> str:
    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[00][error] File not found: {path}")
            sys.exit(1)
        print(f"[00] Reading requirement from {path.name} ...")
        text = _read_input_file(path)
        if not text.strip():
            print(f"[00][error] Could not extract text from {path.name}.")
            sys.exit(1)
        return text

    if args.text:
        return args.text

    # Interactive mode
    print("[00] Paste / type requirement below.")
    print("     Press Enter twice then Ctrl-D (Unix) or Ctrl-Z Enter (Windows) to finish.\n")
    try:
        lines = sys.stdin.read()
    except KeyboardInterrupt:
        print("\n[00] Aborted.")
        sys.exit(0)
    return lines.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="00_clarificator — requirement analysis & Q&A agent"
    )
    parser.add_argument("--input",    metavar="FILE", help="Path to requirement file (.md, .txt, .pdf)")
    parser.add_argument("--text",     metavar="TEXT", help="Requirement as inline text string")
    parser.add_argument("--no-synth", action="store_true",
                        help="Skip synthesis step (don't generate clarified_requirement.md)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run analysis only, print findings, no Q&A and no file writes")
    args = parser.parse_args()

    ensure_dirs()

    # ── Gather input ──────────────────────────────────────────────────────────
    requirement_text = _gather_requirement(args)
    req_hash         = _sha256(requirement_text)
    session_id       = _now_iso()

    # ── Load knowledge context ────────────────────────────────────────────────
    standalone = not (KNOWLEDGE_BASE.exists() or CLARIFICATION_LOG.exists())
    if standalone:
        print("[00] Standalone mode — no knowledge context found.")
        knowledge_context = ""
        answered_ids: set[str] = set()
    else:
        print("[00] Loading knowledge context ...")
        knowledge_context = _load_knowledge_context()
        log_text          = _load_clarification_log()
        answered_ids      = _extract_answered_ids(log_text)
        if answered_ids:
            print(f"[00] Skipping {len(answered_ids)} already-answered findings: {sorted(answered_ids)[:5]}{'...' if len(answered_ids) > 5 else ''}")

    # ── Phase 1+2: Analyze ────────────────────────────────────────────────────
    print("[00] Analyzing requirement ...")
    analysis      = _analyze(requirement_text, knowledge_context, answered_ids)
    project_name  = analysis.get("project_name", "Unknown")
    findings      = analysis.get("findings", [])
    conflicts     = analysis.get("conflicts", [])
    clarified_sum = analysis.get("clarified_summary", "")

    if not findings and not conflicts:
        print("[00] ✓ No ambiguities found — requirement is already clear.")
        if not args.dry_run:
            _write_clarified_req(requirement_text)
        return

    print(f"[00] Found {len(findings)} findings, {len(conflicts)} conflicts.")

    # ── Dry run: just print ───────────────────────────────────────────────────
    if args.dry_run:
        _print_banner("Dry run — findings only")
        for f in _sort_findings(findings):
            _print_finding(f, 0, len(findings))
        if conflicts:
            print("\n⚠️  Conflicts:")
            for c in conflicts:
                print(f"  {c['id']}: {c['description']}")
        return

    # ── Write questions.md (for async client delivery) ────────────────────────
    _write_questions_md(project_name, session_id, _sort_findings(findings), conflicts)

    # ── Phase 3: Interactive loop ─────────────────────────────────────────────
    decisions, unresolved = _run_interactive_loop(findings, project_name)

    if unresolved:
        print(f"\n[00][warn] {len(unresolved)} findings could not be resolved due to circular/unmet dependencies:")
        for u in unresolved:
            print(f"  - {u['id']}: {u['text'][:60]}...")

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_report(session_id, req_hash, project_name, decisions, unresolved, conflicts, findings)
    _append_to_log(session_id, project_name, decisions, conflicts)

    # ── Phase synthesis: clarified requirement ────────────────────────────────
    if not args.no_synth:
        print("\n[00] Synthesizing clarified requirement ...")
        clarified_md = _synthesize_requirement(
            requirement_text, decisions, conflicts, clarified_sum
        )
        _write_clarified_req(clarified_md)

    _print_banner(f"Done — {len(decisions)} decisions recorded")
    print(f"  Next step: python 01_estimator.py --input {CLARIFIED_REQ}\n")


if __name__ == "__main__":
    main()
