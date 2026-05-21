"""
02_clarificator.py
==================
Clarificator agent — upstream nhất trong pipeline.

Nhận raw requirement dưới nhiều dạng:
    - inline text
    - một hoặc nhiều file
    - drag-drop path trong terminal
    - piped stdin
    - attachment-only input

Sau đó phân tích holes/conflicts/assumptions, tổ chức Q&A với user theo
3-tier system, và output clarificator/session.json cùng structured
decision_log.json cho downstream steps.

Khi absorber/codebase_map.md có sẵn, clarificator sẽ hỏi câu hỏi cụ thể
hơn — map được vào files/classes/methods trong codebase hiện tại thay vì
hỏi generic. Findings có codebase_refs[] để downstream trace về code thực.

Usage:
    python pipeline/02_clarificator.py --project my-app --input requirement.pdf
    python pipeline/02_clarificator.py --project my-app --input spec.md notes.md
    python pipeline/02_clarificator.py --project my-app --text "Build a dashboard..."
    python pipeline/02_clarificator.py --project my-app
    python pipeline/02_clarificator.py

When called by harness.py, PIPELINE_PROJECT is already set.
When run directly, --project or interactive project prompt is used and
PIPELINE_PROJECT is set before any artifact path is resolved.

Artifacts produced (owner: clarificator):
    artifacts_<slug>/clarificator/session.json       (short-term, overwrite)
    artifacts_<slug>/clarificator/decision_log.json  (long-term, append)

At the end of each run, prints:
    - artifacts/files read
    - artifacts/files created/updated/overwritten/appended
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Import from artifacts.paths ──────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from artifacts.paths import (  # type: ignore  # noqa: E402
    ABSORBER_CODEBASE_MAP,
    ARCHIVIST_KNOWLEDGE_LOG,
    CLARIFICATOR_DECISION_LOG,
    CLARIFICATOR_SESSION,
    ensure_dirs,
    get_project_name,
)
from artifacts.models import call_model, get_model, get_provider  # noqa: E402

# New shared interactive/drag-drop abstraction.
from modules.drag_and_drop import gather_text_file_bundle  # type: ignore  # noqa: E402
from modules.artifact_tracking import track_read, track_write, print_summary as print_artifact_summary  # noqa: E402
from modules.cost import print_call, print_summary, record_usage  # noqa: E402
from modules.post_interactive import prompt_next_step  # noqa: E402


# Local aliases
KNOWLEDGE_BASE = ARCHIVIST_KNOWLEDGE_LOG


# === WRITE AUTHORITY: clarificator ===
# OWNS  : clarificator/session.json         (short-term - overwrite)
#         clarificator/decision_log.json    (long-term - append only)
# READS : archivist/knowledge_log.md        (knowledge-aware)
#         absorber/codebase_map.md          (upstream-aware/codebase-aware - existing project only)
#         clarificator/decision_log.json    (history-aware)


# ── Model config ─────────────────────────────────────────────────────────────

ROLE = "clarificator"

_TIER3_MIN_CONF = 0.75

_MAX_TOKENS_ANALYZE = 8192
_MAX_TOKENS_DELTA = 2048
_MAX_TOKENS_SYNTHESIS = 4096
_DELTA_REQ_CHARS = 4000


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

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


def _read_pdf(path: Path) -> str:
    """Extract text from PDF via pdftotext or pypdf fallback."""
    track_read(path)

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        import importlib

        pypdf = importlib.import_module("pypdf")
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""


def _read_input_file(path: Path) -> str:
    """
    Read user-provided requirement source.

    This function intentionally converts files to text before model call.
    Clarificator stays model-agnostic and does not rely on provider-native
    file/image attachments.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _read_pdf(path)
        if not text.strip():
            print(f"[clarificator][warn] PDF extraction returned empty text from {path.name}.")
        return text

    track_read(path)

    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
        raise RuntimeError(
            f"Image input is not supported by clarificator yet: {path}\n"
            "Add OCR support first, or provide a text/PDF requirement file."
        )

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())
    return raw


def _parse_json_response(raw: str, label: str) -> dict[str, Any]:
    clean = _strip_json_fences(raw)

    try:
        parsed = json.loads(clean)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{label} parsed as {type(parsed).__name__}, expected object.")
        return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if not isinstance(parsed, dict):
                raise RuntimeError(f"{label} parsed as {type(parsed).__name__}, expected object.")
            return parsed
        except json.JSONDecodeError as exc:
            print(f"[clarificator][error] Failed to parse {label}: {exc}", file=sys.stderr)
            print(f"[clarificator][error] Raw output, first 800 chars:\n{raw[:800]}", file=sys.stderr)
            raise

    raise RuntimeError(f"No JSON object found in {label}.")


def _load_decision_log() -> list[dict[str, Any]]:
    """Load decision_log.json entries list."""
    if not CLARIFICATOR_DECISION_LOG.exists():
        return []
    track_read(CLARIFICATOR_DECISION_LOG)
    try:
        data = json.loads(CLARIFICATOR_DECISION_LOG.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return []
        return entries
    except (json.JSONDecodeError, Exception):
        return []


def _extract_answered_qa_pairs(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract Q/A pairs from decision_log.json entries for semantic dedup."""
    pairs: list[dict[str, str]] = []
    for entry in entries:
        for decision in entry.get("decisions", []):
            if not isinstance(decision, dict):
                continue
            pairs.append(
                {
                    "id": decision.get("id", ""),
                    "question": decision.get("question", ""),
                    "answer": decision.get("answer", ""),
                }
            )
    return pairs


def _load_knowledge_context() -> str:
    parts: list[str] = []

    if KNOWLEDGE_BASE.exists():
        track_read(KNOWLEDGE_BASE)
        parts.append(f"=== archivist/knowledge_log.md ===\n{KNOWLEDGE_BASE.read_text(encoding='utf-8')}")

    # Codebase snapshot — optional, chỉ có khi absorber đã chạy.
    # Clarificator dùng để hỏi câu hỏi map được vào files/classes/methods
    # cụ thể thay vì hỏi generic. Missing = greenfield hoặc absorber chưa chạy.
    if ABSORBER_CODEBASE_MAP.exists():
        track_read(ABSORBER_CODEBASE_MAP)
        parts.append(f"=== absorber/codebase_map.md ===\n{ABSORBER_CODEBASE_MAP.read_text(encoding='utf-8')}")

    entries = _load_decision_log()
    if entries:
        # Render a summary of past decisions for LLM context
        log_lines: list[str] = []
        for entry in entries:
            log_lines.append(f"Session: {entry.get('session_id', '?')}")
            for d in entry.get("decisions", []):
                log_lines.append(f"  [{d.get('id')}] Q: {d.get('question', '')}")
                log_lines.append(f"           A: {d.get('answer', '')}")
        parts.append(f"=== clarificator/decision_log ===\n" + "\n".join(log_lines))

    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# Project resolution
# ════════════════════════════════════════════════════════════════════════════

def _resolve_project(arg_project: str | None) -> str:
    if arg_project and arg_project.strip():
        return arg_project.strip()

    try:
        return get_project_name()
    except RuntimeError:
        pass

    if not sys.stdin.isatty():
        print(
            "[clarificator][error] No --project specified and PIPELINE_PROJECT not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("[clarificator] No --project specified and PIPELINE_PROJECT not set.")
    name = input("  Enter project name: ").strip()
    if not name:
        print("[clarificator] Project name cannot be empty.", file=sys.stderr)
        sys.exit(1)
    return name


def _list_projects() -> None:
    projects = sorted(p for p in _REPO_ROOT.glob("artifacts_*") if p.is_dir())

    if not projects:
        print("[clarificator] No artifacts_* project workspaces found.")
        return

    print("[clarificator] Known project workspaces:")
    for p in projects:
        slug = p.name.removeprefix("artifacts_")
        log = p / "clarificator" / "decision_log.json"
        sessions = 0
        if log.exists():
            try:
                track_read(log)
                data = json.loads(log.read_text(encoding="utf-8"))
                sessions = len(data.get("entries", []))
            except Exception:
                sessions = 0
        print(f"  - {slug:<30} {sessions} clarification session(s)")


# ════════════════════════════════════════════════════════════════════════════
# LLM call
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(
    system: str,
    user: str,
    max_tokens: int = _MAX_TOKENS_ANALYZE,
) -> str:
    try:
        resp = call_model(
            ROLE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            pt        = getattr(usage, "prompt_tokens",     0) or 0
            ct        = getattr(usage, "completion_tokens", 0) or 0
            call_cost = record_usage(usage, model=get_model(ROLE), provider=get_provider(ROLE))
            print_call(__file__, pt, ct, call_cost)
        content = resp.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Model returned empty content.")
        return content
    except RuntimeError as exc:
        if "not set" in str(exc):
            print("\n[clarificator][offline] No API key found. Paste LLM response then EOF:")
            return sys.stdin.read()
        raise
    except Exception as exc:
        print(f"[clarificator][error] LLM call failed: {exc}", file=sys.stderr)
        raise


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 + 2: Analyze & Classify
# ════════════════════════════════════════════════════════════════════════════

_ANALYZE_SYSTEM = """
You are a senior software architect acting as a Requirements Clarificator.
Your job is to deeply read a requirement document and identify EVERY ambiguity,
assumption, conflict, and gap that would block or risk the implementation.

You have access to the project's knowledge base and past clarification history.
Use them to:
1. SEMANTIC DEDUP — Do NOT generate a new finding if ALREADY_ANSWERED_QA
   contains a semantically equivalent or closely related question.
2. CONFLICT DETECTION — Does the new requirement contradict any past decision?
3. ASSUMPTION SURFACING — Does the requirement assume behavior that may not exist yet?
4. CODEBASE-AWARE QUESTIONING — If absorber/codebase_map.md is present in
   KNOWLEDGE CONTEXT, use it to ask concrete questions that reference actual
   files, classes, methods, or patterns in the codebase. Prefer specific
   questions (e.g. "Should the new rate limiter hook into the existing
   `auth/middleware.py` request pipeline, or run as a separate layer?") over
   generic ones ("Where should rate limiting be implemented?").
   If a requirement mentions a feature or module that already exists in the
   codebase, surface that in the finding's citation and codebase_refs fields.
   If codebase_map.md is absent, treat the project as greenfield and skip this rule.

Output ONLY a valid JSON object — no markdown fences, no preamble.
Schema:
{
  "project_name": "<inferred from requirement, or 'Unknown'>",
  "findings": [
    {
      "id": "CLR-001",
      "text": "<natural collaborative clarification question>",
      "tier": 1,
      "category": "business",
      "subcategory": "policy",
      "priority": "blocking",
      "depends_on": [],
      "scenarios": ["option A", "option B"],
      "suggestion": "",
      "confidence": 0.0,
      "citation": "",
      "codebase_refs": []
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
  "clarified_summary": "<one paragraph summary>"
}

TONE RULES:
- Write as a natural, collaborative question — NOT a judgment or critique.
- NEVER start with: "The requirement does not specify", "It is unclear",
  "There is no mention of", or similar.
- Ask directly and warmly.

TIER RULES:
- Tier 1: subjective/business/product decision. Include 2–4 representative scenarios.
- Tier 2: bounded enumerable choice, ≤5 realistic scenarios.
- Tier 3: near-deterministic from context. confidence >= 0.75 and citation required.
  If confidence < 0.75, downgrade to Tier 2.

SUBCATEGORY RULES:
  policy, scoring, approval, routing, output, config, access, integration, other.

"output" and "config" subcategories are NEVER Tier 1 even if category is business.
They are Tier 2 when bounded/enumerable.

Every Tier 1 and Tier 2 finding without scenarios is malformed.

DEPENDENCY RULES:
- If finding B only makes sense after finding A is answered, put A's id in B.depends_on.
- Findings with depends_on should have priority low or medium initially.

PRIORITY:
- blocking: answer must be known before architecture/estimate can proceed.
- high: significantly shapes scope, approval logic, or integration contracts.
- medium: affects one module/workflow/edge case.
- low: nice-to-have.

Generate enough findings for the actual complexity:
- Simple spec: 3–5 findings.
- Medium spec: 6–10 findings.
- Complex enterprise spec: 10–20 findings.

ID FORMAT: CLR-001, CLR-002, ...
"""


def _analyze(
    requirement_text: str,
    knowledge_context: str,
    answered_qa_pairs: list[dict[str, str]],
) -> dict[str, Any]:
    if answered_qa_pairs:
        qa_lines: list[str] = []
        for pair in answered_qa_pairs:
            qa_lines.append(f"  [{pair['id']}] Q: {pair['question']}")
            qa_lines.append(f"         A: {pair['answer']}")
        already_answered_block = (
            "\n\nALREADY_ANSWERED_QA (do NOT re-ask semantically equivalent questions):\n"
            + "\n".join(qa_lines)
        )
    else:
        already_answered_block = ""

    user_msg = f"""KNOWLEDGE CONTEXT:
{knowledge_context if knowledge_context else "(none — standalone mode)"}
{already_answered_block}

REQUIREMENT DOCUMENT:
{requirement_text}

Analyze thoroughly. Apply semantic dedup against ALREADY_ANSWERED_QA above.
If absorber/codebase_map.md is present in KNOWLEDGE CONTEXT, reference specific
files or modules in findings where relevant — populate codebase_refs[] with
repo-relative paths (e.g. ["src/auth/middleware.py"]).
Output only the JSON object."""

    raw = _call_llm(_ANALYZE_SYSTEM, user_msg)
    result = _parse_json_response(raw, "clarification analysis")

    findings = result.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        if finding.get("tier") == 3 and finding.get("confidence", 0) < _TIER3_MIN_CONF:
            finding["tier"] = 2
            finding["confidence"] = None

        if finding.get("tier") in (1, 2) and not finding.get("scenarios"):
            finding["scenarios"] = [
                "Yes / proceed as implied",
                "No / needs different approach",
                "Other (specify below)",
            ]

        # Normalize codebase_refs — ensure list[str], never missing
        refs = finding.get("codebase_refs")
        if not isinstance(refs, list):
            finding["codebase_refs"] = []
        else:
            finding["codebase_refs"] = [r for r in refs if isinstance(r, str)]

    result["findings"] = _enforce_tiers([f for f in findings if isinstance(f, dict)])
    result.setdefault("conflicts", [])
    result.setdefault("clarified_summary", "")
    return result


def _finding_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


def _enforce_tiers(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for finding in findings:
        suggestion = (finding.get("suggestion") or "").strip()
        confidence = finding.get("confidence") or 0.0
        citation = (finding.get("citation") or "").strip()
        scenarios = finding.get("scenarios") or []
        category = finding.get("category", "")
        priority = finding.get("priority", "")
        subcategory = finding.get("subcategory", "other")

        bounded = bool(scenarios) and len(scenarios) <= 5
        near_det_cat = category in ("technical", "design", "logic")
        output_config = subcategory in ("output", "config")

        if finding.get("tier") == 3 and (not citation or confidence < _TIER3_MIN_CONF):
            finding["tier"] = 2
            finding["confidence"] = None

        if suggestion and confidence >= _TIER3_MIN_CONF and citation:
            finding["tier"] = 3
        elif (bounded and near_det_cat) or (bounded and output_config):
            finding["tier"] = 2
        elif (category == "business" or priority == "blocking") and not output_config and not suggestion:
            finding["tier"] = 1
        elif not scenarios and not suggestion:
            finding["tier"] = 1

        if finding.get("tier") in (1, 2) and not finding.get("scenarios"):
            finding["scenarios"] = [
                "Yes — proceed as implied",
                "No — needs a different approach",
                "Other (type custom answer)",
            ]

        if finding.get("tier") == 3 and not finding.get("suggestion"):
            finding["suggestion"] = "(see citation)"

    return findings


# ════════════════════════════════════════════════════════════════════════════
# Delta analysis
# ════════════════════════════════════════════════════════════════════════════

_DELTA_SYSTEM = """
You are a requirements analyst. A blocking clarification question was just answered.
Determine:
1. What NEW questions does this answer reveal that are not already in the queue?
2. Which EXISTING queue questions are now irrelevant or resolved?

Output ONLY JSON:
{
  "new_findings": [
    {
      "id": "NEW-001",
      "text": "<natural collaborative question>",
      "tier": 1,
      "category": "business",
      "priority": "blocking",
      "depends_on": [],
      "scenarios": ["option A", "option B"],
      "suggestion": "",
      "confidence": 0.0,
      "citation": ""
    }
  ],
  "invalidated_ids": ["CLR-XXX"]
}

Rules:
- Only generate questions that could not have been asked before this answer.
- Do not regenerate existing questions.
- If nothing changes, return {"new_findings": [], "invalidated_ids": []}.
- MAXIMUM 3 new findings.
- Only policy-shaping follow-ups.
"""


def _delta_analyze(
    answered_finding: dict[str, Any],
    answer: str,
    requirement_text: str,
    current_queue_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    queue_summary = ", ".join(current_queue_ids) if current_queue_ids else "none"

    req_snippet = requirement_text
    if len(req_snippet) > _DELTA_REQ_CHARS:
        req_snippet = req_snippet[:_DELTA_REQ_CHARS] + (
            f"\n... [truncated — {len(requirement_text)} chars total]"
        )

    user_msg = f"""REQUIREMENT CONTEXT:
{req_snippet}

ANSWERED QUESTION:
  ID: {answered_finding['id']}
  Text: {answered_finding['text']}
  Category: {answered_finding.get('category', '')}
  Priority: {answered_finding.get('priority', '')}

USER ANSWER: {answer}

CURRENT PENDING QUEUE: {queue_summary}

Given this answer, what new questions are revealed and which pending ones are moot?
Output only JSON."""

    try:
        raw = _call_llm(_DELTA_SYSTEM, user_msg, max_tokens=_MAX_TOKENS_DELTA)
        result = _parse_json_response(raw, "delta analysis")
    except Exception as exc:
        print(f"  [clarificator][delta] Delta analysis failed ({exc}), continuing without update.")
        return [], []

    new_findings = result.get("new_findings", [])
    invalidated_ids = result.get("invalidated_ids", [])

    if not isinstance(new_findings, list):
        new_findings = []
    if not isinstance(invalidated_ids, list):
        invalidated_ids = []

    return _enforce_tiers([f for f in new_findings if isinstance(f, dict)]), [
        str(x) for x in invalidated_ids
    ]


# ════════════════════════════════════════════════════════════════════════════
# Interactive answer loop
# ════════════════════════════════════════════════════════════════════════════

_PRIORITY_ORDER = {"blocking": 0, "high": 1, "medium": 2, "low": 3}
_TIER_LABEL = {1: "🔴", 2: "🟡", 3: "🟢"}


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        findings,
        key=lambda f: (
            f.get("tier", 9),
            _PRIORITY_ORDER.get(f.get("priority", "low"), 9),
        ),
    )


def _print_finding(finding: dict[str, Any], index: int, total: int) -> None:
    tid = finding["id"]
    tier = finding.get("tier", 1)
    icon = _TIER_LABEL.get(tier, "⚪")
    category = finding.get("category", "?")
    priority = finding.get("priority", "?")

    print(f"\n{icon} [{index}/{total}] {tid}  tier={tier}  {priority.upper()}  [{category}]")
    print(_wrap(finding["text"], indent=2))

    if finding.get("depends_on"):
        print(f"  ↳ depends on: {', '.join(finding['depends_on'])}")

    if tier in (1, 2) and finding.get("scenarios"):
        print("\n  Options:")
        for i, scenario in enumerate(finding["scenarios"], 1):
            print(f"    {i}. {scenario}")

    if tier == 3:
        print(f"\n  💡 Suggestion: {finding.get('suggestion', '')}")
        conf = finding.get("confidence")
        if conf:
            print(f"  Confidence: {int(conf * 100)}%")
        if finding.get("citation"):
            print(f"  Citation: {finding['citation']}")


def _ask_tier1(finding: dict[str, Any]) -> str:
    scenarios = finding.get("scenarios", [])

    if scenarios:
        print()
        while True:
            choice = input(f"  → Choose 1–{len(scenarios)} or type custom answer: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(scenarios):
                    return str(scenarios[idx])
            if choice:
                return choice
            print("  Please enter a choice.")

    raw = input("  → Your answer: ").strip()
    return raw or "(no answer provided)"


def _ask_tier2(finding: dict[str, Any]) -> str:
    scenarios = finding.get("scenarios", [])

    while True:
        print()
        choice = input(f"  → Choose 1–{len(scenarios)} or type custom answer: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(scenarios):
                return str(scenarios[idx])
        if choice:
            return choice
        print("  Please enter a choice.")


def _ask_tier3(finding: dict[str, Any]) -> tuple[str, bool]:
    print()
    choice = input("  → [A]ccept / [R]eject / [M]odify: ").strip().upper()

    if choice.startswith("A") or choice == "":
        return finding.get("suggestion", "accepted"), True

    if choice.startswith("R"):
        reason = input("  → Rejection reason (optional): ").strip()
        return reason or "rejected", False

    modified = input("  → Modified answer: ").strip()
    return modified or finding.get("suggestion", ""), True


def _dependencies_satisfied(
    finding: dict[str, Any],
    answered: dict[str, str],
    known_ids: set[str] | None = None,
) -> bool:
    for dep in finding.get("depends_on", []):
        if known_ids is not None and dep not in known_ids:
            continue
        if dep not in answered:
            return False
    return True


def _derive_impact(question: str, answer: str, category: str) -> str:
    if not answer or answer.lower() in ("accepted", "yes", "no", "rejected", "(no answer provided)"):
        return ""

    try:
        system = (
            "You are a technical analyst. Given a clarification Q&A pair, "
            "output ONE complete sentence (max 30 words) describing the "
            "implementation impact of this decision. No preamble."
        )
        user = f"Question: {question}\nAnswer: {answer}\nCategory: {category}"
        raw = _call_llm(system, user, max_tokens=256).strip()
        return raw.splitlines()[0].strip().strip('"').strip("'")
    except Exception:
        return ""


def _batch_derive_impacts(decisions: list[dict[str, Any]]) -> None:
    pending = [decision for decision in decisions if not decision.get("impact")]
    if not pending:
        return

    print(f"\n[clarificator] Deriving impact statements ({len(pending)} decisions) ...")
    for decision in pending:
        decision["impact"] = _derive_impact(
            decision["question"],
            decision["answer"],
            decision.get("category", ""),
        )


def _run_interactive_loop(
    findings: list[dict[str, Any]],
    project_name: str,
    requirement_text: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    answered: dict[str, str] = {}
    answered_hashes: set[str] = set()

    delta_depth = 0
    max_delta_depth = 2

    queue = _sort_findings(list(findings))
    deferred: set[str] = set()
    answered_count = 0
    known_ids: set[str] = {finding["id"] for finding in queue if "id" in finding}

    _print_banner(f"Clarification session — {project_name}")
    print(f"  {len(queue)} findings to process.\n")

    i = 0
    while i < len(queue):
        finding = queue[i]
        i += 1

        if finding["id"] in answered:
            continue
        if _finding_hash(finding["text"]) in answered_hashes:
            continue

        if not _dependencies_satisfied(finding, answered, known_ids):
            if finding["id"] not in deferred:
                deferred.add(finding["id"])
                queue.append(finding)
            else:
                unresolved.append(finding)
            continue

        answered_count += 1
        pending_ready = sum(
            1
            for x in queue[i:]
            if x["id"] not in answered and _dependencies_satisfied(x, answered, known_ids)
        )
        display_total = answered_count + pending_ready

        _print_finding(finding, answered_count, display_total)

        tier = finding.get("tier", 1)
        accepted = True

        if tier == 1:
            answer = _ask_tier1(finding)
        elif tier == 2:
            answer = _ask_tier2(finding)
        else:
            answer, accepted = _ask_tier3(finding)

        answered[finding["id"]] = answer
        answered_hashes.add(_finding_hash(finding["text"]))

        decisions.append(
            {
                "id": finding["id"],
                "tier": tier,
                "category": finding.get("category", ""),
                "priority": finding.get("priority", ""),
                "question": finding["text"],
                "answer": answer,
                "accepted": accepted,
                "impact": "",
                "codebase_refs": finding.get("codebase_refs", []),
            }
        )

        is_delta_finding = finding["id"].startswith("NEW-")
        can_delta = (
            tier == 1
            and finding.get("priority") == "blocking"
            and requirement_text
            and delta_depth < max_delta_depth
            and not is_delta_finding
        )

        if can_delta:
            current_queue_ids = [
                x["id"]
                for x in queue[i:]
                if x["id"] not in answered
            ]
            print(f"  [delta] Checking for follow-up questions after {finding['id']}...")
            new_findings, invalidated_ids = _delta_analyze(
                finding,
                answer,
                requirement_text,
                current_queue_ids,
            )

            for inv_id in invalidated_ids:
                if inv_id not in answered:
                    answered[inv_id] = "[invalidated by delta]"
                    print(f"  [delta] Invalidated: {inv_id}")

            injected = 0
            for new_finding in new_findings:
                if "id" not in new_finding or "text" not in new_finding:
                    continue
                if _finding_hash(new_finding["text"]) in answered_hashes:
                    continue
                if new_finding["id"] in answered:
                    continue
                queue.append(new_finding)
                known_ids.add(new_finding["id"])
                injected += 1

            if injected:
                print(f"  [delta] Injected {injected} new finding(s) into queue.")
            elif new_findings:
                print(f"  [delta] {len(new_findings)} potential finding(s) already covered.")
            else:
                print("  [delta] No follow-up questions revealed.")

            delta_depth += 1

    return decisions, unresolved


# ════════════════════════════════════════════════════════════════════════════
# Synthesis
# ════════════════════════════════════════════════════════════════════════════

_SYNTHESIS_SYSTEM = """
You are a technical writer. Given a raw requirement document and clarification
decisions, produce ONE clean unified "Clarified Requirement" markdown document.

Rules:
1. Output the document exactly once. Do not repeat sections.
2. Use the original requirement as the structural template.
3. Incorporate decisions inline naturally.
4. Preserve original bullets/list items unless affected by a decision.
5. Add "## Decisions Log" at the end:
   "- **CLR-XXX**: <one-line summary of answer and impact>"
6. No preamble, no postamble. Output only markdown.
"""


def _synthesize_requirement(
    original: str,
    decisions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    summary: str,
) -> str:
    decisions_text = "\n".join(
        f"- {d['id']} [{d.get('priority', '').upper()}]: {d['question']} → {d['answer']}"
        for d in decisions
    )
    conflicts_text = "\n".join(
        f"- {c['id']}: {c['description']}"
        for c in conflicts
    ) or "None detected."

    user_msg = f"""ORIGINAL REQUIREMENT:
{original}

CLARIFIED SUMMARY:
{summary}

DECISIONS ({len(decisions)} total):
{decisions_text}

CONFLICTS DETECTED:
{conflicts_text}

Produce the clarified requirement document now."""

    return _call_llm(_SYNTHESIS_SYSTEM, user_msg, max_tokens=_MAX_TOKENS_SYNTHESIS)


# ════════════════════════════════════════════════════════════════════════════
# Output writers
# ════════════════════════════════════════════════════════════════════════════

def _normalize_input_sources_for_report(sources: Any) -> list[dict[str, Any]]:
    """
    Normalize source metadata returned from modules.drag_and_drop into JSON-safe dicts.

    The abstraction module may return dataclasses, dicts, or simple objects.
    Clarificator only persists conservative metadata, not full source text.
    """
    normalized: list[dict[str, Any]] = []

    if not sources:
        return normalized

    for src in sources:
        if isinstance(src, dict):
            item = dict(src)
        else:
            item = {
                "kind": getattr(src, "kind", ""),
                "label": getattr(src, "label", ""),
                "path": str(getattr(src, "path", "") or ""),
                "chars": getattr(src, "chars", None),
                "sha256": getattr(src, "sha256", ""),
            }

        # Avoid accidentally storing entire source content in report.
        item.pop("text", None)
        item.pop("content", None)

        if item.get("path") is not None:
            item["path"] = str(item.get("path"))

        normalized.append(item)

    return normalized


def _print_questions_to_terminal(
    project_name: str,
    session_id: str,
    findings: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    """Print questions to terminal only — no file write."""
    print(f"\n{'─' * 60}")
    print(f"  Clarification Questions — {project_name}")
    print(f"  Generated: {session_id[:10]}")
    print(f"{'─' * 60}")

    tier1 = sorted(
        [f for f in findings if f.get("tier") == 1],
        key=lambda f: _PRIORITY_ORDER.get(f.get("priority", "low"), 9),
    )
    tier2 = sorted(
        [f for f in findings if f.get("tier") == 2],
        key=lambda f: _PRIORITY_ORDER.get(f.get("priority", "low"), 9),
    )
    suggestions = [f for f in findings if f.get("tier") == 3]

    tier1_blocking = [f for f in tier1 if f.get("priority") == "blocking"]
    tier1_other = [f for f in tier1 if f.get("priority") != "blocking"]

    if tier1_blocking:
        print("\n  🔴 Blocking — Tier 1 (cần trả lời trước khi estimate)")
        for n, finding in enumerate(tier1_blocking, 1):
            print(f"    {n}. [{finding['id']}] {finding['text']}")
            for scenario in finding.get("scenarios", []):
                print(f"       - {scenario}")

    if tier1_other:
        print("\n  🔴 Tier 1 — Business & Policy Decisions")
        for n, finding in enumerate(tier1_other, 1):
            print(f"    {n}. [{finding['id']}] {finding['text']}")
            for scenario in finding.get("scenarios", []):
                print(f"       - {scenario}")

    if tier2:
        print("\n  🟡 Tier 2 — Bounded Choices")
        for n, finding in enumerate(tier2, 1):
            print(f"    {n}. [{finding['id']}] {finding['text']}")
            for scenario in finding.get("scenarios", []):
                print(f"       - {scenario}")

    if suggestions:
        print("\n  🟢 Tier 3 — Suggestions (confirm nếu đồng ý)")
        for finding in suggestions:
            conf = finding.get("confidence", 0)
            conf_str = f"{int(conf * 100)}%" if conf else "?"
            print(f"    [{finding['id']}] {finding['text']}")
            print(f"      Suggestion: {finding.get('suggestion', '')}")
            print(f"      Confidence: {conf_str} | Citation: {finding.get('citation', 'N/A')}")

    if conflicts:
        print("\n  ⚠️  Conflicts Detected")
        for conflict in conflicts:
            print(f"    [{conflict['id']}] {conflict['description']}")
            if conflict.get("source_a"):
                print(f"      New requirement: {conflict['source_a']}")
            if conflict.get("source_b"):
                print(f"      Existing decision: {conflict['source_b']}")

    print()


def _write_session(
    session_id: str,
    req_hash: str,
    project_name: str,
    decisions: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    input_sources: list[dict[str, Any]] | None = None,
    attachment_only: bool = False,
    requirement_synthesis: str = "",
) -> None:
    """Write clarificator/session.json (short-term, overwrite each run)."""
    tier_counts = {1: 0, 2: 0, 3: 0}
    tier3_accepted = 0
    tier3_rejected = 0

    for decision in decisions:
        tier_counts[decision["tier"]] = tier_counts.get(decision["tier"], 0) + 1
        if decision["tier"] == 3:
            if decision["accepted"]:
                tier3_accepted += 1
            else:
                tier3_rejected += 1

    session_data = {
        "requirement_hash": req_hash,
        "session_id": session_id,
        "project_name": project_name,
        "input_sources": input_sources or [],
        "attachment_only_input": attachment_only,
        "initial_findings": len(findings),
        "delta_injected": sum(1 for d in decisions if d["id"].startswith("NEW-")),
        "tier_counts": tier_counts,
        "total_decisions": len(decisions),
        "tier3_accepted": tier3_accepted,
        "tier3_rejected": tier3_rejected,
        "conflicts_detected": len(conflicts),
        "unresolved": [u["id"] for u in unresolved],
        "decisions": decisions,
        "conflicts": conflicts,
        "requirement_synthesis": requirement_synthesis,
    }

    CLARIFICATOR_SESSION.parent.mkdir(parents=True, exist_ok=True)
    CLARIFICATOR_SESSION.write_text(
        json.dumps(session_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(CLARIFICATOR_SESSION)

    print(f"\n[clarificator] ✓ Session → {CLARIFICATOR_SESSION}")


_IMPACT_TRIM_THRESHOLD = 50  # decisions per entry trước khi trim impacts


def _trim_decision_impacts(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Trim `impacts` list trong mỗi decision nếu entry lớn để tránh log phình.

    Chỉ áp dụng khi số decisions vượt threshold — với session nhỏ giữ nguyên.
    Impact text được truncate ở 200 chars, không xóa hoàn toàn để vẫn track được.
    """
    if len(decisions) <= _IMPACT_TRIM_THRESHOLD:
        return decisions

    trimmed: list[dict[str, Any]] = []
    for d in decisions:
        entry = dict(d)
        impacts = entry.get("impacts")
        if isinstance(impacts, list) and len(impacts) > 3:
            entry["impacts"] = [
                (imp[:200] + "…" if isinstance(imp, str) and len(imp) > 200 else imp)
                for imp in impacts[:3]
            ] + [f"… ({len(impacts) - 3} more impacts trimmed)"]
        elif isinstance(entry.get("impact"), str) and len(entry["impact"]) > 200:
            entry["impact"] = entry["impact"][:200] + "…"
        trimmed.append(entry)
    return trimmed


def _append_decision_log(
    session_id: str,
    project_name: str,
    req_hash: str,
    decisions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    """Append entry to clarificator/decision_log.json (long-term, append-only)."""
    entry = {
        "session_id": session_id,
        "project_name": project_name,
        "requirement_hash": req_hash,
        "generated_at": _now_iso(),
        "total_decisions": len(decisions),
        "conflicts_detected": len(conflicts),
        "decisions": _trim_decision_impacts(decisions),
        "conflicts": conflicts,
    }

    CLARIFICATOR_DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)

    if CLARIFICATOR_DECISION_LOG.exists():
        try:
            data = json.loads(CLARIFICATOR_DECISION_LOG.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "entries" not in data:
                data = {"entries": []}
        except (json.JSONDecodeError, Exception):
            data = {"entries": []}
    else:
        data = {"entries": []}

    data["entries"].append(entry)

    CLARIFICATOR_DECISION_LOG.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(CLARIFICATOR_DECISION_LOG)

    print(f"[clarificator] ✓ Decision log appended → {CLARIFICATOR_DECISION_LOG}")


# ════════════════════════════════════════════════════════════════════════════
# Input gathering — delegated to modules/drag_and_drop.py
# ════════════════════════════════════════════════════════════════════════════

def _gather_requirement_bundle(args: argparse.Namespace) -> Any:
    """
    Gather requirement text and source metadata.

    This function intentionally delegates interactive UX, drag-drop parsing,
    attachment-only detection, stdin handling, and multi-file composition to
    modules.drag_and_drop.

    Clarificator only provides:
      - CLI values
      - file reader callback
      - prompt copy
      - policy around interactivity
    """
    allow_interactive = not args.no_interactive

    bundle = gather_text_file_bundle(
        cli_text=args.text,
        cli_files=args.input or [],
        read_file_fn=_read_input_file,
        prompt_title="Enter requirement",
        prompt_body=(
            "Describe the feature/change, paste requirement text, or drag-drop files. "
            "End with /done (multiline) or press Enter twice (single line)."
        ),
        attachment_prompt="Attach requirement files if any",
        default_attachment_only_prompt="Please analyze the attached requirement source files.",
        allow_interactive=allow_interactive,
    )

    text = getattr(bundle, "text", "")
    if not text or not str(text).strip():
        print("[clarificator][error] Empty requirement input.", file=sys.stderr)
        sys.exit(1)

    return bundle


# ════════════════════════════════════════════════════════════════════════════
# CLI / Main
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="02_clarificator — requirement analysis & Q&A agent"
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Project workspace name. Prompted if omitted and PIPELINE_PROJECT is not set.",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        nargs="+",
        help=(
            "Requirement file(s). Supports text-like files and PDF. "
            "Can be repeated as space-separated paths."
        ),
    )
    parser.add_argument(
        "--text",
        metavar="TEXT",
        help=(
            "Requirement as inline text. If it contains only valid file paths, "
            "the drag/drop layer may treat it as attachment-only input."
        ),
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip synthesis step; do not generate requirement_synthesis field.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis only, print findings, no Q&A and no file writes.",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List all known artifacts_* project workspaces and exit.",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help=(
            "Disable terminal prompts. Intended for CI/harness usage. "
            "Fails if no --input/--text/piped stdin is available."
        ),
    )
    return parser


def main() -> None:
    try:
        parser = _build_parser()
        args = parser.parse_args()

        if args.list_projects and not args.project and not os.environ.get("PIPELINE_PROJECT"):
            _list_projects()
            return

        project_name = _resolve_project(args.project)
        os.environ["PIPELINE_PROJECT"] = project_name
        ensure_dirs()

        if args.list_projects:
            _list_projects()
            return

        print(f"[clarificator] Workspace: {project_name!r}")

        # ── Requirement intake via drag/drop abstraction ─────────────────────
        bundle = _gather_requirement_bundle(args)
        requirement_text = str(getattr(bundle, "text", "")).strip()
        input_sources = _normalize_input_sources_for_report(getattr(bundle, "sources", []))
        attachment_only = bool(getattr(bundle, "attachment_only", False))

        if input_sources:
            print("[clarificator] Requirement sources:")
            for src in input_sources:
                kind = src.get("kind") or "source"
                label = src.get("label") or src.get("path") or "(unknown)"
                chars = src.get("chars")
                suffix = f" ({chars} chars)" if isinstance(chars, int) else ""
                print(f"  - {kind}: {label}{suffix}")

        if attachment_only:
            print("[clarificator] Attachment-only requirement input detected.")

        req_hash = _sha256(requirement_text)
        session_id = _now_iso()

        # ── Load knowledge context ──────────────────────────────────────────
        log_entries = _load_decision_log()
        standalone = not (KNOWLEDGE_BASE.exists() or bool(log_entries))

        if standalone:
            print("[clarificator] Standalone mode — no knowledge context found for this workspace.")
            knowledge_context = ""
            answered_qa_pairs: list[dict[str, str]] = []
        else:
            print("[clarificator] Loading knowledge context ...")
            knowledge_context = _load_knowledge_context()
            answered_qa_pairs = _extract_answered_qa_pairs(log_entries)
            if answered_qa_pairs:
                print(f"[clarificator] Loaded {len(answered_qa_pairs)} past Q/A pairs for semantic dedup")

        # ── Phase 1+2: Analyze ──────────────────────────────────────────────
        print("[clarificator] Analyzing requirement ...")
        analysis = _analyze(requirement_text, knowledge_context, answered_qa_pairs)

        inferred_name = analysis.get("project_name", "Unknown")
        if project_name.lower() == "unknown" and inferred_name != "Unknown":
            project_name = inferred_name
            print(f"[clarificator] Project name inferred from requirement: {project_name!r}")

        findings = analysis.get("findings", [])
        conflicts = analysis.get("conflicts", [])
        clarified_sum = analysis.get("clarified_summary", "")

        if not isinstance(findings, list):
            findings = []
        if not isinstance(conflicts, list):
            conflicts = []

        if not findings and not conflicts:
            print("[clarificator] ✓ No ambiguities found — requirement is already clear.")
            if not args.dry_run:
                # Write session with synthesis = original requirement
                _write_session(
                    session_id=session_id,
                    req_hash=req_hash,
                    project_name=project_name,
                    decisions=[],
                    unresolved=[],
                    conflicts=[],
                    findings=[],
                    input_sources=input_sources,
                    attachment_only=attachment_only,
                    requirement_synthesis=requirement_text,
                )
            return

        print(f"[clarificator] Found {len(findings)} findings, {len(conflicts)} conflicts.")

        if args.dry_run:
            _print_banner("Dry run — findings only")
            for finding in _sort_findings(findings):
                _print_finding(finding, 0, len(findings))
            if conflicts:
                print("\n⚠️  Conflicts:")
                for conflict in conflicts:
                    print(f"  {conflict['id']}: {conflict['description']}")
            return

        # Print questions to terminal (no file write)
        _print_questions_to_terminal(project_name, session_id, _sort_findings(findings), conflicts)

        decisions, unresolved = _run_interactive_loop(
            findings,
            project_name,
            requirement_text,
        )

        if unresolved:
            loud = [
                item for item in unresolved
                if item.get("category") == "business" or item.get("priority") == "blocking"
            ]
            silent = [item for item in unresolved if item not in loud]

            if loud:
                print(
                    f"\n[clarificator][warn] {len(loud)} blocking question(s) could not be resolved "
                    "(unmet or circular dependencies) — review before proceeding:"
                )
                for item in loud:
                    print(f"  ⚠️  {item['id']}: {item['text'][:70]}...")

            if silent:
                print(
                    f"\n[clarificator] {len(silent)} low-priority question(s) skipped due to "
                    "inconsistent dependencies (recorded in report)."
                )

        _batch_derive_impacts(decisions)

        # ── Synthesis ────────────────────────────────────────────────────────
        requirement_synthesis = ""
        if not args.no_synth:
            print("\n[clarificator] Synthesizing clarified requirement ...")
            requirement_synthesis = _synthesize_requirement(
                requirement_text,
                decisions,
                conflicts,
                clarified_sum,
            )

        # ── Write session.json (short-term, overwrite) ───────────────────────
        _write_session(
            session_id=session_id,
            req_hash=req_hash,
            project_name=project_name,
            decisions=decisions,
            unresolved=unresolved,
            conflicts=conflicts,
            findings=findings,
            input_sources=input_sources,
            attachment_only=attachment_only,
            requirement_synthesis=requirement_synthesis,
        )

        # ── Append decision_log.json (long-term, append) ─────────────────────
        _append_decision_log(
            session_id=session_id,
            project_name=project_name,
            req_hash=req_hash,
            decisions=decisions,
            conflicts=conflicts,
        )

        _print_banner(f"Done — {len(decisions)} decisions recorded  [{project_name}]")
        print(f"  Decision log: {CLARIFICATOR_DECISION_LOG}")
        print(f"  Session      → {CLARIFICATOR_SESSION}")

    finally:
        print_summary("[02]")
        print_artifact_summary("[02]")
        prompt_next_step(ROLE, prefix="[02]")


if __name__ == "__main__":
    main()
