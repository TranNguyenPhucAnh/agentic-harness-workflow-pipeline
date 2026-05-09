#!/usr/bin/env python3
"""
03_clarificator.py
==================
Clarificator agent — upstream nhất trong pipeline.

Nhận raw requirement (file hoặc text), phân tích holes/conflicts/assumptions,
tổ chức Q&A với user theo 3-tier system, và output clarified_requirement.md
cùng với structured report cho downstream steps.

Usage:
    python pipeline/00_clarificator.py --project my-app --input requirement.pdf
    python pipeline/00_clarificator.py --project my-app --input spec_draft.md
    python pipeline/00_clarificator.py --project my-app --text "Build a dashboard..."
    python pipeline/00_clarificator.py --project my-app
    python pipeline/00_clarificator.py

When called by harness.py, PIPELINE_PROJECT is already set.
When run directly, --project or interactive project prompt is used and
PIPELINE_PROJECT is set before any artifact path is resolved.

Artifacts produced (owner: 00_clarificator):
    artifacts_<slug>/run/clarification_report.json
    artifacts_<slug>/run/clarification_questions.md
    artifacts_<slug>/knowledge/current/clarification_log.md
    artifacts_<slug>/state/clarified_requirement.md
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

import httpx


# ── Import from artifacts.paths (source of truth cho tất cả paths) ────────────

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from artifacts.paths import (  # type: ignore  # noqa: E402
    KNOWLEDGE_BASE,
    CLARIFICATION_REPORT,
    CLARIFICATION_QUESTIONS,
    CLARIFICATION_LOG,
    CLARIFIED_REQ,
    ensure_dirs,
    get_project_name,
)


# ── Model config ──────────────────────────────────────────────────────────────

_ANALYZE_MODEL = "deepseek/deepseek-chat"
_SUGGEST_MODEL = "deepseek/deepseek-chat"
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
    """Extract text from PDF via pdftotext (poppler) or pypdf fallback."""
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
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
        if not text.strip():
            print(f"[00][warn] PDF extraction returned empty text from {path.name}.")
        return text

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
            print(f"[00][error] Failed to parse {label}: {exc}", file=sys.stderr)
            print(f"[00][error] Raw output, first 800 chars:\n{raw[:800]}", file=sys.stderr)
            raise

    raise RuntimeError(f"No JSON object found in {label}.")


def _load_clarification_log() -> str:
    """Load project clarification log. Requires PIPELINE_PROJECT to be set."""
    if CLARIFICATION_LOG.exists():
        return CLARIFICATION_LOG.read_text(encoding="utf-8")
    return ""


def _extract_answered_qa_pairs(log_text: str) -> list[dict[str, str]]:
    """
    Extract structured Q/A pairs from the log for semantic dedup.
    Returns list of {id, question, answer} dicts.
    """
    pairs: list[dict[str, str]] = []
    blocks = re.split(r"\n(?=###\s+CLR-)", log_text)

    for block in blocks:
        id_match = re.search(r"###\s+(CLR-\d{3})", block)
        q_match = re.search(r"\*\*Q:\*\*\s*(.+?)(?=\n\*\*|\Z)", block, re.DOTALL)
        a_match = re.search(r"\*\*A:\*\*\s*(.+?)(?=\n\*\*|\Z)", block, re.DOTALL)

        if id_match and q_match:
            pairs.append(
                {
                    "id": id_match.group(1),
                    "question": q_match.group(1).strip(),
                    "answer": a_match.group(1).strip() if a_match else "",
                }
            )

    return pairs


def _load_knowledge_context() -> str:
    parts: list[str] = []

    if KNOWLEDGE_BASE.exists():
        parts.append(f"=== base.md ===\n{KNOWLEDGE_BASE.read_text(encoding='utf-8')}")

    log_text = _load_clarification_log()
    if log_text:
        parts.append(f"=== clarification_log.md ===\n{log_text}")

    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# Project resolution
# ════════════════════════════════════════════════════════════════════════════

def _resolve_project(arg_project: str | None) -> str:
    """
    Resolve project name without touching artifact paths prematurely.

    Priority:
      1. --project
      2. PIPELINE_PROJECT
      3. interactive prompt
    """
    if arg_project and arg_project.strip():
        return arg_project.strip()

    try:
        return get_project_name()
    except RuntimeError:
        pass

    print()
    print("[00] No --project specified and PIPELINE_PROJECT not set.")
    name = input("  Enter project name: ").strip()
    if not name:
        print("[00] Project name cannot be empty.", file=sys.stderr)
        sys.exit(1)
    return name


def _list_projects() -> None:
    """
    List artifact workspaces without requiring PIPELINE_PROJECT.
    """
    projects = sorted(
        p for p in _REPO_ROOT.glob("artifacts_*")
        if p.is_dir()
    )

    if not projects:
        print("[00] No artifacts_* project workspaces found.")
        return

    print("[00] Known project workspaces:")
    for p in projects:
        slug = p.name.removeprefix("artifacts_")
        log = p / "knowledge" / "current" / "clarification_log.md"
        sessions = 0
        if log.exists():
            try:
                sessions = len(re.findall(r"^## \d{4}-", log.read_text(encoding="utf-8"), re.MULTILINE))
            except Exception:
                sessions = 0
        print(f"  - {slug:<30} {sessions} clarification session(s)")


# ════════════════════════════════════════════════════════════════════════════
# LLM call
# ════════════════════════════════════════════════════════════════════════════

def _call_llm(
    system: str,
    user: str,
    model: str = _ANALYZE_MODEL,
    max_tokens: int = _MAX_TOKENS_ANALYZE,
) -> str:
    """
    Call an OpenAI-compatible chat completion API.

    OpenRouter model IDs contain '/', e.g. deepseek/deepseek-chat.
    If no API key is present, falls back to stdin mock for offline development.
    """
    if "/" in model:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1"
        model_id = model
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = "https://api.openai.com/v1"
        model_id = model

    if not api_key:
        print("\n[00][offline] No API key found. Paste LLM response then EOF:")
        return sys.stdin.read()

    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"].get("content", "")
        if not content or not content.strip():
            raise RuntimeError(f"LLM returned empty content: {data}")
        return content

    except Exception as exc:
        print(f"[00][error] LLM call failed: {exc}", file=sys.stderr)
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
      "citation": ""
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
- Ask directly and warmly:
  "Which customer segments should be available as filter options?"
  "How should the mobile layout differ from desktop?"
  "Should the dashboard support real-time updates or manual refresh?"

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
    """Run Phase 1+2: call LLM, parse JSON, enforce invariants."""
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

    result["findings"] = _enforce_tiers([f for f in findings if isinstance(f, dict)])
    result.setdefault("conflicts", [])
    result.setdefault("clarified_summary", "")
    return result


def _finding_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


# ════════════════════════════════════════════════════════════════════════════
# Deterministic tier rule engine
# ════════════════════════════════════════════════════════════════════════════

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
- Only policy-shaping follow-ups: architecture, access control, workflow design,
  compliance posture. Do NOT ask about labels, copy, exact thresholds, retry counts,
  scheduling intervals, or developer-decideable details.
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
        print(f"  [00][delta] Delta analysis failed ({exc}), continuing without update.")
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

    print(f"\n[00] Deriving impact statements ({len(pending)} decisions) ...")
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

def _write_report(
    session_id: str,
    req_hash: str,
    project_name: str,
    decisions: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
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

    report = {
        "requirement_hash": req_hash,
        "session_id": session_id,
        "project_name": project_name,
        "initial_findings": len(findings),
        "delta_injected": sum(1 for d in decisions if d["id"].startswith("NEW-")),
        "total_decisions": len(decisions),
        "tier1_answered": tier_counts.get(1, 0),
        "tier2_answered": tier_counts.get(2, 0),
        "tier3_accepted": tier3_accepted,
        "tier3_rejected": tier3_rejected,
        "conflicts_detected": len(conflicts),
        "unresolved": [u["id"] for u in unresolved],
        "decisions": decisions,
        "conflicts": conflicts,
    }

    CLARIFICATION_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[00] ✓ Report → {CLARIFICATION_REPORT}")


def _write_questions_md(
    project_name: str,
    session_id: str,
    findings: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        f"# Clarification Questions — {project_name}",
        f"Generated: {session_id[:10]}",
        "",
    ]

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

    def _render_q(finding: dict[str, Any], numbered: int) -> list[str]:
        out = [f"{numbered}. [{finding['id']}] {finding['text']}"]
        if finding.get("scenarios"):
            for scenario in finding["scenarios"]:
                out.append(f"   - {scenario}")
        return out

    if tier1_blocking:
        lines += ["## 🔴 Blocking — Tier 1 (cần trả lời trước khi estimate)", ""]
        for n, finding in enumerate(tier1_blocking, 1):
            lines += _render_q(finding, n)
            lines.append("")

    if tier1_other:
        lines += ["## 🔴 Tier 1 — Business & Policy Decisions", ""]
        for n, finding in enumerate(tier1_other, 1):
            priority_tag = (
                f"[{finding.get('priority', '').upper()}]"
                if finding.get("priority") != "high"
                else ""
            )
            rendered = _render_q(finding, n)
            if priority_tag:
                rendered[0] = f"{rendered[0]}  {priority_tag}"
            lines += rendered
            lines.append("")

    if tier2:
        lines += ["## 🟡 Tier 2 — Bounded Choices", ""]
        for n, finding in enumerate(tier2, 1):
            lines += _render_q(finding, n)
            lines.append("")

    if suggestions:
        lines += ["## 🟢 Tier 3 — Suggestions (confirm nếu đồng ý)", ""]
        for finding in suggestions:
            conf = finding.get("confidence", 0)
            conf_str = f"{int(conf * 100)}%" if conf else "?"
            lines.append(f"- [{finding['id']}] **Context:** {finding['text']}")
            lines.append(f"  **Suggestion:** {finding.get('suggestion', '')}")
            lines.append(f"  Confidence: {conf_str} | Reasoning: {finding.get('citation', 'N/A')}")
            lines.append("  → Accept / Reject / Modify?")
            lines.append("")

    if conflicts:
        lines += ["---", "## ⚠️ Conflicts Detected", ""]
        for conflict in conflicts:
            lines.append(f"- [{conflict['id']}] {conflict['description']}")
            if conflict.get("source_a"):
                lines.append(f"  New requirement: _{conflict['source_a']}_")
            if conflict.get("source_b"):
                lines.append(f"  Existing decision: _{conflict['source_b']}_")
            lines.append("")

    CLARIFICATION_QUESTIONS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[00] ✓ Questions → {CLARIFICATION_QUESTIONS}")


def _append_to_log(
    session_id: str,
    project_name: str,
    decisions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    blocks: list[str] = [
        f"## {session_id[:10]} | Project: {project_name} | Session: {session_id[11:19]}"
    ]

    for decision in decisions:
        tier_label = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}.get(decision["tier"], "?")
        accepted_label = (
            ""
            if decision["tier"] != 3
            else (" / accepted" if decision["accepted"] else " / rejected")
        )

        entry_lines = [
            f"### {decision['id']} [{tier_label}{accepted_label}]",
            f"**Q:** {decision['question']}",
            f"**A:** {decision['answer']}",
        ]

        if decision.get("impact"):
            entry_lines.append(f"**Impact:** {decision['impact']}")

        blocks.append("\n".join(entry_lines))

    if conflicts:
        conflict_lines = ["### Conflicts resolved this session"]
        for conflict in conflicts:
            conflict_lines.append(f"- [{conflict['id']}] {conflict['description']}")
        blocks.append("\n".join(conflict_lines))

    content = "\n\n" + "\n\n".join(blocks) + "\n"

    with CLARIFICATION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(content)

    print(f"[00] ✓ Log appended → {CLARIFICATION_LOG}")


def _write_clarified_req(content: str) -> None:
    CLARIFIED_REQ.write_text(content, encoding="utf-8")
    print(f"[00] ✓ Clarified requirement → {CLARIFIED_REQ}")


# ════════════════════════════════════════════════════════════════════════════
# Input gathering
# ════════════════════════════════════════════════════════════════════════════

def _gather_requirement(args: argparse.Namespace) -> str:
    if args.input:
        path = Path(args.input)
        if not path.exists():
            print(f"[00][error] File not found: {path}", file=sys.stderr)
            sys.exit(1)

        print(f"[00] Reading requirement from {path.name} ...")
        text = _read_input_file(path)

        if not text.strip():
            print(f"[00][error] Could not extract text from {path.name}.", file=sys.stderr)
            sys.exit(1)

        return text

    if args.text:
        return args.text

    print("[00] Paste / type requirement below.")
    print("     Press Ctrl-D (Unix) or Ctrl-Z Enter (Windows) to finish.\n")

    try:
        return sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n[00] Aborted.")
        sys.exit(0)


# ════════════════════════════════════════════════════════════════════════════
# CLI / Main
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="00_clarificator — requirement analysis & Q&A agent"
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Project workspace name. Prompted if omitted and PIPELINE_PROJECT is not set.",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        help="Path to requirement file (.md, .txt, .pdf)",
    )
    parser.add_argument(
        "--text",
        metavar="TEXT",
        help="Requirement as inline text string",
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip synthesis step; do not generate clarified_requirement.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis only, print findings, no Q&A and no file writes",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List all known artifacts_* project workspaces and exit",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --list-projects must work without PIPELINE_PROJECT.
    if args.list_projects and not args.project and not os.environ.get("PIPELINE_PROJECT"):
        _list_projects()
        return

    # Resolve and set project BEFORE ensure_dirs() or any LazyPath operation.
    project_name = _resolve_project(args.project)
    os.environ["PIPELINE_PROJECT"] = project_name
    ensure_dirs()

    if args.list_projects:
        _list_projects()
        return

    print(f"[00] Workspace: {project_name!r}")

    requirement_text = _gather_requirement(args)
    if not requirement_text.strip():
        print("[00][error] Empty requirement input.", file=sys.stderr)
        sys.exit(1)

    req_hash = _sha256(requirement_text)
    session_id = _now_iso()

    # ── Load knowledge context ────────────────────────────────────────────────
    log_text = _load_clarification_log()
    standalone = not (KNOWLEDGE_BASE.exists() or bool(log_text))

    if standalone:
        print("[00] Standalone mode — no knowledge context found for this workspace.")
        knowledge_context = ""
        answered_qa_pairs: list[dict[str, str]] = []
    else:
        print("[00] Loading knowledge context ...")
        knowledge_context = _load_knowledge_context()
        answered_qa_pairs = _extract_answered_qa_pairs(log_text)
        if answered_qa_pairs:
            print(f"[00] Loaded {len(answered_qa_pairs)} past Q/A pairs for semantic dedup")

    # ── Phase 1+2: Analyze ────────────────────────────────────────────────────
    print("[00] Analyzing requirement ...")
    analysis = _analyze(requirement_text, knowledge_context, answered_qa_pairs)

    inferred_name = analysis.get("project_name", "Unknown")
    if project_name.lower() == "unknown" and inferred_name != "Unknown":
        project_name = inferred_name
        print(f"[00] Project name inferred from requirement: {project_name!r}")

    findings = analysis.get("findings", [])
    conflicts = analysis.get("conflicts", [])
    clarified_sum = analysis.get("clarified_summary", "")

    if not isinstance(findings, list):
        findings = []
    if not isinstance(conflicts, list):
        conflicts = []

    if not findings and not conflicts:
        print("[00] ✓ No ambiguities found — requirement is already clear.")
        if not args.dry_run:
            _write_clarified_req(requirement_text)
        return

    print(f"[00] Found {len(findings)} findings, {len(conflicts)} conflicts.")

    if args.dry_run:
        _print_banner("Dry run — findings only")
        for finding in _sort_findings(findings):
            _print_finding(finding, 0, len(findings))
        if conflicts:
            print("\n⚠️  Conflicts:")
            for conflict in conflicts:
                print(f"  {conflict['id']}: {conflict['description']}")
        return

    _write_questions_md(project_name, session_id, _sort_findings(findings), conflicts)

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
                f"\n[00][warn] {len(loud)} blocking question(s) could not be resolved "
                "(unmet or circular dependencies) — review before proceeding:"
            )
            for item in loud:
                print(f"  ⚠️  {item['id']}: {item['text'][:70]}...")

        if silent:
            print(
                f"\n[00] {len(silent)} low-priority question(s) skipped due to "
                "inconsistent dependencies (recorded in report)."
            )

    _batch_derive_impacts(decisions)

    _write_report(
        session_id,
        req_hash,
        project_name,
        decisions,
        unresolved,
        conflicts,
        findings,
    )
    _append_to_log(session_id, project_name, decisions, conflicts)

    if not args.no_synth:
        print("\n[00] Synthesizing clarified requirement ...")
        clarified_md = _synthesize_requirement(
            requirement_text,
            decisions,
            conflicts,
            clarified_sum,
        )
        _write_clarified_req(clarified_md)

    _print_banner(f"Done — {len(decisions)} decisions recorded  [{project_name}]")
    print(f"  Workspace log: {CLARIFICATION_LOG}")
    print(f"  Artifacts     → {CLARIFIED_REQ}")


if __name__ == "__main__":
    main()
