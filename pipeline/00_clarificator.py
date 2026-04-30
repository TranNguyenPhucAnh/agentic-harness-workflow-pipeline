#!/usr/bin/env python3
"""
00_clarificator.py
==================
Clarificator agent — upstream nhất trong pipeline, chạy trước Estimator.

Nhận raw requirement (file hoặc text), phân tích holes/conflicts/assumptions,
tổ chức Q&A với user theo 3-tier system, và output clarified_requirement.md
cùng với structured report cho downstream steps.

Usage:
    python 00_clarificator.py --project my-app --input requirement.pdf
    python 00_clarificator.py --project my-app --input spec_draft.md
    python 00_clarificator.py --project my-app --text "Build a dashboard..."
    python 00_clarificator.py --project my-app   # interactive multiline prompt
    python 00_clarificator.py                    # prompts for project name

--project is required for dedup to work across sessions. Each project gets its
own clarification_log_<slug>.md so decisions from project A never pollute B.

Artifacts produced (owner: 00_clarificator):
    run/clarification_report.json
    run/clarification_questions.md
    knowledge/current/clarification_log_<project_slug>.md   ← append-only, per-project
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
from typing import Any  # noqa: F401 — kept for future typed helpers

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
# NOTE: CLARIFICATION_LOG is NOT a module-level constant — it is per-project.
#       Use _clarification_log_path(project_slug) to get the correct path.
CLARIFICATION_REPORT    = RUN_DIR     / "clarification_report.json"
CLARIFICATION_QUESTIONS = RUN_DIR     / "clarification_questions.md"
CLARIFIED_REQ           = STATE_DIR   / "clarified_requirement.md"
# legacy single-file log kept for backward compat read-only migration
_LEGACY_CLARIFICATION_LOG = CURRENT_DIR / "clarification_log.md"


def _slugify(name: str) -> str:
    """Convert a project name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug or "unknown"


def _clarification_log_path(project_slug: str) -> Path:
    """Per-project append-only log: knowledge/current/clarification_log_<slug>.md"""
    return CURRENT_DIR / f"clarification_log_{project_slug}.md"


def _list_known_projects() -> list[str]:
    """Return sorted list of project slugs that have existing logs."""
    logs = sorted(CURRENT_DIR.glob("clarification_log_*.md"))
    slugs = []
    for p in logs:
        m = re.match(r"clarification_log_(.+)\.md$", p.name)
        if m:
            slugs.append(m.group(1))
    return slugs

# ── Model config ──────────────────────────────────────────────────────────────
_ANALYZE_MODEL   = "deepseek/deepseek-chat"          # Phase 1+2: reasoning heavy
_SUGGEST_MODEL   = "deepseek/deepseek-chat"          # Phase 3 Tier3: lighter OK but same default
_TIER3_MIN_CONF  = 0.75                              # below this → promote to Tier 2

# ── Token / context limits ────────────────────────────────────────────────────
# Target use case: mini mode with long AC + detailed descriptions.
# DeepSeek V3 context window: 64k tokens. Output cap set conservatively.
#
# _MAX_TOKENS: max output tokens per LLM call.
#   - Analyze call: up to ~15 findings × ~200 tokens each = ~3000 output tokens.
#     8192 is more than enough; kept as ceiling.
#   - Delta call: small output (2–3 new findings max). 2048 is sufficient.
#   - Synthesis call: full clarified_requirement.md, can be long. 4096 safer.
_MAX_TOKENS_ANALYZE   = 8192   # Phase 1+2 analyze
_MAX_TOKENS_DELTA     = 2048   # delta follow-up (small output by design)
_MAX_TOKENS_SYNTHESIS = 4096   # clarified_requirement.md generation

# _DELTA_REQ_CHARS: how many chars of the requirement to include in delta calls.
#   Delta calls only need enough context to understand what was answered and why.
#   Full requirement passed for correctness; truncated only if very long.
#   4000 chars ≈ ~1000 tokens — enough for a detailed AC block.
_DELTA_REQ_CHARS = 4000

# No ceiling on number of findings per session — LLM generates as many as needed.
# The rule engine and dedup filter down organically.

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


def _load_clarification_log(project_slug: str) -> str:
    """Load per-project log. Falls back to legacy global log on first migration."""
    project_log = _clarification_log_path(project_slug)
    if project_log.exists():
        return project_log.read_text(encoding="utf-8")
    # One-time migration: if legacy global log exists, read it but don't trust it
    # for dedup (different projects mixed in) — warn instead.
    if _LEGACY_CLARIFICATION_LOG.exists():
        print(
            f"[00][warn] Found legacy clarification_log.md but no per-project log for "
            f"'{project_slug}'. Legacy log ignored for dedup. "
            f"Consider migrating: cp clarification_log.md clarification_log_{project_slug}.md"
        )
    return ""


def _extract_answered_qa_pairs(log_text: str) -> list[dict]:
    """
    Extract structured Q/A pairs from the log for semantic dedup.
    Returns list of {id, question, answer} dicts.
    No truncation — full text needed so LLM can detect semantic equivalence.
    """
    pairs: list[dict] = []
    # Pattern: ### CLR-NNN [...]\n**Q:** ...\n**A:** ...
    blocks = re.split(r"\n(?=###\s+CLR-)", log_text)
    for block in blocks:
        id_match = re.search(r"###\s+(CLR-\d{3})", block)
        q_match  = re.search(r"\*\*Q:\*\*\s*(.+?)(?=\n\*\*|\Z)", block, re.DOTALL)
        a_match  = re.search(r"\*\*A:\*\*\s*(.+?)(?=\n\*\*|\Z)", block, re.DOTALL)
        if id_match and q_match:
            pairs.append({
                "id":       id_match.group(1),
                "question": q_match.group(1).strip(),
                "answer":   a_match.group(1).strip() if a_match else "",
            })
    return pairs


def _load_knowledge_context(project_slug: str) -> str:
    parts: list[str] = []
    if KNOWLEDGE_BASE.exists():
        parts.append(f"=== base.md ===\n{KNOWLEDGE_BASE.read_text(encoding='utf-8')}")
    log_text = _load_clarification_log(project_slug)
    if log_text:
        parts.append(f"=== clarification_log_{project_slug}.md ===\n{log_text}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call (model-agnostic thin wrapper — swap backend as needed)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, model: str = _ANALYZE_MODEL, max_tokens: int = _MAX_TOKENS_ANALYZE) -> str:
    """
    Call an LLM via OpenAI-compatible API.
    Reads OPENAI_API_KEY / DEEPSEEK_API_KEY / ANTHROPIC_API_KEY from env.
    Falls back to a simple stdin mock when running offline.
    """
    import os

    # Detect which provider to use based on model prefix
    if "/" in model:  # OpenRouter format: "provider/model-name"
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
            "max_tokens": max_tokens,
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
1. SEMANTIC DEDUP — Do NOT generate a new finding if the ALREADY_ANSWERED_QA
   section contains a question that is semantically equivalent or closely related
   to the potential new finding. Equivalence means: same topic, same decision
   space, same impact — even if worded differently.
   If an existing answer already resolves the ambiguity, skip generating it.
   Instead, if the existing answer is relevant, you may reference it in
   "clarified_summary" but do NOT put it in findings[].
2. CONFLICT DETECTION — Does the new requirement contradict any past decision?
3. ASSUMPTION SURFACING — Does the requirement assume behavior that may not exist yet?

Output ONLY a valid JSON object — no markdown fences, no preamble.
Schema:
{
  "project_name": "<inferred from requirement, or 'Unknown'>",
  "findings": [
    {
      "id": "CLR-001",
      "text": "<the clarification question — see TONE RULES below>",
      "tier": 1 | 2 | 3,
      "category": "business" | "logic" | "technical" | "design",
      "priority": "blocking" | "high" | "medium" | "low",
      "depends_on": ["CLR-XXX"],
      "scenarios": ["<option A>", "<option B>", "<option C>"],
      "suggestion": "<recommended approach>",
      "confidence": 0.0,
      "citation": "<source: base.md §X, past decision from log, pattern Y, etc.>"
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
  "clarified_summary": "<one paragraph: what the requirement IS asking for, stating confidently what is already known from past decisions>"
}

TONE RULES (strictly enforced for the "text" field):
- Write as a natural, collaborative question — NOT a judgment or critique.
- NEVER start with: "The requirement does not specify", "The requirement fails to",
  "There is no mention of", "It is unclear", or any phrasing that implies the
  requirement is deficient or that the author made a mistake.
- INSTEAD, ask directly and warmly. Examples of good phrasing:
  ✓ "Which customer segments should be available as filter options?"
  ✓ "How should the mobile layout differ from the desktop view?"
  ✓ "When an order ships, which notification channel should be used?"
  ✓ "Should the dashboard support real-time updates or manual refresh?"
  ✓ "What authentication method should protect this dashboard?"
- The question should feel like a natural follow-up from a thoughtful colleague,
  not an audit finding. Keep it concise (1–2 sentences max).

TIER RULES (strict):
- Tier 1: answer space is subjective OR requires business/product decision from client.
           → MUST include 2–4 concrete representative scenarios[] as starting options.
             These are not exhaustive — user can always type a custom answer.
             Good Tier 1 scenarios: realistic choices a product owner would consider.
           → No suggestion field needed.
- Tier 2: answer space is bounded and fully enumerable (≤5 concrete options cover all cases).
           → MUST include all realistic scenarios[] (2–5 items).
           → No suggestion field needed.
- Tier 3: answer is near-deterministic from context (tech stack, patterns, stated constraints).
           → confidence ≥ 0.75 required. MUST include citation explaining why.
           → confidence < 0.75 → downgrade to Tier 2.
           → scenarios[] can be empty for Tier 3.

SCENARIOS RULE: scenarios[] MUST be non-empty for Tier 1 and Tier 2.
Every Tier 1 and Tier 2 finding without scenarios is malformed — always provide them.

DEPENDENCY RULES:
- If finding B only makes sense after finding A is answered, put A's id in B's depends_on.
- Findings with depends_on should have priority "low" or "medium" initially.

PRIORITY RULES — TWO-TIER HEURISTIC:

Assign priority by asking: "Does this answer change the architecture, approval
semantics, or scope of the system?" If yes → blocking or high. If it only
affects one edge case or one workflow step → medium or low.

POLICY-SHAPING ambiguities (→ blocking / high):
  These questions, if left unanswered, would cause architectural rework or
  incorrect scope. Examples for complex enterprise systems:
  - Risk scoring model: how is residual risk calculated? Who defines thresholds?
  - Approval semantics: can final approval proceed with Approved-with-Conditions?
  - Quorum / committee logic: does Risk Committee require 1 approver or a quorum?
  - "Material change" definition for renewals: what triggers a full re-review?
  - SLA basis: business hours vs calendar hours vs timezone-aware?
  - Rule ownership: who can change routing rules, are rules versioned?
  - Multi-region data residency: does data need to stay within each region?
  - Audit retention scope: which event types are immutable vs admin-correctable?

IMPLEMENTATION-DETAIL ambiguities (→ medium / low):
  These questions affect one feature or edge case, not the overall system shape.
  Examples:
  - Draft save behavior with missing fields
  - Clone field restrictions
  - Reviewer reassignment edge cases
  - Notification preference granularity

CALIBRATION RULE: The number of findings must match the actual complexity of the
requirement — do not stop early. Scale findings to spec size:
  - Simple spec (1–3 features, 1 role): 3–5 findings is normal.
  - Medium spec (4–7 features, 2–4 roles): 6–10 findings is normal.
  - Complex enterprise spec (8+ epics, multiple roles, compliance obligations,
    multi-region, SLA semantics, routing rules): 10–20 findings is expected.

For complex specs, you MUST surface at minimum:
  - All policy-shaping ambiguities in the blocking/high category FIRST.
  - Common enterprise gaps that are almost always ambiguous:
      residual risk scoring model, conditional approval semantics,
      SLA business hours vs calendar hours, quorum for committee approvals,
      what constitutes "material change" for renewals,
      routing rule ownership and versioning,
      audit retention scope and immutability rules.
  - Only AFTER exhausting blocking/high should you surface medium/low findings.

If you only found 5 findings for a complex enterprise spec, you have under-generated.
Re-examine the spec for the gaps listed above before returning.

"blocking": this answer must be known before estimate or architecture can proceed.
"high": significantly shapes scope, approval logic, or integration contracts.
"medium": affects one module, one workflow step, or one edge case.
"low": nice-to-have, can be decided during implementation.

ID FORMAT: CLR-001, CLR-002, ... (3-digit zero-padded, sequential, start from 001 each session)
"""


def _analyze(
    requirement_text: str,
    knowledge_context: str,
    answered_qa_pairs: list[dict],
) -> dict:
    """Run Phase 1+2: call LLM, parse JSON, enforce Tier 3 confidence threshold."""

    # Build semantic dedup block — full Q/A text so LLM can detect meaning, not just IDs
    if answered_qa_pairs:
        qa_lines = []
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

    # Strip markdown fences if model wrapped output anyway
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as exc:
        print(f"[00][error] Failed to parse LLM analysis JSON: {exc}")
        print("Raw output:\n", raw[:500])
        raise

    # Post-process: enforce invariants
    for f in result.get("findings", []):
        # Enforce Tier 3 confidence threshold — downgrade if below
        if f.get("tier") == 3 and f.get("confidence", 0) < _TIER3_MIN_CONF:
            f["tier"] = 2
            f["confidence"] = None
        # Enforce scenarios non-empty for Tier 1 and Tier 2
        # If LLM forgot, add a generic fallback so UI always shows options
        if f.get("tier") in (1, 2) and not f.get("scenarios"):
            f["scenarios"] = ["Yes / proceed as implied", "No / needs different approach", "Other (specify below)"]

    result["findings"] = _enforce_tiers(result.get("findings", []))
    return result


def _finding_hash(text: str) -> str:
    """Stable 8-char hash of normalized finding text — cross-round identity."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Rule engine: enforce tiers deterministically (Test 3)
# LLM proposes → rule engine enforces. Tier is a rule, not a suggestion.
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_tiers(findings: list[dict]) -> list[dict]:
    """
    Correct tier assignment using structural rules only.
    No keywords, no LLM, 100% deterministic. Runs after _analyze().

    Rules (first match wins per finding):
      R1: suggestion + confidence ≥ 0.75 + citation       → Tier 3
      R2: scenarios ≤ 5 + category in (technical/design/logic) → Tier 2
      R3: category == business OR priority == blocking
          OR no scenarios                                  → Tier 1
      R4: Tier 3 missing citation OR confidence < 0.75    → demote to Tier 2

    Post-assignment invariants:
      - Tier 1/2: scenarios must be non-empty
      - Tier 3: suggestion must be present
    """
    for f in findings:
        suggestion = (f.get("suggestion") or "").strip()
        confidence = f.get("confidence") or 0.0
        citation   = (f.get("citation") or "").strip()
        scenarios  = f.get("scenarios") or []
        category   = f.get("category", "")
        priority   = f.get("priority", "")

        bounded      = bool(scenarios) and len(scenarios) <= 5
        near_det_cat = category in ("technical", "design", "logic")

        # R4: validate existing Tier 3 FIRST — before any reassignment
        # If LLM said Tier 3 but evidence is weak, demote to Tier 2 immediately
        # so R1/R2/R3 see the corrected tier (not the original Tier 3 claim)
        if f.get("tier") == 3 and (not citation or confidence < _TIER3_MIN_CONF):
            f["tier"] = 2
            f["confidence"] = None

        # R1: structural promotion to Tier 3 (strongest positive signal)
        if suggestion and confidence >= _TIER3_MIN_CONF and citation:
            f["tier"] = 3
        # R2: bounded enumerable → Tier 2
        elif bounded and near_det_cat:
            f["tier"] = 2
        # R3: subjective / business / no scenarios → Tier 1
        # Exception: if finding has a suggestion (was intended as Tier 3 but
        # demoted by R4), keep it at Tier 2 — not Tier 1. "No scenarios" alone
        # is not enough to force Tier 1 when there is a concrete suggestion.
        elif category == "business" or priority == "blocking" or (
            not scenarios and not suggestion
        ):
            f["tier"] = 1
        # else: keep current tier (already validated by R4 above)

        # Invariants
        if f.get("tier") in (1, 2) and not f.get("scenarios"):
            f["scenarios"] = [
                "Yes — proceed as implied",
                "No — needs a different approach",
                "Other (type custom answer)",
            ]
        if f.get("tier") == 3 and not f.get("suggestion"):
            f["suggestion"] = "(see citation)"

    return findings


# ─────────────────────────────────────────────────────────────────────────────
# Delta analysis: focused follow-up after a Tier 1 blocking answer (Test 4)
# ─────────────────────────────────────────────────────────────────────────────

_DELTA_SYSTEM = """
You are a requirements analyst. A blocking clarification question was just answered.
Your task is ONLY to determine:
  1. What NEW questions does this answer reveal that are not already in the queue?
  2. Which EXISTING queue questions are now irrelevant or resolved by this answer?

Output ONLY a valid JSON object — no markdown, no preamble.
Schema:
{
  "new_findings": [
    {
      "id": "NEW-001",
      "text": "<the clarification question — must follow TONE RULES>",
      "tier": 1 | 2 | 3,
      "category": "business" | "logic" | "technical" | "design",
      "priority": "blocking" | "high" | "medium" | "low",
      "depends_on": [],
      "scenarios": ["<option A>", "<option B>"],
      "suggestion": "",
      "confidence": 0.0,
      "citation": ""
    }
  ],
  "invalidated_ids": ["CLR-XXX", "CLR-YYY"]
}

TONE RULES (same as main analysis — enforced):
- Write as a natural, collaborative question.
- NEVER start with "The requirement does not specify", "It is unclear", or similar
  phrasing that implies the author made a mistake.
- Ask directly: "Which X should be used?", "How should Y behave when Z?", etc.

RULES:
- new_findings[] should only contain questions that COULD NOT have been asked
  before this answer was known. Do not regenerate existing questions.
- invalidated_ids[] should list queue item IDs that this answer makes moot.
- If nothing changes, return {"new_findings": [], "invalidated_ids": []}.
- IDs for new findings use prefix NEW- to avoid collisions with existing CLR- IDs.
- Apply same SCENARIOS RULE: Tier 1 and Tier 2 must have non-empty scenarios[].
- Apply same PRIORITY HEURISTIC: new findings revealed by a blocking answer are
  likely also blocking or high. Only mark medium/low if they are clearly
  implementation-detail questions, not policy-shaping ones.
"""


def _delta_analyze(
    answered_finding: dict,
    answer: str,
    requirement_text: str,
    current_queue_ids: list[str],
) -> tuple[list[dict], list[str]]:
    """
    After a Tier 1 blocking answer, ask LLM:
      - what new questions does this reveal?
      - which pending questions are now invalidated?

    Returns (new_findings, invalidated_ids).
    Cheap: input is small, output is targeted delta only.
    """
    queue_summary = ", ".join(current_queue_ids) if current_queue_ids else "none"

    req_snippet = requirement_text
    if len(req_snippet) > _DELTA_REQ_CHARS:
        req_snippet = req_snippet[:_DELTA_REQ_CHARS] + f"\n... [truncated — {len(requirement_text)} chars total]"

    user_msg = f"""REQUIREMENT CONTEXT:
{req_snippet}

ANSWERED QUESTION:
  ID: {answered_finding['id']}
  Text: {answered_finding['text']}
  Category: {answered_finding.get('category', '')}
  Priority: {answered_finding.get('priority', '')}

USER ANSWER: {answer}

CURRENT PENDING QUEUE (IDs still to be asked): {queue_summary}

Given this answer, what new questions are revealed and which pending ones are now moot?
Output only the JSON object."""

    try:
        raw = _call_llm(_DELTA_SYSTEM, user_msg, max_tokens=_MAX_TOKENS_DELTA)
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        result = json.loads(clean)
    except Exception as exc:
        # Delta failure is non-fatal — log and continue with existing queue
        print(f"  [00][delta] Delta analysis failed ({exc}), continuing without update.")
        return [], []

    new_findings    = result.get("new_findings", [])
    invalidated_ids = result.get("invalidated_ids", [])

    # Apply rule engine to new findings too
    new_findings = _enforce_tiers(new_findings)

    return new_findings, invalidated_ids


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
    """
    Tier 1: show representative options (not exhaustive) + always allow custom answer.
    User can pick a number OR type anything freely.
    """
    scenarios = f.get("scenarios", [])
    if scenarios:
        print("\n  Options:")
        for i, s in enumerate(scenarios, 1):
            print(f"    {i}. {s}")
        print()
        while True:
            choice = input(f"  → Choose 1–{len(scenarios)} or type custom answer: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(scenarios):
                    return scenarios[idx]
            if choice:
                return choice
            print("  Please enter a choice.")
    else:
        # Fallback: pure free-text (should not happen after enforce above)
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


def _dependencies_satisfied(f: dict, answered: dict[str, str], known_ids: set[str] | None = None) -> bool:
    """
    Return True if all deps are answered.
    If known_ids is provided, deps that don't exist in the queue at all
    are treated as already-satisfied (dangling ref from LLM — skip silently).
    """
    for dep in f.get("depends_on", []):
        if known_ids is not None and dep not in known_ids:
            # Dangling reference — dep was never in the queue, treat as resolved
            continue
        if dep not in answered:
            return False
    return True


def _derive_impact(question: str, answer: str, category: str) -> str:
    """
    Produce a one-line impact statement from a Q&A pair via LLM.
    NOTE: Called in batch AFTER the interactive loop completes — never inside it.
    """
    if not answer or answer.lower() in ("accepted", "yes", "no", "rejected", "(no answer provided)"):
        return ""
    try:
        system = (
            "You are a technical analyst. Given a clarification Q&A pair, "
            "output ONE complete sentence (max 30 words) describing the "
            "implementation impact of this decision. "
            "The sentence must be grammatically complete — never cut off mid-word or mid-phrase. "
            "No preamble, no trailing punctuation issues."
        )
        user = f"Question: {question}\nAnswer: {answer}\nCategory: {category}"
        raw = _call_llm(system, user, max_tokens=256).strip()
        # Take first sentence only; strip stray leading quotes
        first_line = raw.splitlines()[0].strip().strip('"').strip("'")
        return first_line
    except Exception:
        return ""


def _batch_derive_impacts(decisions: list[dict]) -> None:
    """
    Enrich decisions[*]["impact"] in-place via LLM calls.
    Called once after the interactive loop — all user Q&A is already done.
    """
    pending = [d for d in decisions if not d.get("impact")]
    if not pending:
        return
    print(f"\n[00] Deriving impact statements ({len(pending)} decisions) ...")
    for d in pending:
        d["impact"] = _derive_impact(d["question"], d["answer"], d.get("category", ""))


def _run_interactive_loop(
    findings: list[dict],
    project_name: str,
    requirement_text: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    Drive the Q&A loop. Returns (decisions, unresolved).
    decisions: list of {id, tier, question, answer, accepted, impact}

    Delta loop (Test 4): after each Tier 1 blocking answer, calls _delta_analyze()
    to inject new findings and remove invalidated ones from the queue.
    Uses _finding_hash() for cross-round identity so NEW-* items are deduped
    against already-answered content even if IDs differ.
    """
    decisions:  list[dict] = []
    unresolved: list[dict] = []
    answered:   dict[str, str] = {}  # id → answer text
    answered_hashes: set[str]  = set()  # content hashes of answered questions

    # Work from a stable sorted list; deferred items go into a separate pending set
    queue:    list[dict] = _sort_findings(list(findings))
    deferred: set[str]   = set()   # ids of items pushed-back at least once
    total    = len(queue)
    answered_count = 0

    # All IDs ever seen in the queue — used to detect dangling depends_on refs
    known_ids: set[str] = {f["id"] for f in queue}

    _print_banner(f"Clarification session — {project_name}")
    print(f"  {total} findings to process.\n")

    i = 0
    while i < len(queue):
        f = queue[i]
        i += 1

        # Already processed — check both ID and content hash
        if f["id"] in answered:
            continue
        if _finding_hash(f["text"]) in answered_hashes:
            # Semantically duplicate from delta injection — skip silently
            continue

        # Dependency not yet satisfied — defer once
        if not _dependencies_satisfied(f, answered, known_ids):
            if f["id"] not in deferred:
                deferred.add(f["id"])
                queue.append(f)
            else:
                # Second time we can't satisfy deps → circular/unresolvable
                unresolved.append(f)
            continue

        answered_count += 1
        # Remaining = items not yet answered and deps satisfied (approximate)
        pending_ready = sum(
            1 for x in queue[i:]
            if x["id"] not in answered and _dependencies_satisfied(x, answered)
        )
        display_total = answered_count + pending_ready
        _print_finding(f, answered_count, display_total)

        tier     = f.get("tier", 1)
        accepted = True

        if tier == 1:
            answer = _ask_tier1(f)
        elif tier == 2:
            answer = _ask_tier2(f)
        else:
            answer, accepted = _ask_tier3(f)

        answered[f["id"]] = answer
        answered_hashes.add(_finding_hash(f["text"]))
        decisions.append({
            "id":       f["id"],
            "tier":     tier,
            "category": f.get("category", ""),
            "priority": f.get("priority", ""),
            "question": f["text"],
            "answer":   answer,
            "accepted": accepted,
            "impact":   "",  # filled by _batch_derive_impacts after loop
        })

        # ── Delta loop: inject new findings after Tier 1 blocking answer ─────
        if tier == 1 and f.get("priority") == "blocking" and requirement_text:
            current_queue_ids = [
                x["id"] for x in queue[i:]
                if x["id"] not in answered
            ]
            print(f"  [delta] Checking for follow-up questions after {f['id']}...")
            new_findings, invalidated_ids = _delta_analyze(
                f, answer, requirement_text, current_queue_ids
            )

            # Remove invalidated items (mark as answered with sentinel)
            for inv_id in invalidated_ids:
                if inv_id not in answered:
                    answered[inv_id] = "[invalidated by delta]"
                    print(f"  [delta] Invalidated: {inv_id}")

            # Inject new findings — dedup by content hash
            injected = 0
            for nf in new_findings:
                if _finding_hash(nf["text"]) in answered_hashes:
                    continue  # semantically already answered
                if nf["id"] in answered:
                    continue  # ID collision
                queue.append(nf)
                known_ids.add(nf["id"])  # register so deps referencing it are not dangling
                injected += 1
            if injected:
                print(f"  [delta] Injected {injected} new finding(s) into queue.")
            elif new_findings:
                print(f"  [delta] {len(new_findings)} potential finding(s) already covered.")
            else:
                print(f"  [delta] No follow-up questions revealed.")

    return decisions, unresolved


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis: produce clarified requirement
# ─────────────────────────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """
You are a technical writer. Given a raw requirement document and a set of
clarification decisions, produce ONE clean, unified "Clarified Requirement"
document in markdown.

STRICT STRUCTURAL RULES — violations produce unusable output:
1. OUTPUT THE DOCUMENT EXACTLY ONCE. Do not repeat any section, heading,
   or block of content. Each section (Context, Workflow, Functional Requirements,
   NFR, Integrations, Out of Scope, Acceptance Criteria) appears exactly one time.
2. Use the ORIGINAL REQUIREMENT as the single structural template.
   Walk through it section by section, top to bottom, in one pass.
   Do not restructure, reorder, or merge sections differently.
3. For each section: incorporate the relevant decisions inline by updating
   the text naturally. Do not add a parallel or duplicate version of the section.
4. Preserve all original bullet points and list items. Do not drop content
   that was not affected by a decision.
5. Every list item must start with "- " on its own line. Never run list items
   into prose without a line break.
6. Add a "## Decisions Log" section at the very end — one line per decision:
   "- **CLR-XXX**: <one-line summary of the answer and its impact>"
7. No preamble, no postamble. Output only the markdown document.

ANTI-PATTERNS — never do these:
- Do not output a short summary version followed by a long full version.
- Do not output the Functional Requirements twice (once short, once detailed).
- Do not copy the NFR or Out of Scope section more than once.
- Do not leave "Automated security questionnaire..." mid-sentence merged into
  an unrelated section.
"""


def _synthesize_requirement(
    original: str,
    decisions: list[dict],
    conflicts: list[dict],
    summary: str,
) -> str:
    decisions_text = "\n".join(
        f"- {d['id']} [{d.get('priority','').upper()}]: {d['question']} → {d['answer']}"
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

    return _call_llm(_SYNTHESIS_SYSTEM, user_msg, max_tokens=_MAX_TOKENS_SYNTHESIS)


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
            if c.get("source_a"):
                lines.append(f"  New requirement: _{c['source_a']}_")
            if c.get("source_b"):
                lines.append(f"  Existing decision: _{c['source_b']}_")
            lines.append("")

    CLARIFICATION_QUESTIONS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[00] ✓ Questions → {CLARIFICATION_QUESTIONS}")


def _append_to_log(
    session_id: str,
    project_name: str,
    project_slug: str,
    decisions: list[dict],
    conflicts: list[dict],
) -> None:
    """Append this session's decisions to the per-project clarification_log_<slug>.md."""
    log_path = _clarification_log_path(project_slug)

    blocks: list[str] = []
    blocks.append(
        f"## {session_id[:10]} | Project: {project_name} | Session: {session_id[11:19]}"
    )

    for d in decisions:
        tier_label = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}.get(d["tier"], "?")
        accepted_label = (
            "" if d["tier"] != 3
            else (" / accepted" if d["accepted"] else " / rejected")
        )
        entry_lines = [
            f"### {d['id']} [{tier_label}{accepted_label}]",
            f"**Q:** {d['question']}",
            f"**A:** {d['answer']}",
        ]
        if d.get("impact"):
            entry_lines.append(f"**Impact:** {d['impact']}")
        blocks.append("\n".join(entry_lines))

    if conflicts:
        conflict_lines = ["### Conflicts resolved this session"]
        for c in conflicts:
            conflict_lines.append(f"- [{c['id']}] {c['description']}")
        blocks.append("\n".join(conflict_lines))

    # Each block separated by blank line; leading newline ensures separation
    # from previous session already in file
    content = "\n\n" + "\n\n".join(blocks) + "\n"

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(content)
    print(f"[00] ✓ Log appended → {log_path}")


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

def _resolve_project(arg_project: str | None) -> tuple[str, str]:
    """
    Resolve (project_name, project_slug).
    If --project not given, show existing workspaces and prompt user.
    Returns (display_name, slug).
    """
    if arg_project:
        name = arg_project.strip()
        return name, _slugify(name)

    ensure_dirs()
    known = _list_known_projects()

    print()
    if known:
        print("Known workspaces:")
        for i, slug in enumerate(known, 1):
            print(f"  {i}. {slug}")
        print()
        raw = input("Enter project name (or number to select existing): ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(known):
                slug = known[idx]
                return slug.replace("-", " ").title(), slug
        name = raw or "unknown"
    else:
        print("[00] No existing workspaces found.")
        name = input("Enter new project name: ").strip() or "unknown"

    return name, _slugify(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="00_clarificator — requirement analysis & Q&A agent"
    )
    parser.add_argument("--project",  metavar="NAME",
                        help="Project workspace name (required for dedup). Prompted if omitted.")
    parser.add_argument("--input",    metavar="FILE",
                        help="Path to requirement file (.md, .txt, .pdf)")
    parser.add_argument("--text",     metavar="TEXT",
                        help="Requirement as inline text string")
    parser.add_argument("--no-synth", action="store_true",
                        help="Skip synthesis step (don't generate clarified_requirement.md)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run analysis only, print findings, no Q&A and no file writes")
    parser.add_argument("--list-projects", action="store_true",
                        help="List all known project workspaces and exit")
    args = parser.parse_args()

    ensure_dirs()

    if args.list_projects:
        known = _list_known_projects()
        if not known:
            print("[00] No project workspaces found.")
        else:
            print("[00] Known workspaces:")
            for slug in known:
                log_path = _clarification_log_path(slug)
                sessions = len(re.findall(r"^## \d{4}-", log_path.read_text(encoding="utf-8"), re.MULTILINE))
                print(f"  • {slug}  ({sessions} session{'s' if sessions != 1 else ''})")
        return

    # ── Resolve project workspace ─────────────────────────────────────────────
    project_name, project_slug = _resolve_project(args.project)
    print(f"[00] Workspace: {project_name!r} (slug: {project_slug})")

    # ── Gather input ──────────────────────────────────────────────────────────
    requirement_text = _gather_requirement(args)
    req_hash         = _sha256(requirement_text)
    session_id       = _now_iso()

    # ── Load knowledge context (project-scoped) ───────────────────────────────
    log_text      = _load_clarification_log(project_slug)
    standalone    = not (KNOWLEDGE_BASE.exists() or bool(log_text))
    if standalone:
        print("[00] Standalone mode — no knowledge context found for this workspace.")
        knowledge_context  = ""
        answered_qa_pairs: list[dict] = []
    else:
        print("[00] Loading knowledge context ...")
        knowledge_context = _load_knowledge_context(project_slug)
        answered_qa_pairs = _extract_answered_qa_pairs(log_text)
        if answered_qa_pairs:
            print(
                f"[00] Loaded {len(answered_qa_pairs)} past Q/A pairs for semantic dedup "
                f"(workspace: {project_slug})"
            )

    # ── Phase 1+2: Analyze ────────────────────────────────────────────────────
    print("[00] Analyzing requirement ...")
    analysis      = _analyze(requirement_text, knowledge_context, answered_qa_pairs)
    # Use --project name as canonical; only fall back to LLM-inferred if project is "unknown"
    inferred_name = analysis.get("project_name", "Unknown")
    if project_name == "unknown" and inferred_name != "Unknown":
        project_name = inferred_name
        project_slug = _slugify(project_name)
        print(f"[00] Project name inferred from requirement: {project_name!r}")

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
    decisions, unresolved = _run_interactive_loop(findings, project_name, requirement_text)

    if unresolved:
        # Split by severity: business/blocking = loud warn, rest = silent to report
        loud = [u for u in unresolved
                if u.get("category") == "business" or u.get("priority") == "blocking"]
        silent = [u for u in unresolved if u not in loud]

        if loud:
            print(f"\n[00][warn] {len(loud)} blocking question(s) could not be resolved "
                  f"(unmet or circular dependencies) — review before proceeding:")
            for u in loud:
                print(f"  ⚠️  {u['id']}: {u['text'][:70]}...")
        if silent:
            print(f"\n[00] {len(silent)} low-priority question(s) skipped due to "
                  f"inconsistent dependencies (recorded in report).")

    # ── Batch derive impact statements (LLM, post-loop) ──────────────────────
    _batch_derive_impacts(decisions)

    # ── Write outputs ─────────────────────────────────────────────────────────
    _write_report(session_id, req_hash, project_name, decisions, unresolved, conflicts, findings)
    _append_to_log(session_id, project_name, project_slug, decisions, conflicts)

    # ── Phase synthesis: clarified requirement ────────────────────────────────
    if not args.no_synth:
        print("\n[00] Synthesizing clarified requirement ...")
        clarified_md = _synthesize_requirement(
            requirement_text, decisions, conflicts, clarified_sum
        )
        _write_clarified_req(clarified_md)

    _print_banner(f"Done — {len(decisions)} decisions recorded  [{project_slug}]")
    print(f"  Workspace log: {_clarification_log_path(project_slug)}")
    print(f"  Next step:     python 01_estimator.py --input {CLARIFIED_REQ}\n")


if __name__ == "__main__":
    main()
