"""
toolkits/devops_mlops/doc_absorber.py
======================================
Handover document absorber — the "DOCUMENTED state" layer of the triangle.

                         Three-source triangle
                         ─────────────────────
    IaC source       → infra_absorber   → infra_map.json      (HIGH confidence)
    Handover doc     → doc_absorber     → doc_map.json         (LOW confidence)
    Live AWS APIs    → live_discovery   → discovery_map.json   (MEDIUM confidence)
                                ↓ all three feed into ↓
                         config_consistency_checker
                         (weighted cross-source aggregation)

This module is the "documented state" layer. Handover docs are:
  - Human-written, narrative, intentionally ambiguous
  - Guaranteed to drift from reality the moment they are written
  - The only source of institutional knowledge (WHY decisions were made,
    known issues, escalation contacts, architectural rationale)

Output schema always carries `confidence: "low"` and `drift_assumption: true`.
Downstream consumers (config_consistency_checker, infra_judge) must treat
this data as signals to verify — not facts to rely on.

────────────────────────────────────────────────────────────────
Pipeline
────────────────────────────────────────────────────────────────

  1. redactor.py runs first (mandatory pre-processing)
       Secrets/IPs redacted → redacted_preview.md written
       Human reviews preview → confirms before LLM sees anything

  2. Human gate: "Proceed with LLM extraction? [Y/n]"
       If N → exit, no LLM call made

  3. LLM extraction (two layers):
       Layer A — Infra facts     → cross-reference with live_discovery
       Layer B — Institutional   → feed to postmortem_archivist

  4. Write artifacts:
       doc_absorber/doc_map.json    (overwrite)
       doc_absorber/doc_map.md      (overwrite)
       doc_absorber/doc_log.json    (append)

────────────────────────────────────────────────────────────────
doc_map.json schema
────────────────────────────────────────────────────────────────

  {
    "confidence": "low",
    "drift_assumption": true,
    "as_of_date": "unknown | YYYY-MM-DD",
    "source_files": [...],
    "infra_facts": [
      {
        "type": "service_mentioned | resource_name | endpoint |
                 credential_key | network_info | dependency",
        "value": "RDS PostgreSQL",
        "key_name": "DB_HOST",           // key name preserved if found
        "section": "Section 3.2",
        "confidence": "low",
        "note": "Mentioned but no version/endpoint specified",
        "cross_ref_hint": "verify against live_discovery.rds"
      }
    ],
    "institutional_knowledge": [
      {
        "type": "architectural_decision | known_issue | runbook_ref |
                 escalation_contact | sla | todo | unstructured_note",
        "content": "...",
        "section": "Section 4",
        "postmortem_relevant": true | false
      }
    ],
    "unstructured_notes": [...]
  }

────────────────────────────────────────────────────────────────
Outputs written
────────────────────────────────────────────────────────────────

  doc_absorber/doc_map.json          overwrite — structured extraction
  doc_absorber/doc_map.md            overwrite — narrative with drift warnings
  doc_absorber/doc_log.json          append    — run history
  doc_absorber/redacted/             written by redactor.py (pre-step)
  doc_absorber/images/               extracted images, human review

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python doc_absorber.py --project iot-mlops --file handover.xlsx
  python doc_absorber.py --project iot-mlops --file handover.pdf notes.docx
  python doc_absorber.py --project iot-mlops --file handover.xlsx --keep-ips
  python doc_absorber.py --project iot-mlops --file handover.xlsx --auto
  python doc_absorber.py --project iot-mlops --show-last
  PIPELINE_PROJECT=iot-mlops python doc_absorber.py --file handover.xlsx

Flags:
  --auto       Skip human review gate (use in CI / trusted environments only)
  --keep-ips   Pass --keep-ips to redactor (preserve IPs for cross-reference)
  --no-redact  Skip redaction step (use only if file already redacted)
  --show-last  Print last doc_map.md and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from modules.artifact_tracking import (                                   # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary as print_cost_summary              # noqa: E402
from modules.call_llm import call_llm                                     # noqa: E402
from modules.post_interactive import prompt_next_step                     # noqa: E402
from artifacts.models import get_model                                    # noqa: E402

ROLE = "doc_absorber"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    base     = Path(override) if override else _REPO_ROOT.parent / "outputs" / "devops_mlops"
    slug     = os.environ.get("PIPELINE_PROJECT", "default")
    return base / f"artifacts_{slug}"


def _doc_absorber_dir() -> Path: return _devops_artifact_root() / "doc_absorber"
def _doc_map_json()     -> Path: return _doc_absorber_dir() / "doc_map.json"
def _doc_map_md()       -> Path: return _doc_absorber_dir() / "doc_map.md"
def _doc_log()          -> Path: return _doc_absorber_dir() / "doc_log.json"
def _redacted_dir()     -> Path: return _doc_absorber_dir() / "redacted"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_EXTRACT = """\
You are a senior DevOps/MLOps architect analyzing handover documentation.

The document you will receive has already been redacted: secrets and credentials
have been replaced with typed placeholders like <PASSWORD_REDACTED>,
<AWS_ACCESS_KEY_ID>, <INTERNAL_IP>. Key names are preserved.

Your task: extract two layers of information.

────────────────────────────────────────────────────────────
LAYER A — Infra facts
────────────────────────────────────────────────────────────
Facts about infrastructure, services, and configuration.
These are LOW CONFIDENCE — the doc may be stale.

Fact types:
  service_mentioned   : A cloud service or technology is referenced
                        (RDS, EKS, SQS, Airflow, MLflow, Grafana…)
  resource_name       : A specific resource name is mentioned
                        (cluster name, bucket name, function name…)
  endpoint            : A URL, hostname, or service endpoint
  credential_key      : A credential key NAME is mentioned
                        (DB_PASSWORD, MLFLOW_TRACKING_URI…)
  network_info        : VPC, subnet, security group, CIDR, port
  dependency          : Service A depends on service B
  configuration       : A config value or setting is described

For each fact, identify:
  - What section of the document it came from
  - Which live_discovery or infra_absorber key to cross-reference against
    (e.g. "verify against live_discovery.rds.instances",
          "verify against infra_map.terraform.iam_roles")

────────────────────────────────────────────────────────────
LAYER B — Institutional knowledge
────────────────────────────────────────────────────────────
Knowledge that CANNOT be derived from code or live AWS state.
This is what makes handover docs valuable despite being stale.

Knowledge types:
  architectural_decision  : "We chose X over Y because…", "We deliberately avoided…"
  known_issue             : "There is a known problem with…", "This sometimes fails…"
  runbook_ref             : Reference to a runbook, wiki page, or procedure
  escalation_contact      : Who to call when X breaks
  sla                     : Service level agreements, uptime targets, RPO/RTO
  todo                    : Pending work, technical debt
  unstructured_note       : Cell comments, margin notes, informal observations
                            (flag these as postmortem_relevant if they describe
                            incidents, failures, or workarounds)

────────────────────────────────────────────────────────────
Output format
────────────────────────────────────────────────────────────
Return ONLY valid JSON (no markdown fences, no prose):
{
  "as_of_date": "YYYY-MM-DD or unknown",
  "document_summary": "1-3 sentences describing what this document covers",
  "infra_facts": [
    {
      "type":          "<fact type>",
      "value":         "<the service/resource/endpoint/key mentioned>",
      "key_name":      "<credential or config key name if present, else null>",
      "section":       "<section heading or sheet name>",
      "note":          "<any qualification: version unknown, endpoint not specified, etc.>",
      "cross_ref_hint":"<which live_discovery or infra_map field to verify against>"
    }
  ],
  "institutional_knowledge": [
    {
      "type":                "<knowledge type>",
      "content":             "<the actual knowledge — verbatim or paraphrased>",
      "section":             "<section heading or sheet name>",
      "postmortem_relevant": true | false,
      "tags":                ["<technology>", "<topic>"]
    }
  ]
}

RULES:
- Output ONLY the JSON. No prose before or after.
- infra_facts: include ONLY things explicitly stated in the document.
  Do not infer or hallucinate services not mentioned.
- institutional_knowledge: this is the high-value layer. Capture it fully.
  A brief architectural note buried in a comment is more valuable than
  a generic "uses EKS" fact.
- postmortem_relevant = true if the knowledge describes: an incident,
  a workaround, a known failure mode, or a "do not do X" lesson.
- Mark as_of_date as "unknown" if not stated in the document.
"""

_SYSTEM_NARRATIVE = """\
You are a senior DevOps/MLOps architect writing a handover analysis report.

Given a structured extraction from a handover document (JSON), write a
concise markdown narrative that:

1. States clearly this is LOW CONFIDENCE / DOCUMENTED STATE data
2. Summarizes what the document covers
3. Lists key infra facts with explicit drift warnings
4. Highlights institutional knowledge (especially postmortem-relevant items)
5. Calls out cross-reference hints for config_consistency_checker

Format:
- Use ## headers for sections
- Keep prose tight — 1-2 sentences per fact
- Use ⚠ for items that need human verification
- Use 💡 for institutional knowledge
- Use 🔁 for cross-reference hints

Do NOT reproduce credential values. Placeholders like <PASSWORD_REDACTED> are fine.
Keep total output under 800 words.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Redactor integration
# ─────────────────────────────────────────────────────────────────────────────

def _run_redactor(
    files:    list[Path],
    keep_ips: bool,
    dry_run:  bool = False,
) -> dict[Path, Any]:
    """
    Run redactor.py on each file.
    Returns {source_path: RedactionResult}.

    ImportError is FATAL — if redactor.py is missing, we refuse to proceed
    with raw files that may contain secrets. Use --no-redact explicitly to
    bypass redaction (only when files are already clean).
    """
    try:
        _here = Path(__file__).parent
        sys.path.insert(0, str(_here))
        from redactor import redact_file  # type: ignore
    except ImportError:
        print()
        print("[doc_absorber][ERROR] redactor.py not found in the same directory.")
        print()
        print("  Refusing to proceed — source files may contain secrets.")
        print("  redactor.py MUST run before any content is sent to the LLM.")
        print()
        print("  To fix: ensure redactor.py is in the same directory as doc_absorber.py.")
        print("  To bypass (only if files are already clean): use --no-redact flag.")
        raise SystemExit(2)

    results: dict[Path, Any] = {}
    for path in files:
        print(f"  Redacting: {path.name} …", end=" ", flush=True)
        try:
            result = redact_file(
                path      = path,
                output_dir = _redacted_dir(),
                keep_ips  = keep_ips,
                dry_run   = dry_run,
            )
            results[path] = result
            print(f"{result.finding_count} findings")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results[path] = None
    return results


def _load_redacted_content(
    source_path:    Path,
    redact_results: dict[Path, Any],
) -> str:
    """Load redacted markdown content for a source file."""
    result = redact_results.get(source_path)

    # If we have a redaction result with content, use it
    if result is not None:
        # RedactionResult.redacted_content (from patched redactor)
        if hasattr(result, "redacted_content") and result.redacted_content:
            return result.redacted_content
        # Fallback: read from written file
        if hasattr(result, "redacted_md_path") and result.redacted_md_path:
            p = Path(str(result.redacted_md_path))
            if p.exists():
                track_read(p)
                return p.read_text(encoding="utf-8", errors="replace")

    # No redaction result — try to find redacted file by convention
    redacted_path = _redacted_dir() / f"{source_path.stem}_redacted.md"
    if redacted_path.exists():
        track_read(redacted_path)
        return redacted_path.read_text(encoding="utf-8", errors="replace")

    # Last resort: read raw (only reached if --no-redact was used without a
    # prior redaction run, meaning no redacted file exists on disk)
    print(
        f"  [warn] No redacted version found for {source_path.name}.\n"
        f"  Reading raw file — ensure this file contains NO secrets.\n"
        f"  If it does, abort now (Ctrl+C) and run without --no-redact."
    )
    track_read(source_path)
    return source_path.read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Human review gate
# ─────────────────────────────────────────────────────────────────────────────

def _flush_stdin() -> None:
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def _human_review_gate(
    redact_results: dict[Path, Any],
    auto:           bool,
) -> bool:
    """
    Show human what was redacted, ask for confirmation before LLM call.
    Returns True if human approves (or auto=True).
    """
    print()
    print("=" * 68)
    print("  REDACTION REVIEW GATE")
    print("=" * 68)

    total_findings = 0
    for path, result in redact_results.items():
        if result is None:
            print(f"  ⚠  {path.name}: redaction failed — will use raw file")
            continue
        fc  = getattr(result, "finding_count", 0)
        rmd = getattr(result, "redacted_md_path", None)
        imgs = getattr(result, "image_paths", [])
        total_findings += fc
        print(f"  {path.name}:")
        print(f"    Secrets redacted:  {fc}")
        if rmd:
            print(f"    Preview:           {rmd}")
        if imgs:
            print(f"    Images extracted:  {len(imgs)} file(s) — review manually")

    print()

    if total_findings > 0:
        print("  ⚠  Review the redacted preview(s) above before proceeding.")
        print("     Verify all secrets are properly replaced with placeholders.")
        print("     The LLM will ONLY see the redacted version.")
    else:
        print("  ✓  No secrets detected in documents.")

    print()

    if auto:
        print("  [--auto] Skipping human review gate — proceeding with LLM extraction.")
        return True

    if not sys.stdin.isatty():
        print("  [non-interactive] Proceeding with LLM extraction.")
        return True

    _flush_stdin()
    try:
        ans = input("  Proceed with LLM extraction? [Y/n]: ").strip().lower()
        return ans in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LLM extraction
# ─────────────────────────────────────────────────────────────────────────────

_total_cost: float = 0.0


def _call_llm(system: str, user: str, max_tokens: int = 8192) -> str:
    global _total_cost
    raw, cost = call_llm(
        ROLE, system, user,
        max_tokens   = max_tokens,
        caller_file  = __file__,
        label        = f"[doc_absorber] {get_model(ROLE)}",
    )
    _total_cost += (cost or 0.0)
    return raw


def _parse_json_response(raw: str) -> dict[str, Any]:
    import re
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-counting fallback
    depth, start, in_str, esc = 0, None, False, False
    candidates: list[str] = []
    for i, ch in enumerate(text):
        if esc:        esc = False;  continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"':  in_str = not in_str; continue
        if in_str:     continue
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"No valid JSON in LLM response ({len(raw)} chars)", raw, 0
    )


def _extract_from_document(
    contents:     list[tuple[str, str]],   # [(filename, redacted_content), ...]
    verbose:      bool = False,
) -> dict[str, Any]:
    """
    LLM call 1: extract infra facts + institutional knowledge.
    contents: list of (filename, redacted_md_content) tuples.

    Context budget: _MAX_PER_FILE chars per file, _MAX_TOTAL_CHARS overall.
    Budget is distributed evenly across files — no single file can crowd out others.
    """
    _MAX_PER_FILE    = 30_000
    _MAX_TOTAL_CHARS = 80_000
    per_file_cap     = min(_MAX_PER_FILE, _MAX_TOTAL_CHARS // max(len(contents), 1))

    print("  [LLM] Extracting infra facts and institutional knowledge …")

    parts: list[str] = []
    total_chars = 0
    for filename, content in contents:
        if len(content) > per_file_cap:
            print(f"    Truncating {filename}: {len(content):,} → {per_file_cap:,} chars")
            content = content[:per_file_cap] + "\n\n… (truncated to fit context)"
        if total_chars + len(content) > _MAX_TOTAL_CHARS:
            remaining = _MAX_TOTAL_CHARS - total_chars
            if remaining > 500:
                content = content[:remaining] + "\n\n… (truncated: overall context limit reached)"
            else:
                print(f"    Skipping {filename} — overall context limit reached")
                continue
        parts.append(f"## Document: {filename}\n\n{content}")
        total_chars += len(content)
        if verbose:
            print(f"    {filename}: {len(content):,} chars  (total so far: {total_chars:,})")

    user = (
        "Extract infra facts and institutional knowledge from the following "
        f"{len(parts)} document(s):\n\n"
        + "\n\n---\n\n".join(parts)
    )

    raw    = _call_llm(_SYSTEM_EXTRACT, user, max_tokens=8192)
    result = _parse_json_response(raw)

    n_facts = len(result.get("infra_facts", []))
    n_know  = len(result.get("institutional_knowledge", []))
    print(f"  [LLM] Extracted: {n_facts} infra facts, {n_know} institutional knowledge items")

    return result


def _generate_narrative(
    extraction:    dict[str, Any],
    source_files:  list[str],
) -> str:
    """
    LLM call 2: generate human-readable markdown narrative with drift warnings.
    """
    print("  [LLM] Generating narrative report …")

    user = (
        f"Source files: {', '.join(source_files)}\n\n"
        f"Extraction result:\n\n"
        f"{json.dumps(extraction, indent=2, ensure_ascii=False)}"
    )

    return _call_llm(_SYSTEM_NARRATIVE, user, max_tokens=2048)


# ─────────────────────────────────────────────────────────────────────────────
# doc_map.json builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_doc_map(
    extraction:    dict[str, Any],
    source_files:  list[Path],
    redact_results: dict[Path, Any],
    keep_ips:      bool,
    run_at:        str,
) -> dict[str, Any]:
    """
    Assemble the final doc_map.json.
    Always carries confidence=low and drift_assumption=true.
    """
    # Aggregate redaction stats
    total_redacted = sum(
        getattr(r, "finding_count", 0)
        for r in redact_results.values()
        if r is not None
    )
    all_categories: dict[str, int] = {}
    for r in redact_results.values():
        if r is None:
            continue
        for cat, count in getattr(r, "categories", {}).items():
            all_categories[cat] = all_categories.get(cat, 0) + count

    # Flag postmortem-relevant institutional knowledge
    pm_items = [
        item for item in extraction.get("institutional_knowledge", [])
        if item.get("postmortem_relevant")
    ]

    return {
        # ── Confidence metadata — ALWAYS present ──────────────────────────────
        "confidence":          "low",
        "drift_assumption":    True,
        "as_of_date":          extraction.get("as_of_date", "unknown"),
        "generated_at":        run_at,

        # ── Source provenance ─────────────────────────────────────────────────
        "source_files":        [str(p) for p in source_files],
        "keep_ips":            keep_ips,
        "redaction": {
            "total_findings":  total_redacted,
            "categories":      all_categories,
            "note":            "Secret values were redacted before LLM processing.",
        },

        # ── Document summary ──────────────────────────────────────────────────
        "document_summary":    extraction.get("document_summary", ""),

        # ── Extracted layers ──────────────────────────────────────────────────
        "infra_facts":           extraction.get("infra_facts", []),
        "institutional_knowledge": extraction.get("institutional_knowledge", []),

        # ── Convenience indexes ───────────────────────────────────────────────
        "stats": {
            "infra_facts_count":            len(extraction.get("infra_facts", [])),
            "institutional_knowledge_count": len(extraction.get("institutional_knowledge", [])),
            "postmortem_relevant_count":     len(pm_items),
            "services_mentioned": list({
                f["value"]
                for f in extraction.get("infra_facts", [])
                if f.get("type") == "service_mentioned"
            }),
            "credential_keys_found": [
                f["key_name"]
                for f in extraction.get("infra_facts", [])
                if f.get("type") == "credential_key" and f.get("key_name")
            ],
        },

        # ── Consumer hints ────────────────────────────────────────────────────
        "consumer_notes": {
            "config_consistency_checker": (
                "Use infra_facts for cross-reference. Weight all findings as LOW "
                "confidence. Treat mismatches with live_discovery as 'doc stale' "
                "before assuming live is wrong."
            ),
            "postmortem_archivist": (
                f"{len(pm_items)} items flagged as postmortem_relevant. "
                "Consider ingesting institutional_knowledge items with "
                "postmortem_relevant=true."
            ),
            "infra_judge": (
                "Do not use doc_map as ground truth. Use as supplementary "
                "context only. Prefer live_discovery and infra_map for verdicts."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Artifact writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_doc_map(doc_map: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(json.dumps(doc_map, indent=2, ensure_ascii=False))
        return
    p = _doc_map_json()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc_map, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(p)


def _write_doc_map_md(narrative: str, doc_map: dict[str, Any], dry_run: bool) -> None:
    """Write doc_map.md — narrative + header with drift warning."""
    if dry_run:
        return

    # Prepend standard drift warning header
    run_at  = doc_map.get("generated_at", _now_iso())
    sources = ", ".join(Path(s).name for s in doc_map.get("source_files", []))
    header  = textwrap.dedent(f"""\
        <!-- doc_map.md — generated by doc_absorber.py on {run_at} -->
        <!-- confidence: LOW | drift_assumption: true -->
        <!-- sources: {sources} -->

        > ⚠ **LOW CONFIDENCE — DOCUMENTED STATE ONLY**
        > This document was written by humans and is assumed to have drifted
        > from actual deployed state. Verify all facts against `live_discovery`
        > and `infra_absorber` before acting on them.
        > As of date: **{doc_map.get("as_of_date", "unknown")}**

        ---

    """)

    content = header + narrative.strip() + "\n"

    p = _doc_map_md()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    track_write(p)


def _append_doc_log(
    doc_map:  dict[str, Any],
    dry_run:  bool,
) -> None:
    if dry_run:
        return
    log = _doc_log()
    log.parent.mkdir(parents=True, exist_ok=True)

    try:
        track_read(log)
        data    = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        entries = []

    stats = doc_map.get("stats", {})
    entries.append({
        "run_at":                  doc_map["generated_at"],
        "source_files":            [Path(s).name for s in doc_map.get("source_files", [])],
        "as_of_date":              doc_map.get("as_of_date", "unknown"),
        "infra_facts_count":       stats.get("infra_facts_count", 0),
        "institutional_count":     stats.get("institutional_knowledge_count", 0),
        "postmortem_relevant":     stats.get("postmortem_relevant_count", 0),
        "redaction_findings":      doc_map.get("redaction", {}).get("total_findings", 0),
        "services_mentioned":      stats.get("services_mentioned", []),
        "credential_keys":         stats.get("credential_keys_found", []),
    })

    log.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(log)


def _maybe_commit_log() -> None:
    log = _doc_log()
    if not log.exists():
        return
    try:
        data    = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        return
    if not entries:
        return
    try:
        ans = input(f"  Keep this entry in {log.name}? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  [doc_absorber] Entry kept (non-interactive).")
        return
    if ans in ("n", "no"):
        entries.pop()
        try:
            log.write_text(
                json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("  [doc_absorber] Entry discarded.")
        except Exception as exc:
            print(f"  [doc_absorber][warn] Could not revert: {exc}")
    else:
        print(f"  [doc_absorber] Entry kept (total: {len(entries)}).")


def _suggest_postmortem_routing(doc_map: dict[str, Any]) -> None:
    """
    Print actionable suggestion if postmortem-relevant items were found.
    """
    pm_items = [
        item for item in doc_map.get("institutional_knowledge", [])
        if item.get("postmortem_relevant")
    ]
    if not pm_items:
        return

    slug = os.environ.get("PIPELINE_PROJECT", "<name>")
    print()
    print(f"  💡 {len(pm_items)} institutional knowledge item(s) flagged as postmortem-relevant:")
    for item in pm_items[:3]:
        content_preview = item.get("content", "")[:70]
        print(f"     [{item.get('type', '?')}] {content_preview}…")
    if len(pm_items) > 3:
        print(f"     … and {len(pm_items) - 3} more")
    print()
    print("  Route to postmortem_archivist:")
    print(f"    python postmortem_archivist.py --project {slug} --mode capture")


# ─────────────────────────────────────────────────────────────────────────────
# show-last
# ─────────────────────────────────────────────────────────────────────────────

def _show_last() -> None:
    md = _doc_map_md()
    if md.exists():
        track_read(md)
        print(md.read_text(encoding="utf-8"))
        return
    js = _doc_map_json()
    if js.exists():
        track_read(js)
        print(js.read_text(encoding="utf-8"))
        return
    print("[doc_absorber] No output found. Run without --show-last first.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# File gathering — CLI flag or drag-and-drop interactive prompt
# ─────────────────────────────────────────────────────────────────────────────

def _read_input_file(path: Path) -> str:
    """
    Read callback for drag_and_drop.gather_text_file_bundle.
    Returns the file path as a string — doc_absorber does not read content
    here. Content reading happens later inside redactor.py per file.
    This prevents any secret-containing file being read before redaction.
    """
    return str(path)


def _gather_files(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Path]:
    """
    Collect handover document paths from one of three sources:
      1. --file flag (CLI, non-interactive)
      2. drag-and-drop into terminal (interactive, no --file given)
      3. typed/pasted file paths in interactive prompt

    Returns a list of resolved, existing Paths.
    Exits with error if no valid files are found.
    """
    # ── Path 1: --file flag provided ─────────────────────────────────────────
    if args.file:
        resolved: list[Path] = []
        for f in args.file:
            p = Path(f).expanduser().resolve()
            if not p.exists():
                print(f"  [warn] File not found: {p}")
            else:
                resolved.append(p)
        if not resolved:
            parser.error("No valid files found from --file arguments.")
        return resolved

    # ── Path 2 + 3: no --file → interactive drag-and-drop prompt ─────────────
    if args.no_interactive:
        parser.error("--file is required when --no-interactive is set.")

    try:
        from modules.drag_and_drop import gather_text_file_bundle  # type: ignore
    except ImportError:
        print("[doc_absorber][warn] drag_and_drop module not found.")
        print("  Falling back to manual input — enter file paths one per line,")
        print("  then press Enter on an empty line to finish:")
        paths: list[Path] = []
        while True:
            try:
                line = input("  Path: ").strip().strip("'\"")
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            p = Path(line).expanduser().resolve()
            if p.exists():
                paths.append(p)
            else:
                print(f"  [warn] Not found: {p}")
        if not paths:
            print("[doc_absorber] No files provided. Exiting.")
            sys.exit(1)
        return paths

    print()
    bundle = gather_text_file_bundle(
        cli_text   = None,
        cli_files  = [],
        read_file_fn = _read_input_file,
        prompt_title = "Handover documents",
        prompt_body  = (
            "Drag handover files into terminal (Excel, PDF, DOCX, MD), "
            "or type/paste file paths.\n"
            "  End input with /done, or press Enter twice."
        ),
        attachment_prompt              = "Drop files here",
        default_attachment_only_prompt = "",
        allow_interactive              = True,
    )

    # Collect paths from bundle.sources (drag-drop detection) or bundle.text
    paths = []

    # Sources from drag-drop detection
    # bundle.sources is list[InputSource] dataclass — use .path attribute directly
    for source in getattr(bundle, "sources", []):
        source_path = getattr(source, "path", None)
        if source_path is not None:
            p = Path(str(source_path)).expanduser().resolve()
            if p.exists():
                paths.append(p)
            else:
                print(f"  [warn] Path from drag-drop not found: {p}")

    # Fallback: parse bundle.text as newline-separated paths
    if not paths:
        for line in (getattr(bundle, "text", "") or "").splitlines():
            candidate = line.strip().strip("'\"")
            if candidate:
                p = Path(candidate).expanduser().resolve()
                if p.exists():
                    paths.append(p)
                elif candidate:
                    print(f"  [warn] Path not found: {candidate}")

    if not paths:
        print("[doc_absorber] No valid files gathered. Exiting.")
        sys.exit(1)

    return paths


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doc_absorber.py",
        description="Handover document absorber — documented state layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python doc_absorber.py --project iot-mlops --file handover.xlsx
              python doc_absorber.py --project iot-mlops --file handover.pdf notes.docx
              python doc_absorber.py --project iot-mlops --file handover.xlsx --auto
              python doc_absorber.py --project iot-mlops --show-last
        """),
    )
    p.add_argument("--project",    default=os.environ.get("PIPELINE_PROJECT"))
    p.add_argument("--file",       nargs="*", metavar="FILE",
                   help=(
                       "Handover document file(s) to absorb. "
                       "Optional — if omitted, an interactive prompt is shown "
                       "where you can drag-drop files into the terminal."
                   ))
    p.add_argument("--no-interactive", action="store_true",
                   help="Disable interactive prompt. --file is required if set.")
    p.add_argument("--keep-ips",   action="store_true",
                   help="Preserve IP addresses in redacted content (for cross-reference).")
    p.add_argument("--auto",       action="store_true",
                   help="Skip human review gate. Use only in trusted CI environments.")
    p.add_argument("--no-redact",  action="store_true",
                   help=(
                       "Skip redaction step. Use ONLY if files are already clean "
                       "(no secrets). If a redacted/<stem>_redacted.md from a prior "
                       "run exists, that file is used automatically. "
                       "If not, raw file is read with a strong warning."
                   ))
    p.add_argument("--dry-run",    action="store_true",
                   help="Run extraction but do not write artifacts.")
    p.add_argument("--show-last",  action="store_true",
                   help="Print last doc_map.md and exit.")
    p.add_argument("--verbose",    action="store_true")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error("Use --project <name> or export PIPELINE_PROJECT=<name>.")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    _doc_absorber_dir().mkdir(parents=True, exist_ok=True)

    # --show-last short circuit
    if args.show_last:
        _show_last()
        sys.exit(0)

    print("=" * 68)
    print("  DOC ABSORBER")
    print("=" * 68)
    print()

    exit_code = 0
    doc_map: dict[str, Any] = {}

    try:
        # Resolve file paths
        # Gather source files via --file flag or interactive drag-drop
        source_files = _gather_files(args, parser)

        print(f"  Files: {', '.join(p.name for p in source_files)}")
        print()

        # ── Step 1: Redaction ─────────────────────────────────────────────────
        redact_results: dict[Path, Any] = {}

        if args.no_redact:
            print("  [--no-redact] Skipping redaction step.")
            redact_results = {p: None for p in source_files}
        else:
            print("  Step 1 — Redacting secrets …")
            redact_results = _run_redactor(
                files    = source_files,
                keep_ips = args.keep_ips,
                dry_run  = False,   # always write redacted files for human review
            )

        # ── Step 2: Human review gate ─────────────────────────────────────────
        print()
        print("  Step 2 — Human review gate …")
        approved = _human_review_gate(
            redact_results = redact_results,
            auto           = args.auto or args.no_redact,
        )
        if not approved:
            print("[doc_absorber] Aborted by user.")
            sys.exit(0)

        # ── Step 3: Load redacted content ─────────────────────────────────────
        print()
        print("  Step 3 — Loading redacted content …")
        contents: list[tuple[str, str]] = []
        for path in source_files:
            content = _load_redacted_content(path, redact_results)
            if content.strip():
                contents.append((path.name, content))
                if args.verbose:
                    print(f"    {path.name}: {len(content):,} chars")

        if not contents:
            print("[doc_absorber] No content to extract. Exiting.")
            sys.exit(1)

        # ── Step 4: LLM extraction ────────────────────────────────────────────
        print()
        print("  Step 4 — LLM extraction …")
        run_at     = _now_iso()
        extraction = _extract_from_document(contents, verbose=args.verbose)

        # ── Step 5: Narrative generation ──────────────────────────────────────
        print()
        print("  Step 5 — Generating narrative …")
        source_names = [p.name for p in source_files]
        narrative    = _generate_narrative(extraction, source_names)

        # ── Step 6: Assemble and write ────────────────────────────────────────
        print()
        print("  Step 6 — Writing artifacts …")
        doc_map = _build_doc_map(
            extraction     = extraction,
            source_files   = source_files,
            redact_results = redact_results,
            keep_ips       = args.keep_ips,
            run_at         = run_at,
        )

        _write_doc_map(doc_map, dry_run=args.dry_run)
        _write_doc_map_md(narrative, doc_map, dry_run=args.dry_run)
        _append_doc_log(doc_map, dry_run=args.dry_run)

        if not args.dry_run:
            print(f"    doc_map.json → {_doc_map_json()}")
            print(f"    doc_map.md   → {_doc_map_md()}")
            print(f"    doc_log      → {_doc_log()}")

        # ── Summary ───────────────────────────────────────────────────────────
        stats = doc_map.get("stats", {})
        print()
        print("=" * 68)
        print("  EXTRACTION SUMMARY")
        print("=" * 68)
        print(f"  Infra facts:            {stats.get('infra_facts_count', 0)}")
        print(f"  Institutional knowledge: {stats.get('institutional_knowledge_count', 0)}")
        print(f"  Postmortem-relevant:     {stats.get('postmortem_relevant_count', 0)}")
        svcs = stats.get("services_mentioned", [])
        if svcs:
            print(f"  Services mentioned:     {', '.join(svcs[:8])}"
                  + (" …" if len(svcs) > 8 else ""))
        creds = stats.get("credential_keys_found", [])
        if creds:
            print(f"  Credential keys found:  {', '.join(creds[:6])}"
                  + (" …" if len(creds) > 6 else ""))
        print()
        print("  ⚠  confidence: LOW | drift_assumption: true")
        print("  All facts must be verified against live_discovery and infra_map.")

        # Suggest postmortem routing
        _suggest_postmortem_routing(doc_map)

    except KeyboardInterrupt:
        print("\n[doc_absorber] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[doc_absorber][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[doc_absorber]")
        print()
        print_cost_summary("[doc_absorber]")
        prompt_next_step(ROLE, prefix="[doc_absorber]")

        # Long-term log commit — inside finally so it always runs on clean exit
        if not args.dry_run and exit_code == 0:
            try:
                _maybe_commit_log()
            except Exception as e:
                print(f"[doc_absorber][warn] Could not commit log: {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
