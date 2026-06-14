"""
pipeline/04c_spec_risk_assessor.py
===================================
Spec Risk & Impact Assessor — reads the current spec and the absorber's
codebase_map.md, then produces a structured risk/impact report before
the spec enters spectracker and downstream generation.

Position in pipeline:
    specwright → spec_validator (npm facts) → [04c_spec_risk_assessor] → spectracker

Unlike spec_validator (deterministic fact-check via npm registry),
this script is reasoning-heavy: it asks an LLM to evaluate how the spec's
requirements will interact with the existing codebase.

────────────────────────────────────────────────────────────────
What it assesses
────────────────────────────────────────────────────────────────

  Risk classification (overall)
    tiny   — isolated change, touches ≤1 module, no shared state, no API change
    normal — multi-file, clear boundary, no breaking public API or data model change
    high   — cross-cutting, touches shared state / public API / auth / data model /
              storage schema, or requires coordinated changes across layers

  Per-finding classification
    For each spec requirement (AC, NFR, architecture decision, data model change,
    API contract), the LLM produces a finding with:
      - risk_level:   high | medium | low
      - impact_area:  which files/modules/layers are likely affected
      - confidence:   high | medium | low  (how certain the LLM is)
      - kind:         conflict | ambiguity | missing_detail | breaking_change |
                      assumption | scope_creep | dependency_risk

  Two treatment tracks based on confidence:

    HIGH confidence finding
      → Interactive Q&A: LLM suggests a concrete spec patch (1 sentence)
      → Human prompted y/n
      → If y: patch applied in-place to spec (no version bump — not at spectracker yet)
      → If n: finding recorded as "deferred to human" in report

    LOW/MEDIUM confidence finding (needs judgment)
      → No Q&A, no auto-patch
      → Report highlights the relevant spec section + approximate line
      → Lists 2–3 concrete options the human could take
      → Human decides independently — no prompt

────────────────────────────────────────────────────────────────
Relationship to clarificator and spec_validator
────────────────────────────────────────────────────────────────

  clarificator (03)   reads raw REQUIREMENT from human, finds holes/assumptions/
                      conflicts via Q&A loop, outputs clarification synthesis
                      → operates before spec exists

  spec_validator (04b) deterministically fact-checks package names/versions
                      against npm registry, auto-patches clear typos
                      → operates on spec, no codebase context needed

  spec_risk_assessor (04c)  reads SPEC + CODEBASE MAP, evaluates how the spec
                      will land on the existing code, surfaces risk and impact
                      → operates on spec with full codebase context

────────────────────────────────────────────────────────────────
Inputs consumed
────────────────────────────────────────────────────────────────

  spec/<slug>.md              canonical spec (specwright output, post-validator)
  absorber/codebase_map.md    LLM narrative of current codebase state
                              If missing: proceeds with spec only (lower quality)

────────────────────────────────────────────────────────────────
Artifacts written
────────────────────────────────────────────────────────────────

  spec/spec_risk_report.md     short-term OVERWRITE
    Structured report for human reading. Sections:
      Overall Risk Classification, Summary, High-Confidence Patches Applied,
      Manual Review Required (with section refs + options), Low-Risk Notes,
      Patch Decisions Log

  spec/spec_risk_log.json      long-term APPEND
    One entry per run: timestamp, spec_version, overall_risk, finding counts,
    patches applied, patches deferred

  spec/<slug>.md               PATCHED IN-PLACE (only if human confirms y)
    No version bump — spec_risk_assessor runs before spectracker.
    Human can review diff via: git diff spec/<slug>.md

────────────────────────────────────────────────────────────────
Artifact impact by command
────────────────────────────────────────────────────────────────

  Command                  spec_risk_report.md   spec_risk_log.json   spec (patched)
  ──────────────────────── ─────────────────────  ─────────────────── ──────────────
  (normal run)             OVERWRITE              APPEND               PATCHED if y
  --no-patch               OVERWRITE              APPEND               unchanged
  --dry-run                –                      –                    unchanged

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python 04c_spec_risk_assessor.py --project my-app
    Full run. Reads spec + codebase_map.md. Presents high-confidence
    patches one by one (y/n). Writes report + log.
    Exit 0 = ran successfully (report written regardless of risk level)
    Exit 1 = spec not found or LLM call failed
    Exit 2 = high-risk findings present (for CI gating if needed)

  python 04c_spec_risk_assessor.py --project my-app --no-patch
    Run full assessment but skip all y/n prompts. No spec patching.
    All high-confidence findings appear in report as "deferred".
    Use in CI / non-interactive environments.

  python 04c_spec_risk_assessor.py --project my-app --dry-run
    Build briefing and print to stdout. No LLM call, no writes.
    Use to verify context quality before spending tokens.

  python 04c_spec_risk_assessor.py --project my-app --show-last
    Print the most recent spec_risk_report.md without re-running.

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifacts.paths import (  # noqa: E402
    ABSORBER_CODEBASE_MD,
    artifact_root,
    ensure_dirs,
    get_spec_path,
)
from modules.call_llm import call_llm_json  # noqa: E402
from modules.artifact_tracking import (  # noqa: E402
    track_read,
    track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary as print_cost_summary  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402


ROLE = "spec_validator"   # reasoning-capable, reuse existing role

MAX_SPEC_CHARS     = 100_000
MAX_CODEBASE_CHARS =  60_000
MAX_BRIEFING_CHARS = 180_000


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths  (all under spec/ — same directory as the spec itself)
# ─────────────────────────────────────────────────────────────────────────────

def _spec_dir() -> Path:
    return artifact_root() / "spec"

def _report_path() -> Path:
    return _spec_dir() / "spec_risk_report.md"

def _log_path() -> Path:
    return _spec_dir() / "spec_risk_log.json"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="04c_spec_risk_assessor.py",
        description="Assess risk and codebase impact of the current spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python 04c_spec_risk_assessor.py --project my-app
              python 04c_spec_risk_assessor.py --project my-app --no-patch
              python 04c_spec_risk_assessor.py --project my-app --dry-run
              python 04c_spec_risk_assessor.py --project my-app --show-last
        """),
    )
    p.add_argument("--project", default=os.environ.get("PIPELINE_PROJECT"),
                   metavar="NAME", help="Project workspace name.")
    p.add_argument("--no-patch", action="store_true",
                   help="Skip all y/n prompts. No spec patching. All high-confidence "
                        "findings recorded as deferred.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build briefing and print to stdout. No LLM call, no writes.")
    p.add_argument("--show-last", action="store_true",
                   help="Print most recent spec_risk_report.md and exit.")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Safe readers
# ─────────────────────────────────────────────────────────────────────────────

def _read(path: Path, label: str) -> str:
    if not path.exists():
        print(f"[risk_assessor][warn] {label} not found: {path}")
        return ""
    track_read(path)
    try:
        return path.read_text(errors="replace")
    except Exception as e:
        print(f"[risk_assessor][warn] Could not read {label}: {e}")
        return ""


def _trunc(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    print(f"[risk_assessor] Truncating {label}: {len(text):,} → {limit:,} chars")
    return text[:limit] + f"\n\n[truncated: {len(text):,} chars total]"


# ─────────────────────────────────────────────────────────────────────────────
# Spec helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_version(text: str) -> str:
    m = re.search(r"^#\s*Version:\s*(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def _spec_section_lines(spec_text: str) -> list[tuple[int, str]]:
    """Return [(line_number, heading_text), ...] for all ## headings."""
    result = []
    for i, line in enumerate(spec_text.splitlines(), start=1):
        m = re.match(r"^#{1,3}\s+(.+)$", line)
        if m:
            result.append((i, m.group(1).strip()))
    return result


def _find_line_for_section(spec_text: str, section_heading: str) -> int | None:
    """Return approximate 1-based line number of a section heading."""
    heading_lower = section_heading.lower()
    for i, line in enumerate(spec_text.splitlines(), start=1):
        if re.match(r"^#{1,3}\s+", line):
            if heading_lower in line.lower():
                return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Briefing builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_briefing(spec_text: str, codebase_map: str) -> str:
    sections = _spec_section_lines(spec_text)
    section_index = "\n".join(f"  L{ln}: {h}" for ln, h in sections) or "  (none found)"

    parts: list[str] = []

    parts.append(textwrap.dedent(f"""\
        ## 0. Context

        **Task:** Assess the risk and codebase impact of this spec.
        **Spec version:** {_parse_version(spec_text)}

        **Spec section index (line numbers):**
        {section_index}

        {"**Codebase map available:** yes" if codebase_map else "**Codebase map:** NOT AVAILABLE — assess from spec only."}
    """).strip())

    parts.append(
        "## 1. Spec\n\n"
        "```markdown\n"
        + _trunc(spec_text, MAX_SPEC_CHARS, "spec")
        + "\n```"
    )

    if codebase_map:
        parts.append(
            "## 2. Codebase map (absorber narrative — current state of src/)\n\n"
            + _trunc(codebase_map, MAX_CODEBASE_CHARS, "codebase_map.md")
        )
    else:
        parts.append(
            "## 2. Codebase map\n\n"
            "_[Not available. Assess from spec only. "
            "Run absorber for higher-quality risk assessment.]_"
        )

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a senior software architect performing a pre-implementation risk and
impact assessment of a technical spec against an existing codebase.

Your job is NOT to rewrite the spec. Your job is to:
  1. Classify the overall risk of implementing this spec on the existing codebase.
  2. Identify specific findings — conflicts, ambiguities, assumptions, breaking
     changes, scope risks, dependency risks — per spec requirement.
  3. For each finding, decide whether you are confident enough to suggest a
     concrete one-sentence spec patch (high confidence), or whether the finding
     requires human judgment (medium/low confidence).

────────────────────────────────────────────────────────────────
Overall risk levels
────────────────────────────────────────────────────────────────
  tiny   — isolated change, ≤1 module, no shared state, no public API change,
            no data model change, no auth/storage impact.
            Executor can implement with minimal risk of regression.
  normal — multi-file change, clear module boundary, no breaking public API,
            no data model migration required, limited cross-module coordination.
            Standard execution with careful review of integration points.
  high   — cross-cutting concern, touches shared state / public API surface /
            auth flow / storage schema / data model, OR requires coordinated
            changes across multiple layers, OR introduces external dependencies
            with breaking version constraints.
            Human should review impact carefully before proceeding.

────────────────────────────────────────────────────────────────
Finding kinds
────────────────────────────────────────────────────────────────
  conflict        — spec requirement directly contradicts something in codebase
  ambiguity       — requirement is underspecified; multiple valid interpretations
  missing_detail  — a value/behaviour is required but not defined in spec
  breaking_change — implementing this will break existing callers/contracts
  assumption      — spec assumes something about codebase that may not be true
  scope_creep     — requirement implicitly pulls in work not listed in spec
  dependency_risk — declared library/version/API has known compatibility risk

────────────────────────────────────────────────────────────────
Confidence levels for patches
────────────────────────────────────────────────────────────────
  high   — evidence from BOTH spec and codebase map clearly shows the issue
            AND the fix is a simple, unambiguous one-sentence change to the spec.
            These findings will be presented to the human as y/n patches.
  medium — issue is real but the right fix requires judgment or tradeoffs.
            Do NOT suggest as auto-patch. Provide 2–3 options instead.
  low    — issue is speculative or depends on context not visible in the inputs.
            Flag as informational only.

────────────────────────────────────────────────────────────────
Output format — raw JSON only, no markdown fences
────────────────────────────────────────────────────────────────
{
  "spec_version": "<string>",
  "overall_risk": "tiny" | "normal" | "high",
  "risk_rationale": "<1–2 sentences explaining the overall risk classification>",
  "summary": "<3–5 sentence executive summary of the assessment>",
  "findings": [
    {
      "id":            "F-01",
      "kind":          "conflict" | "ambiguity" | "missing_detail" | "breaking_change"
                       | "assumption" | "scope_creep" | "dependency_risk",
      "risk_level":    "high" | "medium" | "low",
      "confidence":    "high" | "medium" | "low",
      "spec_section":  "<heading text of the relevant spec section>",
      "spec_line":     <approximate 1-based line number or null>,
      "impact_areas":  ["src/lib/opfs.ts", "src/hooks/useAudio.ts"],
      "description":   "<1–2 sentences: what the finding is>",
      "patch_suggestion": "<one-sentence concrete spec text change if confidence=high, else null>",
      "options": [
        "<option A>",
        "<option B>",
        "<option C>"
      ]
    }
  ]
}

Rules:
- patch_suggestion: non-null ONLY if confidence is "high". Must be one sentence,
  directly usable as a spec text replacement. Must not change the spec's intent —
  only clarify or correct a clear error/conflict.
- options: non-null and non-empty ONLY if confidence is "medium". Provide 2–3
  concrete, actionable options the human can choose between.
- For low-confidence findings: both patch_suggestion and options may be null.
  Just describe the concern.
- impact_areas: list of most likely affected file paths or module names.
  Use paths from the codebase map if available. Use spec-derived names otherwise.
- Do not invent findings. If the spec and codebase are well-aligned, say so in summary
  and return an empty findings array with overall_risk "tiny" or "normal".
- Prioritise: breaking_change and conflict findings first, then missing_detail,
  then ambiguity and assumption, then scope_creep and dependency_risk.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Interactive patch loop — one finding at a time
# ─────────────────────────────────────────────────────────────────────────────

def _run_patch_loop(
    spec_text:  str,
    findings:   list[dict[str, Any]],
    no_patch:   bool,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    For each high-confidence finding, prompt human y/n and apply patch if y.
    Returns (patched_spec_text, applied_patches, deferred_patches).
    """
    high_conf = [f for f in findings if f.get("confidence") == "high"
                 and f.get("patch_suggestion")]

    applied:  list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []

    if not high_conf:
        return spec_text, applied, deferred

    current_spec = spec_text

    print()
    print("─" * 60)
    print(f"  HIGH-CONFIDENCE PATCHES ({len(high_conf)} found)")
    print("─" * 60)

    for finding in high_conf:
        fid         = finding.get("id", "?")
        section     = finding.get("spec_section", "")
        line_hint   = finding.get("spec_line")
        description = finding.get("description", "")
        suggestion  = finding.get("patch_suggestion", "")
        kind        = finding.get("kind", "")
        risk        = finding.get("risk_level", "")

        print()
        print(f"  [{fid}] {kind.upper()}  risk={risk}")
        if section:
            loc = f"§ {section}"
            if line_hint:
                loc += f"  (line ~{line_hint})"
            print(f"  Location: {loc}")
        print(f"  Finding:  {description}")
        print(f"  Patch:    {suggestion}")
        print()

        if no_patch:
            print("  [--no-patch] Deferred.")
            deferred.append(finding)
            continue

        try:
            choice = input("  Apply this patch? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "n"

        if choice == "y":
            # Apply: find the section in spec and append the patch suggestion
            # as an inline note prefixed with "<!-- RISK-PATCH: ... -->"
            # so it's visible to spectracker / human without corrupting spec structure.
            patch_comment = f"\n<!-- RISK-PATCH {fid}: {suggestion} -->\n"
            if section:
                # Insert after the first line of the matched section
                lines = current_spec.splitlines(keepends=True)
                for i, line in enumerate(lines):
                    if re.match(r"^#{1,3}\s+", line) and section.lower() in line.lower():
                        lines.insert(i + 1, patch_comment)
                        current_spec = "".join(lines)
                        break
                else:
                    current_spec += patch_comment
            else:
                current_spec += patch_comment

            finding["_patch_applied"] = True
            applied.append(finding)
            print(f"  ✓ Patch applied (added as <!-- RISK-PATCH {fid} --> comment).")
        else:
            deferred.append(finding)
            print("  Deferred.")

    return current_spec, applied, deferred


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

_RISK_EMOJI = {"high": "🔴", "normal": "🟡", "tiny": "🟢"}
_RISK_LEVEL_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def _write_report(
    result:      dict[str, Any],
    spec_version: str,
    applied:     list[dict[str, Any]],
    deferred:    list[dict[str, Any]],
    no_patch:    bool,
) -> str:
    overall_risk    = result.get("overall_risk", "normal")
    risk_rationale  = result.get("risk_rationale", "")
    summary         = result.get("summary", "")
    findings        = result.get("findings", [])
    ts              = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    emoji = _RISK_EMOJI.get(overall_risk, "🟡")
    L: list[str] = []

    L += [
        "# Spec Risk & Impact Assessment", "",
        f"**Overall risk:** {emoji} `{overall_risk.upper()}`",
        f"**Spec version:** {spec_version}",
        f"**Assessed at:** {ts}",
        f"**Findings:** {len(findings)} total  "
        f"({sum(1 for f in findings if f.get('risk_level') == 'high')} high · "
        f"{sum(1 for f in findings if f.get('risk_level') == 'medium')} medium · "
        f"{sum(1 for f in findings if f.get('risk_level') == 'low')} low)",
        "",
        "## Risk Rationale", "",
        risk_rationale, "",
        "## Summary", "",
        summary, "",
    ]

    # Patches applied
    if applied:
        L += ["## High-Confidence Patches Applied", ""]
        for f in applied:
            L += [
                f"- **{f['id']}** [{f['kind']}]: {f['patch_suggestion']}",
                f"  *Added as `<!-- RISK-PATCH {f['id']} -->` comment in spec.*",
                "",
            ]

    # Deferred high-confidence
    if deferred:
        label = "High-Confidence Patches (deferred — --no-patch)" if no_patch \
                else "High-Confidence Patches Deferred by Human"
        L += [f"## {label}", ""]
        for f in deferred:
            L += [
                f"- **{f['id']}** [{f['kind']}]: {f['patch_suggestion']}",
                f"  *To apply manually: edit spec section § {f.get('spec_section', '?')}*",
                "",
            ]

    # Manual review — medium/low confidence findings
    manual = [f for f in findings if f.get("confidence") in ("medium", "low")]
    if manual:
        L += ["## Manual Review Required", "",
              "_The following findings require human judgment. "
              "No automatic patches are suggested._", ""]

        for f in manual:
            fid      = f.get("id", "?")
            kind     = f.get("kind", "")
            risk     = f.get("risk_level", "low")
            conf     = f.get("confidence", "low")
            section  = f.get("spec_section", "")
            line_h   = f.get("spec_line")
            desc     = f.get("description", "")
            areas    = f.get("impact_areas", [])
            opts     = f.get("options") or []
            re_emoji = _RISK_LEVEL_EMOJI.get(risk, "🟢")

            loc = ""
            if section:
                loc = f"§ **{section}**"
                if line_h:
                    loc += f" (line ~{line_h})"

            L += [f"### {fid} — {re_emoji} {kind.replace('_', ' ').title()}  "
                  f"_(risk: {risk} · confidence: {conf})_", ""]
            if loc:
                L += [f"**Location:** {loc}", ""]
            L += [f"**Finding:** {desc}", ""]
            if areas:
                L += [f"**Impact areas:** {', '.join(f'`{a}`' for a in areas)}", ""]
            if opts:
                L += ["**Options:**", ""]
                for i, opt in enumerate(opts, 1):
                    L += [f"  {i}. {opt}"]
                L += [""]
            L += [""]

    # Low-risk informational notes
    info = [f for f in findings
            if f.get("confidence") == "low" and f.get("risk_level") == "low"
            and f not in manual]
    if info:
        L += ["## Low-Risk Notes (informational)", ""]
        for f in info:
            L += [f"- **{f['id']}** [{f.get('kind', '')}]: {f.get('description', '')}"]
        L += [""]

    # No findings case
    if not findings:
        L += [
            "## ✅ No significant risk findings", "",
            "The spec appears well-aligned with the existing codebase.",
            "Proceed to spectracker when ready.", "",
        ]

    # Footer
    L += ["---", "",
          "**Next step:** Review any manual items above, then proceed to spectracker:", "",
          "```",
          f"python pipeline/05_spectracker.py --project "
          f"{os.environ.get('PIPELINE_PROJECT', '<name>')}",
          "```", ""]

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Log writer
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(
    result:       dict[str, Any],
    spec_version: str,
    applied:      list[dict[str, Any]],
    deferred:     list[dict[str, Any]],
) -> None:
    log = _log_path()
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text())
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    findings = result.get("findings", [])
    entry = {
        "assessed_at":   datetime.now(timezone.utc).isoformat(),
        "spec_version":  spec_version,
        "overall_risk":  result.get("overall_risk"),
        "total_findings": len(findings),
        "high_risk":     sum(1 for f in findings if f.get("risk_level") == "high"),
        "medium_risk":   sum(1 for f in findings if f.get("risk_level") == "medium"),
        "low_risk":      sum(1 for f in findings if f.get("risk_level") == "low"),
        "patches_applied":  [f["id"] for f in applied],
        "patches_deferred": [f["id"] for f in deferred],
    }
    existing.append(entry)

    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"entries": existing}, indent=2, ensure_ascii=False))
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    project = args.project
    if not project:
        print("[risk_assessor][error] --project not specified and PIPELINE_PROJECT not set.")
        sys.exit(1)
    os.environ["PIPELINE_PROJECT"] = project
    ensure_dirs()

    print("=" * 60)
    print("  SPEC RISK & IMPACT ASSESSOR")
    print("=" * 60)
    print()

    # ── --show-last ──────────────────────────────────────────────────────────
    if args.show_last:
        p = _report_path()
        if not p.exists():
            print("[risk_assessor] No previous report found.")
            sys.exit(1)
        print(p.read_text())
        sys.exit(0)

    # ── Load inputs ──────────────────────────────────────────────────────────
    spec_path    = get_spec_path()
    spec_text    = _read(spec_path, "spec")
    codebase_map = _read(Path(str(ABSORBER_CODEBASE_MD)), "codebase_map.md")

    if not spec_text:
        print(f"[risk_assessor][error] Spec not found: {spec_path}")
        print("  Run 05_specwright.py first.")
        sys.exit(1)

    if not codebase_map:
        print("[risk_assessor][warn] codebase_map.md not found.")
        print("  Run absorber first for higher-quality assessment.")
        print("  Proceeding with spec only.\n")

    spec_version = _parse_version(spec_text)
    sections     = _spec_section_lines(spec_text)
    print(f"  Spec:          {spec_path.name}  (version {spec_version})")
    print(f"  Sections:      {len(sections)}")
    print(f"  Codebase map:  {'yes' if codebase_map else 'NOT AVAILABLE'}")
    print()

    # ── Build briefing ───────────────────────────────────────────────────────
    briefing = _build_briefing(spec_text, codebase_map)[:MAX_BRIEFING_CHARS]
    print(f"  Briefing size: {len(briefing):,} chars")
    print()

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("[risk_assessor] DRY RUN — briefing follows:\n")
        print(briefing)
        sys.exit(0)

    # ── LLM call ─────────────────────────────────────────────────────────────
    print("[risk_assessor] Calling LLM …")
    t0 = time.time()

    try:
        result, cost = call_llm_json(
            role        = ROLE,
            system      = _SYSTEM,
            user        = briefing,
            max_tokens  = 8192,
            caller_file = __file__,
            label       = "risk_assessor",
        )
    except Exception as e:
        print(f"[risk_assessor][error] LLM call failed: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed  = time.time() - t0
    findings = result.get("findings", [])
    overall  = result.get("overall_risk", "normal")
    emoji    = _RISK_EMOJI.get(overall, "🟡")

    print(f"  Elapsed: {elapsed:.1f}s  |  cost: ${cost:.4f}")
    print(f"  Overall risk: {emoji} {overall.upper()}")
    print(f"  Findings: {len(findings)}  "
          f"({sum(1 for f in findings if f.get('risk_level') == 'high')} high, "
          f"{sum(1 for f in findings if f.get('confidence') == 'high')} high-confidence patches)")
    print()

    # ── Interactive patch loop ───────────────────────────────────────────────
    patched_spec, applied, deferred = _run_patch_loop(
        spec_text = spec_text,
        findings  = findings,
        no_patch  = args.no_patch,
    )

    # Write patched spec back if anything changed
    if applied:
        spec_path.write_text(patched_spec, encoding="utf-8")
        print(f"\n  Spec patched in-place: {spec_path}")
        print(f"  {len(applied)} patch(es) applied as <!-- RISK-PATCH --> comments.")
        print("  Review with: git diff")
        print()

    # ── Write report ─────────────────────────────────────────────────────────
    _spec_dir().mkdir(parents=True, exist_ok=True)

    report_md = _write_report(result, spec_version, applied, deferred, args.no_patch)
    report_path = _report_path()
    report_with_header = apply_md_header(
        content = report_md,
        path    = report_path,
        owner   = "04c_spec_risk_assessor.py",
    )
    report_path.write_text(report_with_header, encoding="utf-8")
    track_write(report_path)

    _append_log(result, spec_version, applied, deferred)

    print(f"  Written:  {report_path}")
    print(f"  Appended: {_log_path()}")
    print()

    print_artifact_summary()
    print()
    print_cost_summary()

    # ── Summary & exit code ──────────────────────────────────────────────────
    print()
    high_findings = [f for f in findings if f.get("risk_level") == "high"]
    manual_items  = [f for f in findings if f.get("confidence") in ("medium", "low")]

    if overall == "high" or high_findings:
        print(f"⚠  HIGH RISK — {len(high_findings)} high-risk finding(s).")
        print(f"   Review {report_path} before proceeding to spectracker.")
        sys.exit(2)
    else:
        if manual_items:
            print(f"  {len(manual_items)} finding(s) require manual review.")
            print(f"  See: {report_path}")
        else:
            print("  ✅ No significant risk findings. Ready for spectracker.")
        print()
        slug = os.environ.get("PIPELINE_PROJECT", "<name>")
        print(f"  Next: python pipeline/05_spectracker.py --project {slug}")
        sys.exit(0)


if __name__ == "__main__":
    main()