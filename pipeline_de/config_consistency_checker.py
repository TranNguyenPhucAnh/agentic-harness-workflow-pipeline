"""
toolkits/devops_mlops/config_consistency_checker.py
====================================================
LLM-assisted cross-file configuration consistency checker.

Detects synchronization failures across config files before they cause incidents:
  - IAM role name in Terraform vs IRSA annotation in values.yaml
  - ServiceAccount name in values.yaml vs pod template in Jenkinsfile
  - Image repository in ECR vs image.repository in values.yaml vs Image Updater annotation
  - S3 path structure in DAG files vs Athena Partition Projection in Terraform
  - Namespace in Helm release vs namespace in ArgoCD Application manifest
  - Resource limits in values.yaml vs actual node allocatable capacity
  - Chart dependency name (wrapper key) vs values.yaml top-level key

Input: infra_absorber output (codebase_map.json) — no raw file re-reading needed.
       Optionally accepts raw file paths for targeted deep-check.

Output:
  consistency/consistency_report.md    (short-term overwrite)
  consistency/consistency_log.json     (long-term append)

Usage:
  python config_consistency_checker.py --project iot-mlops
  python config_consistency_checker.py --project iot-mlops --files terraform/ apps/
  python config_consistency_checker.py --project iot-mlops --focus auth_iam
  python config_consistency_checker.py --project iot-mlops --dry-run
  python config_consistency_checker.py --project iot-mlops --show-last
  PIPELINE_PROJECT=iot-mlops python config_consistency_checker.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
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
from modules.md_header import apply_header as apply_md_header             # noqa: E402
from modules.post_interactive import prompt_next_step                     # noqa: E402
from artifacts.models import get_model                                    # noqa: E402

ROLE = "config_consistency_checker"

# ── Finding types ─────────────────────────────────────────────────────────────
FINDING_TYPES = {
    "MISMATCH":     "Two files reference the same logical value but with different strings",
    "MISSING_REF":  "File A references a value that does not appear to exist in file B",
    "STALE_VALUE":  "A value appears outdated relative to a more recent definition elsewhere",
    "NAMESPACE_DRIFT": "Namespace/scope mismatch between K8s, Helm, ArgoCD, Terraform",
    "PATH_MISMATCH":   "S3/file path structure differs between writer (DAG) and reader (Athena/Terraform)",
    "FORMAT_MISMATCH": "Same logical value in different formats (short name vs full ARN vs URL)",
}

SEVERITY = ["HIGH", "MEDIUM", "LOW"]

# Focus areas map to check groups
FOCUS_AREAS = {
    "auth_iam":        "IRSA role names, OIDC annotations, ServiceAccount names, namespace conditions",
    "image_registry":  "Image repository, tag strategy, Image Updater annotations, chart key paths",
    "data_pipeline":   "S3 path structure, partition format, Athena table schema, DAG connection IDs",
    "networking":      "Namespaces, service names, ports, ingress hostnames, Cloudflare tunnel targets",
    "resources":       "CPU/memory requests/limits, node capacity, HPA targets",
    "all":             "All categories above",
}

_MAX_BRIEFING_CHARS = 80_000   # LLM context budget for briefing
_MAX_JSON_CHARS     = 60_000   # max chars from codebase_map.json inventory


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    if override:
        base = Path(override)
    else:
        base = _REPO_ROOT.parent / "outputs" / "devops_mlops"
    slug = os.environ.get("PIPELINE_PROJECT", "default")
    return base / f"artifacts_{slug}"


def _consistency_dir() -> Path:
    return _devops_artifact_root() / "consistency"


def _report_path() -> Path:
    return _consistency_dir() / "consistency_report.md"


def _log_path() -> Path:
    return _consistency_dir() / "consistency_log.json"


def _absorber_map_json() -> Path:
    return _devops_artifact_root() / "infra_absorber" / "infra_map.json"


def _absorber_map_md() -> Path:
    return _devops_artifact_root() / "infra_absorber" / "infra_map.md"


def _discovery_map_json() -> Path:
    return _devops_artifact_root() / "live_discovery" / "discovery_map.json"


def _doc_map_json() -> Path:
    return _devops_artifact_root() / "doc_absorber" / "doc_map.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# Load absorber output
# ─────────────────────────────────────────────────────────────────────────────

def _load_absorber_map() -> dict[str, Any]:
    """Load infra_map.json from infra_absorber output."""
    p = _absorber_map_json()
    if not p.exists():
        return {}
    track_read(p)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] Failed to load absorber map: {exc}")
        return {}


def _load_absorber_md() -> str:
    """Load infra_map.md summary (compact, good for context)."""
    p = _absorber_map_md()
    if not p.exists():
        return ""
    track_read(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _load_discovery_map() -> dict[str, Any]:
    """
    Load discovery_map.json from live_discovery output.
    Confidence: MEDIUM — actual deployed state but may include orphans/abandoned resources.
    """
    p = _discovery_map_json()
    if not p.exists():
        return {}
    track_read(p)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] Failed to load discovery map: {exc}")
        return {}


def _load_doc_map() -> dict[str, Any]:
    """
    Load doc_map.json from doc_absorber output.
    Confidence: LOW — human-written, assume drift vs actual state.
    """
    p = _doc_map_json()
    if not p.exists():
        return {}
    track_read(p)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [warn] Failed to load doc map: {exc}")
        return {}


def _sources_available(
    absorber_map:   dict[str, Any],
    discovery_map:  dict[str, Any],
    doc_map:        dict[str, Any],
) -> dict[str, bool]:
    """Return which sources are available — used for confidence weighting."""
    return {
        "infra_absorber":  bool(absorber_map),
        "live_discovery":  bool(discovery_map),
        "doc_absorber":    bool(doc_map),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Briefing builder — extract cross-reference candidates from absorber map
# ─────────────────────────────────────────────────────────────────────────────

def _build_briefing(
    absorber_map:   dict[str, Any],
    absorber_md:    str,
    discovery_map:  dict[str, Any],
    doc_map:        dict[str, Any],
    focus:          str,
    extra_files:    list[Path],
    mode:           str,
) -> str:
    """
    Build LLM briefing from all available sources.

    Three sources with explicit confidence levels:
      A  infra_absorber  → DESIRED state   confidence: HIGH  (IaC source of truth)
      B  live_discovery  → DEPLOYED state  confidence: MEDIUM (actual but may have orphans)
      C  doc_absorber    → DOCUMENTED state confidence: LOW   (human-written, assume drift)

    Mode:
      consistency  — check within-IaC synchronization (file_a vs file_b mismatches)
      drift        — check IaC/docs vs live deployment (desired vs actual)
      full         — both consistency and drift

    The LLM uses source agreement to weight confidence:
      3/3 sources agree  → HIGH confidence finding
      2/3 sources agree  → MEDIUM confidence
      1/3 source only    → LOW confidence / signal only
    """
    sections: list[str] = []
    available = _sources_available(absorber_map, discovery_map, doc_map)
    avail_str = ", ".join(k for k, v in available.items() if v) or "none"

    sections.append(
        f"## Context\n\n"
        f"Mode: {mode}\n"
        f"Focus: {focus}\n"
        f"Sources available: {avail_str}\n\n"
        f"Confidence model:\n"
        f"  infra_absorber (A) = HIGH   — IaC source files, deterministic\n"
        f"  live_discovery (B) = MEDIUM — actual AWS state, may include orphans\n"
        f"  doc_absorber   (C) = LOW    — human-written handover docs, assume drift\n"
    )

    # ── Source A: infra_absorber ──────────────────────────────────────────────
    if absorber_map or absorber_md:
        parts_a: list[str] = ["## Source A — IaC (infra_absorber) | confidence: HIGH\n"]

        if absorber_md:
            topo_match = re.search(
                r"## Infrastructure Topology.*?(?=\n## |\Z)", absorber_md, re.DOTALL
            )
            if topo_match:
                parts_a.append(topo_match.group().strip()[:4000])

        topo = absorber_map.get("infra_topology", {})
        if topo:
            xref = _extract_crossref_candidates(topo, focus)
            if xref:
                parts_a.append("\n### Cross-reference candidates (IaC)\n\n" + xref)

        sections.append("\n".join(parts_a))
    else:
        sections.append(
            "## Source A — IaC (infra_absorber) | confidence: HIGH\n\n"
            "_[Not available — run infra_absorber.py to generate infra_map.json]_"
        )

    # ── Source B: live_discovery ──────────────────────────────────────────────
    if discovery_map:
        parts_b: list[str] = [
            "## Source B — Live AWS (live_discovery) | confidence: MEDIUM\n"
            "_Note: may include orphaned/abandoned resources not in IaC._\n"
        ]
        svc_data   = discovery_map.get("services", {})
        crawler_st = discovery_map.get("crawler_status", {})
        ok_svcs    = [k for k, v in crawler_st.items() if v.get("status") == "ok"]

        if ok_svcs:
            parts_b.append(f"Services crawled: {', '.join(ok_svcs)}\n")

        # IAM roles from live
        iam_roles = [
            r.get("name") for r in svc_data.get("iam", {}).get("roles", []) if r.get("name")
        ]
        if iam_roles:
            parts_b.append(
                "### IAM roles (live)\n"
                + "\n".join(f"  - {r}" for r in iam_roles[:30])
            )

        # S3 buckets
        buckets = [
            b.get("name") for b in svc_data.get("s3", {}).get("buckets", []) if b.get("name")
        ]
        if buckets:
            parts_b.append(
                "\n### S3 buckets (live)\n"
                + "\n".join(f"  - {b}" for b in buckets[:30])
            )

        # ECR repos and image tags
        repos = svc_data.get("ecr", {}).get("repositories", [])
        if repos:
            lines = []
            for r in repos[:15]:
                tags = r.get("image_tags", [])[:5]
                lines.append(f"  - {r.get('name')}: tags=[{', '.join(tags)}]")
            parts_b.append("\n### ECR repositories (live)\n" + "\n".join(lines))

        # EKS — SA annotations
        for cluster in svc_data.get("eks", {}).get("clusters", []):
            parts_b.append(f"\n### EKS cluster: {cluster.get('name')} (live)")
            parts_b.append(f"  OIDC issuer: {cluster.get('oidc_issuer', '—')}")

        # SQS queues
        queues = [
            q.get("name") for q in svc_data.get("sqs", {}).get("queues", []) if q.get("name")
        ]
        if queues:
            parts_b.append(
                "\n### SQS queues (live)\n"
                + "\n".join(f"  - {q}" for q in queues[:20])
            )

        # Secrets manager — names only
        secrets = [
            s.get("name") for s in svc_data.get("secretsmanager", {}).get("secrets", [])
            if s.get("name")
        ]
        if secrets:
            parts_b.append(
                "\n### Secrets Manager entries (live, names only)\n"
                + "\n".join(f"  - {s}" for s in secrets[:20])
            )

        sections.append("\n".join(parts_b))
    else:
        sections.append(
            "## Source B — Live AWS (live_discovery) | confidence: MEDIUM\n\n"
            "_[Not available — run live_discovery.py to generate discovery_map.json]_"
        )

    # ── Source C: doc_absorber ────────────────────────────────────────────────
    if doc_map:
        parts_c: list[str] = [
            "## Source C — Handover docs (doc_absorber) | confidence: LOW\n"
            "_Note: human-written, assume drift vs actual state. Treat as signals, not facts._\n"
        ]
        infra_facts = doc_map.get("infra_facts", [])
        if infra_facts:
            parts_c.append("### Infra facts mentioned in docs")
            for fact in infra_facts[:30]:
                # doc_map schema: type, value, key_name, cross_ref_hint
                fact_type  = fact.get("type", "")
                key_name   = fact.get("key_name", "")
                value      = fact.get("value", "")
                cross_hint = fact.get("cross_ref_hint", "")
                line = f"  - [{fact_type}] {key_name + ': ' if key_name else ''}{value}"
                if cross_hint:
                    line += f"  → cross-ref: {cross_hint}"
                parts_c.append(line)

        inst_knowledge = doc_map.get("institutional_knowledge", [])
        if inst_knowledge:
            parts_c.append("\n### Architectural decisions / known issues")
            for item in inst_knowledge[:10]:
                parts_c.append(
                    f"  - [{item.get('type','')}] {item.get('content','')[:120]}"
                )

        sections.append("\n".join(parts_c))
    else:
        sections.append(
            "## Source C — Handover docs (doc_absorber) | confidence: LOW\n\n"
            "_[Not available — run doc_absorber.py to generate doc_map.json]_"
        )

    # ── Extra files ───────────────────────────────────────────────────────────
    if extra_files:
        extra_parts: list[str] = []
        for path in extra_files[:10]:
            if not path.exists():
                continue
            track_read(path)
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 3000:
                content = content[:3000] + "\n… (truncated)"
            extra_parts.append(f"### {path}\n```\n{content}\n```")
        if extra_parts:
            sections.append(
                "## Additional files (user-specified)\n\n"
                + "\n\n".join(extra_parts)
            )

    briefing = "\n\n---\n\n".join(sections)
    if len(briefing) > _MAX_BRIEFING_CHARS:
        briefing = briefing[:_MAX_BRIEFING_CHARS] + "\n\n… (truncated to fit context)"

    return briefing


def _extract_crossref_candidates(
    topo:  dict[str, Any],
    focus: str,
) -> str:
    """
    Render infra_topology into a structured cross-reference block for the LLM.

    Reads directly from the aggregated topology dict in infra_map.json —
    not per-file inventory. Each list item schema matches what
    infra_absorber.build_infra_topology() produces.
    """
    groups: dict[str, list[str]] = defaultdict(list)

    # ── auth_iam ──────────────────────────────────────────────────────────────
    if focus in ("auth_iam", "all"):
        for sa in topo.get("service_accounts", []):
            name = sa.get("name", "?")
            irsa = sa.get("irsa", "")
            file_ = sa.get("file", "")
            src  = sa.get("source", "")
            label = f"{file_}" + (f" [{src}]" if src else "")
            if irsa:
                groups["service_account_annotations"].append(
                    f"  {label}: name={name}  irsa={irsa}"
                )
            else:
                groups["service_account_names"].append(
                    f"  {label}: name={name}"
                )

        for role in topo.get("iam_roles", []):
            groups["iam_roles"].append(
                f"  {role.get('file', '?')}: {role.get('name', '?')}"
            )

    # ── image_registry ────────────────────────────────────────────────────────
    if focus in ("image_registry", "all"):
        for img in topo.get("image_refs", []):
            groups["image_refs"].append(f"  {img}")

    # ── data_pipeline ─────────────────────────────────────────────────────────
    if focus in ("data_pipeline", "all"):
        for bucket in topo.get("s3_buckets", []):
            groups["s3_paths"].append(f"  s3://{bucket}")

        for conn in topo.get("connections_used", []):
            groups["connection_ids"].append(f"  {conn}")

        for dag in topo.get("dag_inventory", []):
            tasks_str = ", ".join(dag.get("tasks", [])[:8])
            groups["dag_tasks"].append(
                f"  {dag.get('file', '?')}: dag_id={dag.get('dag_id', '?')}  tasks=[{tasks_str}]"
            )

    # ── networking ────────────────────────────────────────────────────────────
    if focus in ("networking", "all"):
        for ns in topo.get("k8s_namespaces", []):
            groups["namespaces"].append(f"  {ns}")

        for app in topo.get("argocd_apps", []):
            groups["argocd_apps"].append(
                f"  {app.get('file', '?')}: name={app.get('name', '?')} "
                f"repo={app.get('repoURL', '?')} rev={app.get('targetRevision', '?')}"
            )

    # ── resources (terraform + helm, cross-cutting) ───────────────────────────
    if focus in ("resources", "all"):
        for res in topo.get("terraform_resources", []):
            rtype = res.get("type", "")
            if rtype in ("aws_eks_node_group", "aws_autoscaling_group",
                         "kubernetes_deployment", "kubernetes_stateful_set"):
                groups["resource_limits"].append(
                    f"  {res.get('file', '?')}: {rtype}.{res.get('name', '?')}"
                )

    if not any(groups.values()):
        return ""

    group_labels = {
        "service_account_annotations": "ServiceAccount names + IRSA annotations (cross-file)",
        "service_account_names":        "ServiceAccount names (no IRSA found)",
        "iam_roles":                    "IAM role names from Terraform",
        "image_refs":                   "Container image refs (all sources)",
        "s3_paths":                     "S3 bucket names",
        "connection_ids":               "Airflow connection IDs",
        "dag_tasks":                    "Airflow DAG inventory",
        "namespaces":                   "Kubernetes namespaces",
        "argocd_apps":                  "ArgoCD Application targets",
        "resource_limits":              "Compute resource definitions",
    }

    parts: list[str] = []
    for key, label in group_labels.items():
        items = groups.get(key, [])
        if items:
            parts.append(f"### {label}\n" + "\n".join(items))

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

def _build_system(focus: str) -> str:
    focus_desc = FOCUS_AREAS.get(focus, FOCUS_AREAS["all"])
    finding_types_str = "\n".join(f"  {k}: {v}" for k, v in FINDING_TYPES.items())

    return f"""\
You are a DevOps/MLOps infrastructure consistency auditor.

You will receive configuration data from up to THREE sources, each with a
different confidence level:

  Source A — infra_absorber (IaC source files)   confidence: HIGH
             Machine-readable, deterministic. Use as primary reference.

  Source B — live_discovery (live AWS state)      confidence: MEDIUM
             Actual deployed state — may include orphaned/abandoned resources.

  Source C — doc_absorber (handover docs)         confidence: LOW
             Human-written, assume drift vs actual. Treat as signals only.

Focus area: {focus} — {focus_desc}

## Your task
1. CONSISTENCY mode: Find sync failures within IaC (Source A vs A — file_a vs file_b).
2. DRIFT mode: Find divergence between desired (A) and deployed (B), or doc (C) vs live (B).
3. FULL mode: Both of the above.

Use source agreement to weight confidence:
  3/3 sources agree  → HIGH confidence finding
  2/3 sources agree  → MEDIUM confidence
  1/3 source only    → LOW confidence — signal only, verify manually

## Normalization rules (do NOT flag these as mismatches)
- Short name vs full ARN: `my-role` and `arn:aws:iam::123456789:role/my-role` are the SAME value.
- Short hostname vs FQDN: `svc-name` and `svc-name.namespace.svc.cluster.local` are the SAME.
- `latest` tag is intentionally dynamic — do not flag as mismatch unless image_updater is involved.
- Values prefixed with `${{` or `${{` are template variables — skip if you cannot resolve them.
- Intentional multi-environment differences (dev/staging/prod) — flag only if in same environment.
- doc_absorber (Source C) mismatch with A or B ALONE does not constitute a HIGH finding —
  it is a LOW signal indicating stale docs unless A and B also disagree.

## Finding types
{finding_types_str}

## Severity guidelines
HIGH   — will cause a runtime failure if not fixed. Must be backed by Source A or B.
MEDIUM — likely to cause issues but has a workaround. May be A-only or A+B.
LOW    — informational, doc-only signal, or needs manual verification.

Return ONLY valid JSON (no markdown fences, no prose):
{{
  "summary": "<1-2 sentences overall assessment>",
  "risk_level": "HIGH" | "MEDIUM" | "LOW" | "CLEAN",
  "findings": [
    {{
      "id":               "CC-01",
      "type":             "<one of the finding type keys above>",
      "severity":         "HIGH" | "MEDIUM" | "LOW",
      "sources_agree":    ["infra_absorber", "live_discovery"],
      "sources_disagree": ["doc_absorber"],
      "file_a":           "<relative path or source label>",
      "value_a":          "<the value in source A>",
      "file_b":           "<relative path or source label>",
      "value_b":          "<the value in source B>",
      "description":      "<what is inconsistent and why it matters>",
      "fix_hint":         "<specific action: which file to change, to what value>",
      "likely_explanation": "<e.g. handover doc outdated / orphan resource / IaC drift>"
    }}
  ]
}}

RULES:
- Do NOT invent findings. Only report inconsistencies clearly visible in the provided data.
- If no inconsistencies found: {{"summary": "No issues found.", "risk_level": "CLEAN", "findings": []}}
- Maximum 20 findings. Prioritize HIGH severity.
- fix_hint must name the specific file and the exact change needed.
- sources_agree / sources_disagree must list which of the three source labels agree/disagree.
- Output ONLY the JSON object.

"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

def _call_checker_llm(briefing: str, focus: str) -> dict[str, Any]:
    system = _build_system(focus)
    user   = (
        f"Focus: {focus}\n\n"
        f"Check the following configuration data for consistency issues:\n\n"
        f"{briefing}"
    )

    print(f"  Briefing size: {len(briefing):,} chars → LLM …")

    raw, _ = call_llm(
        ROLE, system, user,
        max_tokens=4096,
        caller_file=__file__,
        label=f"[consistency] {get_model(ROLE)}",
    )

    return _parse_json(raw)


def _parse_json(raw: str) -> dict[str, Any]:
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
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
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
        f"No valid JSON in LLM response ({len(raw)} chars). First 200: {raw[:200]!r}",
        raw, 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
_RISK_EMOJI     = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "CLEAN": "✅"}


def _write_report(
    result:     dict[str, Any],
    focus:      str,
    run_at:     str,
) -> str:
    """Build markdown report content (without md_header — returned for caller to apply)."""
    findings = result.get("findings", [])
    summary  = result.get("summary", "")
    risk     = result.get("risk_level", "CLEAN")
    emoji    = _RISK_EMOJI.get(risk, "")

    by_severity: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_severity[f.get("severity", "LOW")].append(f)

    lines: list[str] = [
        f"# Config Consistency Report",
        f"",
        f"**Run at:** {run_at}  ",
        f"**Focus:** {focus}  ",
        f"**Risk level:** {emoji} {risk}  ",
        f"**Findings:** {len(findings)} "
        f"({len(by_severity.get('HIGH', []))} HIGH, "
        f"{len(by_severity.get('MEDIUM', []))} MEDIUM, "
        f"{len(by_severity.get('LOW', []))} LOW)",
        f"",
        f"**Summary:** {summary}",
        f"",
        f"---",
        f"",
    ]

    if not findings:
        lines += ["## ✅ No issues found", "", "All cross-referenced values appear consistent."]
        return "\n".join(lines)

    for severity in ("HIGH", "MEDIUM", "LOW"):
        sev_findings = by_severity.get(severity, [])
        if not sev_findings:
            continue

        sem = _SEVERITY_EMOJI.get(severity, "")
        lines += [f"## {sem} {severity} ({len(sev_findings)})", ""]

        for f in sev_findings:
            fid   = f.get("id", "?")
            ftype = f.get("type", "")
            fa    = f.get("file_a", "?")
            fb    = f.get("file_b", "?")
            va    = f.get("value_a", "?")
            vb    = f.get("value_b", "?")
            desc  = f.get("description", "")
            fix   = f.get("fix_hint", "")

            agree    = f.get("sources_agree", [])
            disagree = f.get("sources_disagree", [])
            explain  = f.get("likely_explanation", "")
            agree_str    = ", ".join(agree)    if agree    else "—"
            disagree_str = ", ".join(disagree) if disagree else "—"

            lines += [
                f"### {fid} — {ftype}",
                f"",
                f"| | File / Source | Value |",
                f"|---|---|---|",
                f"| A | `{fa}` | `{va}` |",
                f"| B | `{fb}` | `{vb}` |",
                f"",
                f"**Sources agree:** {agree_str}  ",
                f"**Sources disagree:** {disagree_str}",
                f"",
                f"**Issue:** {desc}",
                f"",
                f"**Fix:** {fix}",
                *([ f"", f"**Likely explanation:** {explain}" ] if explain else []),
                f"",
                f"---",
                f"",
            ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Log I/O
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(
    result:      dict[str, Any],
    focus:       str,
    run_at:      str,
    dry_run:     bool,
) -> None:
    if dry_run:
        return

    findings   = result.get("findings", [])
    risk       = result.get("risk_level", "CLEAN")
    high_count = sum(1 for f in findings if f.get("severity") == "HIGH")
    sev_ids    = [f.get("id") for f in findings if f.get("severity") == "HIGH"]

    log = _log_path()
    log.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    try:
        track_read(log)
        data = json.loads(log.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else data.get("entries", [])
    except Exception:
        entries = []

    entries.append({
        "run_at":          run_at,
        "focus":           focus,
        "mode":            result.get("_mode", "full"),
        "sources_used":    result.get("_sources_used", []),
        "risk_level":      risk,
        "total_findings":  len(findings),
        "high_findings":   high_count,
        "high_ids":        sev_ids,
        "summary":         result.get("summary", ""),
    })

    log.write_text(
        json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(log)


def _show_last() -> None:
    """Print the last consistency report to stdout."""
    p = _report_path()
    if not p.exists():
        print("[consistency] No report found. Run without --show-last first.")
        return
    track_read(p)
    print(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Maybe commit log
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_commit_log() -> None:
    """Y/n prompt to keep the just-appended log entry."""
    log = _log_path()
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
        print(f"  [consistency] Entry kept (non-interactive).")
        return

    if ans in ("n", "no"):
        entries.pop()
        try:
            log.write_text(
                json.dumps({"entries": entries}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [consistency] Entry discarded.")
        except Exception as exc:
            print(f"  [consistency][warn] Could not revert log: {exc}")
    else:
        print(f"  [consistency] Entry kept (total: {len(entries)}).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="config_consistency_checker.py",
        description="LLM-assisted cross-file config consistency checker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",    default=os.environ.get("PIPELINE_PROJECT"))
    p.add_argument(
        "--focus",
        choices=list(FOCUS_AREAS.keys()),
        default="all",
        help="Limit check to a specific focus area (default: all).",
    )
    p.add_argument(
        "--files",
        nargs="+",
        metavar="PATH",
        help="Additional file/dir paths to include alongside absorber output.",
    )
    p.add_argument("--dry-run",   action="store_true",
                   help="Run analysis but do not write artifacts.")
    p.add_argument("--show-last", action="store_true",
                   help="Print last report and exit.")
    p.add_argument("--no-absorber", action="store_true",
                   help="Skip loading infra_absorber output (use only --files input).")
    p.add_argument(
        "--mode",
        choices=["consistency", "drift", "full"],
        default="full",
        help=(
            "consistency — within-IaC sync check (file_a vs file_b). "
            "drift — IaC/docs vs live deployment. "
            "full — both (default)."
        ),
    )
    p.add_argument("--verbose",   action="store_true")
    return p


def _configure_project(project: str | None, parser: argparse.ArgumentParser) -> None:
    if project:
        os.environ["PIPELINE_PROJECT"] = project
        return
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error("Use --project <name> or export PIPELINE_PROJECT=<name>.")


def _resolve_extra_files(paths: list[str]) -> list[Path]:
    """Expand file/dir args to list of individual file paths."""
    result: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir():
            for ext in ("*.tf", "*.yaml", "*.yml", "*.py", "Jenkinsfile", "*.json"):
                result.extend(p.rglob(ext))
        elif p.is_file():
            result.append(p)
    # Deduplicate, keep order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for f in result:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    _configure_project(args.project, parser)
    _consistency_dir().mkdir(parents=True, exist_ok=True)

    # --show-last short circuit
    if args.show_last:
        _show_last()
        sys.exit(0)

    print("=" * 68)
    print("  CONFIG CONSISTENCY CHECKER")
    print("=" * 68)
    print(f"  Focus:    {args.focus}")
    print(f"  Dry-run:  {args.dry_run}")
    print()

    exit_code  = 0
    result: dict[str, Any] = {}
    run_at = _now_display()

    try:
        # Load absorber output
        absorber_map = {}
        absorber_md  = ""
        if not args.no_absorber:
            absorber_map = _load_absorber_map()
            absorber_md  = _load_absorber_md()
            if not absorber_map and not absorber_md:
                print("[consistency][warn] infra_absorber output not found.")
                print(f"  Expected: {_absorber_map_json()}")
                print("  Run infra_absorber.py first, or use --files to pass files directly.")
                if not args.files:
                    sys.exit(1)
            else:
                inv_count = len(absorber_map.get("infra_topology", {}).get("terraform_resources", []))
                print(f"  Absorber: loaded (topology resources: {inv_count})")

        # Load live_discovery and doc_absorber outputs (optional — degrade gracefully)
        discovery_map = _load_discovery_map()
        doc_map       = _load_doc_map()

        if discovery_map:
            svc_count = len(discovery_map.get("services", {}))
            print(f"  Discovery: {svc_count} service(s) from live_discovery")
        else:
            print("  Discovery: not available (run live_discovery.py to enable drift detection)")

        if doc_map:
            fact_count = len(doc_map.get("infra_facts", []))
            print(f"  Doc map:   {fact_count} infra fact(s) from doc_absorber")
        else:
            print("  Doc map:   not available (run doc_absorber.py to enable doc cross-check)")

        # Resolve extra files
        extra_files: list[Path] = []
        if args.files:
            extra_files = _resolve_extra_files(args.files)
            print(f"  Extra files: {len(extra_files)}")

        # Build briefing — pass all 3 sources + mode
        print("  Building briefing …")
        briefing = _build_briefing(
            absorber_map  = absorber_map,
            absorber_md   = absorber_md,
            discovery_map = discovery_map,
            doc_map       = doc_map,
            focus         = args.focus,
            extra_files   = extra_files,
            mode          = args.mode,
        )

        if not briefing.strip():
            print("[consistency] No configuration data found to check. Exiting.")
            sys.exit(1)

        if args.verbose:
            print(f"  Briefing: {len(briefing):,} chars")

        # LLM call
        print("  Calling LLM …")
        result = _call_checker_llm(briefing, args.focus)

        # Annotate result with metadata for log
        result["_mode"]         = args.mode
        result["_sources_used"] = [
            s for s, avail in {
                "infra_absorber": bool(absorber_map),
                "live_discovery": bool(discovery_map),
                "doc_absorber":   bool(doc_map),
            }.items() if avail
        ]

        findings   = result.get("findings", [])
        risk       = result.get("risk_level", "CLEAN")
        high_count = sum(1 for f in findings if f.get("severity") == "HIGH")

        print()
        print(f"  Risk level: {_RISK_EMOJI.get(risk, '')} {risk}")
        print(f"  Findings:   {len(findings)} "
              f"({high_count} HIGH, "
              f"{sum(1 for f in findings if f.get('severity') == 'MEDIUM')} MEDIUM, "
              f"{sum(1 for f in findings if f.get('severity') == 'LOW')} LOW)")
        print()

        # Print HIGH findings to terminal immediately
        if high_count:
            print("  ── HIGH severity findings ──────────────────────────────")
            for f in findings:
                if f.get("severity") == "HIGH":
                    print(f"  {f['id']}: {f.get('description', '')[:80]}")
                    print(f"    Fix: {f.get('fix_hint', '')[:80]}")
            print()

        # Write report
        if not args.dry_run:
            report_md = _write_report(result, args.focus, run_at)
            report_md = apply_md_header(
                content = report_md,
                path    = _report_path(),
                owner   = "config_consistency_checker.py",
            )
            _report_path().write_text(report_md, encoding="utf-8")
            track_write(_report_path())
            print(f"  Report: {_report_path()}")

            _append_log(result, args.focus, run_at, dry_run=False)
            print(f"  Log:    {_log_path()}")
        else:
            print("[consistency] Dry-run — no artifacts written.")
            # Print report to stdout in dry-run
            print()
            print(_write_report(result, args.focus, run_at))

        # Exit code: 2 for HIGH findings (CI gating), 0 otherwise
        if high_count:
            exit_code = 2
        elif findings:
            exit_code = 1
        else:
            exit_code = 0

    except KeyboardInterrupt:
        print("\n[consistency] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[consistency][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[consistency]")
        print()
        print_cost_summary("[consistency]")
        prompt_next_step(ROLE, prefix="[consistency]")

    # Long-term artifact commit
    if not args.dry_run and exit_code in (0, 1, 2):
        _maybe_commit_log()

    # Final message
    if exit_code == 2:
        print(f"\n[consistency] ⚠  HIGH risk — review {_report_path()} before deploying.")
    elif exit_code == 0:
        print(f"\n[consistency] ✅ No issues found.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
