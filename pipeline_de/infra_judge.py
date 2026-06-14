"""
toolkits/devops_mlops/infra_judge.py
=====================================
Periodic infrastructure verdict — reads aggregated metrics, consistency
findings, and history trends, then produces a structured verdict with
actionable recommendations.

Position in toolkit:
  metrics_reporter  →  metrics_report.json   ┐
  config_checker    →  consistency_log.json  ├─► infra_judge  →  verdict
  postmortem_archivist → postmortem_kb.json  ┘

Unlike judge.py in the SWE toolkit (which reviews code quality after
green tests), infra_judge operates on continuous operational data — there
is no "done" state, only "current health snapshot" and trends over time.

────────────────────────────────────────────────────────────────
What infra_judge evaluates
────────────────────────────────────────────────────────────────

  1. Infra health
     OOM events, pod restarts, node CPU/memory utilisation.
     Trend: are restarts increasing week-over-week?

  2. ML pipeline health
     Model registered and in Production? Training running regularly?
     Data quality flags (flat data, low f1, skipped registration)?
     Trend: has f1 been improving or stagnating?

  3. Cost
     Total cost vs previous period. Week-over-week delta.
     Anomalies flagged by metrics_reporter.
     Trend: cost creep vs seasonal variation?

  4. Drift risk (from config_consistency_checker)
     Latest consistency run risk level. How many HIGH findings?
     Are findings recurring (same ids across multiple runs)?

  5. Known incident patterns (from postmortem_archivist)
     Are current flags matching known incident taxonomies?
     Preventable incidents that have not been addressed yet?

────────────────────────────────────────────────────────────────
Verdict levels
────────────────────────────────────────────────────────────────

  HEALTHY       All dimensions green, no concerning trends.
  DEGRADED      One or more dimensions yellow. Attention needed.
  CRITICAL      Active issue requiring immediate action.
  INSUFFICIENT  Not enough data to make a meaningful verdict
                (missing collectors, first run, etc.)

────────────────────────────────────────────────────────────────
Outputs written
────────────────────────────────────────────────────────────────

  infra_judge/verdict.md         (short-term, OVERWRITE)
    Human-readable verdict with per-dimension assessment,
    trend analysis, recommendations, and postmortem cross-references.

  infra_judge/verdict.json       (short-term, OVERWRITE)
    Structured verdict for downstream consumption.
    Schema: meta, overall, dimensions, recommendations,
            trends, postmortem_refs, flags_summary.

  infra_judge/verdict_log.json   (long-term, APPEND)
    One entry per run. Enables trend detection across verdicts.

────────────────────────────────────────────────────────────────
Artifact impact by command
────────────────────────────────────────────────────────────────

  Command                  verdict.md  verdict.json  log.json
  ─────────────────────── ──────────  ────────────  ────────
  (normal run)             OVERWRITE   OVERWRITE     APPEND
  --dry-run                –           –             –
  --show-last              –           –             –
  --bootstrap              –           –             APPEND (data only)

────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────

  python infra_judge.py --project iot-mlops
    Full verdict. Reads all available sources.

  python infra_judge.py --project iot-mlops --history-runs 4
    Use last N history entries for trend analysis (default: 4).

  python infra_judge.py --project iot-mlops --dry-run
    Build briefing and print it. No LLM call, no writes.

  python infra_judge.py --project iot-mlops --show-last
    Print most recent verdict.md without re-running.

  python infra_judge.py --project iot-mlops --bootstrap
    First-run mode: collect and store current snapshot in history
    without making a verdict (no trend data yet for meaningful assessment).

────────────────────────────────────────────────────────────────
Environment variables
────────────────────────────────────────────────────────────────

  PIPELINE_PROJECT      Required. Project slug.
  DEVOPS_ARTIFACT_ROOT  Override artifact output root.

For taxonomy details see artifacts/TAXONOMY.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from modules.artifact_tracking import (  # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.call_llm import call_llm                      # noqa: E402
from modules.cost import print_summary as print_cost_summary  # noqa: E402
from modules.md_header import apply_header as apply_md_header  # noqa: E402
from artifacts.models import get_model                     # noqa: E402

ROLE = "infra_judge"

_DEFAULT_HISTORY_RUNS = 4
_MAX_BRIEFING_CHARS   = 120_000


# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────

def _devops_artifact_root() -> Path:
    override = os.environ.get("DEVOPS_ARTIFACT_ROOT")
    base     = Path(override) if override else _REPO_ROOT.parent / "outputs" / "devops_mlops"
    slug     = os.environ.get("PIPELINE_PROJECT", "default")
    return base / f"artifacts_{slug}"

def _judge_dir()        -> Path: return _devops_artifact_root() / "infra_judge"
def _verdict_md()       -> Path: return _judge_dir() / "verdict.md"
def _verdict_json()     -> Path: return _judge_dir() / "verdict.json"
def _verdict_log()      -> Path: return _judge_dir() / "verdict_log.json"

# Input artifact paths from other modules
def _metrics_report()   -> Path: return _devops_artifact_root() / "metrics_reporter" / "metrics_report.json"
def _metrics_history()  -> Path: return _devops_artifact_root() / "metrics_reporter" / "metrics_history.json"
def _consistency_log()  -> Path: return _devops_artifact_root() / "consistency"       / "consistency_log.json"
def _postmortem_kb()    -> Path: return _devops_artifact_root() / "postmortem"         / "postmortem_kb.json"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Loaders — all optional, graceful empty on missing
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path, label: str) -> dict[str, Any] | list[Any]:
    if not path.exists():
        print(f"  [judge] {label}: not found — {path}")
        return {}
    track_read(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [judge] {label}: failed to load — {exc}")
        return {}


def _load_metrics_report() -> dict[str, Any]:
    data = _load_json(_metrics_report(), "metrics_report.json")
    return data if isinstance(data, dict) else {}


def _load_metrics_history(n: int) -> list[dict[str, Any]]:
    """Return last N entries from metrics_history.json."""
    data = _load_json(_metrics_history(), "metrics_history.json")
    entries = data.get("entries", []) if isinstance(data, dict) else []
    return entries[-n:] if entries else []


def _load_consistency_log() -> list[dict[str, Any]]:
    """Return all entries from consistency_log.json, most recent last."""
    data = _load_json(_consistency_log(), "consistency_log.json")
    if isinstance(data, dict):
        return data.get("entries", [])
    return []


def _load_postmortem_kb() -> list[dict[str, Any]]:
    """Return postmortem entries for pattern matching."""
    data = _load_json(_postmortem_kb(), "postmortem_kb.json")
    if isinstance(data, dict):
        return data.get("entries", [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Trend analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trend(values: list[float | None], label: str) -> dict[str, Any]:
    """
    Compute simple trend from a list of values (oldest → newest).
    Returns: {direction, magnitude, values, label}
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return {"direction": "insufficient_data", "values": clean, "label": label}

    delta  = clean[-1] - clean[0]
    pct    = (delta / abs(clean[0]) * 100) if clean[0] != 0 else 0
    if pct > 15:
        direction = "increasing"
    elif pct < -15:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "direction":  direction,
        "magnitude":  round(pct, 1),
        "first":      clean[0],
        "last":       clean[-1],
        "values":     clean,
        "label":      label,
    }


def _compute_trends(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute trends from metrics_history entries."""
    if len(history) < 2:
        return {"note": "Insufficient history for trend analysis (need ≥2 runs)."}

    return {
        "oom_events":          _trend([h.get("oom_events")        for h in history], "OOM events"),
        "pod_restarts":        _trend([h.get("total_pod_restarts") for h in history], "Pod restarts"),
        "node_cpu_avg_pct":    _trend([h.get("node_cpu_avg_pct")   for h in history], "Node CPU %"),
        "node_memory_avg_pct": _trend([h.get("node_memory_avg_pct") for h in history], "Node Memory %"),
        "total_cost_usd":      _trend([h.get("total_cost_usd")     for h in history], "Total cost USD"),
        "warning_count":       _trend([h.get("warning_count")      for h in history], "Warning flags"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Postmortem cross-reference
# ─────────────────────────────────────────────────────────────────────────────

_FLAG_CODE_TO_TAXONOMY = {
    "OOM_EVENTS":         "resource_constraint",
    "HIGH_RESTART_COUNT": "resource_constraint",
    "HIGH_NODE_MEMORY":   "resource_constraint",
    "HIGH_NODE_CPU":      "resource_constraint",
    "COST_SPIKE":         "other",
    "MODEL_NOT_REGISTERED": "mlops",
    "FLAT_DATA":          "mlops",
    "LOW_F1":             "mlops",
    "NO_TRAINING":        "mlops",
}


def _find_relevant_postmortems(
    flags:      list[dict[str, Any]],
    postmortems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Match current warning flags against postmortem taxonomy.
    Returns postmortem entries likely relevant to current issues.
    """
    if not flags or not postmortems:
        return []

    relevant_taxonomies = set()
    for flag in flags:
        code     = flag.get("code", "")
        taxonomy = _FLAG_CODE_TO_TAXONOMY.get(code)
        if taxonomy:
            relevant_taxonomies.add(taxonomy)

    refs: list[dict[str, Any]] = []
    for pm in postmortems:
        if pm.get("taxonomy") in relevant_taxonomies:
            refs.append({
                "incident_id":   pm.get("incident_id"),
                "taxonomy":      pm.get("taxonomy"),
                "symptom":       pm.get("symptom", "")[:100],
                "resolution":    (pm.get("resolution") or "unresolved")[:100],
                "preventable_by": pm.get("preventable_by"),
            })

    return refs[:5]  # cap to avoid overloading briefing


# ─────────────────────────────────────────────────────────────────────────────
# Briefing builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_briefing(
    report:        dict[str, Any],
    trends:        dict[str, Any],
    consistency:   list[dict[str, Any]],
    postmortem_refs: list[dict[str, Any]],
    history_runs:  int,
) -> str:
    parts: list[str] = []

    # ── 0. Context ────────────────────────────────────────────────────────────
    run_at     = report.get("report_at", "unknown")
    period     = report.get("period_days", 7)
    flags      = report.get("flags", [])
    flag_count = report.get("flag_count", len(flags))

    infra = report.get("infra_health", {})
    ml    = report.get("ml_health",   {})
    cost  = report.get("cost",        {})

    avail_sources = []
    if infra.get("status") == "ok":   avail_sources.append("cloudwatch")
    if ml.get("status") == "ok":      avail_sources.append("mlflow")
    if cost.get("status") == "ok":    avail_sources.append("cost_explorer")

    parts.append(textwrap.dedent(f"""\
        ## 0. Assessment context

        Report generated: {run_at}
        Period: {period} days
        Available data sources: {', '.join(avail_sources) or 'none'}
        Active flags: {flag_count}
        History entries used for trends: {history_runs}
    """).strip())

    # ── 1. Infra health ───────────────────────────────────────────────────────
    infra_lines = ["## 1. Infrastructure health\n"]
    if infra.get("status") == "no_data":
        infra_lines.append("_No CloudWatch data available._")
    else:
        infra_lines += [
            f"Status: {infra.get('status')}",
            f"Node count: {infra.get('node_count')}",
            f"Node CPU avg: {infra.get('node_cpu_avg_pct')}%",
            f"Node memory avg: {infra.get('node_memory_avg_pct')}%",
            f"OOM events ({period}d): {infra.get('oom_events', 0)}",
            f"Total pod restarts ({period}d): {infra.get('total_pod_restarts', 0)}",
        ]
        by_ns = infra.get("pod_restarts_by_ns", {})
        if by_ns:
            infra_lines.append("Pod restarts by namespace:")
            for ns, count in sorted(by_ns.items(), key=lambda x: -x[1])[:8]:
                infra_lines.append(f"  {ns}: {count}")
    parts.append("\n".join(infra_lines))

    # ── 2. ML health ──────────────────────────────────────────────────────────
    ml_lines = ["## 2. ML pipeline health\n"]
    if ml.get("status") == "no_data":
        ml_lines.append("_No MLflow data available._")
    elif ml.get("status") not in ("ok",):
        ml_lines.append(f"Status: {ml.get('status')} — {ml.get('reason', '')} {ml.get('hint', '')}")
    else:
        ml_lines += [
            f"Experiments: {ml.get('experiment_count')}",
            f"Registered models: {ml.get('registered_models')} (int: count)",
            f"Production models: {ml.get('production_models')}",
            f"Last training: {ml.get('last_training_at') or 'never'}",
        ]
        dq_flags = ml.get("data_quality_flags", [])
        if dq_flags:
            ml_lines.append("Data quality flags:")
            for f in dq_flags[:5]:
                ml_lines.append(f"  [{f.get('flag')}] {f.get('experiment')}: {f.get('hint', '')}")
        for exp in ml.get("experiments", [])[:5]:
            best_f1    = exp.get("best_f1") or exp.get("best_metrics", {}).get("f1")
            registered = exp.get("registered")
            ml_lines.append(
                f"  Exp '{exp.get('name')}': runs={exp.get('run_count', '?')} "
                f"best_f1={best_f1} registered={registered}"
            )
    parts.append("\n".join(ml_lines))

    # ── 3. Cost ───────────────────────────────────────────────────────────────
    cost_lines = ["## 3. Cost\n"]
    if cost.get("status") == "no_data":
        cost_lines.append("_No Cost Explorer data available._")
    else:
        cost_lines.append(f"Total ({period}d): ${cost.get('total_usd')} USD")
        wow = cost.get("week_over_week", {})
        if wow:
            cost_lines.append(
                f"Week-over-week: {wow.get('delta_pct', '?')}% "
                f"(prev: ${wow.get('prev_total_usd', '?')})"
            )
        anomaly = cost.get("cost_anomaly")
        if anomaly:
            cost_lines.append(f"Anomaly detected: {anomaly.get('hint', str(anomaly))}")
        by_svc = cost.get("by_service", {})
        if by_svc:
            cost_lines.append("Top services by cost:")
            for svc, amt in sorted(by_svc.items(), key=lambda x: -(x[1] or 0))[:8]:
                cost_lines.append(f"  {svc}: ${amt:.4f}")
    parts.append("\n".join(cost_lines))

    # ── 4. Active flags ───────────────────────────────────────────────────────
    if flags:
        flag_lines = [f"## 4. Active flags ({len(flags)})\n"]
        for f in flags:
            flag_lines.append(
                f"[{f.get('level')}] {f.get('code')} ({f.get('source')}): "
                f"{f.get('detail')} → {f.get('hint', '')}"
            )
        parts.append("\n".join(flag_lines))
    else:
        parts.append("## 4. Active flags\n\n_No active flags._")

    # ── 5. Trends ─────────────────────────────────────────────────────────────
    if trends.get("note"):
        parts.append(f"## 5. Trends\n\n{trends['note']}")
    else:
        trend_lines = [f"## 5. Trends (last {history_runs} runs)\n"]
        for key, t in trends.items():
            if not isinstance(t, dict):
                continue
            direction = t.get("direction", "?")
            magnitude = t.get("magnitude", 0)
            label     = t.get("label", key)
            arrow     = {"increasing": "↑", "decreasing": "↓", "stable": "→"}.get(direction, "?")
            trend_lines.append(
                f"  {label}: {arrow} {direction} {magnitude:+.1f}%  "
                f"({t.get('first')} → {t.get('last')})"
            )
        parts.append("\n".join(trend_lines))

    # ── 6. Consistency findings ───────────────────────────────────────────────
    if consistency:
        latest = consistency[-1] if consistency else {}
        risk   = latest.get("risk_level", "?")
        n_high = sum(1 for e in consistency[-3:] for f in e.get("findings_summary", [])
                     if f.get("severity") == "HIGH")
        cons_lines = [
            f"## 6. Config consistency (last {min(len(consistency), 3)} runs)\n",
            f"Latest risk level: {risk}",
            f"HIGH findings in last 3 runs: {n_high}",
        ]
        # Recurring HIGH findings
        finding_ids: dict[str, int] = {}
        for entry in consistency[-3:]:
            for fid in entry.get("high_finding_ids", []):
                finding_ids[fid] = finding_ids.get(fid, 0) + 1
        recurring = [fid for fid, cnt in finding_ids.items() if cnt >= 2]
        if recurring:
            cons_lines.append(
                f"Recurring (seen ≥2 runs): {', '.join(recurring[:5])}"
            )
        parts.append("\n".join(cons_lines))
    else:
        parts.append(
            "## 6. Config consistency\n\n"
            "_No consistency log found. Run config_consistency_checker.py._"
        )

    # ── 7. Postmortem cross-references ────────────────────────────────────────
    if postmortem_refs:
        pm_lines = [
            f"## 7. Relevant past incidents ({len(postmortem_refs)} matches)\n",
            "Current flags match these known incident patterns:",
        ]
        for ref in postmortem_refs:
            pm_lines.append(
                f"  {ref['incident_id']} [{ref['taxonomy']}]: {ref['symptom']}"
                f" → {ref['resolution']}"
            )
        parts.append("\n".join(pm_lines))
    else:
        parts.append(
            "## 7. Relevant past incidents\n\n"
            "_No matching postmortem patterns found._"
        )

    briefing = "\n\n---\n\n".join(parts)
    if len(briefing) > _MAX_BRIEFING_CHARS:
        briefing = briefing[:_MAX_BRIEFING_CHARS] + "\n\n… (truncated)"
    return briefing


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a pragmatic DevOps/MLOps infrastructure judge.

Your job is to assess the current state of an IoT ML pipeline infrastructure
and produce a clear, actionable verdict. You are NOT reviewing code quality —
you are reviewing operational health.

You will receive:
  Section 0: Context (period, sources available, active flag count)
  Section 1: Infrastructure health (nodes, OOM, pod restarts)
  Section 2: ML pipeline health (experiments, models, training recency)
  Section 3: Cost (total, WoW delta, anomalies)
  Section 4: Active flags (pre-computed warnings from metrics_reporter)
  Section 5: Trends (changes across recent history runs)
  Section 6: Config consistency (recent checker results)
  Section 7: Relevant past incidents (postmortem cross-references)

────────────────────────────────────────────────────────────────
Verdict levels
────────────────────────────────────────────────────────────────
HEALTHY       All dimensions within normal range. No concerning trends.
DEGRADED      One or more dimensions need attention. No immediate outage risk.
CRITICAL      Active issue that requires immediate action.
INSUFFICIENT  Not enough data for a meaningful verdict (first run, missing
              collectors, or all sources show no_data).

────────────────────────────────────────────────────────────────
Dimension assessment
────────────────────────────────────────────────────────────────
For each of the 4 dimensions (infra, ml, cost, drift), output:
  status: HEALTHY | DEGRADED | CRITICAL | NO_DATA
  summary: 1-2 sentences
  key_metrics: 2-4 most important numbers
  concerns: list of specific issues (empty if HEALTHY)

────────────────────────────────────────────────────────────────
Recommendations
────────────────────────────────────────────────────────────────
Produce 1-5 concrete, actionable recommendations ordered by priority.
Each must name:
  - What to do (specific command or action, not vague advice)
  - Why (which metric or trend drives this)
  - Expected impact

────────────────────────────────────────────────────────────────
Postmortem cross-reference
────────────────────────────────────────────────────────────────
If Section 7 has matching past incidents, reference them in your
recommendations: "This matches PM-003 (OOMKilled airflow-worker) —
resolution was to increase memory limit from 512Mi to 768Mi."

────────────────────────────────────────────────────────────────
Output format
────────────────────────────────────────────────────────────────
Return raw JSON only (no markdown fences):
{
  "overall": "HEALTHY" | "DEGRADED" | "CRITICAL" | "INSUFFICIENT",
  "overall_rationale": "<1-2 sentences explaining the overall verdict>",
  "dimensions": {
    "infra": {
      "status":       "HEALTHY|DEGRADED|CRITICAL|NO_DATA",
      "summary":      "<1-2 sentences>",
      "key_metrics":  {"oom_events": 0, "restarts": 0, ...},
      "concerns":     ["<specific issue>", ...]
    },
    "ml": {
      "status":       "HEALTHY|DEGRADED|CRITICAL|NO_DATA",
      "summary":      "<1-2 sentences>",
      "key_metrics":  {"production_models": 0, "last_training_at": null, ...},
      "concerns":     []
    },
    "cost": {
      "status":       "HEALTHY|DEGRADED|CRITICAL|NO_DATA",
      "summary":      "<1-2 sentences>",
      "key_metrics":  {"total_usd": 0, "wow_delta_pct": 0},
      "concerns":     []
    },
    "drift": {
      "status":       "HEALTHY|DEGRADED|CRITICAL|NO_DATA",
      "summary":      "<1-2 sentences>",
      "key_metrics":  {"risk_level": "CLEAN", "recurring_findings": 0},
      "concerns":     []
    }
  },
  "recommendations": [
    {
      "priority":      1,
      "action":        "<specific command or action>",
      "reason":        "<which metric/trend drives this>",
      "impact":        "<expected result>",
      "postmortem_ref": "PM-003" or null
    }
  ],
  "trend_summary": "<1-2 sentences on the most important trends>",
  "flags_summary": {
    "total":    0,
    "warnings": 0,
    "by_code":  {"OOM_EVENTS": 0, ...}
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────────────

_total_cost: float = 0.0


def _call_judge_llm(briefing: str) -> dict[str, Any]:
    global _total_cost
    raw, cost = call_llm(
        ROLE,
        system      = _SYSTEM,
        user        = briefing,
        max_tokens  = 4096,
        caller_file = __file__,
        label       = f"[infra_judge] {get_model(ROLE)}",
    )
    _total_cost += (cost or 0.0)

    # Strip fences
    import re
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        print(f"  [judge][warn] Could not parse LLM response as JSON.")
        print(f"  First 300 chars: {cleaned[:300]}")
        return {
            "overall":           "INSUFFICIENT",
            "overall_rationale": "LLM response could not be parsed.",
            "dimensions":        {},
            "recommendations":   [],
            "trend_summary":     "",
            "flags_summary":     {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────

_VERDICT_EMOJI = {
    "HEALTHY":      "✅",
    "DEGRADED":     "⚠️",
    "CRITICAL":     "🔴",
    "INSUFFICIENT": "❓",
}
_DIM_EMOJI = {
    "HEALTHY":  "✅",
    "DEGRADED": "🟡",
    "CRITICAL": "🔴",
    "NO_DATA":  "—",
}


def _write_verdict_md(
    verdict:     dict[str, Any],
    run_at:      str,
    period_days: int,
    trends:      dict[str, Any],
    pm_refs:     list[dict[str, Any]],
) -> str:
    overall  = verdict.get("overall", "INSUFFICIENT")
    rationale = verdict.get("overall_rationale", "")
    dims     = verdict.get("dimensions", {})
    recs     = verdict.get("recommendations", [])
    t_sum    = verdict.get("trend_summary", "")
    fl_sum   = verdict.get("flags_summary", {})
    emoji    = _VERDICT_EMOJI.get(overall, "?")
    slug     = os.environ.get("PIPELINE_PROJECT", "<name>")

    L: list[str] = [
        "# Infra Verdict", "",
        f"**Overall:** {emoji} {overall}",
        f"**Assessed:** {run_at}",
        f"**Period:** {period_days} days",
        "",
        f"> {rationale}",
        "",
        "---", "",
        "## Dimensions", "",
    ]

    dim_order = [("infra", "Infrastructure"), ("ml", "ML Pipeline"),
                 ("cost", "Cost"), ("drift", "Drift Risk")]
    for key, label in dim_order:
        d     = dims.get(key, {})
        st    = d.get("status", "NO_DATA")
        de    = _DIM_EMOJI.get(st, "—")
        summ  = d.get("summary", "No data.")
        kms   = d.get("key_metrics", {})
        cons  = d.get("concerns", [])

        L.append(f"### {de} {label} — {st}")
        L.append(f"_{summ}_")
        if kms:
            L.append("")
            for mk, mv in kms.items():
                L.append(f"- **{mk}:** {mv}")
        if cons:
            L.append("")
            L.append("**Concerns:**")
            for c in cons:
                L.append(f"- {c}")
        L.append("")

    # Recommendations
    if recs:
        L += ["---", "", "## Recommendations", ""]
        for r in recs:
            pri   = r.get("priority", "?")
            act   = r.get("action", "")
            why   = r.get("reason", "")
            imp   = r.get("impact", "")
            pmref = r.get("postmortem_ref")

            L.append(f"### {pri}. {act}")
            L.append(f"**Why:** {why}")
            L.append(f"**Expected impact:** {imp}")
            if pmref:
                L.append(f"**See also:** {pmref}")
            L.append("")

    # Trend summary
    if t_sum:
        L += ["---", "", "## Trend Summary", "", t_sum, ""]

    # Postmortem refs
    if pm_refs:
        L += ["---", "", "## Related Past Incidents", ""]
        for ref in pm_refs:
            L.append(
                f"- **{ref['incident_id']}** [{ref['taxonomy']}]: "
                f"{ref['symptom']} → {ref['resolution']}"
            )
        L.append("")

    # Flag summary
    if fl_sum.get("total", 0) > 0:
        L += ["---", "", "## Active Flags", ""]
        L.append(f"Total: {fl_sum.get('total', 0)}  "
                 f"(Warnings: {fl_sum.get('warnings', 0)})")
        by_code = fl_sum.get("by_code", {})
        if by_code:
            for code, cnt in by_code.items():
                L.append(f"- {code}: {cnt}")
        L.append("")

    # Next step
    L += [
        "---", "",
        "## Next steps", "",
        "```",
        f"# Re-run metrics collection",
        f"python toolkits/devops_mlops/metrics_reporter.py --project {slug}",
        f"",
        f"# Re-run consistency check",
        f"python toolkits/devops_mlops/config_consistency_checker.py --project {slug}",
        f"",
        f"# Re-run judge after fixing issues",
        f"python toolkits/devops_mlops/infra_judge.py --project {slug}",
        "```", "",
    ]

    return "\n".join(L)


def _write_verdict_json(
    verdict:      dict[str, Any],
    run_at:       str,
    period_days:  int,
    trends:       dict[str, Any],
    pm_refs:      list[dict[str, Any]],
    sources_used: list[str],
) -> dict[str, Any]:
    return {
        "meta": {
            "run_at":        run_at,
            "period_days":   period_days,
            "sources_used":  sources_used,
            "judge_version": 1,
            "model":         get_model(ROLE),
        },
        "overall":           verdict.get("overall"),
        "overall_rationale": verdict.get("overall_rationale"),
        "dimensions":        verdict.get("dimensions", {}),
        "recommendations":   verdict.get("recommendations", []),
        "trend_summary":     verdict.get("trend_summary", ""),
        "flags_summary":     verdict.get("flags_summary", {}),
        "trends_raw":        trends,
        "postmortem_refs":   pm_refs,
    }


def _append_log(verdict_json: dict[str, Any]) -> None:
    log = _verdict_log()
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass

    entry = {
        "run_at":          verdict_json["meta"]["run_at"],
        "overall":         verdict_json["overall"],
        "period_days":     verdict_json["meta"]["period_days"],
        "sources_used":    verdict_json["meta"]["sources_used"],
        "dim_infra":       verdict_json["dimensions"].get("infra", {}).get("status"),
        "dim_ml":          verdict_json["dimensions"].get("ml", {}).get("status"),
        "dim_cost":        verdict_json["dimensions"].get("cost", {}).get("status"),
        "dim_drift":       verdict_json["dimensions"].get("drift", {}).get("status"),
        "rec_count":       len(verdict_json["recommendations"]),
        "flag_count":      verdict_json["flags_summary"].get("total", 0),
    }
    existing.append(entry)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"entries": existing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap mode — collect without verdict
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap(report: dict[str, Any], period_days: int) -> None:
    """
    First-run mode: store current snapshot in verdict_log.json without
    making a verdict. Needed to seed trends for the first real verdict run.
    """
    print("[infra_judge] Bootstrap mode — recording baseline snapshot, no verdict.")
    run_at = _now_iso()
    entry: dict[str, Any] = {
        "run_at":       run_at,
        "overall":      "BOOTSTRAP",
        "period_days":  period_days,
        "sources_used": [],
        "dim_infra":    report.get("infra_health", {}).get("status"),
        "dim_ml":       report.get("ml_health",   {}).get("status"),
        "dim_cost":     report.get("cost",         {}).get("status"),
        "dim_drift":    None,
        "rec_count":    0,
        "flag_count":   report.get("flag_count", 0),
    }
    log = _verdict_log()
    existing: list[dict[str, Any]] = []
    if log.exists():
        try:
            track_read(log)
            data     = json.loads(log.read_text(encoding="utf-8"))
            existing = data if isinstance(data, list) else data.get("entries", [])
        except Exception:
            pass
    existing.append(entry)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"entries": existing}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    track_write(log)
    print(f"  Snapshot stored: {log}")
    print(f"  Run again without --bootstrap to get a full verdict.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="infra_judge.py",
        description="Produce infrastructure verdict from metrics, consistency, and postmortem data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python infra_judge.py --project iot-mlops
              python infra_judge.py --project iot-mlops --history-runs 8
              python infra_judge.py --project iot-mlops --dry-run
              python infra_judge.py --project iot-mlops --show-last
              python infra_judge.py --project iot-mlops --bootstrap
        """),
    )
    p.add_argument("--project",       default=os.environ.get("PIPELINE_PROJECT"),
                   help="Project slug. Sets PIPELINE_PROJECT.")
    p.add_argument("--history-runs",  type=int, default=_DEFAULT_HISTORY_RUNS,
                   metavar="N",
                   help=f"Number of history entries to use for trends (default: {_DEFAULT_HISTORY_RUNS}).")
    p.add_argument("--dry-run",       action="store_true",
                   help="Build briefing and print it. No LLM call, no writes.")
    p.add_argument("--show-last",     action="store_true",
                   help="Print most recent verdict.md without re-running.")
    p.add_argument("--bootstrap",     action="store_true",
                   help="Record baseline snapshot without making a verdict (first run).")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.project:
        os.environ["PIPELINE_PROJECT"] = args.project
    if not os.environ.get("PIPELINE_PROJECT"):
        parser.error("Use --project <name> or export PIPELINE_PROJECT=<name>.")

    print("=" * 60)
    print("  INFRA JUDGE")
    print("=" * 60)
    print()

    exit_code = 0

    try:
        # --show-last
        if args.show_last:
            p = _verdict_md()
            if not p.exists():
                print("[infra_judge] No verdict found. Run without --show-last first.")
                sys.exit(1)
            track_read(p)
            print(p.read_text())
            sys.exit(0)

        # ── Load inputs ───────────────────────────────────────────────────────
        print("  Loading inputs …")
        report      = _load_metrics_report()
        history     = _load_metrics_history(args.history_runs)
        consistency = _load_consistency_log()
        postmortems = _load_postmortem_kb()

        period_days  = report.get("period_days", 7)
        run_at       = _now_iso()

        # Determine available sources
        sources_used: list[str] = []
        if report:                                sources_used.append("metrics_reporter")
        if history:                               sources_used.append("metrics_history")
        if consistency:                           sources_used.append("consistency_checker")
        if postmortems:                           sources_used.append("postmortem_archivist")

        print(f"  Sources: {', '.join(sources_used) or 'none'}")
        print(f"  History entries: {len(history)}")
        print(f"  Consistency runs: {len(consistency)}")
        print(f"  Postmortem entries: {len(postmortems)}")
        print()

        if not report:
            print("[infra_judge][warn] No metrics_report.json found.")
            print(f"  Expected: {_metrics_report()}")
            print("  Run metrics_reporter.py first.")
            if not history and not consistency:
                print("[infra_judge] No data sources available — cannot produce verdict.")
                sys.exit(2)

        # --bootstrap
        if args.bootstrap:
            run_bootstrap(report, period_days)
            return

        # ── Compute trends ────────────────────────────────────────────────────
        trends = _compute_trends(history)

        # ── Postmortem cross-reference ────────────────────────────────────────
        flags   = report.get("flags", [])
        pm_refs = _find_relevant_postmortems(flags, postmortems)
        if pm_refs:
            print(f"  Postmortem matches: {len(pm_refs)}")

        # ── Build briefing ────────────────────────────────────────────────────
        briefing = _build_briefing(
            report        = report,
            trends        = trends,
            consistency   = consistency,
            postmortem_refs = pm_refs,
            history_runs  = len(history),
        )
        print(f"  Briefing: {len(briefing):,} chars")

        if args.dry_run:
            print("\n[infra_judge] DRY RUN — briefing follows:\n")
            print(briefing)
            sys.exit(0)

        # ── LLM call ──────────────────────────────────────────────────────────
        print("\n  Calling LLM …")
        t0      = time.time()
        verdict = _call_judge_llm(briefing)
        elapsed = time.time() - t0
        overall = verdict.get("overall", "INSUFFICIENT")

        emoji = _VERDICT_EMOJI.get(overall, "?")
        print(f"  Elapsed: {elapsed:.1f}s  cost: ${_total_cost:.4f}")
        print(f"  Verdict: {emoji} {overall}")
        print()

        # ── Build output ──────────────────────────────────────────────────────
        verdict_md   = _write_verdict_md(verdict, run_at, period_days, trends, pm_refs)
        verdict_json = _write_verdict_json(
            verdict, run_at, period_days, trends, pm_refs, sources_used
        )

        # ── Write artifacts ───────────────────────────────────────────────────
        _judge_dir().mkdir(parents=True, exist_ok=True)

        # verdict.md
        md_with_header = apply_md_header(
            content = verdict_md,
            path    = _verdict_md(),
            owner   = "infra_judge.py",
        )
        _verdict_md().write_text(md_with_header, encoding="utf-8")
        track_write(_verdict_md())

        # verdict.json
        _verdict_json().write_text(
            json.dumps(verdict_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        track_write(_verdict_json())

        # log
        _append_log(verdict_json)

        print(f"  Written:  {_verdict_md()}")
        print(f"  Written:  {_verdict_json()}")
        print(f"  Appended: {_verdict_log()}")

        # Set exit code based on verdict
        exit_code = {
            "HEALTHY":      0,
            "DEGRADED":     1,
            "CRITICAL":     2,
            "INSUFFICIENT": 3,
        }.get(overall, 3)

    except KeyboardInterrupt:
        print("\n[infra_judge] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[infra_judge][error] {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[infra_judge]")
        print()
        print_cost_summary("[infra_judge]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
