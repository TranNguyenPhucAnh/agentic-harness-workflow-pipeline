"""
toolkits/devops_mlops/postmortem_archivist.py
=============================================
Structured postmortem storage for DevOps/MLOps incidents.

Two modes of operation:
  1. ingest    — parse free-form notes (markdown) into structured postmortem entries.
                 Idempotent: SHA256 per paragraph, skips already-ingested chunks.
  2. capture   — add a single new incident interactively (live capture after an incident).
  3. search    — query the knowledge base by taxonomy, tags, or keyword.
  4. list      — list all postmortem entries with summary.
  5. export    — dump all entries as markdown report.

Artifacts written:
  postmortem/postmortem_kb.json        (long-term, APPEND-ONLY — full knowledge base, never overwritten)
  postmortem/postmortem_log.md         (long-term append — human-readable history)
  postmortem/ingest_cache.json         (internal, OVERWRITE — SHA256 dedup checkpoint)

Usage:
  python postmortem_archivist.py --project iot-mlops --mode ingest --notes notes/IOT_AIR_QUALITY_NOTE_1.md
  python postmortem_archivist.py --project iot-mlops --mode ingest --notes notes/*.md
  python postmortem_archivist.py --project iot-mlops --mode capture
  python postmortem_archivist.py --project iot-mlops --mode search --query "IRSA annotation"
  python postmortem_archivist.py --project iot-mlops --mode search --taxonomy auth_iam
  python postmortem_archivist.py --project iot-mlops --mode list
  python postmortem_archivist.py --project iot-mlops --mode export
  python postmortem_archivist.py --project iot-mlops --mode stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLKIT_DIR = Path(__file__).parent
_REPO_ROOT   = _TOOLKIT_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from artifacts.paths import get_project_slug                              # noqa: E402
from modules.artifact_tracking import (                                   # noqa: E402
    track_read, track_write,
    print_summary as print_artifact_summary,
)
from modules.cost import print_summary as print_cost_summary              # noqa: E402
from modules.call_llm import call_llm                                     # noqa: E402
from artifacts.models import get_model                                    # noqa: E402

ROLE = "postmortem_archivist"

# ── Taxonomy ──────────────────────────────────────────────────────────────────
TAXONOMIES = {
    "auth_iam":           "IAM, IRSA, OIDC, ServiceAccount, AssumeRole, permissions",
    "resource_constraint":"OOMKilled, CPU throttle, memory limit, node capacity",
    "config_drift":       "values.yaml mismatch, annotation wrong, indent error, key path wrong",
    "image_registry":     "ECR auth, image pull, Image Updater, writeBackMethod, tag override",
    "data_pipeline":      "SQS, S3, Airflow DAG, XCom, SqsSensor, medallion architecture",
    "infra_connectivity": "connection refused, DNS, port, network, VPC, subnet, security group",
    "k8s_orchestration":  "ArgoCD sync, Helm chart, PVC, StorageClass, EBS CSI, finalizers",
    "ci_cd":              "Jenkins, JNLP, DinD, pipeline, gitSync, image build",
    "terraform":          "provider, module, state, destroy, dependency cycle, pagination",
    "observability":      "Grafana, Prometheus, Loki, Cloudflare Tunnel, node_exporter, Athena",
    "ml_model":           "training, inference, f1, precision, threshold, MLflow, synthetic anomaly",
    "other":              "does not fit any specific category",
}

# preventable_by values (what tool/practice would have caught this)
PREVENTABLE_BY = [
    "config_consistency_checker",    # value mismatch across IaC files
    "doc_absorber_cross_check",      # drift between handover docs and IaC/live
    "live_discovery_drift",          # orphaned/abandoned resource not in IaC
    "resource_monitor",              # OOM, CPU throttle, capacity issues
    "incident_clarificator",         # structured diagnosis would have narrowed faster
    "pre_deploy_validation",         # caught before apply/deploy
    "manual_review",                 # requires human architectural judgement
    "documentation",                 # missing or stale runbook/doc
    None,                            # not preventable by a tool
]

_MIN_CHUNK_WORDS = 20    # chunks shorter than this are skipped during ingest
_MAX_CHUNK_CHARS = 3000  # max chars sent to LLM per chunk
_LLM_BATCH_SIZE  = 5    # number of chunks per LLM call during ingest


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


def _postmortem_dir() -> Path:
    return _devops_artifact_root() / "postmortem"


def _kb_path() -> Path:
    return _postmortem_dir() / "postmortem_kb.json"


def _log_path() -> Path:
    return _postmortem_dir() / "postmortem_log.md"


def _ingest_cache_path() -> Path:
    return _postmortem_dir() / "ingest_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge base I/O
# ─────────────────────────────────────────────────────────────────────────────

def _load_kb() -> dict[str, Any]:
    p = _kb_path()
    if not p.exists():
        return {"entries": [], "meta": {"version": 1, "total": 0}}
    try:
        track_read(p)
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": [], "meta": {"version": 1, "total": 0}}


def _save_kb(kb: dict[str, Any]) -> None:
    """
    APPEND-ONLY write. Loads existing KB first, merges new entries by
    incident_id, then writes. Never discards existing entries.
    This is the primary store — data loss here is unrecoverable.
    """
    p = _kb_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Load existing to merge — never replace
    existing = _load_kb()
    existing_ids = {e.get("incident_id") for e in existing.get("entries", [])}

    # Append only entries that don't already exist
    new_only = [
        e for e in kb.get("entries", [])
        if e.get("incident_id") not in existing_ids
    ]
    merged = existing.get("entries", []) + new_only

    out = {
        "entries": merged,
        "meta": {
            "version":    1,
            "total":      len(merged),
            "updated_at": _now_iso(),
        },
    }
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    track_write(p)


def _load_ingest_cache() -> set[str]:
    p = _ingest_cache_path()
    if not p.exists():
        return set()
    try:
        track_read(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data.get("ingested_hashes", []))
    except Exception:
        return set()


def _save_ingest_cache(hashes: set[str]) -> None:
    p = _ingest_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"ingested_hashes": sorted(hashes)}, indent=2),
        encoding="utf-8",
    )
    track_write(p)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_display() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ─────────────────────────────────────────────────────────────────────────────
# Postmortem entry schema
# ─────────────────────────────────────────────────────────────────────────────

def _next_incident_id() -> str:
    """Sequential, stable ID. Reads current KB to find highest existing number."""
    entries = _load_kb().get("entries", [])
    nums = []
    for e in entries:
        m = re.search(r"PM-(\d+)", e.get("incident_id", ""))
        if m:
            nums.append(int(m.group(1)))
    return f"PM-{(max(nums, default=0) + 1):03d}"


def _make_entry(
    source:          str,
    taxonomy:        str,
    symptom:         str,
    causal_chain:    list[str],
    root_cause:      str,
    resolution:      str,
    failed_attempts: list[str],
    preventable_by:  str | None,
    files_affected:  list[str],
    tags:            list[str],
    raw_excerpt:     str = "",
    incident_id:     str = "",
) -> dict[str, Any]:
    ts = _now_iso()
    if not incident_id:
        incident_id = _next_incident_id()
    return {
        "incident_id":     incident_id,
        "created_at":      ts,
        "source":          source,   # "notes_ingestion" | "live_capture"
        "taxonomy":        taxonomy,
        "symptom":         symptom,
        "causal_chain":    causal_chain,
        "root_cause":      root_cause,
        "resolution":      resolution,
        "failed_attempts": failed_attempts,
        "preventable_by":  preventable_by,
        "files_affected":  files_affected,
        "tags":            tags,
        "raw_excerpt":     raw_excerpt[:500] if raw_excerpt else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ingest: notes → structured entries
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INGEST = f"""\
You are a DevOps/MLOps incident analyst. You will receive one or more raw
text chunks from an engineer's personal notes about building an IoT MLOps
pipeline on AWS (EKS, Airflow, MLflow, Terraform, ArgoCD, Jenkins, Grafana).

The notes are written informally in Vietnamese and English, mixing explanations,
commands, and incident accounts. Your job is to extract ONLY the incidents —
"Stuck:" blocks, failed attempts, and resolution patterns — and convert them
into structured postmortem entries.

TAXONOMY options (pick the single best fit):
{chr(10).join(f'  {k}: {v}' for k, v in TAXONOMIES.items())}

PREVENTABLE_BY options:
  config_consistency_checker  — value mismatch across IaC config files
  doc_absorber_cross_check    — drift between handover docs and IaC/live state
  live_discovery_drift        — orphaned/abandoned resource not tracked in IaC
  resource_monitor            — OOM, CPU throttle, capacity issues
  incident_clarificator       — structured diagnosis would have narrowed root cause faster
  pre_deploy_validation       — would have been caught before apply/deploy
  manual_review               — requires human architectural judgement
  documentation               — missing or stale runbook/operational doc
  null                        — not preventable by any tool

Return a JSON object with this exact schema (no markdown fences):
{{
  "entries": [
    {{
      "taxonomy":        "<one of the taxonomy keys above>",
      "symptom":         "<what the engineer observed — error message, behavior>",
      "causal_chain":    ["<step 1>", "<step 2>", "..."],
      "root_cause":      "<single sentence, the actual underlying cause>",
      "resolution":      "<what finally fixed it>",
      "failed_attempts": ["<attempt 1>", "<attempt 2>"],
      "preventable_by":  "<one of PREVENTABLE_BY options or null>",
      "files_affected":  ["<e.g. terraform/modules/iam/main.tf>", "..."],
      "tags":            ["<technology>", "<concept>", "..."],
      "raw_excerpt":     "<verbatim 1-2 sentences from source that best describe the incident>"
    }}
  ]
}}

RULES:
- Output ONLY the JSON object. No prose, no explanation.
- Skip chunks that are purely educational/explanatory (no incident, no "Stuck:").
- Skip TODO items that have not been resolved yet.
- Keep symptom, root_cause, resolution in English for consistency.
- If a chunk describes multiple distinct incidents, emit multiple entries.
- failed_attempts may be empty list [] if the engineer solved it on first try.
- tags: include technology names (eks, irsa, argocd, helm, mlflow, airflow...)
  and concept names (connection-refused, oom, image-pull, config-drift...).
- If no incidents found in the chunk, return: {{"entries": []}}
"""


def _chunk_notes(text: str) -> list[str]:
    """
    Split notes into incident-focused chunks.

    Strategy:
    1. Split on "Stuck:" markers — these are explicit incident blocks.
    2. For sections without "Stuck:", split on paragraph boundaries (double newline).
    3. Filter out chunks that are too short to be incidents.
    """
    chunks: list[str] = []

    # Split on "Stuck:" blocks — preserve the marker
    stuck_pattern = re.compile(r"(?=Stuck\s*\d*\s*:)", re.IGNORECASE)
    parts = stuck_pattern.split(text)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split long parts on paragraph breaks
        if len(part) > _MAX_CHUNK_CHARS:
            paras = re.split(r"\n\n+", part)
            current = ""
            for para in paras:
                if len(current) + len(para) < _MAX_CHUNK_CHARS:
                    current = (current + "\n\n" + para).strip()
                else:
                    if current and len(current.split()) >= _MIN_CHUNK_WORDS:
                        chunks.append(current)
                    current = para
            if current and len(current.split()) >= _MIN_CHUNK_WORDS:
                chunks.append(current)
        else:
            if len(part.split()) >= _MIN_CHUNK_WORDS:
                chunks.append(part)

    return chunks


def _chunk_hash(chunk: str) -> str:
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]


def _call_ingest_llm(chunks: list[str]) -> list[dict[str, Any]]:
    """Send a batch of chunks to LLM, return parsed entries."""
    combined = "\n\n---CHUNK---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)
    )
    user_msg = (
        f"Extract incidents from the following {len(chunks)} chunk(s). "
        f"Each chunk is separated by ---CHUNK---.\n\n{combined}"
    )

    try:
        raw, _ = call_llm(
            ROLE,
            _SYSTEM_INGEST,
            user_msg,
            max_tokens=4096,
            caller_file=__file__,
            label=f"[postmortem] {get_model(ROLE)}",
        )
    except Exception as exc:
        print(f"  [postmortem][warn] LLM call failed: {exc}")
        return []

    # Parse JSON — strip markdown fences if present
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```\s*$", "", raw)

    try:
        data = json.loads(raw)
        return data.get("entries", [])
    except json.JSONDecodeError as exc:
        # Try to find JSON object in output
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return data.get("entries", [])
            except Exception:
                pass
        print(f"  [postmortem][warn] JSON parse failed: {exc}")
        return []


def mode_ingest(args: argparse.Namespace) -> None:
    """Parse free-form notes files into structured postmortem entries."""
    note_paths = []
    if args.notes:
        for pattern in args.notes:
            matched = list(Path(".").glob(pattern))
            if not matched:
                p = Path(pattern)
                if p.exists():
                    matched = [p]
            note_paths.extend(matched)

    if not note_paths:
        print("[postmortem] No notes files found. Use --notes <path> or <glob>.")
        return

    print(f"[postmortem] Ingesting {len(note_paths)} file(s):")
    for p in note_paths:
        print(f"  {p}")
    print()

    kb             = _load_kb()
    ingest_cache   = _load_ingest_cache()
    new_entries    = 0
    skipped_chunks = 0
    total_chunks   = 0

    for note_path in note_paths:
        note_path = Path(note_path)
        if not note_path.exists():
            print(f"  [skip] {note_path} not found")
            continue
        track_read(note_path)
        text   = note_path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_notes(text)
        total_chunks += len(chunks)
        print(f"  {note_path.name}: {len(chunks)} chunks")

        # Filter already-ingested chunks
        new_chunks  = []
        new_hashes  = []
        for chunk in chunks:
            h = _chunk_hash(chunk)
            if h in ingest_cache and not args.force:
                skipped_chunks += 1
            else:
                new_chunks.append(chunk)
                new_hashes.append(h)

        if not new_chunks:
            print(f"    All chunks already ingested — skipping.")
            continue

        print(f"    {len(new_chunks)} new chunk(s) to process…")

        # Batch LLM calls
        for i in range(0, len(new_chunks), _LLM_BATCH_SIZE):
            batch   = new_chunks[i:i + _LLM_BATCH_SIZE]
            print(f"    Batch {i // _LLM_BATCH_SIZE + 1}: {len(batch)} chunk(s)…", end=" ", flush=True)
            entries = _call_ingest_llm(batch)
            print(f"{len(entries)} incident(s) found")

            for raw_entry in entries:
                if not raw_entry.get("root_cause"):
                    continue
                entry = _make_entry(
                    source          = "notes_ingestion",
                    taxonomy        = raw_entry.get("taxonomy", "other"),
                    symptom         = raw_entry.get("symptom", ""),
                    causal_chain    = raw_entry.get("causal_chain", []),
                    root_cause      = raw_entry.get("root_cause", ""),
                    resolution      = raw_entry.get("resolution", ""),
                    failed_attempts = raw_entry.get("failed_attempts", []),
                    preventable_by  = raw_entry.get("preventable_by"),
                    files_affected  = raw_entry.get("files_affected", []),
                    tags            = raw_entry.get("tags", []),
                    raw_excerpt     = raw_entry.get("raw_excerpt", ""),
                )
                kb["entries"].append(entry)
                new_entries += 1

            for h in new_hashes[i:i + _LLM_BATCH_SIZE]:
                ingest_cache.add(h)

            # Checkpoint after every batch — survive crashes mid-ingest
            _save_ingest_cache(ingest_cache)

    print()
    print(f"[postmortem] Ingestion complete:")
    print(f"  Total chunks:   {total_chunks}")
    print(f"  Skipped:        {skipped_chunks}")
    print(f"  New entries:    {new_entries}")

    _save_kb(kb)
    _append_log_ingest(new_entries, note_paths)

    print(f"  KB:  {_kb_path()} ({len(_load_kb()['entries'])} total entries)")
    print(f"  Log: {_log_path()}")


# ─────────────────────────────────────────────────────────────────────────────
# Live capture: interactive incident entry
# ─────────────────────────────────────────────────────────────────────────────

def mode_capture(args: argparse.Namespace) -> None:
    """Interactively capture a new incident after it happens."""
    print("[postmortem] Live incident capture")
    print("=" * 60)
    print()

    def _ask(prompt: str, required: bool = True) -> str:
        while True:
            val = input(f"  {prompt}: ").strip()
            if val or not required:
                return val
            print("    (required — please enter a value)")

    def _ask_list(prompt: str) -> list[str]:
        print(f"  {prompt} (one per line, empty line to finish):")
        items = []
        while True:
            item = input("    > ").strip()
            if not item:
                break
            items.append(item)
        return items

    # Taxonomy selection
    print("  Taxonomy:")
    for i, (k, v) in enumerate(TAXONOMIES.items(), 1):
        print(f"    {i:2}. {k:<25} — {v[:50]}")
    while True:
        try:
            choice = int(input("  Select [1-{}]: ".format(len(TAXONOMIES))))
            if 1 <= choice <= len(TAXONOMIES):
                taxonomy = list(TAXONOMIES.keys())[choice - 1]
                break
        except ValueError:
            pass
        print("    Invalid — enter a number")

    print()
    symptom         = _ask("Symptom (what you observed / error message)")
    causal_chain    = _ask_list("Causal chain (ordered steps leading to the issue)")
    root_cause      = _ask("Root cause (single sentence)")
    resolution      = _ask("Resolution (what fixed it)")
    failed_attempts = _ask_list("Failed attempts (what you tried that didn't work)")
    files_affected  = _ask_list("Files affected (e.g. terraform/modules/iam/main.tf)")
    tags_raw        = _ask("Tags (comma-separated: eks, irsa, argocd, ...)", required=False)
    tags            = [t.strip() for t in tags_raw.split(",") if t.strip()]

    # preventable_by
    print()
    print("  Preventable by:")
    valid_prev = [p for p in PREVENTABLE_BY if p is not None] + ["none"]
    for i, p in enumerate(valid_prev, 1):
        print(f"    {i}. {p}")
    try:
        p_choice = int(input(f"  Select [1-{len(valid_prev)}]: "))
        preventable_by = valid_prev[p_choice - 1] if 1 <= p_choice <= len(valid_prev) else None
        if preventable_by == "none":
            preventable_by = None
    except (ValueError, IndexError):
        preventable_by = None

    entry = _make_entry(
        source          = "live_capture",
        taxonomy        = taxonomy,
        symptom         = symptom,
        causal_chain    = causal_chain,
        root_cause      = root_cause,
        resolution      = resolution,
        failed_attempts = failed_attempts,
        preventable_by  = preventable_by,
        files_affected  = files_affected,
        tags            = tags,
    )

    print()
    print(f"  Entry: {entry['incident_id']}")
    confirm = input("  Save? [Y/n]: ").strip().lower()
    if confirm in ("", "y", "yes"):
        kb = _load_kb()
        kb["entries"].append(entry)
        _save_kb(kb)
        _append_log_capture(entry)
        print(f"[postmortem] Saved → {_kb_path()}")
    else:
        print("[postmortem] Discarded.")


# ─────────────────────────────────────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────────────────────────────────────

def mode_search(args: argparse.Namespace) -> None:
    """Search postmortem KB by keyword, taxonomy, or tag."""
    kb      = _load_kb()
    entries = kb.get("entries", [])

    results = entries

    if args.taxonomy:
        results = [e for e in results if e.get("taxonomy") == args.taxonomy]

    if getattr(args, "tag", None):
        results = [e for e in results if args.tag in e.get("tags", [])]

    if args.query:
        q = args.query.lower()
        results = [
            e for e in results
            if any(
                q in str(v).lower()
                for v in [
                    e.get("symptom", ""),
                    e.get("root_cause", ""),
                    e.get("resolution", ""),
                    " ".join(e.get("tags", [])),
                    " ".join(e.get("causal_chain", [])),
                    e.get("raw_excerpt", ""),
                ]
            )
        ]

    if not results:
        print("[postmortem] No matching entries found.")
        return

    print(f"[postmortem] {len(results)} result(s):\n")
    for e in results:
        _print_entry_summary(e)


def _print_entry_summary(e: dict[str, Any]) -> None:
    print(f"  ── {e['incident_id']} [{e['taxonomy']}] ──")
    print(f"     Symptom:    {e['symptom'][:80]}")
    print(f"     Root cause: {e['root_cause'][:80]}")
    print(f"     Resolution: {e['resolution'][:80]}")
    if e.get("failed_attempts"):
        print(f"     Failed ({len(e['failed_attempts'])}): {e['failed_attempts'][0][:60]}…")
    if e.get("tags"):
        print(f"     Tags:       {', '.join(e['tags'][:8])}")
    if e.get("preventable_by"):
        print(f"     Preventable by: {e['preventable_by']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# List
# ─────────────────────────────────────────────────────────────────────────────

def mode_list(args: argparse.Namespace) -> None:
    """List all entries with one-line summary."""
    kb      = _load_kb()
    entries = kb.get("entries", [])

    if not entries:
        print("[postmortem] Knowledge base is empty. Run --mode ingest first.")
        return

    print(f"[postmortem] {len(entries)} entries:\n")
    for e in entries:
        ts       = e.get("created_at", "")[:10]
        tid      = e.get("incident_id", "?")
        taxonomy = e.get("taxonomy", "?")
        rc       = e.get("root_cause", "?")[:60]
        print(f"  {ts}  {tid:<35} [{taxonomy:<20}]  {rc}")


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def mode_stats(args: argparse.Namespace) -> None:
    """Print knowledge base statistics."""
    kb      = _load_kb()
    entries = kb.get("entries", [])

    if not entries:
        print("[postmortem] Knowledge base is empty.")
        return

    from collections import Counter
    tax_counts  = Counter(e.get("taxonomy", "other") for e in entries)
    prev_counts = Counter(e.get("preventable_by") or "none" for e in entries)
    src_counts  = Counter(e.get("source", "?") for e in entries)

    print(f"[postmortem] Knowledge Base Statistics")
    print(f"{'=' * 50}")
    print(f"  Total entries: {len(entries)}")
    print()
    print(f"  By taxonomy:")
    for k, n in tax_counts.most_common():
        bar = "█" * n
        print(f"    {k:<25} {n:3}  {bar}")
    print()
    print(f"  By source:")
    for k, n in src_counts.most_common():
        print(f"    {k:<25} {n:3}")
    print()
    print(f"  Preventable by:")
    for k, n in prev_counts.most_common():
        print(f"    {k:<30} {n:3}")
    print()

    # Most common tags
    all_tags = []
    for e in entries:
        all_tags.extend(e.get("tags", []))
    tag_counts = Counter(all_tags)
    print(f"  Top tags:")
    for tag, n in tag_counts.most_common(15):
        print(f"    {tag:<25} {n:3}")


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def mode_export(args: argparse.Namespace) -> None:
    """Export all entries as structured markdown report."""
    kb      = _load_kb()
    entries = kb.get("entries", [])

    if not entries:
        print("[postmortem] Nothing to export.")
        return

    from collections import defaultdict
    by_taxonomy: dict[str, list] = defaultdict(list)
    for e in entries:
        by_taxonomy[e.get("taxonomy", "other")].append(e)

    lines: list[str] = [
        f"# Postmortem Knowledge Base",
        f"",
        f"Generated: {_now_display()}  |  "
        f"Project: {os.environ.get('PIPELINE_PROJECT', '?')}  |  "
        f"Entries: {len(entries)}",
        f"",
        f"---",
        f"",
    ]

    for taxonomy, tax_entries in sorted(by_taxonomy.items()):
        desc = TAXONOMIES.get(taxonomy, "")
        lines += [
            f"## {taxonomy}",
            f"_{desc}_",
            f"",
        ]
        for e in tax_entries:
            lines += [
                f"### {e['incident_id']}",
                f"",
                f"**Symptom:** {e['symptom']}",
                f"",
                f"**Root cause:** {e['root_cause']}",
                f"",
            ]
            if e.get("causal_chain"):
                lines.append("**Causal chain:**")
                for step in e["causal_chain"]:
                    lines.append(f"1. {step}")
                lines.append("")

            lines.append(f"**Resolution:** {e['resolution']}")
            lines.append("")

            if e.get("failed_attempts"):
                lines.append("**Failed attempts:**")
                for attempt in e["failed_attempts"]:
                    lines.append(f"- {attempt}")
                lines.append("")

            if e.get("files_affected"):
                lines.append(f"**Files:** {', '.join(f'`{f}`' for f in e['files_affected'])}")
                lines.append("")

            if e.get("preventable_by"):
                lines.append(f"**Preventable by:** `{e['preventable_by']}`")
                lines.append("")

            if e.get("tags"):
                lines.append(f"**Tags:** {', '.join(e['tags'])}")
                lines.append("")

            lines.append("---")
            lines.append("")

    report = "\n".join(lines)

    out_path = _postmortem_dir() / "postmortem_export.md"
    out_path.write_text(report, encoding="utf-8")
    track_write(out_path)
    print(f"[postmortem] Exported {len(entries)} entries → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Log writers
# ─────────────────────────────────────────────────────────────────────────────

def _append_log_ingest(new_entries: int, note_paths: list[Path]) -> None:
    log = _log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    block = (
        f"## Ingest — {_now_display()}\n\n"
        f"- **Files**: {', '.join(p.name for p in note_paths)}\n"
        f"- **New entries**: {new_entries}\n\n"
        f"---\n\n"
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(block)
    track_write(log)


def _append_log_capture(entry: dict[str, Any]) -> None:
    log = _log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    block = (
        f"## {entry['incident_id']} — {_now_display()}\n\n"
        f"- **Taxonomy**: {entry['taxonomy']}\n"
        f"- **Symptom**: {entry['symptom']}\n"
        f"- **Root cause**: {entry['root_cause']}\n"
        f"- **Resolution**: {entry['resolution']}\n"
        f"- **Preventable by**: {entry.get('preventable_by') or 'none'}\n\n"
        f"---\n\n"
    )
    with log.open("a", encoding="utf-8") as f:
        f.write(block)
    track_write(log)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — for incident_clarificator.py
# ─────────────────────────────────────────────────────────────────────────────

def get_relevant_context(
    taxonomy:  str | None       = None,
    keywords:  list[str] | None = None,
    max_items: int               = 5,
) -> str:
    """
    Return a compact postmortem context string for injection into
    incident_clarificator.py LLM prompts.

    Usage:
        from toolkits.devops_mlops.postmortem_archivist import get_relevant_context
        ctx = get_relevant_context(taxonomy="auth_iam", keywords=["IRSA", "AssumeRole"])
        # inject ctx into clarificator system prompt
    """
    entries = _load_kb().get("entries", [])
    if not entries:
        return ""

    matched: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _entry_matches(e: dict[str, Any], kw: str | None) -> bool:
        if taxonomy and e.get("taxonomy") != taxonomy:
            return False
        if kw:
            searchable = " ".join([
                e.get("symptom", ""),
                e.get("root_cause", ""),
                e.get("resolution", "") or "",
                " ".join(e.get("tags", [])),
                " ".join(e.get("causal_chain", [])),
                " ".join(e.get("failed_attempts", [])),
                e.get("raw_excerpt", ""),
            ]).lower()
            if kw.lower() not in searchable:
                return False
        return True

    for kw in (keywords or [None]):
        for e in entries:
            iid = e.get("incident_id", "")
            if iid not in seen and _entry_matches(e, kw):
                seen.add(iid)
                matched.append(e)

    if not matched:
        return ""

    lines = ["--- HISTORICAL POSTMORTEMS (similar past incidents) ---"]
    for e in matched[:max_items]:
        iid    = e.get("incident_id", "?")
        sym    = e.get("symptom", "")
        rc     = e.get("root_cause", "")
        res    = e.get("resolution") or "unresolved"
        failed = e.get("failed_attempts", [])

        lines.append(f"\n{iid}: {sym}")
        lines.append(f"  Root cause: {rc}")
        lines.append(f"  Resolution: {res}")
        if failed:
            lines.append(f"  ✗ Did NOT work: {'; '.join(failed[:3])}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="postmortem_archivist.py",
        description="DevOps/MLOps postmortem knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",  default=os.environ.get("PIPELINE_PROJECT"))
    p.add_argument(
        "--mode",
        choices=["ingest", "capture", "search", "list", "export", "stats"],
        default="list",
    )
    p.add_argument("--notes",    nargs="+", metavar="FILE", help="Note files for --mode ingest")
    p.add_argument("--query",    default=None, help="Keyword search for --mode search")
    p.add_argument("--taxonomy", default=None,
                   choices=list(TAXONOMIES.keys()),
                   help="Filter by taxonomy for --mode search")
    p.add_argument("--tag",      default=None,
                   help="Filter by tag for --mode search (e.g. --tag irsa)")
    p.add_argument("--force",    action="store_true", help="Re-ingest already-cached chunks")
    p.add_argument("--verbose",  action="store_true")
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

    _postmortem_dir().mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  POSTMORTEM ARCHIVIST — {args.mode.upper()}")
    print("=" * 60)
    print()

    exit_code = 0
    try:
        dispatch = {
            "ingest":  mode_ingest,
            "capture": mode_capture,
            "search":  mode_search,
            "list":    mode_list,
            "export":  mode_export,
            "stats":   mode_stats,
        }
        dispatch[args.mode](args)

    except KeyboardInterrupt:
        print("\n[postmortem] Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"[postmortem][error] {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        print()
        print_artifact_summary("[postmortem]")
        print()
        print_cost_summary("[postmortem]")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
