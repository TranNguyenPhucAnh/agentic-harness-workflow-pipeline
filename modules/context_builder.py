"""
modules/context_builder.py
==========================
Pure-logic context builder — no LLM calls, deterministic, testable.

Builds a structured context dict for each pipeline step by reading:
  [upstream]  Aggregate short-term artifacts of ALL prior steps (cumulative)
  [history]   Own long-term log (*_log.json) — last N entries
  [knowledge] archivist/knowledge_log.md + archivist/spec_gaps.md

RULE: This module only reads. It never writes any artifact.
      Caller (each pipeline script) injects the returned dict into its prompt.

Usage
─────
    from modules.context_builder import build_context

    ctx = build_context("executor", artifact_root, scope="full")

    # Inject into prompt:
    system_prompt += ctx["knowledge"]
    user_msg      += ctx["upstream"] + ctx["history"]

Upstream accumulation rule
──────────────────────────
Each step consumes the short-term artifacts of ALL steps that precede it
in pipeline order. E.g. executor sees absorber + clarificator + enricher +
spectracker + scaffolder + planner outputs — not just the immediate predecessor.

Scope-awareness
───────────────
executor and debugger may operate in "full" or "mini" scope.
build_context() accepts an optional `scope` param that selects the correct
planner artifact (full_plan.json vs mini_plan.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ── Pipeline order ────────────────────────────────────────────────────────────
# Defines the canonical step sequence used for upstream accumulation.
# Each step consumes short-term artifacts of all steps BEFORE it in this list.

_PIPELINE_ORDER: list[str] = [
    "absorber",
    "clarificator",
    "enricher",
    "specwright",
    "spectracker",
    "scaffolder",
    "planner",
    "executor",
    "debugger",
    "reporter",
    "judge",
    "patcher",
    "archivist",
]


# ── Selective upstream override ───────────────────────────────────────────────
# Late-stage steps do not need cumulative upstream from ALL prior steps.
# Only the listed prior steps are included. Unlisted steps → full cumulative.
# This keeps token budget bounded for judge/patcher/reporter.

_SELECTIVE_UPSTREAM: dict[str, list[str]] = {
    "judge":    ["specwright", "scaffolder", "planner", "executor", "debugger"],
    "patcher":  ["specwright", "planner", "executor", "debugger", "judge"],
    "reporter": ["scaffolder", "planner", "executor", "debugger"],
    "archivist":["specwright", "executor", "debugger", "judge", "patcher"],
}

# Hard cap on total upstream chars (all projected blocks combined).
# Prevents runaway token usage for very large artifacts.
_UPSTREAM_MAX_CHARS: int = 50_000


# ── Short-term artifact map ───────────────────────────────────────────────────
# Maps each step → list of short-term artifact relative paths it PRODUCES.
# context_builder reads these when building upstream context for later steps.
#
# scope-sensitive entries use a callable that receives scope → path string.
# "spec" is a special case: path is resolved dynamically via get_spec_path().

def _short_term_artifacts(scope: str = "full") -> dict[str, list[str]]:
    """
    Returns map of step → list of short-term artifact relpaths.
    scope: "full" | "mini" — affects which planner artifact is listed.
    """
    return {
        "absorber": [
            "absorber/codebase_map.md",
        ],
        "clarificator": [
            "clarificator/session.json",
        ],
        "enricher": [
            "enricher/enriched_prompt.md",
        ],
        "specwright": [
            # spec path is slug-dependent — handled specially in _read_upstream
            "__spec__",
        ],
        "spectracker": [
            "spectracker/version_delta.json",
        ],
        "scaffolder": [
            "scaffolder/blueprint.json",
        ],
        "planner": [
            "planner/full_plan.json" if scope == "full" else "planner/mini_plan.json",
        ],
        "executor": [
            "executor/manifest.json",
        ],
        "debugger": [
            "debugger/test_summary.json",
        ],
        "reporter": [
            "reporter/execution_summary.md",
        ],
        "judge": [
            "judge/verdict_raw.json",
            "judge/verdict_summary.md",
        ],
        "patcher": [
            "patcher/fix_summary.md",
        ],
        "archivist": [
            # archivist's outputs are knowledge artifacts, not short-term
        ],
    }


# ── Long-term log map ─────────────────────────────────────────────────────────
# Maps each step → relative path of its own *_log.json (C2 audit artifact).
# Used by _own_history().

_OWN_LOG: dict[str, str] = {
    "absorber":     "absorber/codebase_log.json",
    "clarificator": "clarificator/decision_log.json",
    "enricher":     "enricher/prompt_log.json",
    "spectracker":  "spectracker/version_log.json",
    "scaffolder":   "scaffolder/skeleton_log.json",
    "planner":      "planner/plan_log.json",
    "executor":     "executor/manifest_log.json",
    "debugger":     "debugger/test_log.json",
    "reporter":     "reporter/execution_log.json",
    "judge":        "judge/verdict_log.json",
    "patcher":      "patcher/attempt_log.json",
    "archivist":    "archivist/curation_log.json",
    # specwright has no own log — spectracker is its long-term pair
}


# ── Projection config ─────────────────────────────────────────────────────────
# Controls how much of each artifact is included in upstream context.
# Keeps token count bounded without LLM summarization.

_PROJECTION: dict[str, dict[str, Any]] = {
    # absorber/codebase_map.md — can be large; first N lines captures summary
    "absorber/codebase_map.md": {
        "type": "md",
        "max_lines": 80,
    },
    # clarificator/session.json — extract key fields only
    "clarificator/session.json": {
        "type": "json_fields",
        "fields": ["requirement_synthesis", "decisions", "conflicts", "unresolved"],
    },
    # enricher/enriched_prompt.md — medium size; first N lines
    "enricher/enriched_prompt.md": {
        "type": "md",
        "max_lines": 60,
    },
    # spec — can be large; first N lines captures overview + key sections
    "__spec__": {
        "type": "md",
        "max_lines": 100,
    },
    # spectracker/version_delta.json — small, keep full
    "spectracker/version_delta.json": {
        "type": "json_full",
    },
    # scaffolder/blueprint.json — extract structure only, omit file lists
    "scaffolder/blueprint.json": {
        "type": "json_fields",
        "fields": ["spec_version", "modules", "summary"],
        # modules: keep module name + purpose, truncate files list
        "modules_max_files": 5,
    },
    # planner/full_plan.json — large; extract tasks summary only
    "planner/full_plan.json": {
        "type": "json_fields",
        "fields": ["implementation_order", "global_notes"],
        "tasks_summary": True,   # include per-task role + behavior_summary only
        "tasks_max": 20,
    },
    # planner/mini_plan.json — extract plan portion only
    "planner/mini_plan.json": {
        "type": "json_fields",
        "fields": ["plan"],
    },
    # executor/manifest.json — extract summary fields
    "executor/manifest.json": {
        "type": "json_fields",
        "fields": ["scope", "mode", "generated_at", "files", "failed_files"],
        "files_max": 10,
    },
    # debugger/test_summary.json — extract final status + last iteration
    "debugger/test_summary.json": {
        "type": "json_fields",
        "fields": ["final_status", "scope", "total_iterations", "escalated"],
    },
    # reporter/execution_summary.md — first N lines
    "reporter/execution_summary.md": {
        "type": "md",
        "max_lines": 30,
    },
    # judge/verdict_raw.json — extract verdict + blocking issues
    # NOTE: archivist wraps model output as {"response": "<json string>"}.
    # _project_json handles the unwrap before field extraction.
    "judge/verdict_raw.json": {
        "type": "json_fields",
        "fields": ["verdict", "blocking_issues", "spec_gaps_found"],
        "unwrap_response": True,
    },
    # judge/verdict_summary.md — first N lines
    "judge/verdict_summary.md": {
        "type": "md",
        "max_lines": 40,
    },
    # patcher/fix_summary.md — first N lines
    "patcher/fix_summary.md": {
        "type": "md",
        "max_lines": 40,
    },
}


# ── Internal readers ──────────────────────────────────────────────────────────

def _read_text_safe(path: Path) -> str | None:
    """Read text file; return None if missing or unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def _read_json_safe(path: Path) -> Any | None:
    """Read + parse JSON file; return None if missing or parse error."""
    text = _read_text_safe(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _project_md(text: str, max_lines: int) -> str:
    """Return first max_lines of a markdown file."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... [{len(lines) - max_lines} more lines truncated]"


def _project_json(data: Any, cfg: dict[str, Any]) -> Any:
    """
    Apply projection rules to a parsed JSON object.
    Returns a new dict/list suitable for serialization.
    """
    proj_type = cfg.get("type", "json_full")

    if proj_type == "json_full":
        return data

    if proj_type == "json_fields":
        if not isinstance(data, dict):
            return data

        # Unwrap nested JSON-in-string (e.g. verdict_raw {"response": "<json>"})
        if cfg.get("unwrap_response") and "response" in data:
            raw_response = data["response"]
            if isinstance(raw_response, str):
                try:
                    data = json.loads(raw_response)
                except json.JSONDecodeError:
                    pass  # keep original if parse fails

        fields = cfg.get("fields", [])
        result: dict[str, Any] = {}

        for field in fields:
            if field not in data:
                continue
            val = data[field]

            # Special handling for "files" in executor manifest
            if field == "files":
                max_f = cfg.get("files_max", 10)
                if isinstance(val, list) and len(val) > max_f:
                    result[field] = val[:max_f] + [f"... {len(val) - max_f} more"]
                else:
                    result[field] = val

            # Special handling for "modules" in blueprint
            elif field == "modules" and "modules_max_files" in cfg:
                max_files = cfg["modules_max_files"]
                trimmed = []
                for mod in (val if isinstance(val, list) else []):
                    files = mod.get("files", [])
                    entry = {k: v for k, v in mod.items() if k != "files"}
                    if len(files) > max_files:
                        entry["files"] = files[:max_files] + [
                            {"note": f"... {len(files) - max_files} more files"}
                        ]
                    else:
                        entry["files"] = files
                    trimmed.append(entry)
                result[field] = trimmed

            # Special handling for planner tasks summary
            elif field == "tasks" and cfg.get("tasks_summary"):
                max_t = cfg.get("tasks_max", 20)
                if isinstance(val, list):
                    summary = [
                        {
                            "file_path": t.get("file_path"),
                            "role": t.get("role"),
                            "behavior_summary": t.get("behavior_summary"),
                        }
                        for t in val[:max_t]
                    ]
                    if len(val) > max_t:
                        summary.append({"note": f"... {len(val) - max_t} more tasks"})
                    result[field] = summary
                else:
                    result[field] = val

            else:
                result[field] = val

        # For planner: also extract tasks summary alongside other fields
        if cfg.get("tasks_summary") and "tasks" in data and "tasks" not in fields:
            tasks = data.get("tasks", [])
            max_t = cfg.get("tasks_max", 20)
            summary = [
                {
                    "file_path": t.get("file_path"),
                    "role": t.get("role"),
                    "behavior_summary": t.get("behavior_summary"),
                }
                for t in tasks[:max_t]
            ]
            if len(tasks) > max_t:
                summary.append({"note": f"... {len(tasks) - max_t} more tasks"})
            result["tasks_summary"] = summary

        return result

    return data


def _format_artifact_block(label: str, content: str) -> str:
    """Wrap projected content in a labeled block for prompt injection."""
    return f"### {label}\n\n{content}\n\n"


# ── Core builders ─────────────────────────────────────────────────────────────

def _read_upstream(
    step: str,
    artifact_dir: Path,
    scope: str = "full",
    slug: str = "",
) -> str:
    """
    Build cumulative upstream context for `step`.
    Reads short-term artifacts of ALL steps that precede `step` in pipeline order.
    Returns formatted string ready for prompt injection.
    slug: used to resolve spec filename deterministically (avoids glob fragility).
    """
    if step not in _PIPELINE_ORDER:
        return ""

    step_idx = _PIPELINE_ORDER.index(step)
    all_prior = _PIPELINE_ORDER[:step_idx]

    if not all_prior:
        return ""

    # Apply selective filter for late-stage steps
    if step in _SELECTIVE_UPSTREAM:
        allowed = set(_SELECTIVE_UPSTREAM[step])
        prior_steps = [s for s in all_prior if s in allowed]
    else:
        prior_steps = all_prior

    if not prior_steps:
        return ""

    artifacts_map = _short_term_artifacts(scope)
    blocks: list[str] = []
    total_chars: int = 0

    for i, prior in enumerate(prior_steps):
        rel_paths = artifacts_map.get(prior, [])
        for rel in rel_paths:

            # Special case: spec path is slug-dependent
            if rel == "__spec__":
                # Prefer explicit slug; fall back to glob
                if slug:
                    path = artifact_dir / "spec" / f"specwright_spec_{slug}.md"
                    if not path.exists():
                        continue
                else:
                    spec_candidates = sorted(
                        (artifact_dir / "spec").glob("specwright_spec_*.md")
                    )
                    if not spec_candidates:
                        continue
                    path = spec_candidates[-1]  # sorted → take last (most recent slug)
                label = f"upstream.{prior}.spec"
                text = _read_text_safe(path)
                if text is None:
                    continue
                cfg = _PROJECTION.get("__spec__", {"type": "md", "max_lines": 100})
                projected = _project_md(text, cfg.get("max_lines", 100))
                block = _format_artifact_block(label, projected)
                if total_chars + len(block) > _UPSTREAM_MAX_CHARS:
                    blocks.append(
                        f"[upstream truncated at spec — total chars would exceed "
                        f"{_UPSTREAM_MAX_CHARS:,}]\n\n"
                    )
                    return "".join(blocks)
                blocks.append(block)
                total_chars += len(block)
                continue

            path = artifact_dir / rel
            cfg  = _PROJECTION.get(rel, {})
            proj_type = cfg.get("type", "json_full") if cfg else None
            label = f"upstream.{prior}.{Path(rel).name}"

            if rel.endswith(".md"):
                text = _read_text_safe(path)
                if text is None:
                    continue
                max_lines = cfg.get("max_lines", 60) if cfg else 60
                projected = _project_md(text, max_lines)
                block = _format_artifact_block(label, projected)
                if total_chars + len(block) > _UPSTREAM_MAX_CHARS:
                    remaining = len(prior_steps) - i
                    blocks.append(
                        f"[upstream truncated — {remaining} step(s) omitted "
                        f"(total chars would exceed {_UPSTREAM_MAX_CHARS:,})]\n\n"
                    )
                    return "".join(blocks)
                blocks.append(block)
                total_chars += len(block)

            elif rel.endswith(".json"):
                data = _read_json_safe(path)
                if data is None:
                    continue
                if cfg and proj_type != "json_full":
                    projected_data = _project_json(data, cfg)
                else:
                    projected_data = data
                projected = json.dumps(projected_data, indent=2, ensure_ascii=False)
                block = _format_artifact_block(label, f"```json\n{projected}\n```")
                if total_chars + len(block) > _UPSTREAM_MAX_CHARS:
                    remaining = len(prior_steps) - i
                    blocks.append(
                        f"[upstream truncated — {remaining} step(s) omitted "
                        f"(total chars would exceed {_UPSTREAM_MAX_CHARS:,})]\n\n"
                    )
                    return "".join(blocks)
                blocks.append(block)
                total_chars += len(block)

    return "".join(blocks)


_HISTORY_ENTRY_MAX_CHARS: int = 3_000


def _slim_entry(entry: Any, max_chars: int = _HISTORY_ENTRY_MAX_CHARS) -> Any:
    """
    Truncate a single log entry if its JSON representation is too large.
    Preserves the entry as-is if small enough; otherwise returns a trimmed
    version with a truncation note.
    """
    serialized = json.dumps(entry, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return entry
    # Return slimmed version: keep all scalar fields, truncate long string fields
    if not isinstance(entry, dict):
        return str(entry)[:max_chars] + "... [truncated]"
    slimmed: dict[str, Any] = {}
    for k, v in entry.items():
        if isinstance(v, str) and len(v) > 500:
            slimmed[k] = v[:500] + f"... [{len(v) - 500} chars truncated]"
        elif isinstance(v, list) and len(json.dumps(v, ensure_ascii=False)) > 1_000:
            slimmed[k] = v[:3] + [f"... {len(v) - 3} more items"]
        else:
            slimmed[k] = v
    return slimmed


def _own_history(
    step: str,
    artifact_dir: Path,
    max_entries: int = 3,
) -> str:
    """
    Read own *_log.json (C2 audit artifact) and return last N entries
    formatted for prompt injection. Returns empty string if log missing.
    Each entry is slimmed to _HISTORY_ENTRY_MAX_CHARS to prevent runaway size
    from large entries (e.g. debugger cluster details).
    """
    rel = _OWN_LOG.get(step)
    if not rel:
        return ""

    data = _read_json_safe(artifact_dir / rel)
    if data is None:
        return ""

    entries = data if isinstance(data, list) else data.get("entries", [])
    if not entries:
        return ""

    recent = [_slim_entry(e) for e in entries[-max_entries:]]
    serialized = json.dumps(recent, indent=2, ensure_ascii=False)
    label = f"history.{step}.{Path(rel).name} (last {len(recent)} entries)"
    return _format_artifact_block(label, f"```json\n{serialized}\n```")


# Max chars to include from knowledge_log.md (keep tail = most recent patterns).
_KNOWLEDGE_LOG_MAX_CHARS: int = 8_000

def _knowledge_excerpt(
    step: str,
    artifact_dir: Path,
) -> str:
    """
    Read archivist/knowledge_log.md and archivist/spec_gaps.md.
    Returns formatted string. Both files are optional — silently skipped if missing.

    knowledge_log.md is append-only and grows unbounded. We keep the tail
    (_KNOWLEDGE_LOG_MAX_CHARS) so the most recent patterns are always included.

    Future: if knowledge_log has structured section headers, filter to sections
    relevant to `step` only.
    """
    blocks: list[str] = []

    # knowledge_log.md — keep tail (most recent)
    kl_path = artifact_dir / "archivist" / "knowledge_log.md"
    kl_text = _read_text_safe(kl_path)
    if kl_text and kl_text.strip():
        kl_text = kl_text.strip()
        if len(kl_text) > _KNOWLEDGE_LOG_MAX_CHARS:
            kl_text = (
                f"[... older entries truncated — showing last "
                f"{_KNOWLEDGE_LOG_MAX_CHARS:,} chars]\n\n"
                + kl_text[-_KNOWLEDGE_LOG_MAX_CHARS:]
            )
        blocks.append(_format_artifact_block(
            "knowledge.archivist.knowledge_log",
            kl_text,
        ))

    # spec_gaps.md — relevant to specwright, judge, patcher, archivist
    _SPEC_GAPS_CONSUMERS = {
        "specwright", "scaffolder", "planner",
        "executor", "judge", "patcher", "archivist",
    }
    if step in _SPEC_GAPS_CONSUMERS:
        sg_path = artifact_dir / "archivist" / "spec_gaps.md"
        sg_text = _read_text_safe(sg_path)
        if sg_text and sg_text.strip():
            blocks.append(_format_artifact_block(
                "knowledge.archivist.spec_gaps",
                sg_text.strip(),
            ))

    return "".join(blocks)


# ── Public API ────────────────────────────────────────────────────────────────

def build_context(
    step: str,
    artifact_dir: Path,
    *,
    slug: str = "",
    scope: str = "full",
    history_entries: int = 3,
) -> dict[str, str]:
    """
    Build context dict for a pipeline step.

    Args:
        step:            Pipeline step name (e.g. "executor", "judge").
        artifact_dir:    Path to artifacts_<slug>/ directory.
        slug:            Project slug used to resolve spec filename
                         (e.g. "my-app"). If empty, falls back to glob.
        scope:           "full" or "mini" — affects which planner artifact is used.
                         Only relevant for executor and debugger.
        history_entries: How many recent entries to include from own *_log.json.

    Returns:
        {
          "upstream":  Formatted string — cumulative short-term artifacts of all
                       prior steps. Empty string if step is first.
          "history":   Formatted string — last N entries from own *_log.json.
                       Empty string if no log or no entries yet.
          "knowledge": Formatted string — archivist knowledge_log.md + spec_gaps.md.
                       Empty string if archivist artifacts not yet created.
        }

    All values are safe to concatenate directly into a prompt.
    Missing files are silently skipped — no exceptions raised.

    Example:
        ctx = build_context("executor", artifact_root, scope="full")

        system_prompt += "\\n\\n" + ctx["knowledge"]
        user_msg = ctx["upstream"] + ctx["history"] + task_block + stub
    """
    if step not in _PIPELINE_ORDER:
        raise ValueError(
            f"[context_builder] Unknown step: {step!r}.\n"
            f"  Valid steps: {_PIPELINE_ORDER}"
        )

    return {
        "upstream":  _read_upstream(step, artifact_dir, scope=scope, slug=slug),
        "history":   _own_history(step, artifact_dir, max_entries=history_entries),
        "knowledge": _knowledge_excerpt(step, artifact_dir),
    }


def context_token_estimate(ctx: dict[str, str], chars_per_token: int = 4) -> dict[str, int]:
    """
    Rough token estimate for each context section.
    Uses chars_per_token as approximation (default 4 chars ≈ 1 token).
    Useful for logging / prompt budget checks before sending to API.

    Example:
        ctx = build_context("executor", artifact_root)
        est = context_token_estimate(ctx)
        print(f"Context tokens: upstream={est['upstream']}, "
              f"history={est['history']}, knowledge={est['knowledge']}, "
              f"total={est['total']}")
    """
    result = {k: len(v) // chars_per_token for k, v in ctx.items()}
    result["total"] = sum(result.values())
    return result


# ── CLI: python -m modules.context_builder (or python context_builder.py) ─────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python context_builder.py <step> <artifact_dir> [scope]")
        print(f"  steps: {_PIPELINE_ORDER}")
        sys.exit(1)

    _step         = sys.argv[1]
    _artifact_dir = Path(sys.argv[2])
    _scope        = sys.argv[3] if len(sys.argv) > 3 else "full"

    _ctx = build_context(_step, _artifact_dir, scope=_scope)
    _est = context_token_estimate(_ctx)

    print(f"=== Context for step: {_step!r} (scope={_scope}) ===")
    print(f"Token estimates: {_est}")
    print()
    for _key, _val in _ctx.items():
        if _val:
            print(f"--- {_key} ({len(_val)} chars) ---")
            print(_val[:500] + ("..." if len(_val) > 500 else ""))
            print()
        else:
            print(f"--- {_key}: (empty) ---")
