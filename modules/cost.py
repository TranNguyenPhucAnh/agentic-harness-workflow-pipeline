"""
modules/cost.py — Cost tracking for agentic pipeline.

Strategy
────────
For OpenRouter calls  → read `resp.usage.cost` directly from the response.
                        OpenRouter always includes this field (no extra params needed).
                        Unit: credits (1 credit = $0.000001 USD → divide by 1_000_000).

For non-OpenRouter    → fall back to _PRICE_TABLE (hardcoded $/MTok).
                        Covers Google (gemini), direct Anthropic, etc.

Provider detection: check the `provider` param passed alongside `usage`.
If provider is "openrouter" → use native cost.
Otherwise → use price table lookup on model string.

Usage per script
────────────────
    from modules.cost import record_usage, print_call, print_summary, summary
    from artifacts.models import get_model, get_provider, call_model

    ROLE  = "executor"
    MODEL = get_model(ROLE)

    resp      = call_model(ROLE, messages=[...])
    call_cost = record_usage(resp.usage, model=MODEL, provider=get_provider(ROLE))
    print_call(__file__, resp.usage.prompt_tokens, resp.usage.completion_tokens, call_cost)

    # end of run:
    print_summary("[08]")
    manifest["token_summary"] = summary()

Design note
───────────
Module-level globals are safe here because harness launches each script
in a separate subprocess (subprocess.run), so state never leaks across scripts.
Call reset() only in tests or multi-run single-process scenarios.
"""

from __future__ import annotations

# ─── Fallback pricing table ───────────────────────────────────────────────────
# Used only for NON-OpenRouter providers (e.g. google/gemini direct).
# OpenRouter responses carry native cost — no lookup needed for those.
#
# Format: list of (model_id_prefix, input_$/MTok, output_$/MTok).
# Matched via str.startswith; ordered most-specific first to avoid collisions.
# Values in USD per million tokens.

_PRICE_TABLE: list[tuple[str, float, float]] = [
    # ── Gemini (Google direct) ────────────────────────────────────────────────
    # Prices as of May 2026 — verify at https://ai.google.dev/pricing
    ("gemini-2.5-pro",               1.25,  10.00),   # >200k ctx: 2.50/15.00
    ("gemini-2.5-flash",             0.15,   0.60),
    ("gemini-2.0-flash",             0.10,   0.40),
    ("gemini-1.5-pro",               1.25,   5.00),
    ("gemini-1.5-flash",             0.075,  0.30),
    # ── Anthropic direct (if ever used without OpenRouter) ───────────────────
    ("claude-opus-4-7",              5.00,  25.00),
    ("claude-opus-4-6",              5.00,  25.00),
    ("claude-sonnet-4-6",            3.00,  15.00),
    ("claude-opus-4-5",              5.00,  25.00),
    ("claude-sonnet-4-5",            3.00,  15.00),
    ("claude-haiku-4-5",             1.00,   5.00),
    ("claude-opus-4-1",             15.00,  75.00),
    ("claude-opus-4",               15.00,  75.00),
    ("claude-sonnet-4",              3.00,  15.00),
    ("claude-haiku-3-5",             0.80,   4.00),
    ("claude-haiku-3",               0.25,   1.25),
]

_MTOK           = 1_000_000   # tokens per pricing unit
_CREDITS_TO_USD = 1 / _MTOK   # 1 credit = $0.000001 (OpenRouter unit)


def _lookup_price(model: str) -> tuple[float, float] | None:
    """Return (input_$/MTok, output_$/MTok) or None if model unknown."""
    m = model.lower()
    for prefix, inp, out in _PRICE_TABLE:
        if m.startswith(prefix):
            return inp, out
    return None


# ─── Module-level accumulators ────────────────────────────────────────────────

_total_prompt:     int   = 0
_total_completion: int   = 0
_total_cost:       float = 0.0
_cost_missing:     bool  = False   # True if any call couldn't determine cost


# ─── Public API ───────────────────────────────────────────────────────────────

def record_usage(
    usage,
    *,
    model:    str = "",
    provider: str = "",
) -> float | None:
    """
    Accumulate token counts and cost from one API response.

    Parameters
    ----------
    usage    : response.usage object (openai-compat: .prompt_tokens / .completion_tokens,
               or anthropic-native: .input_tokens / .output_tokens).
               OpenRouter also sets usage.cost (in credits).
    model    : model string — used for fallback price-table lookup on non-OpenRouter.
    provider : provider key from models.py (e.g. "openrouter", "google").
               Pass get_provider(role) here.

    Returns
    -------
    float | None — cost for this call in USD, or None if cost couldn't be determined.
    """
    global _total_prompt, _total_completion, _total_cost, _cost_missing

    # ── Token counts (handle both openai-compat and anthropic-native field names)
    pt = (
        getattr(usage, "prompt_tokens",     None)
        or getattr(usage, "input_tokens",   None)
        or 0
    )
    ct = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens",  None)
        or 0
    )

    _total_prompt     += pt
    _total_completion += ct

    # ── Cost resolution ───────────────────────────────────────────────────────

    # Path 1: OpenRouter native cost (always present, most accurate).
    # usage.cost is in credits; 1 credit = $0.000001.
    # Reflects real routed price including any caching/discounts applied.
    if provider.lower() == "openrouter":
        raw_cost = getattr(usage, "cost", None)
        if raw_cost is not None:
            call_cost = float(raw_cost) * _CREDITS_TO_USD
            _total_cost += call_cost
            return call_cost
        # OpenRouter but cost field missing — unusual, flag it
        _cost_missing = True
        return None

    # Path 2: Fallback price-table lookup for other providers (e.g. google direct)
    if model:
        prices = _lookup_price(model)
        if prices is not None:
            inp_price, out_price = prices
            call_cost = (pt * inp_price + ct * out_price) / _MTOK
            _total_cost += call_cost
            return call_cost

    # Path 3: Unknown — accumulate tokens but mark cost incomplete
    _cost_missing = True
    return None


def summary() -> dict:
    """
    Return a dict for writing into a session manifest's token_summary field.

    {
        "prompt_tokens":     int,
        "completion_tokens": int,
        "total_tokens":      int,
        "total_cost_usd":    float | None,   # None if any call was unresolvable
        "cost_complete":     bool,
    }
    """
    return {
        "prompt_tokens":     _total_prompt,
        "completion_tokens": _total_completion,
        "total_tokens":      _total_prompt + _total_completion,
        "total_cost_usd":    round(_total_cost, 6) if not _cost_missing else None,
        "cost_complete":     not _cost_missing,
    }


def print_call(
    file_path: str,
    pt: int,
    ct: int,
    cost: float | None,
    *,
    label: str = "",
) -> None:
    """
    Print a single-line cost entry for one API call.

    Example:
        [cost] 08_executor.py              │ prompt=  1,234  completion=   567  cost=$0.0231
    """
    import os
    name     = label or os.path.basename(file_path)
    cost_str = f"${cost:.4f}" if cost is not None else "n/a"
    print(
        f"[cost] {name:<30} │ "
        f"prompt={pt:>8,}  completion={ct:>7,}  cost={cost_str}"
    )


def print_summary(prefix: str = "[cost]") -> None:
    """
    Print cumulative totals at end of a script run.

    Example:
        [08] TOTAL │ prompt=  12,345  completion=  3,210  total=  15,555  cost=$0.1847
    """
    total    = _total_prompt + _total_completion
    cost_str = (
        f"${_total_cost:.4f}"
        if not _cost_missing
        else f"${_total_cost:.4f}+ (incomplete)"
    )
    print(
        f"{prefix} TOTAL │ "
        f"prompt={_total_prompt:>9,}  "
        f"completion={_total_completion:>8,}  "
        f"total={total:>9,}  "
        f"cost={cost_str}"
    )


def reset() -> None:
    """
    Reset all accumulators.

    Call at the start of a script when running multiple logical "runs"
    in one process (tests, REPL). Not needed in normal harness usage
    because each script runs in its own subprocess.
    """
    global _total_prompt, _total_completion, _total_cost, _cost_missing
    _total_prompt     = 0
    _total_completion = 0
    _total_cost       = 0.0
    _cost_missing     = False
