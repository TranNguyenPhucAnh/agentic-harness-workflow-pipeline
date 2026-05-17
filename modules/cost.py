"""
modules/cost.py — Cost tracking for agentic pipeline.

Strategy
────────
For OpenRouter calls  → read `resp.usage.cost` directly from the response.
                        OpenRouter usually includes this field in credits
                        (1 credit = $0.000001 USD → divide by 1_000_000).
                        BUT: some routes return cost=0 even when tokens were
                        consumed (upstream accounting fail, certain provider
                        routings). In that case we fall through to the price
                        table so cost isn't silently lost.

For non-OpenRouter    → fall back to _PRICE_TABLE (hardcoded $/MTok).
                        Covers Google (gemini), direct Anthropic, etc.

Provider detection: check the `provider` param passed alongside `usage`.
If provider is "openrouter" AND usage.cost > 0 → use native cost.
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
# Used for:
#   1) NON-OpenRouter providers (e.g. google/gemini direct).
#   2) OpenRouter responses where usage.cost is 0 or missing.
#
# Format: list of (model_id_prefix, input_$/MTok, output_$/MTok).
# Matched via str.startswith on the LOWERCASED model string passed to
# record_usage(). Order is most-specific first to avoid prefix collisions.
# Values in USD per million tokens.
#
# IMPORTANT: prefixes for OpenRouter routes must include the org prefix
# (e.g. "anthropic/claude-..."), because the model string passed in is the
# part AFTER the provider key — i.e. exactly what was sent to the API.
#
# VERIFY all OpenRouter prices against https://openrouter.ai/models before
# trusting cost numbers in production. Prices below are best-effort placeholders.

_PRICE_TABLE: list[tuple[str, float, float]] = [
    # ── Gemini (Google direct) ────────────────────────────────────────────────
    # Prices as of May 2026 — verify at https://ai.google.dev/pricing
    ("gemini-2.5-pro",                       1.25,  10.00),   # >200k ctx: 2.50/15.00
    ("gemini-2.5-flash",                     0.15,   0.60),
    ("gemini-2.0-flash",                     0.10,   0.40),
    ("gemini-1.5-pro",                       1.25,   5.00),
    ("gemini-1.5-flash",                     0.075,  0.30),

    # ── OpenRouter routes (fallback when usage.cost == 0 / missing) ───────────
    # Prefix-matched, so e.g. "moonshotai/kimi-k2" matches "moonshotai/kimi-k2.6".
    # VERIFY at https://openrouter.ai/models — values below are placeholders.
    ("anthropic/claude-opus-4.7",              5.00,  25.00),
    ("anthropic/claude-sonnet-4.6",            3.00,  15.00),
    ("anthropic/claude-haiku-4.5",             1.00,   5.00),
    ("moonshotai/kimi-k2.6",                   0.73,   3.49),
    ("deepseek/deepseek-v4-pro",               0.435,  0.87),
    ("qwen/qwen3.6-plus",                      0.325,  1.95),
    ("z-ai/glm-5.1",                           0.98,   3.08),
    ("minimax/minimax-m2.7",                   0.279,  1.20),

    # ── Anthropic direct (if ever used without OpenRouter) ────────────────────
    ("claude-opus-4-7",                      5.00,  25.00),
    ("claude-opus-4-6",                      5.00,  25.00),
    ("claude-sonnet-4-6",                    3.00,  15.00),
    ("claude-opus-4-5",                      5.00,  25.00),
    ("claude-sonnet-4-5",                    3.00,  15.00),
    ("claude-haiku-4-5",                     1.00,   5.00),
    ("claude-opus-4-1",                     15.00,  75.00),
    ("claude-opus-4",                       15.00,  75.00),
    ("claude-sonnet-4",                      3.00,  15.00),
    ("claude-haiku-3-5",                     0.80,   4.00),
    ("claude-haiku-3",                       0.25,   1.25),
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
               OpenRouter also sets usage.cost (in credits) — but sometimes 0.
    model    : model string — used for fallback price-table lookup. Pass the
               value returned by get_model(role) here (no provider prefix).
    provider : provider key from models.py (e.g. "openrouter", "gemini").
               Pass get_provider(role) here.

    Returns
    -------
    float | None — cost for this call in USD, or None if cost couldn't be determined.

    Resolution order
    ----------------
    1. OpenRouter native cost (if provider==openrouter AND usage.cost > 0)
    2. Price-table lookup on `model` string
    3. Mark cost as missing, return None
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

    # Path 1: OpenRouter native cost (most accurate when present AND non-zero).
    # usage.cost is in credits; 1 credit = $0.000001.
    # Reflects real routed price including any caching/discounts applied.
    #
    # Some routes return cost=0 even when tokens were consumed (upstream
    # accounting issue, certain provider routings). In that case we DON'T
    # accept the zero — we fall through to the price table below so the
    # number isn't silently lost.
    if provider.lower() == "openrouter":
        raw_cost = getattr(usage, "cost", None)
        if raw_cost is not None and float(raw_cost) > 0:
            call_cost = float(raw_cost) * _CREDITS_TO_USD
            _total_cost += call_cost
            return call_cost
        # cost is None or 0 → fall through to Path 2

    # Path 2: Fallback price-table lookup. Covers:
    #   - Non-OpenRouter providers (e.g. gemini direct)
    #   - OpenRouter when usage.cost was 0 or missing
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
