"""
artifacts/models.py
===================
SOURCE OF TRUTH cho tất cả model assignments và provider config trong pipeline.

RULE: Không script nào hard-code model string hay endpoint URL — chỉ import từ đây.

Usage
─────
    from artifacts.models import get_client, get_model

    client   = get_client("executor")
    model    = get_model("executor")
    response = client.chat.completions.create(model=model, messages=[...])

Reasoning toggle
────────────────
    # Default: reasoning OFF toàn bộ (xem REASONING_OVERRIDES bên dưới)
    # Với các provider trong REASONING_EXPLICIT_DISABLE_PROVIDERS (vd openrouter),
    # reasoning OFF sẽ được gửi EXPLICIT {"reasoning": {"enabled": False}} thay vì
    # không gửi gì — tránh model dùng default-on reasoning của provider (kimi, glm, ...).

    # Bật cho một call cụ thể:
    extra = reasoning_params("judge")
    client.chat.completions.create(model=model, messages=[...], **extra)

    # Hoặc dùng helper all-in-one:
    call_model("executor", messages=[...])
    call_model("judge",    messages=[...], reasoning=True)   # override per-call

Đổi model/provider
──────────────────
    Sửa ROLES hoặc PROVIDERS trong file này.
    Không cần đụng bất kỳ script pipeline nào.
"""

from __future__ import annotations

import os
from typing import Any

# ── Provider registry ─────────────────────────────────────────────────────────
#
# Thêm provider mới: thêm entry vào dict này, không sửa gì khác.
#
# key        : tên ngắn dùng làm prefix trong ROLES (vd "openrouter/...")
# base_url   : OpenAI-compatible endpoint
# api_key_env: tên env var chứa API key
# headers    : extra headers gửi kèm mọi request (vd HTTP-Referer cho OpenRouter)

PROVIDERS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "headers": {
            "HTTP-Referer": os.environ.get("PIPELINE_HTTP_REFERER", "https://github.com/pipeline"),
            "X-Title":      os.environ.get("PIPELINE_X_TITLE",      "llm-pipeline"),
        },
    },
    "gemini": {
        "base_url":    "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GOOGLE_API_KEY",
        "headers": {},
    },
    # Ví dụ fallback providers (uncomment khi cần):
    # "openrouter-eu": {
    #     "base_url":    "https://openrouter.ai/api/v1",   # same URL, diff key
    #     "api_key_env": "OPENROUTER_EU_API_KEY",
    #     "headers": {},
    # },
}


# ── Model role registry ───────────────────────────────────────────────────────
#
# Format: "<provider_key>/<model_string_on_that_provider>"
#
# provider_key phải có trong PROVIDERS dict ở trên.
# model_string là chuỗi truyền vào API, không bao gồm provider_key.
#
# Để đổi model cho một step: sửa value ở đây, không đụng script nào.
# Để thêm role mới: thêm entry, sau đó dùng get_client/get_model trong script.

ROLES: dict[str, str] = {
    # ── Pipeline steps ─────────────────────────────────────────────────────
    # NOTE: ~ prefix trên OpenRouter = "route directly via provider đó".
    # Khi dùng ~, model string PHẢI là exact model ID của provider
    # (vd ~anthropic/claude-sonnet-4-5-20251001).
    # "claude-sonnet-latest" không phải valid Anthropic model ID → OpenRouter
    # trả về HTML 404 thay vì JSON → JSONDecodeError trong openai SDK.
    # Drop ~ để dùng load-balanced routing (resilient hơn, auto-failover).
    "absorber":           "gemini/gemini-2.5-flash",
    "clarificator":       "openrouter/z-ai/glm-5.1",
    "enricher":           "openrouter/qwen/qwen3.6-plus",
    "specwright":         "openrouter/anthropic/claude-sonnet-4.6",
    #"spectracker":        "gemini/gemini-2.5-flash",
    "scaffolder":         "openrouter/z-ai/glm-5.1",
    "planner":            "openrouter/moonshotai/kimi-k2.6",
    "executor":           "openrouter/anthropic/claude-sonnet-4.6",
    "debugger":           "openrouter/minimax/minimax-m2.7",
    "debugger_secondary": "openrouter/qwen/qwen3.6-plus",
    "reporter":           "openrouter/deepseek/deepseek-v4-pro",
    "judge":              "openrouter/moonshotai/kimi-k2.6",
    "patcher":            "openrouter/minimax/minimax-m2.7",
    #"archivist":         "openrouter/deepseek/deepseek-v4-pro",

    # ── Aux / utility roles (thêm khi cần) ─────────────────────────────────
    # "summarizer":       "deepseek/deepseek-v4-flash",
    # "validator":        "z-ai/glm-5.1",
}


# ── Reasoning config ──────────────────────────────────────────────────────────
#
# REASONING_DEFAULT   : True/False — áp dụng cho mọi role không có entry riêng
# REASONING_OVERRIDES : per-role override; nếu role không có entry → dùng DEFAULT
#
# Thứ tự ưu tiên (cao → thấp):
#   1. per-call argument trong call_model(..., reasoning=True/False)
#   2. REASONING_OVERRIDES[role]
#   3. REASONING_DEFAULT
#
# NOTE: {"reasoning": {"enabled": false}} là best-effort — provider/model có thể
# tự kích hoạt reasoning khi input phức tạp. Không có client-side hard guarantee.
# Để tắt cứng: dùng model không có reasoning mode (vd deepseek-v4-pro, qwen3.6-plus).

REASONING_DEFAULT: bool = False

REASONING_OVERRIDES: dict[str, bool] = {
    # Bật reasoning cho các role cần deep thinking:
    "judge":     True,
    "planner":   True,
    # "executor":  True,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_role(role: str) -> tuple[str, str]:
    """
    Return (provider_key, model_string) for a role.
    Raises ValueError if role unknown or provider not in PROVIDERS.
    """
    if role not in ROLES:
        raise ValueError(
            f"[models] Unknown role: {role!r}.\n"
            f"  Valid roles: {sorted(ROLES)}"
        )
    entry = ROLES[role]
    # entry format: "<provider_key>/<model_string>"
    # model_string itself may contain slashes (e.g. "deepseek/deepseek-v4-pro")
    slash = entry.index("/")
    provider_key = entry[:slash]
    model_string  = entry[slash + 1:]

    if provider_key not in PROVIDERS:
        raise ValueError(
            f"[models] Provider {provider_key!r} not in PROVIDERS.\n"
            f"  (role={role!r}, entry={entry!r})"
        )
    return provider_key, model_string


def _resolve_reasoning(role: str, override: bool | None) -> bool:
    """Return effective reasoning flag for a role + optional per-call override."""
    if override is not None:
        return override
    return REASONING_OVERRIDES.get(role, REASONING_DEFAULT)


# ── Public API ────────────────────────────────────────────────────────────────

def get_model(role: str) -> str:
    """Return the model string (as passed to the API) for a given role."""
    _, model_string = _parse_role(role)
    return model_string


def get_provider(role: str) -> str:
    """Return the provider key for a given role."""
    provider_key, _ = _parse_role(role)
    return provider_key


def get_client(role: str):
    """
    Return a configured openai.OpenAI client for a given role.

    The client's base_url and api_key are set from PROVIDERS.
    Extra headers (e.g. HTTP-Referer for OpenRouter) are injected via
    default_headers.

    Example:
        client = get_client("executor")
        resp   = client.chat.completions.create(
                     model=get_model("executor"),
                     messages=[...],
                 )
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "[models] openai package not installed. Run: pip install openai"
        ) from exc

    provider_key, _ = _parse_role(role)
    cfg = PROVIDERS[provider_key]

    api_key = os.environ.get(cfg["api_key_env"], "").strip()
    if not api_key:
        raise RuntimeError(
            f"[models] Env var {cfg['api_key_env']!r} is not set.\n"
            f"  Required for role={role!r}, provider={provider_key!r}."
        )

    return OpenAI(
        base_url=cfg["base_url"],
        api_key=api_key,
        default_headers=cfg.get("headers", {}),
    )


def reasoning_params(role: str, override: bool | None = None) -> dict[str, Any]:
    """
    Return extra kwargs to pass to chat.completions.create() for reasoning.

    reasoning ON  → {"extra_body": {"reasoning": {"enabled": True}}}
    reasoning OFF → {} (không gửi param — best-effort, không hard guarantee)

    Example:
        extra = reasoning_params("judge")
        client.chat.completions.create(model=model, messages=msgs, **extra)
    """
    enabled = _resolve_reasoning(role, override)
    if not enabled:
        return {}
    return {"extra_body": {"reasoning": {"enabled": True}}}


def call_model(
    role: str,
    messages: list[dict[str, Any]],
    *,
    reasoning: bool | None = None,
    **kwargs: Any,
) -> Any:
    """
    Convenience wrapper: get_client + get_model + reasoning_params in one call.

    Returns the raw API response object (same as client.chat.completions.create).

    Args:
        role      : role name from ROLES
        messages  : chat messages list
        reasoning : per-call reasoning override (None = use REASONING_OVERRIDES/DEFAULT)
        **kwargs  : forwarded to chat.completions.create (e.g. temperature, max_tokens)

    Example:
        resp = call_model("executor", messages=[{"role": "user", "content": "..."}])
        text = resp.choices[0].message.content

        # Force reasoning on for this call only:
        resp = call_model("judge", messages=[...], reasoning=True)
    """
    import json as _json
    client = get_client(role)
    model  = get_model(role)
    extra  = reasoning_params(role, reasoning)
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            **extra,
            **kwargs,
        )
    except _json.JSONDecodeError as exc:
        # OpenRouter (and other proxies) return HTML error pages or malformed JSON
        # when the model string is invalid, quota is exceeded, or upstream times out.
        # The openai SDK then fails trying to parse the HTTP response body as JSON.
        #
        # Common causes:
        #   - Invalid model string (e.g. "~anthropic/claude-sonnet-latest" — the
        #     ~ prefix requires an exact provider model ID, not an alias)
        #   - Model not available on this provider
        #   - Rate limit / quota exceeded (some providers return HTML for these)
        #   - Upstream timeout returning a partial/empty body
        provider_key, _ = _parse_role(role)
        cfg = PROVIDERS[provider_key]
        raise RuntimeError(
            f"[models] call_model({role!r}) received a non-JSON response from "
            f"{cfg['base_url']} — provider likely returned an HTML error page.\n"
            f"  model string : {model!r}\n"
            f"  Check: (1) model ID is valid on {provider_key!r}, "
            f"(2) API key is correct, (3) quota not exceeded.\n"
            f"  Original parse error: {exc}"
        ) from exc


def model_info(role: str | None = None) -> dict | list[dict]:
    """
    Return config info for a role (or all roles if role is None).
    Useful for logging/debugging. Does NOT expose API keys.

    Example:
        print(model_info("executor"))
        # {'role': 'executor', 'provider': 'openrouter',
        #   'model': 'anthropic/claude-sonnet-4.6', 'reasoning': False}
    """
    def _info(r: str) -> dict:
        pk, ms = _parse_role(r)
        return {
            "role":      r,
            "provider":  pk,
            "model":     ms,
            "base_url":  PROVIDERS[pk]["base_url"],
            "reasoning": _resolve_reasoning(r, None),
        }

    if role is not None:
        return _info(role)
    return [_info(r) for r in sorted(ROLES)]


# ── CLI: python -m artifacts.models (or python models.py) ────────────────────

if __name__ == "__main__":
    import json
    print(json.dumps(model_info(), indent=2))
