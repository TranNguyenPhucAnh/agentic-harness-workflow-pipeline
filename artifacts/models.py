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

import json
import os
import sys
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
    "ckey": {
        "base_url":    "https://ckey.vn",   # Anthropic SDK — không có /v1 (SDK tự handle)
        "api_key_env": "CKEY_API_KEY",
        "headers":     {},
        "sdk":         "anthropic",          # dùng Anthropic SDK thay vì OpenAI SDK
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
    "absorber":           "gemini/gemini-3.5-flash",
    "clarificator":       "openrouter/z-ai/glm-5.1",
    "enricher":           "openrouter/minimax/minimax-m2.7",
    "specwright":         "openrouter/anthropic/claude-sonnet-4.6",
    "spec_validator":     "openrouter/minimax/minimax-m2.7",
    #"spectracker":        "gemini/gemini-3.5-flash",
    "scaffolder":         "openrouter/z-ai/glm-5.1",
    "planner":            "openrouter/moonshotai/kimi-k2.6",
    "executor":           "openrouter/xiaomi/mimo-v2.5-pro",
    "test_writer":        "openrouter/deepseek/deepseek-v4-pro",
    "debugger":           "ckey/gpt-5.5",
    "debugger_secondary": "openrouter/xiaomi/mimo-v2.5-pro",
    "reporter":           "openrouter/minimax/minimax-m2.7",
    "judge":              "openrouter/moonshotai/kimi-k2.6",
    "done_checker":       "openrouter/moonshotai/kimi-k2.6",
    "patcher":            "openrouter/deepseek/deepseek-v4-pro",
    "patcher_secondary":  "openrouter/xiaomi/mimo-v2.5-pro",
    "error_fixer":        "openrouter/deepseek/deepseek-v4-pro",
    "summarizer":         "gemini/gemini-3.5-flash",
    #"archivist":         "openrouter/minimax/minimax-m2.7",

    # ── DevOps / MLOps toolkit roles ────────────────────────────────────────
    # Dùng model mạnh cho analysis + low-cost cho summarization tasks.
    # Điều chỉnh theo budget — các roles này gọi AWS APIs nên ít LLM calls hơn SWE.
    "infra_absorber":           "openrouter/deepseek/deepseek-v4-pro",   # HCL/YAML extraction
    "doc_absorber":             "openrouter/moonshotai/kimi-k2.6",       # unstructured doc analysis
    "config_consistency_checker": "openrouter/moonshotai/kimi-k2.6",    # cross-file reasoning
    "incident_clarificator":    "openrouter/deepseek/deepseek-v4-pro",   # Q&A diagnosis loop
    "postmortem_archivist":     "openrouter/minimax/minimax-m2.7",       # structured extraction
    "metrics_reporter":         "gemini/gemini-3.5-flash",               # no LLM (AWS APIs only)
    "live_discovery":           "gemini/gemini-3.5-flash",               # no LLM (AWS APIs only)
    "spec_risk_assessor":       "openrouter/moonshotai/kimi-k2.6",       # risk + impact analysis
    "spec_impact_assessor":     "openrouter/moonshotai/kimi-k2.6",       # same — separate role

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

class _OpenAICompatResponseAdapter:
    """Wrap OpenAI-format response from ckey.vn into same interface as _AnthropicResponseAdapter."""

    def __init__(self, content: str, usage: dict, raw_body: dict) -> None:
        self.choices = [self._Choice(content)]
        self.usage = self._Usage(usage)

    class _Choice:
        def __init__(self, content: str) -> None:
            self.finish_reason = "stop"
            self.message = self._Message(content)

        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content
                self.tool_calls = None

    class _Usage:
        def __init__(self, usage: dict) -> None:
            self.prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            self.completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

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

def _is_anthropic_sdk(role: str) -> bool:
    """Return True nếu provider của role này dùng Anthropic SDK thay vì OpenAI SDK."""
    provider_key, _ = _parse_role(role)
    return PROVIDERS[provider_key].get("sdk") == "anthropic"


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


def get_anthropic_client(role: str):
    """
    Return a configured anthropic.Anthropic client cho các provider dùng Anthropic SDK
    (vd ckey). base_url lấy từ PROVIDERS — KHÔNG append /v1, Anthropic SDK tự handle.

    Example:
        client = get_anthropic_client("debugger")
        resp   = client.messages.create(
                     model=get_model("debugger"),
                     max_tokens=1024,
                     messages=[{"role": "user", "content": "..."}],
                 )
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "[models] anthropic package not installed. Run: pip install anthropic"
        ) from exc

    provider_key, _ = _parse_role(role)
    cfg = PROVIDERS[provider_key]

    api_key = os.environ.get(cfg["api_key_env"], "").strip()
    if not api_key:
        raise RuntimeError(
            f"[models] Env var {cfg['api_key_env']!r} is not set.\n"
            f"  Required for role={role!r}, provider={provider_key!r}."
        )

    return Anthropic(
        api_key=api_key,
        base_url=cfg["base_url"],
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

    Tự động route sang Anthropic SDK nếu provider config có "sdk": "anthropic"
    (vd ckey). Response luôn được wrap thành _AnthropicResponseAdapter để
    call_llm.py dùng interface thống nhất (.choices[0].message.content, .usage, ...).

    Returns the raw API response object (same as client.chat.completions.create
    với OpenAI SDK, hoặc _AnthropicResponseAdapter với Anthropic SDK).

    Args:
        role      : role name from ROLES
        messages  : chat messages list
        reasoning : per-call reasoning override (None = use REASONING_OVERRIDES/DEFAULT)
        **kwargs  : forwarded to create() (e.g. temperature, max_tokens)

    Example:
        resp = call_model("executor", messages=[{"role": "user", "content": "..."}])
        text = resp.choices[0].message.content

        # Force reasoning on for this call only:
        resp = call_model("judge", messages=[...], reasoning=True)
    """
    if _is_anthropic_sdk(role):
        return _call_model_anthropic(role, messages, **kwargs)

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


def _parse_ckey_response_body(raw_text: str) -> dict:
    """
    ckey.vn đôi khi trả về SSE stream thay vì single JSON object.
    SSE format: nhiều dòng "data: {...}\n\ndata: {...}\n\ndata: [DONE]"
    
    Strategy:
    1. Thử parse thẳng → nếu được thì dùng luôn (non-streaming response)
    2. Nếu JSONDecodeError → scan từng dòng data: {...}, gom choices/content
       từ delta chunks hoặc lấy dòng message-level cuối cùng
    """
    text = raw_text.strip()
    
    # Pass 1: single JSON (non-streaming)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Pass 2: SSE stream — tìm dòng data: có "choices" hoặc "content" (message-level)
    # ckey trả về OpenAI streaming format:
    #   data: {"id":"...","choices":[{"delta":{"content":"..."},...}],...}
    # hoặc một dòng summary cuối: {"choices":[{"message":{"content":"..."}}],...}
    
    content_parts: list[str] = []
    last_full_body: dict | None = None
    usage_data: dict = {}
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == "data: [DONE]":
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
        else:
            payload = line  # bare JSON line (không có data: prefix)
        
        if not payload:
            continue
        
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        
        # Chunk có message đầy đủ (non-delta) → đây là winner
        choices = chunk.get("choices", [])
        if choices:
            msg = choices[0].get("message")
            if msg and msg.get("content"):
                last_full_body = chunk
                break  # đã có message đầy đủ, không cần scan tiếp
            
            # Streaming delta chunk
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
        
        # Usage có thể nằm ở chunk cuối
        if chunk.get("usage"):
            usage_data = chunk["usage"]
    
    # Nếu tìm thấy message đầy đủ trong một chunk
    if last_full_body:
        return last_full_body
    
    # Nếu chỉ có delta chunks → ghép lại thành synthetic response
    if content_parts:
        return {
            "choices": [{"message": {"content": "".join(content_parts)}}],
            "usage": usage_data,
        }
    
    # Fallback: trả về dict rỗng, caller sẽ xử lý empty content
    return {}


def _call_model_anthropic(
    role: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> "_OpenAICompatResponseAdapter":
    client = get_anthropic_client(role)
    model  = get_model(role)

    system_parts  = [m["content"] for m in messages if m["role"] == "system"]
    user_messages = [m for m in messages if m["role"] != "system"]
    system        = system_parts[0] if system_parts else None

    max_tokens = kwargs.pop("max_tokens", 8192)
    temperature = kwargs.pop("temperature", None)
    kwargs.pop("extra_body", None)

    create_kwargs: dict[str, Any] = {
        "model":      model,
        "messages":   user_messages,
        "max_tokens": max_tokens,
    }
    if system:
        create_kwargs["system"] = system
    if temperature is not None:
        create_kwargs["temperature"] = temperature

    raw_resp = client.messages.with_raw_response.create(**create_kwargs)
    # === DEBUG LOG ===
    import sys as _sys
    _raw_preview = raw_resp.text[:800] if raw_resp.text else "<EMPTY>"
    print(f"[DEBUG ckey] raw_resp.text ({len(raw_resp.text)} chars):\n{_raw_preview}", file=_sys.stderr)
    # === END DEBUG ===

    body = _parse_ckey_response_body(raw_resp.text)   # ← thay json.loads trực tiếp
    # === DEBUG LOG 2 ===
    print(f"[DEBUG ckey] parsed body keys: {list(body.keys())}", file=_sys.stderr)
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        ct = msg.get("content", "")
        print(f"[DEBUG ckey] content_text ({len(ct or '')} chars): {repr((ct or '')[:200])}", file=_sys.stderr)
    else:
        print(f"[DEBUG ckey] NO choices in body. body={repr(str(body)[:300])}", file=_sys.stderr)
    # === END DEBUG LOG 2 ===

    content_text = ""
    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content_text = msg.get("content", "") or ""

    usage_data = body.get("usage", {})
    return _OpenAICompatResponseAdapter(content_text, usage_data, body)

class _AnthropicResponseAdapter:
    """
    Wrap anthropic.types.Message thành duck-type interface giống OpenAI ChatCompletion.

    call_llm.py dùng các attribute sau — tất cả đều được implement:
      resp.choices[0].message.content   → _extract_text()
      resp.choices[0].message.tool_calls → _guard_tool_calls()
      resp.choices[0].finish_reason     → _get_finish_reason()
      resp.usage.prompt_tokens          → _record_cost()
      resp.usage.completion_tokens      → _record_cost()

    stop_reason mapping (Anthropic → OpenAI):
      "end_turn"      → "stop"
      "max_tokens"    → "length"   (trigger auto-continue trong call_llm.py)
      "stop_sequence" → "stop"
      anything else   → "stop"
    """

    _STOP_REASON_MAP = {
        "end_turn":      "stop",
        "max_tokens":    "length",
        "stop_sequence": "stop",
    }

    def __init__(self, msg: Any) -> None:
        self._msg  = msg
        self.choices = [_AnthropicResponseAdapter._Choice(msg)]
        self.usage   = _AnthropicResponseAdapter._Usage(msg.usage)

    class _Choice:
        def __init__(self, msg: Any) -> None:
            stop_reason = getattr(msg, "stop_reason", None) or ""
            self.finish_reason = _AnthropicResponseAdapter._STOP_REASON_MAP.get(
                stop_reason, "stop"
            )
            self.message = _AnthropicResponseAdapter._Message(msg)

    class _Message:
        def __init__(self, msg: Any) -> None:
            # Guard against content=None — Anthropic API trả về None khi
            # response bị truncate, rate-limited, hoặc lỗi upstream.
            # getattr default [] không đủ vì attribute tồn tại nhưng value là None.
            content_blocks = getattr(msg, "content", None) or []

            # Ưu tiên text blocks (normal output).
            text_parts = [
                b.text
                for b in content_blocks
                if getattr(b, "type", "") == "text"
            ]

            if text_parts:
                self.content = "\n".join(text_parts).strip()
            else:
                # Debug: log block types để dễ diagnose khi empty
                block_types = [getattr(b, "type", "?") for b in content_blocks]
                if block_types:
                    import sys
                    print(
                        f"[models][warn] No text blocks found. "
                        f"Block types: {block_types}. "
                        f"stop_reason={getattr(msg, 'stop_reason', '?')}",
                        file=sys.stderr,
                    )

                # Fallback: nếu model chỉ trả về thinking blocks (extended thinking
                # mode hoặc Claude 3.7+/4.x với thinking bật), lấy thinking text.
                # Điều này xảy ra khi REASONING_OVERRIDES bật cho role này,
                # hoặc provider tự bật thinking khi input phức tạp.
                thinking_parts = [
                    getattr(b, "thinking", "") or getattr(b, "text", "")
                    for b in content_blocks
                    if getattr(b, "type", "") == "thinking"
                ]
                self.content = "\n".join(thinking_parts).strip()

            self.tool_calls = None  # không dùng tool_calls

    class _Usage:
        def __init__(self, usage: Any) -> None:
            self.prompt_tokens     = getattr(usage, "input_tokens",  0) or 0
            self.completion_tokens = getattr(usage, "output_tokens", 0) or 0


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
    print(json.dumps(model_info(), indent=2))