"""
modules/call_llm.py — Unified LLM call wrapper for all pipeline scripts.
...
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from typing import Any

from artifacts.models import call_model, get_model, get_provider
from modules.cost import print_call, record_usage

# ─── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_RETRIES        = 3
_DEFAULT_TEMPERATURE    = None   # None = use model default
_BACKOFF_BASE           = 2.0    # seconds; actual wait = base^(attempt-1) + jitter
_BACKOFF_JITTER         = 1.0    # max random jitter seconds
_FLAT_RETRY_SLEEP       = 3.0    # used when backoff=False
_DEFAULT_CONTINUATIONS  = 5      # max continuation rounds for finish_reason=length
_CONTINUE_PROMPT        = "continue from where you left off"
_CONTINUE_JSON_PROMPT   = (
    "Your previous response was cut off before the JSON was complete. "
    "Please output the COMPLETE JSON object from the very beginning, "
    "ensuring it is valid and fully closed."
)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _extract_text(resp: Any) -> str:
    """Extract text content from a call_model response object."""
    # Guard: nếu resp không phải object hợp lệ (vd string từ bad provider response)
    if not hasattr(resp, "choices"):
        raise RuntimeError(
            f"[_extract_text] Expected OpenAI response object, got {type(resp).__name__!r}. "
            f"Value: {str(resp)[:200]!r}"
        )
    content = getattr(resp.choices[0].message, "content", None)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in content
        )
    return (content or "").strip()


def _get_finish_reason(resp: Any) -> str:
    """Return the finish_reason string from the first choice, or '' if unavailable."""
    return getattr(resp.choices[0], "finish_reason", None) or ""


def _guard_tool_calls(resp: Any) -> None:
    """
    Raise RuntimeError if the model returned tool_calls instead of text.
    This guards planner/executor/debugger/patcher against models that
    hallucinate tool use when none was requested.
    """
    tool_calls = getattr(resp.choices[0].message, "tool_calls", None)
    if tool_calls:
        raise RuntimeError(f"Model returned tool_calls instead of text: {tool_calls}")


def _record_cost(resp: Any, role: str, caller_file: str, label: str) -> float:
    """Record token usage and return cost. Returns 0.0 on any error."""
    usage = getattr(resp, "usage", None)
    if not usage:
        return 0.0
    try:
        pt        = getattr(usage, "prompt_tokens",     0) or 0
        ct        = getattr(usage, "completion_tokens", 0) or 0
        call_cost = record_usage(usage, model=get_model(role), provider=get_provider(role))
        print_call(caller_file, pt, ct, call_cost, **({ "label": label } if label else {}))
        return call_cost
    except Exception:
        return 0.0


def _sleep_before_retry(attempt: int, backoff: bool) -> None:
    if backoff:
        wait = (_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, _BACKOFF_JITTER)
        time.sleep(wait)
    else:
        time.sleep(_FLAT_RETRY_SLEEP)


def _handle_offline(exc: Exception, role: str) -> str | None:
    """
    If the exception indicates a missing API key, prompt user to paste
    the LLM response manually (offline/dev mode).
    Returns the pasted text, or None if this is not an offline situation.
    """
    if "not set" in str(exc):
        prefix = f"[{role}]" if role else "[llm]"
        print(f"\n{prefix}[offline] No API key found. Paste LLM response then EOF (Ctrl-D):")
        return sys.stdin.read()
    return None


def _sanitize_json_string(raw: str) -> str:
    """
    Remove or escape ASCII control characters (0x00–0x1F, 0x7F) that are
    invalid inside JSON strings but sometimes appear in model output when
    a response is truncated mid-string and then continued.

    Only characters that are not valid JSON whitespace (\\t \\n \\r) are
    replaced with a space so the overall structure is preserved.
    """
    def _replace(m: re.Match) -> str:
        ch = m.group(0)
        if ch in ("\t", "\n", "\r"):
            return ch
        return " "

    return re.sub(r"[\x00-\x1f\x7f]", _replace, raw)


def _escape_code_newlines(raw: str) -> str:
    """
    Escape literal newlines/tabs/carriage-returns inside JSON string values.
    Models sometimes write actual newline chars inside "code": "..." instead
    of the escaped \\n sequence, producing invalid JSON.
    Walks the string tracking in-string state to only escape within quotes.
    """
    result = []
    in_string = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == '\\' and in_string:
            # Escaped char — pass through both backslash and next char
            result.append(ch)
            i += 1
            if i < len(raw):
                result.append(raw[i])
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
        if in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\t':
            result.append('\\t')
        elif in_string and ch == '\r':
            result.append('\\r')
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def _try_ast_literal(s: str) -> dict[str, Any] | None:
    """Parse Python dict literal (single-quoted) via ast.literal_eval (safe)."""
    import ast
    try:
        parsed = ast.literal_eval(s.strip())
        if isinstance(parsed, dict):
            return json.loads(json.dumps(parsed))
    except Exception:
        pass
    return None


def _parse_json_object(raw: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output.

    Pass 0 — escape literal newlines inside JSON strings (code fields)
    Pass 1 — direct json.loads ± control-char sanitization
    Pass 2 — find outermost {...} then json.loads ± sanitization
    Pass 3 — ast.literal_eval (Python dict with single-quoted keys/values)
    Raises ValueError if all passes fail.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    # Pass 1: direct parse (with escalating sanitization)
    for candidate in (
        text,
        _sanitize_json_string(text),
        _escape_code_newlines(text),
        _escape_code_newlines(_sanitize_json_string(text)),
    ):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            return parsed
        except json.JSONDecodeError:
            pass

    # Pass 2: find outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        block = m.group()
        for candidate in (
            block,
            _sanitize_json_string(block),
            _escape_code_newlines(block),
            _escape_code_newlines(_sanitize_json_string(block)),
        ):
            try:
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
                return parsed
            except json.JSONDecodeError:
                pass

    # Pass 3: Python dict literal (single-quoted keys/values from some models)
    for candidate in (text, m.group() if m else None):
        if candidate is None:
            continue
        result = _try_ast_literal(candidate)
        if result is not None:
            return result

    raise ValueError(
        "No valid JSON object found in model output "
        "(tried json.loads, regex extract, ast.literal_eval)."
    )


def _continue_on_length(
    role: str,
    messages: list[dict[str, str]],
    first_resp: Any,
    kwargs: dict[str, Any],
    max_continuations: int,
    caller_file: str,
    label: str,
    *,
    json_mode: bool = False,
) -> tuple[str, float]:
    """
    Given a response that hit finish_reason=length, keep calling the model
    until finish_reason != length or max_continuations is exhausted.

    json_mode=True changes the strategy: instead of appending the truncated
    output and asking the model to continue (which produces broken JSON),
    we ask the model to re-output the COMPLETE JSON from scratch.  The last
    complete response replaces accumulated content rather than being appended.

    Returns (full_content: str, total_cost: float).
    The cost of the *first* call is NOT included — callers add it separately.
    """
    accumulated = _extract_text(first_resp)
    total_cost  = 0.0
    conv        = list(messages)  # local copy

    for turn in range(1, max_continuations + 1):
        print(
            f"[call_llm][continue] {role} finish_reason=length "
            f"— continuation {turn}/{max_continuations} …"
        )

        if json_mode:
            # Don't append the broken partial JSON — just ask for a full redo.
            # We keep the original system+user messages and add a single user
            # follow-up so the model has full context.
            conv_for_call = conv + [{"role": "user", "content": _CONTINUE_JSON_PROMPT}]
        else:
            # Plain-text mode: append what the model produced so far, then continue.
            conv.append({"role": "assistant", "content": accumulated})
            conv.append({"role": "user",      "content": _CONTINUE_PROMPT})
            conv_for_call = conv

        resp = call_model(role, conv_for_call, **kwargs)

        cost        = _record_cost(resp, role, caller_file, label)
        total_cost += cost
        chunk       = _extract_text(resp)

        if json_mode:
            # Replace accumulated with the new (hopefully complete) response.
            if chunk:
                accumulated = chunk
        else:
            if chunk:
                accumulated += chunk

        finish = _get_finish_reason(resp)
        if finish != "length":
            break

        if json_mode:
            # Update conv so next iteration's follow-up has the latest attempt.
            conv = list(messages)
    else:
        print(
            f"[call_llm][warn] {role} still finish_reason=length after "
            f"{max_continuations} continuations — returning accumulated content."
        )

    return accumulated, total_cost


# ─── Public API ───────────────────────────────────────────────────────────────

def call_llm(
    role: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 8192,
    temperature: float | None = _DEFAULT_TEMPERATURE,
    retries: int = _DEFAULT_RETRIES,
    backoff: bool = True,
    label: str = "",
    caller_file: str = "",
    offline_fallback: bool = True,
    max_continuations: int = _DEFAULT_CONTINUATIONS,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[str, float]:
    """
    Call an LLM model with automatic retry on empty content.

    Parameters
    ──────────
    role               : Role key registered in artifacts/models.py (e.g. "enricher").
    system             : System prompt string.
    user               : User message string.
    max_tokens         : Max completion tokens.
    temperature        : Sampling temperature. None = model default.
    retries            : Total attempts before giving up (default 3).
    backoff            : If True, use exponential backoff between retries.
                         If False, use flat 3s sleep (matches old debugger/patcher behaviour).
    label              : Optional label appended to print_call output (e.g. "[09] secondary").
    caller_file        : Pass __file__ from the calling script for accurate cost log paths.
    offline_fallback   : If True, prompt user to paste response when API key is missing.
    max_continuations  : Max extra calls when finish_reason=length (default 5).
                         Set to 0 to disable auto-continue.

    Returns
    ───────
    (content: str, cost: float)
      content — model response text, stripped. Concatenated across continuations.
      cost    — total USD cost across all calls; 0.0 if unavailable.
    """
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = call_model(role, messages, **kwargs)

            cost    = _record_cost(resp, role, caller_file, label)
            content = _extract_text(resp)

            _guard_tool_calls(resp)

            if not content:
                raise RuntimeError("Model returned empty content.")

            # ── Auto-continue on truncation ───────────────────────────────────
            if max_continuations > 0 and _get_finish_reason(resp) == "length":
                extra_content, extra_cost = _continue_on_length(
                    role, messages, resp, kwargs, max_continuations, caller_file, label,
                    json_mode=False,
                )
                return extra_content, cost + extra_cost

            return content, cost

        except RuntimeError as exc:
            if offline_fallback:
                pasted = _handle_offline(exc, role)
                if pasted is not None:
                    return pasted, 0.0

            last_exc = exc
            if attempt < retries:
                wait = (_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, _BACKOFF_JITTER) if backoff else _FLAT_RETRY_SLEEP
                print(
                    f"[call_llm][warn] {role} attempt {attempt}/{retries} failed: {exc} "
                    f"— retrying in {wait:.1f}s …"
                )
                time.sleep(wait)
            else:
                print(f"[call_llm][warn] {role} attempt {attempt}/{retries} failed: {exc} — giving up.")

        except Exception as exc:
            print(f"[call_llm][error] {role} call failed: {exc}", file=sys.stderr)
            raise

    raise last_exc or RuntimeError(f"Model returned empty content after {retries} retries.")


def call_llm_json(
    role: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 8192,
    temperature: float | None = _DEFAULT_TEMPERATURE,
    retries: int = _DEFAULT_RETRIES,
    backoff: bool = True,
    label: str = "",
    caller_file: str = "",
    max_continuations: int = _DEFAULT_CONTINUATIONS,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Call an LLM model and parse the response as a JSON object.

    Retries on both empty content AND invalid/non-object JSON.
    Does not support offline_fallback (JSON mode only used in automated steps).

    When finish_reason=length, asks the model to re-output the COMPLETE JSON
    from scratch rather than appending the truncated fragment (which would
    produce invalid JSON with control characters or mismatched braces).

    Parameters
    ──────────
    Same as call_llm(), minus offline_fallback.

    Returns
    ───────
    (data: dict, cost: float)
      data — parsed JSON object.
      cost — cumulative USD cost across all attempts.
    """
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    last_exc: Exception | None = None
    total_cost = 0.0

    for attempt in range(1, retries + 1):
        try:
            resp = call_model(role, messages, **kwargs)

            cost        = _record_cost(resp, role, caller_file, label)
            total_cost += cost
            content     = _extract_text(resp)

            _guard_tool_calls(resp)

            if not content:
                raise RuntimeError("Model returned empty content.")

            # ── Auto-continue on truncation (JSON-aware) ──────────────────────
            if max_continuations > 0 and _get_finish_reason(resp) == "length":
                extra_content, extra_cost = _continue_on_length(
                    role, messages, resp, kwargs, max_continuations, caller_file, label,
                    json_mode=True,
                )
                total_cost += extra_cost
                content     = extra_content

            data = _parse_json_object(content)
            return data, total_cost

        except (RuntimeError, ValueError) as exc:
            last_exc = exc
            if attempt < retries:
                wait = (_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, _BACKOFF_JITTER) if backoff else _FLAT_RETRY_SLEEP
                print(
                    f"[call_llm][warn] {role} JSON attempt {attempt}/{retries} failed: {exc} "
                    f"— retrying in {wait:.1f}s …"
                )
                time.sleep(wait)
            else:
                print(f"[call_llm][warn] {role} JSON attempt {attempt}/{retries} failed: {exc} — giving up.")

        except Exception as exc:
            print(f"[call_llm][error] {role} JSON call failed: {exc}", file=sys.stderr)
            raise

    raise last_exc or RuntimeError(f"call_llm_json failed after {retries} retries.")


def call_llm_messages(
    role: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 8192,
    temperature: float | None = _DEFAULT_TEMPERATURE,
    retries: int = _DEFAULT_RETRIES,
    backoff: bool = True,
    label: str = "",
    caller_file: str = "",
    max_continuations: int = _DEFAULT_CONTINUATIONS,
) -> tuple[str, float]:
    """
    Low-level variant: accepts a pre-built messages list instead of system+user strings.

    Use when a script builds multi-turn conversation history or needs
    fine-grained control over message roles (e.g. executor file-by-file calls).

    Returns
    ───────
    (content: str, cost: float)
    """
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature

    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = call_model(role, messages, **kwargs)

            cost    = _record_cost(resp, role, caller_file, label)
            content = _extract_text(resp)

            _guard_tool_calls(resp)

            if not content:
                raise RuntimeError("Model returned empty content.")

            # ── Auto-continue on truncation ───────────────────────────────────
            if max_continuations > 0 and _get_finish_reason(resp) == "length":
                extra_content, extra_cost = _continue_on_length(
                    role, messages, resp, kwargs, max_continuations, caller_file, label,
                    json_mode=False,
                )
                return extra_content, cost + extra_cost

            return content, cost

        except RuntimeError as exc:
            last_exc = exc
            if attempt < retries:
                wait = (_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, _BACKOFF_JITTER) if backoff else _FLAT_RETRY_SLEEP
                print(
                    f"[call_llm][warn] {role} attempt {attempt}/{retries} failed: {exc} "
                    f"— retrying in {wait:.1f}s …"
                )
                time.sleep(wait)
            else:
                print(f"[call_llm][warn] {role} attempt {attempt}/{retries} failed: {exc} — giving up.")

        except Exception as exc:
            print(f"[call_llm][error] {role} messages call failed: {exc}", file=sys.stderr)
            raise

    raise last_exc or RuntimeError(f"call_llm_messages failed after {retries} retries.")