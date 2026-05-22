"""
modules/call_llm.py — Unified LLM call wrapper for all pipeline scripts.

Replaces the duplicated _call_llm / _call_model / _model_call boilerplate
found in every pipeline script with a single, consistent interface.

Features
────────
  • Retry on empty content  — configurable count, exponential backoff
  • Retry on tool_calls     — guard against model returning tool_calls instead of text
  • Offline fallback        — stdin paste when API key is missing
  • Cost tracking           — record_usage + print_call on every attempt
  • JSON mode               — parse + retry on invalid/non-object JSON
  • Consistent return type  — always (content: str, cost: float)

────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────

Plain text call (most scripts):

    from modules.call_llm import call_llm

    text, cost = call_llm(
        role        = ROLE,
        system      = SYSTEM_PROMPT,
        user        = user_msg,
        max_tokens  = 8192,
        caller_file = __file__,
    )

JSON call (scaffolder, planner, executor, debugger, judge, patcher):

    from modules.call_llm import call_llm_json

    data, cost = call_llm_json(
        role        = ROLE,
        system      = SYSTEM_PROMPT,
        user        = user_msg,
        max_tokens  = 32768,
        temperature = 0.1,
        caller_file = __file__,
        label       = "[07] full_plan",
    )

Multi-role scripts (debugger, patcher pass role as variable):

    text, cost = call_llm(
        role        = "debugger_secondary",
        system      = SYSTEM_PROMPT,
        user        = user_msg,
        caller_file = __file__,
        label       = "[09] secondary",
    )

────────────────────────────────────────────────────────────────────────
Migration cheatsheet
────────────────────────────────────────────────────────────────────────

Old pattern → new pattern:

  # enricher (returned tuple)
  content, cost = _call_llm(system, user, max_tokens=N)
  →
  content, cost = call_llm(ROLE, system, user, max_tokens=N, caller_file=__file__)

  # clarificator/specwright (returned str)
  content = _call_llm(system, user, max_tokens=N)
  →
  content, _ = call_llm(ROLE, system, user, max_tokens=N, caller_file=__file__)

  # debugger/patcher (role param, returned str)
  content = _model_call(role, messages, max_tokens=N)
  →
  content, _ = call_llm(role, system, user, max_tokens=N, caller_file=__file__)

  # scaffolder/planner/executor (JSON, inline retry)
  data = _call_and_parse(system, user, max_tokens=N, max_retries=5)
  →
  data, cost = call_llm_json(ROLE, system, user, max_tokens=N, retries=5, caller_file=__file__)
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

_DEFAULT_RETRIES     = 3
_DEFAULT_TEMPERATURE = None   # None = use model default
_BACKOFF_BASE        = 2.0    # seconds; actual wait = base^(attempt-1) + jitter
_BACKOFF_JITTER      = 1.0    # max random jitter seconds
_FLAT_RETRY_SLEEP    = 3.0    # used when backoff=False


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _extract_text(resp: Any) -> str:
    """Extract text content from a call_model response object."""
    content = getattr(resp.choices[0].message, "content", None)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            for part in content
        )
    return (content or "").strip()


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


def _parse_json_object(raw: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output.
    Strips markdown fences if present, then tries direct parse.
    Falls back to finding the outermost {...} block via regex.
    Raises ValueError if no valid JSON object is found.
    """
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())

    # Direct parse
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find outermost {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            return parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"No valid JSON object found in model output: {exc}") from exc

    raise ValueError("No JSON object found in model output.")


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
) -> tuple[str, float]:
    """
    Call an LLM model with automatic retry on empty content.

    Parameters
    ──────────
    role             : Role key registered in artifacts/models.py (e.g. "enricher").
    system           : System prompt string.
    user             : User message string.
    max_tokens       : Max completion tokens.
    temperature      : Sampling temperature. None = model default.
    retries          : Total attempts before giving up (default 3).
    backoff          : If True, use exponential backoff between retries.
                       If False, use flat 3s sleep (matches old debugger/patcher behaviour).
    label            : Optional label appended to print_call output (e.g. "[09] secondary").
    caller_file      : Pass __file__ from the calling script for accurate cost log paths.
    offline_fallback : If True, prompt user to paste response when API key is missing.

    Returns
    ───────
    (content: str, cost: float)
      content — model response text, stripped.
      cost    — USD cost from record_usage(); 0.0 if unavailable.
    """
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature

    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = call_model(
                role,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                **kwargs,
            )

            cost    = _record_cost(resp, role, caller_file, label)
            content = _extract_text(resp)

            # Guard: model must not return tool_calls when we expect text
            _guard_tool_calls(resp)

            if not content:
                raise RuntimeError("Model returned empty content.")

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
            # Non-retryable: network error, auth, etc.
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
) -> tuple[dict[str, Any], float]:
    """
    Call an LLM model and parse the response as a JSON object.

    Retries on both empty content AND invalid/non-object JSON.
    Does not support offline_fallback (JSON mode only used in automated steps).

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

    last_exc: Exception | None = None
    total_cost = 0.0

    for attempt in range(1, retries + 1):
        try:
            resp = call_model(
                role,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                **kwargs,
            )

            cost        = _record_cost(resp, role, caller_file, label)
            total_cost += cost
            content     = _extract_text(resp)

            # Guard: model must not return tool_calls when we expect JSON text
            _guard_tool_calls(resp)

            if not content:
                raise RuntimeError("Model returned empty content.")

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

            # Guard: model must not return tool_calls when we expect text
            _guard_tool_calls(resp)

            if not content:
                raise RuntimeError("Model returned empty content.")

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
