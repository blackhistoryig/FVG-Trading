"""
agents/scout.py

Scout: thin LLM wrapper that explains and contextualizes a raw FVG signal
already produced by the deterministic core. Scout does NOT invent trade
ideas, does NOT decide position sizing, and does NOT have veto power --
it only proposes a ScoutOutput that Risk Guardian will judge.

Design mirrors risk_guardian.py's pattern for symmetry:
- Deterministic input in, LLM call, validated pydantic output out.
- Fail closed on parse/validation errors (retry once, then raise/flag
  rather than silently returning garbage upstream).
- No math, no execution logic here -- that lives in the FVG core /
  Position Monitor.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from groq import Groq
from pydantic import ValidationError

from agent_schemas import (
    Direction,
    FVGContext,
    OptionsStrategy,
    ScoutOutput,
    SCOUT_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Config -- keep provider/model swap-able in case a free-tier model
# rate-limits or degrades close to the deadline.
#
# NOTE (checked 2026-09-01): Groq deprecated llama-3.1-8b-instant and
# llama-3.3-70b-versatile on the Free/Developer tier as of 2026-08-16.
# Default here is openai/gpt-oss-120b, Groq's own recommended
# replacement and still free-tier eligible. If this also gets rotated
# by the deadline, check https://console.groq.com/docs/models for the
# current list -- a deprecated model ID fails as a clean 400, but it's
# easy to mistake for a bad API key if you don't check the model list
# first.
# ---------------------------------------------------------------------------

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REQUEST_TIMEOUT_SECONDS = 10
MAX_API_RETRIES = 1  # one retry on the network/API call itself
MAX_PARSE_RETRIES = 1  # one retry with a stricter follow-up on bad JSON


def _get_groq_client() -> Groq:
    """
    Colab gotcha: GROQ_API_KEY must already be in os.environ before this
    module is imported/run. In Colab, load it explicitly first:

        from google.colab import userdata
        import os
        os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

    userdata.get() does NOT auto-populate os.environ -- forgetting this
    line is the single most common reason this client fails with an
    auth error that looks like a bad key when the key is actually fine.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not found in environment. In Colab: "
            "os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY') "
            "before importing scout.py."
        )
    return Groq(api_key=GROQ_API_KEY)


def _build_user_message(raw_signal: dict[str, Any]) -> str:
    """Serialize the raw FVG signal dict for the LLM. Kept as a separate
    function so the prompt-assembly logic is easy to unit test / tweak
    without touching the API-calling logic.

    NOTE (2026-09-01): openai/gpt-oss-120b was observed inventing its
    own field names (suggested_stop_loss, suggested_max_hold_time
    instead of the schema's recommended_max_loss_usd,
    recommended_max_hold_hours) and putting a free-text description
    into suggested_strategy instead of the exact enum value. Spelling
    out every required key and the exact enum strings here (rather than
    relying on the system prompt alone) fixes this -- same root cause
    and same fix pattern as risk_guardian.py's earlier bug."""
    signal_id = raw_signal.get("signal_id", "UNKNOWN")
    return (
        "Raw FVG signal from the deterministic core (JSON):\n\n"
        f"{json.dumps(raw_signal, default=str, indent=2)}\n\n"
        "Respond with a single JSON object with EXACTLY these keys (use these "
        "EXACT key names, not synonyms):\n"
        f'  "signal_id": copy this exact string: "{signal_id}"\n'
        '  "symbol": string, copy from the input signal\n'
        '  "direction": must be EXACTLY "BUY" or "SELL", copy from the input signal\n'
        '  "underlying_price": number, copy from the input signal\n'
        '  "fvg_context": object, copy the fvg_context object from the input signal '
        "unchanged\n"
        '  "thesis": string, max 500 characters, 2-3 sentence plain-language rationale\n'
        '  "market_context": string, max 300 characters\n'
        '  "confidence_score": number between 0.0 and 1.0\n'
        '  "suggested_strategy": must be EXACTLY "debit_call_spread" or '
        '"debit_put_spread" -- a short enum code, NOT a sentence or description '
        '(e.g. NOT "Long SPY call spread targeting 452" -- just the exact code '
        '"debit_call_spread")\n'
        '  "suggested_expiration_bias": string, e.g. "nearest_liquid_weekly" '
        "(optional, has a default)\n"
        '  "risk_flags": array of short strings, e.g. ["earnings_this_week"] '
        "(optional, can be empty array)\n"
        '  "recommended_max_loss_usd": number > 0 -- use THIS EXACT key name, '
        'NOT "suggested_stop_loss" or any other name\n'
        '  "recommended_max_hold_hours": number > 0, in HOURS -- use THIS EXACT '
        'key name, NOT "suggested_max_hold_time" or any other name, and do not '
        'use a string like "48h", use a plain number like 48\n\n'
        "No markdown, no code fences, no commentary outside the JSON object. "
        "Do not add extra keys beyond what is listed above. Use the exact key "
        "names given -- do not invent synonyms."
    )


def _call_groq(client: Groq, messages: list[dict[str, str]]) -> str:
    """One API call with a short timeout and one retry on transient
    failure (network blip, 429, 5xx). Raises on final failure so the
    caller can decide how to fail closed."""
    last_err: Optional[Exception] = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,  # low temp: this is explanation, not creative generation
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see retry loop
            last_err = exc
            if attempt < MAX_API_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
    raise RuntimeError(f"Groq API call failed after retries: {last_err}") from last_err


def _parse_scout_output(raw_text: str) -> ScoutOutput:
    """Parse raw LLM text into a validated ScoutOutput. Raises
    (ValueError/ValidationError/json.JSONDecodeError) on failure --
    caller handles the retry-then-fail-closed decision."""
    data = json.loads(raw_text)
    return ScoutOutput.model_validate(data)


def run_scout(
    raw_signal: dict[str, Any],
    client: Optional[Groq] = None,
) -> ScoutOutput:
    """
    Main entry point. Takes a raw FVG signal dict from the deterministic
    core, returns a validated ScoutOutput.

    raw_signal is expected to contain at least the fields needed to
    construct FVGContext plus signal_id / symbol / direction /
    underlying_price -- but Scout is told to return the FULL ScoutOutput
    shape (thesis, market_context, confidence_score, etc.), so we parse
    the model's JSON directly into ScoutOutput rather than merging
    fields manually.

    Fails closed: if the LLM output can't be parsed/validated after one
    stricter retry, raises RuntimeError rather than returning a
    fabricated or partially-filled ScoutOutput. Risk Guardian and the
    Executor should never receive a Scout object that didn't pass
    schema validation.
    """
    client = client or _get_groq_client()

    messages = [
        {"role": "system", "content": SCOUT_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(raw_signal)},
    ]

    raw_text = _call_groq(client, messages)

    try:
        return _parse_scout_output(raw_text)
    except (json.JSONDecodeError, ValidationError, ValueError) as first_err:
        # One stricter retry: tell the model exactly what went wrong.
        messages.append({"role": "assistant", "content": raw_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response did not parse as valid JSON matching "
                    "the ScoutOutput schema. Error: "
                    f"{first_err}\n\n"
                    'Remember: "suggested_strategy" must be EXACTLY "debit_call_spread" '
                    'or "debit_put_spread" (a short code, not a sentence). Use the exact '
                    'key names "recommended_max_loss_usd" and "recommended_max_hold_hours" '
                    "(plain numbers, not strings like \"48h\"). "
                    "Return ONLY a single valid JSON object matching ScoutOutput. "
                    "No markdown, no code fences, no explanation text."
                ),
            }
        )
        retry_text = _call_groq(client, messages)
        try:
            return _parse_scout_output(retry_text)
        except (json.JSONDecodeError, ValidationError, ValueError) as second_err:
            raise RuntimeError(
                "Scout output failed schema validation twice. "
                f"First error: {first_err}. Second error: {second_err}. "
                f"Last raw response: {retry_text!r}"
            ) from second_err


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime, timezone

    example_raw_signal = {
        "signal_id": "SPY-20260901-1",
        "symbol": "SPY",
        "direction": "BUY",
        "underlying_price": 449.10,
        "fvg_context": {
            "gap_type": "bullish",
            "mss_confirmed": True,
            "displacement_strength": 1.8,
            "measured_move_target": 452.30,
            "entry_bar_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    print(f"Calling Groq ({GROQ_MODEL}) with example FVG signal...")
    try:
        result = run_scout(example_raw_signal)
        print(result.model_dump_json(indent=2))
        print("Scout smoke test passed: valid ScoutOutput returned.")
    except RuntimeError as e:
        print(f"Scout smoke test FAILED: {e}")
