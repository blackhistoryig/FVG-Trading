"""
agents/risk_guardian.py

Risk Guardian: the only agent output the Executor is allowed to act on.
Reviews a Scout proposal (ScoutOutput) against account/risk context and
either approves, approves-with-modification, or vetoes the trade, then
sets the FINAL enforceable stop-loss and max-hold-time.

Design principle: the LLM is never the sole safety layer. Hard
deterministic guardrails (max daily loss, max concurrent positions,
per-symbol exposure) are checked in PLAIN CODE first. If any hard
limit is breached, we short-circuit to a VETO and skip the LLM call
entirely -- no ambiguity, no wasted API call, no chance of the model
talking itself into approving something the account can't safely take.

Only when all hard limits pass do we call Groq for the qualitative
risk read (sizing nuance, thesis-quality check, stop-loss reasoning
grounded in the backtest's "right direction, still lost big" finding).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from groq import Groq
from pydantic import ValidationError

from agent_schemas import (
    RiskDecision,
    RiskGuardianOutput,
    ScoutOutput,
    RISK_GUARDIAN_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Config
#
# NOTE (checked 2026-09-01): Groq deprecated llama-3.1-8b-instant and
# llama-3.3-70b-versatile on the Free/Developer tier as of 2026-08-16.
# Default here is openai/gpt-oss-120b, matching agents/scout.py. If this
# also gets rotated, check https://console.groq.com/docs/models before
# assuming it's an auth error.
# ---------------------------------------------------------------------------

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
REQUEST_TIMEOUT_SECONDS = 10
MAX_API_RETRIES = 1
MAX_PARSE_RETRIES = 1


# ---------------------------------------------------------------------------
# Deterministic hard limits -- checked BEFORE any LLM call.
# These numbers are placeholders; tune them to your actual risk
# tolerance for the hackathon paper account.
# ---------------------------------------------------------------------------

MAX_DAILY_LOSS_USD = 500.0
MAX_CONCURRENT_POSITIONS = 5
MAX_PER_SYMBOL_EXPOSURE_USD = 1000.0


@dataclass
class AccountRiskContext:
    """Snapshot of account/risk state at decision time. The Executor or
    Position Monitor is responsible for keeping this accurate -- Risk
    Guardian only reads it."""

    current_daily_pnl_usd: float
    open_position_count: int
    current_symbol_exposure_usd: float  # exposure already open in this signal's symbol
    proposed_trade_cost_usd: float  # estimated debit for the proposed spread


def _check_hard_limits(ctx: AccountRiskContext) -> Optional[str]:
    """Returns a veto_reason string if a hard limit is breached, else
    None. Plain code, no LLM involved -- this is the safety layer that
    must hold even if the LLM call fails, hangs, or hallucinates."""
    if ctx.current_daily_pnl_usd <= -MAX_DAILY_LOSS_USD:
        return (
            f"Daily loss limit breached: current daily P&L "
            f"${ctx.current_daily_pnl_usd:.2f} <= -${MAX_DAILY_LOSS_USD:.2f} cap."
        )
    if ctx.open_position_count >= MAX_CONCURRENT_POSITIONS:
        return (
            f"Max concurrent positions reached: {ctx.open_position_count} "
            f">= {MAX_CONCURRENT_POSITIONS} cap."
        )
    projected_symbol_exposure = ctx.current_symbol_exposure_usd + ctx.proposed_trade_cost_usd
    if projected_symbol_exposure > MAX_PER_SYMBOL_EXPOSURE_USD:
        return (
            f"Per-symbol exposure cap breached: projected exposure "
            f"${projected_symbol_exposure:.2f} > ${MAX_PER_SYMBOL_EXPOSURE_USD:.2f} cap."
        )
    return None


# ---------------------------------------------------------------------------
# Groq call plumbing (mirrors agents/scout.py for consistency)
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    """
    Colab gotcha (not relevant in Codespaces, but kept for parity with
    scout.py): GROQ_API_KEY must be in os.environ before this runs. In
    Colab: os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY").
    In Codespaces: export GROQ_API_KEY=... in the terminal, or set it
    as a Codespaces secret in repo settings.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not found in environment. Set it with "
            "export GROQ_API_KEY=... in the terminal (Codespaces) or "
            "via userdata.get() (Colab) before running risk_guardian.py."
        )
    return Groq(api_key=GROQ_API_KEY)


def _build_user_message(scout_output: ScoutOutput, ctx: AccountRiskContext) -> str:
    payload = {
        "scout_proposal": json.loads(scout_output.model_dump_json()),
        "account_risk_context": {
            "current_daily_pnl_usd": ctx.current_daily_pnl_usd,
            "open_position_count": ctx.open_position_count,
            "current_symbol_exposure_usd": ctx.current_symbol_exposure_usd,
            "proposed_trade_cost_usd": ctx.proposed_trade_cost_usd,
        },
    }
    return (
        "Scout proposal and current account risk context (JSON):\n\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Respond with a single JSON object matching the RiskGuardianOutput schema. "
        "No markdown, no code fences, no commentary outside the JSON object."
    )


def _call_groq(client: Groq, messages: list[dict[str, str]]) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,  # even lower than Scout -- this is a risk verdict, not prose
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < MAX_API_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
    raise RuntimeError(f"Groq API call failed after retries: {last_err}") from last_err


def _parse_risk_output(raw_text: str) -> RiskGuardianOutput:
    data = json.loads(raw_text)
    return RiskGuardianOutput.model_validate(data)


def _fail_closed_veto(signal_id: str, reason: str) -> RiskGuardianOutput:
    """Construct a safe, schema-valid VETO when we can't get (or trust)
    an LLM verdict. Never return an unvalidated or fabricated APPROVE."""
    return RiskGuardianOutput(
        signal_id=signal_id,
        decision=RiskDecision.VETO,
        veto_reason=reason,
        risk_rationale=reason,
        position_size_contracts=0,
        final_max_loss_usd=0.0,
        final_max_hold_hours=0.0,
    )


def run_risk_guardian(
    scout_output: ScoutOutput,
    account_context: AccountRiskContext,
    client: Optional[Groq] = None,
) -> RiskGuardianOutput:
    """
    Main entry point. Order of operations:
      1. Deterministic hard-limit check (plain code, no LLM).
         If breached -> immediate VETO, no API call made.
      2. Otherwise, call Groq for the qualitative risk verdict.
      3. Validate into RiskGuardianOutput; retry once on bad JSON.
      4. If both LLM attempts fail to produce valid output -> fail
         closed with VETO. The Executor must never receive anything
         other than a schema-valid RiskGuardianOutput.
    """
    hard_limit_breach = _check_hard_limits(account_context)
    if hard_limit_breach is not None:
        return _fail_closed_veto(scout_output.signal_id, hard_limit_breach)

    client = client or _get_groq_client()

    messages = [
        {"role": "system", "content": RISK_GUARDIAN_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(scout_output, account_context)},
    ]

    try:
        raw_text = _call_groq(client, messages)
    except RuntimeError as api_err:
        return _fail_closed_veto(
            scout_output.signal_id, f"llm_call_failed: {api_err}"
        )

    try:
        return _parse_risk_output(raw_text)
    except (json.JSONDecodeError, ValidationError, ValueError) as first_err:
        messages.append({"role": "assistant", "content": raw_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response did not parse as valid JSON matching "
                    "the RiskGuardianOutput schema. Error: "
                    f"{first_err}\n\n"
                    "Return ONLY a single valid JSON object matching RiskGuardianOutput. "
                    "If you intend to VETO, veto_reason is required. "
                    "No markdown, no code fences, no explanation text."
                ),
            }
        )
        try:
            retry_text = _call_groq(client, messages)
        except RuntimeError as api_err:
            return _fail_closed_veto(
                scout_output.signal_id, f"llm_retry_call_failed: {api_err}"
            )
        try:
            return _parse_risk_output(retry_text)
        except (json.JSONDecodeError, ValidationError, ValueError) as second_err:
            return _fail_closed_veto(
                scout_output.signal_id,
                "llm_output_invalid: failed schema validation twice "
                f"(first: {first_err}; second: {second_err})",
            )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import datetime, timezone
    from agent_schemas import Direction, FVGContext, OptionsStrategy

    example_scout = ScoutOutput(
        signal_id="SPY-20260901-1",
        symbol="SPY",
        direction=Direction.BUY,
        underlying_price=449.10,
        fvg_context=FVGContext(
            gap_type="bullish",
            mss_confirmed=True,
            displacement_strength=1.8,
            measured_move_target=452.30,
            entry_bar_timestamp=datetime.now(timezone.utc),
        ),
        thesis="Bullish FVG with confirmed MSS, confluent with 1h bias.",
        market_context="Low realized volatility, no FOMC/earnings this week.",
        confidence_score=0.72,
        suggested_strategy=OptionsStrategy.DEBIT_CALL_SPREAD,
        recommended_max_loss_usd=45.0,
        recommended_max_hold_hours=18.0,
    )

    print("=== Case 1: hard daily-loss limit breached -> expect auto-VETO, no LLM call ===")
    breached_ctx = AccountRiskContext(
        current_daily_pnl_usd=-600.0,  # already past -$500 cap
        open_position_count=1,
        current_symbol_exposure_usd=0.0,
        proposed_trade_cost_usd=200.0,
    )
    result_breach = run_risk_guardian(example_scout, breached_ctx)
    print(result_breach.model_dump_json(indent=2))
    assert result_breach.decision == RiskDecision.VETO
    assert result_breach.veto_reason is not None
    print("Case 1 passed: hard limit correctly forced a VETO with no LLM call.\n")

    print(f"=== Case 2: normal case, calling Groq ({GROQ_MODEL}) ===")
    normal_ctx = AccountRiskContext(
        current_daily_pnl_usd=-50.0,
        open_position_count=1,
        current_symbol_exposure_usd=0.0,
        proposed_trade_cost_usd=200.0,
    )
    try:
        result_normal = run_risk_guardian(example_scout, normal_ctx)
        print(result_normal.model_dump_json(indent=2))
        print("Case 2 passed: valid RiskGuardianOutput returned.")
    except Exception as e:  # noqa: BLE001 -- smoke test, want to see any failure clearly
        print(f"Case 2 FAILED: {e}")
