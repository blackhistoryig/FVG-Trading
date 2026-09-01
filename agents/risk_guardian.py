"""
Risk Guardian — thin LLM wrapper for the options-trading AI agent stack.

Deterministic math/execution lives elsewhere. This file:
  1. Runs hard-limit checks in PLAIN CODE first (no LLM in the loop for vetoes).
  2. If clear, calls Groq for a reasoned APPROVE/APPROVE_MODIFIED/VETO.
  3. Validates the LLM response into RiskGuardianOutput, with one retry on
     parse failure and fail-closed (VETO) if that retry also fails.

NOTE: field names for ScoutOutput/RiskGuardianOutput below were inferred from
spec (symbol, direction, position_size_contracts, final_max_loss_usd,
veto_reason) and NOT independently verified against the live agent_schemas.py
content at push time (tooling limitation). Verify against the actual pydantic
models before first real run.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Union

from pydantic import ValidationError

from agents.agent_schemas import (
    RiskGuardianOutput,
    ScoutOutput,
    RiskDecision,
    RISK_GUARDIAN_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
API_TIMEOUT_SECONDS = 10
API_MAX_RETRIES = 1  # network/HTTP-level retry on the call itself


def _get_groq_client():
    """Lazy import + init so this module can be imported without the SDK
    installed (e.g. during unit tests that only exercise the guardrails)."""
    from openai import OpenAI

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. In Colab: "
            "os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')"
        )
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        timeout=API_TIMEOUT_SECONDS,
        max_retries=API_MAX_RETRIES,
    )


# ---------------------------------------------------------------------------
# Account / risk context passed in alongside the Scout proposal
# ---------------------------------------------------------------------------

@dataclass
class AccountRiskContext:
    daily_pnl_usd: float
    open_position_count: int
    per_symbol_exposure_usd: dict = field(default_factory=dict)
    max_daily_loss_usd: float = 500.0
    max_concurrent_positions: int = 5
    max_symbol_exposure_usd: float = 1000.0


# ---------------------------------------------------------------------------
# Deterministic hard-limit checks — run BEFORE any LLM call
# ---------------------------------------------------------------------------

def check_hard_limits(
    scout: ScoutOutput, ctx: AccountRiskContext
) -> Optional[str]:
    """Returns a veto_reason string if a hard limit is breached, else None.
    This function must never call the LLM and must be pure/deterministic."""

    if ctx.daily_pnl_usd <= -abs(ctx.max_daily_loss_usd):
        return (
            f"max_daily_loss_breached: daily_pnl_usd={ctx.daily_pnl_usd:.2f} "
            f"<= -{ctx.max_daily_loss_usd:.2f}"
        )

    if ctx.open_position_count >= ctx.max_concurrent_positions:
        return (
            f"max_concurrent_positions_breached: "
            f"open={ctx.open_position_count} >= limit={ctx.max_concurrent_positions}"
        )

    symbol = scout.symbol
    current_exposure = ctx.per_symbol_exposure_usd.get(symbol, 0.0)
    if current_exposure >= ctx.max_symbol_exposure_usd:
        return (
            f"symbol_exposure_limit_breached: symbol={symbol} "
            f"exposure_usd={current_exposure:.2f} >= limit={ctx.max_symbol_exposure_usd:.2f}"
        )

    return None


def _veto_output(reason: str) -> RiskGuardianOutput:
    """Builds a fail-closed VETO output. Field names beyond decision/
    veto_reason are left at whatever safe defaults your schema requires
    (e.g. position_size_contracts=0). Adjust here if your validators need
    additional fields populated on VETO."""
    return RiskGuardianOutput(
        decision=RiskDecision.VETO,
        veto_reason=reason,
        position_size_contracts=0,
        final_max_loss_usd=0.0,
    )


# ---------------------------------------------------------------------------
# LLM call + validation
# ---------------------------------------------------------------------------

def _build_user_message(scout: ScoutOutput, ctx: AccountRiskContext) -> str:
    scout_dict = scout.model_dump() if hasattr(scout, "model_dump") else dict(scout)
    payload = {
        "scout_proposal": scout_dict,
        "account_risk_context": {
            "daily_pnl_usd": ctx.daily_pnl_usd,
            "open_position_count": ctx.open_position_count,
            "per_symbol_exposure_usd": ctx.per_symbol_exposure_usd,
            "max_daily_loss_usd": ctx.max_daily_loss_usd,
            "max_concurrent_positions": ctx.max_concurrent_positions,
            "max_symbol_exposure_usd": ctx.max_symbol_exposure_usd,
        },
    }
    return (
        "Evaluate this trade proposal against the account risk context. "
        "Respond with ONLY a single JSON object matching the required schema "
        "(decision, veto_reason, position_size_contracts, final_max_loss_usd, "
        "and any other required fields).\n\n"
        f"{json.dumps(payload, default=str)}"
    )


def _call_groq(messages: list) -> str:
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=600,
    )
    return resp.choices[0].message.content


def _parse_llm_json(raw: str) -> RiskGuardianOutput:
    data = json.loads(raw)
    return RiskGuardianOutput(**data)


def evaluate_trade(
    scout: Union[ScoutOutput, dict], ctx: AccountRiskContext
) -> RiskGuardianOutput:
    if isinstance(scout, dict):
        scout = ScoutOutput(**scout)

    veto_reason = check_hard_limits(scout, ctx)
    if veto_reason is not None:
        return _veto_output(veto_reason)

    user_msg = _build_user_message(scout, ctx)
    messages = [
        {"role": "system", "content": RISK_GUARDIAN_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = _call_groq(messages)
    except Exception as e:
        return _veto_output(f"llm_call_failed: {e}")

    try:
        return _parse_llm_json(raw)
    except (json.JSONDecodeError, ValidationError) as first_err:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "That response was not valid JSON matching the required "
                    "schema. Return ONLY valid JSON, with no markdown fences, "
                    "no commentary — just the JSON object with the exact "
                    f"required fields. Error was: {first_err}"
                ),
            },
        ]
        try:
            raw_retry = _call_groq(retry_messages)
            return _parse_llm_json(raw_retry)
        except Exception:
            return _veto_output("llm_output_invalid")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_scout = ScoutOutput(
        symbol="SPY",
        direction="BUY",
        signal_strength=0.72,
        entry_price=520.15,
        proposed_contract="SPY241206C00521000",
        rationale="Bullish MSS + displacement FVG on 15m, confluent with 1h bias.",
    )

    print("=== Case A: hard limit breached (max daily loss) — should auto-VETO, no LLM call ===")
    breached_ctx = AccountRiskContext(
        daily_pnl_usd=-600.0,
        open_position_count=1,
        per_symbol_exposure_usd={"SPY": 200.0},
        max_daily_loss_usd=500.0,
    )
    result_a = evaluate_trade(example_scout, breached_ctx)
    print(result_a.model_dump_json(indent=2))
    assert result_a.decision == RiskDecision.VETO

    print("\n=== Case B: normal case — should go through to Groq ===")
    if not GROQ_API_KEY:
        print(
            "SKIPPED: GROQ_API_KEY not set in this environment. "
            "In Colab, run:\n"
            "  from google.colab import userdata\n"
            "  import os\n"
            "  os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')\n"
            "then re-run this smoke test."
        )
    else:
        normal_ctx = AccountRiskContext(
            daily_pnl_usd=-50.0,
            open_position_count=1,
            per_symbol_exposure_usd={"SPY": 200.0},
            max_daily_loss_usd=500.0,
        )
        result_b = evaluate_trade(example_scout, normal_ctx)
        print(result_b.model_dump_json(indent=2))
