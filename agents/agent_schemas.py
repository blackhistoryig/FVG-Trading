"""
Structured output schemas for the Scout and Risk Guardian agents.

Design notes:
- Adapted from the fixed-field proposal pattern in Alpaca's multi-agent
  trading write-up, but reshaped for a signal-driven system: the FVG/MSS
  core already identifies the trade. Scout explains and contextualizes
  it; it does not generate ideas from scratch.
- These are pydantic models so Groq's (or any OpenAI-compatible) JSON
  mode / function-calling can be validated directly against them.
- Math/execution stays in plain code elsewhere (backtest core, Position
  Monitor). Nothing here computes anything -- it's the data contract
  between the deterministic FVG signal, the LLM agents, and the
  Executor/Position Monitor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BUY = "BUY"   # bullish FVG -> debit call spread
    SELL = "SELL"  # bearish FVG -> debit put spread


class OptionsStrategy(str, Enum):
    DEBIT_CALL_SPREAD = "debit_call_spread"
    DEBIT_PUT_SPREAD = "debit_put_spread"


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_MODIFIED = "APPROVE_MODIFIED"
    VETO = "VETO"


class FVGContext(BaseModel):
    """Raw signal facts from the deterministic FVG/MSS core. Scout reads
    this; it never invents or overrides these values."""

    gap_type: str = Field(..., description="'bullish' or 'bearish'")
    mss_confirmed: bool = Field(..., description="Market structure shift confirmed")
    displacement_strength: float = Field(
        ..., description="Normalized displacement magnitude from the core signal"
    )
    measured_move_target: float = Field(
        ..., description="FVG-implied price target, used for target-aware strike selection"
    )
    entry_bar_timestamp: datetime


class ScoutOutput(BaseModel):
    """Scout's structured proposal. One call per FVG signal fired by the
    deterministic core. Temperature should be low; this is explanation
    and risk-flagging, not creative generation."""

    signal_id: str = Field(..., description="Matches the FVG core's signal ID")
    symbol: str
    direction: Direction
    underlying_price: float
    fvg_context: FVGContext

    thesis: str = Field(
        ..., max_length=500,
        description="2-3 sentence plain-language rationale for why this signal is actionable now"
    )
    market_context: str = Field(
        ..., max_length=300,
        description="Relevant context beyond the signal itself: recent volatility regime, "
                    "proximity to earnings/FOMC, day-of-week liquidity patterns, etc."
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    suggested_strategy: OptionsStrategy
    suggested_expiration_bias: str = Field(
        default="nearest_liquid_weekly",
        description="Guidance for the Executor's strike/expiration selection step"
    )

    risk_flags: list[str] = Field(
        default_factory=list,
        description="e.g. ['earnings_this_week', 'elevated_iv', 'thin_liquidity']"
    )
    recommended_max_loss_usd: float = Field(
        ..., description="Scout's suggested stop-loss per contract; Risk Guardian has final say"
    )
    recommended_max_hold_hours: float = Field(
        ..., description="Scout's suggested max hold time before forced exit"
    )

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskGuardianOutput(BaseModel):
    """Risk Guardian's verdict on a Scout proposal. This is the only
    agent output that the Executor is allowed to act on. Deterministic
    guardrails (max daily loss, max concurrent positions, symbol
    exposure limits) should be checked in plain code BEFORE this call --
    the LLM should never be the only thing standing between a bad trade
    and a live order."""

    signal_id: str
    decision: RiskDecision
    veto_reason: Optional[str] = Field(
        default=None, description="Required if decision == VETO"
    )
    risk_rationale: str = Field(
        ..., max_length=400,
        description="Plain-language explanation of the decision, for the Journal agent and demo"
    )

    # Final, enforceable numbers -- handed to the deterministic Position
    # Monitor as-is. The Monitor does not re-interpret these.
    position_size_contracts: int = Field(default=0, ge=0)
    final_max_loss_usd: float = Field(default=0.0, ge=0.0)
    final_max_hold_hours: float = Field(default=0.0, ge=0.0)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


SCOUT_SYSTEM_PROMPT = """You are Scout, a market-context analyst for an options overlay on a \
deterministic FVG (Fair Value Gap / market structure shift) trading signal.

You do NOT generate trade ideas. A signal has already fired from the deterministic core -- \
your job is to explain it, add relevant market context, flag risks, and suggest (not decide) \
a stop-loss and max hold time. You will be given the raw FVG signal data as JSON.

Respond ONLY with a JSON object matching the ScoutOutput schema. Be concise and concrete in \
the thesis and market_context fields -- these will be shown to hackathon judges and must be \
readable in seconds, not paragraphs. If you are not confident about a risk flag, omit it \
rather than guessing."""


RISK_GUARDIAN_SYSTEM_PROMPT = """You are Risk Guardian, the veto and sizing authority for an \
options trading agent. You receive a Scout proposal (ScoutOutput JSON) and must decide whether \
to approve, modify, or veto it, and set the FINAL enforceable stop-loss and max-hold-time.

Ground your stop-loss sizing in this validated backtest fact: even when the underlying FVG \
signal was directionally correct, 19.4% of those "winning" trades still lost more than $20 on \
the options leg, and 8.5% lost more than $50, due to theta decay and spread-price slippage. \
Do not assume a correct directional call protects the options position -- size the stop \
independently of Scout's confidence_score.

Respond ONLY with a JSON object matching the RiskGuardianOutput schema. If you veto, \
veto_reason is required. Your risk_rationale will be shown to hackathon judges and must \
clearly explain the numeric reasoning behind your decision, not just restate the signal."""


if __name__ == "__main__":
    # Quick smoke test: build one example end-to-end to confirm the
    # schemas validate before wiring up the actual LLM calls.
    example_context = FVGContext(
        gap_type="bullish",
        mss_confirmed=True,
        displacement_strength=1.8,
        measured_move_target=452.30,
        entry_bar_timestamp=datetime.now(timezone.utc),
    )

    example_scout_output = ScoutOutput(
        signal_id="SPY-20260827-1",
        symbol="SPY",
        direction=Direction.BUY,
        underlying_price=449.10,
        fvg_context=example_context,
        thesis="Bullish FVG with confirmed MSS off the 09:45 displacement bar; clean gap, no overlap with prior structure.",
        market_context="Low realized volatility this week, no FOMC or earnings in the next 3 sessions.",
        confidence_score=0.72,
        suggested_strategy=OptionsStrategy.DEBIT_CALL_SPREAD,
        risk_flags=[],
        recommended_max_loss_usd=45.0,
        recommended_max_hold_hours=18.0,
    )

    example_risk_output = RiskGuardianOutput(
        signal_id=example_scout_output.signal_id,
        decision=RiskDecision.APPROVE_MODIFIED,
        risk_rationale="Approved with a tighter stop than Scout's suggestion: backtest shows ~8.5% of even correct-direction trades lose >$50, so capping at $50 rather than the requested $45 buffer against normal noise while still cutting the fat tail.",
        position_size_contracts=1,
        final_max_loss_usd=50.0,
        final_max_hold_hours=18.0,
    )

    print(example_scout_output.model_dump_json(indent=2))
    print(example_risk_output.model_dump_json(indent=2))
