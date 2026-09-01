"""
agents/pipeline.py

First real Scout -> Risk Guardian chain. Takes one raw FVG signal dict
from the deterministic core, runs it through Scout to get a proposal,
then runs that proposal through Risk Guardian to get the final,
enforceable verdict. This is the core loop the Executor will eventually
wrap (Executor adds: place the actual order via Alpaca CLI, log to the
Journal agent, and hand off to the Position Monitor).

Deliberately kept as a standalone script (not yet folded into
executor.py) so it can be run and demoed on its own -- this is the
first end-to-end proof that Scout's output feeds Risk Guardian's input
correctly, which is the core "agent pipeline" claim for the submission.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from agent_schemas import RiskDecision
from scout import run_scout
from risk_guardian import AccountRiskContext, run_risk_guardian


def run_pipeline(
    raw_signal: dict[str, Any],
    account_context: AccountRiskContext,
) -> dict[str, Any]:
    """
    Runs the full Scout -> Risk Guardian chain for one raw FVG signal.

    Returns a dict with both intermediate and final results so a caller
    (or a demo script) can print/log the full reasoning chain, not just
    the final verdict. This is deliberately verbose -- for a hackathon
    demo, showing the full chain (raw signal -> Scout's thesis -> Risk
    Guardian's verdict) is the point, not just the end result.

    Design note: Scout has no veto power and Risk Guardian is the only
    output the Executor may act on. If Scout's call fails outright
    (e.g. bad API key, network error), this raises -- there is no
    "proceed without Scout" path, since Risk Guardian needs a real
    proposal to evaluate. If Risk Guardian's call fails, it fails
    closed to VETO internally (see risk_guardian.py) rather than
    raising, so run_risk_guardian() itself never throws for LLM-side
    failures -- only for programming errors.
    """
    print(f"--- Step 1: Scout evaluates raw FVG signal ({raw_signal.get('signal_id')}) ---")
    scout_output = run_scout(raw_signal)
    print(scout_output.model_dump_json(indent=2))
    print()

    print("--- Step 2: Risk Guardian reviews Scout's proposal ---")
    risk_output = run_risk_guardian(scout_output, account_context)
    print(risk_output.model_dump_json(indent=2))
    print()

    final_action = (
        "PLACE ORDER" if risk_output.decision in (RiskDecision.APPROVE, RiskDecision.APPROVE_MODIFIED)
        else "NO TRADE (vetoed)"
    )
    print(f"--- Final action: {final_action} ---")

    return {
        "raw_signal": raw_signal,
        "scout_output": scout_output,
        "risk_output": risk_output,
        "final_action": final_action,
    }


# ---------------------------------------------------------------------------
# Smoke test / demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_raw_signal = {
        "signal_id": f"SPY-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
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

    # Placeholder account state -- in the live bot, the Executor/Position
    # Monitor will supply real numbers pulled from the Alpaca CLI
    # (current day's realized P&L, open position count, per-symbol
    # exposure). Kept as a simple, editable literal here for demo runs.
    demo_account_context = AccountRiskContext(
        current_daily_pnl_usd=-50.0,
        open_position_count=1,
        current_symbol_exposure_usd=0.0,
        proposed_trade_cost_usd=200.0,
    )

    print("=== Scout -> Risk Guardian pipeline demo ===\n")
    result = run_pipeline(example_raw_signal, demo_account_context)

    assert result["scout_output"].signal_id == result["risk_output"].signal_id, (
        "signal_id mismatch between Scout and Risk Guardian output -- "
        "the handoff is broken."
    )
    print("\nPipeline smoke test passed: signal_id matched end-to-end, "
          "Scout's proposal was consumed and judged by Risk Guardian.")
