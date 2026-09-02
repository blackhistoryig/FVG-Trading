"""
agents/pipeline.py

Orchestrates the full Scout -> Risk Guardian -> Executor pipeline for the
hackathon options-trading agent system.

Fail-closed by design: if any stage raises or returns an invalid result, the
pipeline stops at that stage and reports the error rather than fabricating a
downstream result. Mirrors the fail-closed pattern already validated in
risk_guardian.py and executor.py.

Import style: FLAT, matching scout.py / risk_guardian.py (this project fixed
a real ModuleNotFoundError earlier from using `agents.xxx`-style imports --
do not reintroduce that).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from agent_schemas import (
    ScoutOutput,
    RiskGuardianOutput,
)
from scout import run_scout
from risk_guardian import run_risk_guardian, AccountRiskContext
from executor import run_executor


def _enum_str(value: Any) -> str:
    """
    Safely coerce a (str, Enum) member (Direction, RiskDecision, etc.) to its
    plain string value.

    CRITICAL GOTCHA (already bit this project twice, once in executor.py):
    Direction/RiskDecision inherit (str, Enum). Calling str() directly on an
    instance returns 'RiskDecision.VETO' (Enum.__str__), NOT 'VETO', even
    though the instance IS a str. Always route through this helper before
    comparing to literal strings or printing for humans/logs.
    """
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_pipeline(
    raw_signal: Dict[str, Any],
    account_context: AccountRiskContext,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Runs Scout -> Risk Guardian -> Executor end to end.

    Args:
        raw_signal: raw FVG signal dict, same shape Scout already accepts.
        account_context: AccountRiskContext (current_daily_pnl_usd,
            open_position_count, current_symbol_exposure_usd,
            proposed_trade_cost_usd) for the "Fable FVG" paper account.
        dry_run: forwarded to run_executor. Defaults True (log-only, no real
            order submitted). Do NOT flip to False without re-confirming
            with the user first -- the mleg POST /v2/orders call has never
            been exercised live as of this handoff.

    Returns:
        dict with keys: signal_id, scout, risk_guardian, executor,
        final_status, error. Every stage's raw model output is included
        (via model_dump) for logging/demo capture, even when a later stage
        vetoes or fails.
    """
    result: Dict[str, Any] = {
        "signal_id": raw_signal.get("signal_id"),
        "scout": None,
        "risk_guardian": None,
        "executor": None,
        "final_status": None,
        "error": None,
    }

    # ---- Stage 1: Scout (LLM) ----
    _print_section("STAGE 1: SCOUT")
    try:
        scout_output: ScoutOutput = run_scout(raw_signal)
    except Exception as exc:
        result["error"] = f"Scout failed: {exc}"
        result["final_status"] = "ERROR"
        print(result["error"])
        return result

    result["scout"] = json.loads(scout_output.model_dump_json())
    print(f"Symbol: {scout_output.symbol}")
    print(f"Direction: {_enum_str(scout_output.direction)}")
    print(f"Suggested strategy: {_enum_str(scout_output.suggested_strategy)}")
    print(f"Confidence: {scout_output.confidence_score}")
    print(f"Thesis: {scout_output.thesis}")

    # ---- Stage 2: Risk Guardian (deterministic hard limits + LLM) ----
    _print_section("STAGE 2: RISK GUARDIAN")
    try:
        risk_output: RiskGuardianOutput = run_risk_guardian(scout_output, account_context)
    except Exception as exc:
        result["error"] = f"Risk Guardian failed: {exc}"
        result["final_status"] = "ERROR"
        print(result["error"])
        return result

    result["risk_guardian"] = json.loads(risk_output.model_dump_json())
    decision = _enum_str(risk_output.decision)
    print(f"Decision: {decision}")
    if decision == "VETO":
        print(f"Veto reason: {risk_output.veto_reason}")
    print(f"Rationale: {risk_output.risk_rationale}")

    # ---- Stage 3: Executor (deterministic, non-agentic) ----
    # run_executor already no-ops on VETO internally, so it is safe to call
    # unconditionally here -- this keeps the printed reasoning chain
    # complete for demo purposes (a VETO'd signal still shows an explicit
    # "Executor: no-op" instead of silently vanishing from the trace).
    _print_section(f"STAGE 3: EXECUTOR (dry_run={dry_run})")
    try:
        executor_output: Dict[str, Any] = run_executor(scout_output, risk_output, dry_run=dry_run)
    except Exception as exc:
        result["error"] = f"Executor failed: {exc}"
        result["final_status"] = "ERROR"
        print(result["error"])
        return result

    result["executor"] = executor_output
    print(json.dumps(executor_output, indent=2, default=str))

    # ---- Final status ----
    result["final_status"] = "VETOED" if decision == "VETO" else "PROCESSED"

    _print_section("PIPELINE COMPLETE")
    print(f"Final status: {result['final_status']}")

    return result


if __name__ == "__main__":
    # Smoke test using the REAL AccountRiskContext dataclass fields,
    # confirmed field-for-field against the live risk_guardian.py on
    # 2026-09-01: current_daily_pnl_usd, open_position_count,
    # current_symbol_exposure_usd, proposed_trade_cost_usd.
    import sys
    from datetime import datetime, timezone

    example_signal = {
        "signal_id": "SPY-20260901-PIPELINE-TEST",
        "symbol": "SPY",
        "direction": "BUY",
        "underlying_price": 762.15,
        "fvg_context": {
            "gap_type": "bullish",
            "mss_confirmed": True,
            "displacement_strength": 1.2,
            "measured_move_target": 777.0,
            "entry_bar_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Mirrors risk_guardian.py's own "Case 2: normal case" smoke test --
    # daily P&L at -$50, one open position, no existing SPY exposure.
    example_account_context = AccountRiskContext(
        current_daily_pnl_usd=-50.0,
        open_position_count=1,
        current_symbol_exposure_usd=0.0,
        proposed_trade_cost_usd=200.0,
    )

    dry = "--live" not in sys.argv
    output = run_pipeline(example_signal, example_account_context, dry_run=dry)
    print("\nFinal result keys:", list(output.keys()))
