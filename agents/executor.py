"""
agents/executor.py

Deterministic, non-agentic order-placement step for the FVG Copilot
options overlay (Alpaca AI Trading Agents Hackathon).

Design contract (see Notion "Alpaca AI Trading Agents Hackathon" page,
Decision: Alpaca CLI over MCP Server, Aug 27):
  - This module is intentionally NOT an LLM agent. No reasoning calls.
    It only turns an already-approved RiskGuardianOutput into an actual
    multi-leg options order on Alpaca's paper account, or does nothing.
  - Order placement shells out to the Alpaca CLI (`alpaca` binary),
    NOT the alpaca-py SDK. The SDK stays reserved for the deterministic
    FVG core / backtest / Position Monitor in live_bot.py (untouched).
  - As of the Alpaca CLI's current release, `alpaca order submit` only
    exposes single-leg flags (--symbol/--side/--qty/--type/...). There is
    no first-class multi-leg (mleg) flag yet, so multi-leg debit spreads
    are placed via the CLI's documented raw-API escape hatch:
        echo '<json>' | alpaca api POST /v2/orders
    This is still "the Alpaca CLI", satisfying the required-technology
    rule, and matches Alpaca's own multi-leg payload shape
    (order_class="mleg", legs=[...]).
  - Fail-closed philosophy carried over from scout.py / risk_guardian.py:
    every function either returns a clean, explicit result dict, or logs
    a clear reason and returns a NO_ORDER / ERROR result. It never raises
    an uncaught exception up to a caller, and it never fabricates an order
    confirmation.

VERIFIED AGAINST LIVE agents/agent_schemas.py (Sep 1, ~2:24 PM PDT):
  - ScoutOutput: signal_id, symbol, direction (Direction: "BUY"/"SELL"),
    underlying_price, fvg_context (FVGContext: gap_type, mss_confirmed,
    displacement_strength, measured_move_target, entry_bar_timestamp),
    thesis, market_context, confidence_score, suggested_strategy
    (OptionsStrategy), suggested_expiration_bias, risk_flags,
    recommended_max_loss_usd, recommended_max_hold_hours, generated_at.
  - RiskGuardianOutput: signal_id, decision (RiskDecision: APPROVE /
    APPROVE_MODIFIED / VETO), veto_reason, risk_rationale,
    position_size_contracts, final_max_loss_usd, final_max_hold_hours,
    generated_at.
  - FVGContext.measured_move_target IS the real target-price field --
    _extract_price_target() below already matched this name correctly.
  - REAL BUG CAUGHT AND FIXED during this verification: Direction and
    RiskDecision are `class X(str, Enum)`. Calling plain str(Direction.BUY)
    returns 'Direction.BUY' (Enum's __str__), NOT 'BUY', even though the
    instance IS a str. The original draft called str(direction).upper()
    and would have silently misrouted every BUY signal to a put spread
    instead of raising an error. Fixed via the _enum_str() helper, which
    reads .value when present. RiskDecision's "VETO"/"APPROVE" substring
    check happened to survive the same bug by luck (both are substrings
    of "RiskDecision.APPROVE_MODIFIED" etc.) but is now routed through
    the same safe helper for consistency, not left as an accidental pass.

KNOWN OPEN ITEM: Executor does not yet persist final_max_hold_hours
anywhere -- Position Monitor (not yet built) is the piece meant to
enforce it on open positions. For now it is logged and returned in the
result dict so it isn't silently dropped, but there is no durable
store (DB/JSON sidecar) wiring it to a not-yet-existing Position Monitor.
Flag this explicitly when Position Monitor is built.

Usage:
    from executor import run_executor
    result = run_executor(scout_output, risk_output, dry_run=True)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("executor")

ALPACA_BIN = os.environ.get("ALPACA_CLI_BIN", "alpaca")

# ---------------------------------------------------------------------------
# MVP contract-selection rules (Notion: "MVP Contract Selection Rules (draft)")
# ---------------------------------------------------------------------------
MIN_DTE = 14
MAX_DTE = 45
MAX_BID_ASK_SPREAD_PCT = 0.15       # reject legs wider than 15% of mid
MIN_REWARD_RISK_RATIO = 1.0         # (width - debit) / debit must clear this
DEFAULT_SPREAD_WIDTH = 5.0          # fallback if no target-aware distance found
STRIKE_INCREMENT_GUESS = 1.0        # SPY/QQQ/XLF/XLV/IWM are ~$1 increments
CLIENT_ORDER_ID_PREFIX = "fvgcopilot"


class ExecutorError(Exception):
    """Raised only for programmer errors (bad call shape), never for
    market/CLI failures -- those are caught and returned as result dicts."""


@dataclass
class ExecResult:
    action: str                     # "NO_ORDER" | "ORDER_SUBMITTED" | "ORDER_REJECTED" | "ERROR"
    reason: str
    order_payload: Optional[dict] = None
    cli_response: Optional[dict] = None
    signal_id: Optional[str] = None
    final_max_hold_hours: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "order_payload": self.order_payload,
            "cli_response": self.cli_response,
            "signal_id": self.signal_id,
            "final_max_hold_hours": self.final_max_hold_hours,
        }


# ---------------------------------------------------------------------------
# Defensive field access helpers
# ---------------------------------------------------------------------------
def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Try several possible attribute names in order; log once if none hit."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    log.warning("None of the expected fields %s found on %r; using default=%r", names, type(obj).__name__, default)
    return default


def _enum_str(value: Any) -> str:
    """Safely turn a value that might be a str-mixin Enum (e.g. Direction.BUY,
    RiskDecision.APPROVE_MODIFIED from agent_schemas.py) or a plain string
    into its plain string form.

    BUG THIS FIXES: for `class Direction(str, Enum): BUY = "BUY"`, calling
    plain str(Direction.BUY) returns 'Direction.BUY' (Enum's __str__), NOT
    'BUY' -- even though the instance IS a str. Always route enum-or-str
    fields through this helper before comparing against literal strings."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _extract_price_target(scout_output: Any) -> Optional[float]:
    """Best-effort extraction of a measured-move / target price from Scout's
    output, to drive target-aware strike selection (backtest bug-fix #1:
    fixed-width spreads mismatched actual price targets -- see Notion
    Backtest Log). Confirmed live: FVGContext.measured_move_target is the
    real field name, tried first below; thesis-text regex is a fallback
    for robustness only, DEFAULT_SPREAD_WIDTH is the last resort."""
    for attr_path in (("fvg_context", "measured_move_target"), ("fvg_context", "target_price"),
                       ("fvg_context", "target"), ("target_price",)):
        obj = scout_output
        ok = True
        for part in attr_path:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and isinstance(obj, (int, float)):
            return float(obj)

    thesis = _get(scout_output, "thesis", default="") or ""
    match = re.search(r"target[^0-9]{0,15}([0-9]+(?:\.[0-9]+)?)", thesis, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _run_cli(args: list[str], input_json: Optional[dict] = None, timeout: int = 20) -> dict:
    """Run an `alpaca` CLI subcommand and parse its JSON stdout.
    Never raises -- returns {"_error": "..."} on any failure so callers
    can fail closed instead of crashing."""
    cmd = [ALPACA_BIN] + args
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(input_json) if input_json is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"_error": f"alpaca CLI binary not found (looked for '{ALPACA_BIN}'); is it installed and on PATH?"}
    except subprocess.TimeoutExpired:
        return {"_error": f"alpaca CLI call timed out after {timeout}s: {' '.join(cmd)}"}

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        return {"_error": f"alpaca CLI exited {proc.returncode}: {stderr or proc.stdout.strip()}"}

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {"_error": "alpaca CLI returned empty stdout"}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {"_error": f"alpaca CLI returned non-JSON stdout: {exc}; raw={stdout[:300]}"}


def fetch_option_chain(symbol: str) -> dict:
    """`alpaca data option chain --underlying-symbol <symbol>`"""
    return _run_cli(["data", "option", "chain", "--underlying-symbol", symbol, "--quiet"])


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------
def _parse_expiration(contract: dict) -> Optional[date]:
    exp_str = contract.get("expiration_date") or contract.get("expiration")
    if not exp_str:
        return None
    try:
        return datetime.strptime(exp_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _mid_and_spread_pct(contract: dict) -> tuple[Optional[float], Optional[float]]:
    quote = contract.get("latest_quote") or contract
    bid = quote.get("bid_price") or quote.get("bp")
    ask = quote.get("ask_price") or quote.get("ap")
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None, None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None, None
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid if mid else None
    return mid, spread_pct


def select_option_contracts(symbol: str, direction: Any, underlying_price: float,
                             price_target: Optional[float], max_loss_usd: float) -> dict:
    """Select a long debit spread (call spread if bullish/BUY, put spread if
    bearish/SELL) per the MVP Contract Selection Rules:
      - 14-45 DTE, no same-day expiry
      - buy near-ATM/one-strike-ITM, sell 1-2 strikes further OTM
      - target-aware width when a measured-move target is available,
        else DEFAULT_SPREAD_WIDTH
      - reject wide bid/ask spreads or an implied debit above max_loss_usd

    `direction` may be a plain string ("BUY"/"SELL") or a Direction enum
    instance -- routed through _enum_str() so either works correctly.

    Returns a dict describing the chosen legs, or {"_error": "..."} if no
    valid combination is found (executor caller must not place an order then).
    """
    is_bullish = _enum_str(direction).upper() in ("BUY", "BULLISH", "CALL")
    option_type = "call" if is_bullish else "put"

    chain = fetch_option_chain(symbol)
    if "_error" in chain:
        return {"_error": chain["_error"]}

    contracts = chain.get("option_contracts") or chain.get("contracts") or chain.get("snapshots") or []
    if not contracts:
        return {"_error": f"option chain for {symbol} returned no contracts"}

    today = date.today()
    candidates = []
    for c in contracts:
        c_type = (c.get("type") or c.get("contract_type") or "").lower()
        if c_type != option_type:
            continue
        exp = _parse_expiration(c)
        if exp is None:
            continue
        dte = (exp - today).days
        if not (MIN_DTE <= dte <= MAX_DTE):
            continue
        strike = c.get("strike_price") or c.get("strike")
        try:
            strike = float(strike)
        except (TypeError, ValueError):
            continue
        c["_strike"] = strike
        c["_dte"] = dte
        c["_exp"] = exp
        candidates.append(c)

    if not candidates:
        return {"_error": f"no {option_type} contracts for {symbol} in {MIN_DTE}-{MAX_DTE} DTE window"}

    candidates.sort(key=lambda c: c["_dte"])
    chosen_exp = candidates[0]["_exp"]
    same_exp = [c for c in candidates if c["_exp"] == chosen_exp]
    same_exp.sort(key=lambda c: c["_strike"])

    if price_target is not None:
        target_width = abs(price_target - underlying_price)
        width = max(STRIKE_INCREMENT_GUESS, round(target_width / STRIKE_INCREMENT_GUESS) * STRIKE_INCREMENT_GUESS)
    else:
        width = DEFAULT_SPREAD_WIDTH

    if is_bullish:
        long_leg = min(same_exp, key=lambda c: abs(c["_strike"] - underlying_price))
        otm_target_strike = long_leg["_strike"] + width
        short_candidates = [c for c in same_exp if c["_strike"] > long_leg["_strike"]]
        if not short_candidates:
            return {"_error": "no OTM strike available above the long call leg"}
        short_leg = min(short_candidates, key=lambda c: abs(c["_strike"] - otm_target_strike))
    else:
        long_leg = min(same_exp, key=lambda c: abs(c["_strike"] - underlying_price))
        otm_target_strike = long_leg["_strike"] - width
        short_candidates = [c for c in same_exp if c["_strike"] < long_leg["_strike"]]
        if not short_candidates:
            return {"_error": "no OTM strike available below the long put leg"}
        short_leg = min(short_candidates, key=lambda c: abs(c["_strike"] - otm_target_strike))

    for leg, label in ((long_leg, "long"), (short_leg, "short")):
        _, spread_pct = _mid_and_spread_pct(leg)
        if spread_pct is None:
            return {"_error": f"{label} leg {leg.get('symbol')} missing usable bid/ask quote"}
        if spread_pct > MAX_BID_ASK_SPREAD_PCT:
            return {"_error": f"{label} leg {leg.get('symbol')} bid/ask spread {spread_pct:.1%} exceeds {MAX_BID_ASK_SPREAD_PCT:.0%} liquidity cap"}

    long_mid, _ = _mid_and_spread_pct(long_leg)
    short_mid, _ = _mid_and_spread_pct(short_leg)
    net_debit_per_share = round(long_mid - short_mid, 2)
    if net_debit_per_share <= 0:
        return {"_error": f"computed non-positive net debit ({net_debit_per_share}); refusing to trade a credit here under a debit-spread-only rule"}

    spread_width_actual = abs(long_leg["_strike"] - short_leg["_strike"])
    max_loss_per_contract = net_debit_per_share * 100
    max_profit_per_contract = (spread_width_actual - net_debit_per_share) * 100
    reward_risk = (max_profit_per_contract / max_loss_per_contract) if max_loss_per_contract else 0

    if reward_risk < MIN_REWARD_RISK_RATIO:
        return {"_error": f"reward:risk {reward_risk:.2f} below minimum {MIN_REWARD_RISK_RATIO}"}
    if max_loss_per_contract > max_loss_usd:
        return {"_error": f"1-contract max loss ${max_loss_per_contract:.2f} exceeds risk cap ${max_loss_usd:.2f} even before sizing"}

    return {
        "option_type": option_type,
        "expiration": chosen_exp.isoformat(),
        "dte": long_leg["_dte"],
        "long_symbol": long_leg.get("symbol"),
        "long_strike": long_leg["_strike"],
        "short_symbol": short_leg.get("symbol"),
        "short_strike": short_leg["_strike"],
        "net_debit_per_share": net_debit_per_share,
        "spread_width": spread_width_actual,
        "max_loss_per_contract_usd": max_loss_per_contract,
        "max_profit_per_contract_usd": max_profit_per_contract,
        "reward_risk_ratio": round(reward_risk, 2),
    }


# ---------------------------------------------------------------------------
# Order construction + placement
# ---------------------------------------------------------------------------
def build_mleg_order(contract_selection: dict, qty: int, client_order_id: str) -> dict:
    """Build the raw /v2/orders JSON payload for a long debit spread.
    Buying the near-the-money leg, selling the further-OTM leg, both
    same expiration -- matches Alpaca's documented mleg order shape."""
    return {
        "type": "market",
        "time_in_force": "day",
        "order_class": "mleg",
        "qty": str(qty),
        "client_order_id": client_order_id,
        "legs": [
            {
                "symbol": contract_selection["long_symbol"],
                "side": "buy",
                "position_intent": "buy_to_open",
                "ratio_qty": "1",
            },
            {
                "symbol": contract_selection["short_symbol"],
                "side": "sell",
                "position_intent": "sell_to_open",
                "ratio_qty": "1",
            },
        ],
    }


def submit_mleg_order(order_payload: dict, dry_run: bool = True) -> dict:
    """Place the multi-leg order via the Alpaca CLI's raw API passthrough:
        echo '<json>' | alpaca api POST /v2/orders
    dry_run=True (the safe default) prints the exact payload and command
    that would run, without shelling out, so this can be reviewed before
    ever hitting the live paper account."""
    if dry_run:
        log.info("[DRY RUN] Would run: echo '<payload>' | %s api POST /v2/orders", ALPACA_BIN)
        log.info("[DRY RUN] Payload: %s", json.dumps(order_payload))
        return {"_dry_run": True, "would_submit": order_payload}
    return _run_cli(["api", "POST", "/v2/orders"], input_json=order_payload)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_executor(scout_output: Any, risk_output: Any, dry_run: bool = True) -> dict:
    """Chain: RiskGuardianOutput -> (VETO -> no-op) | (APPROVE* -> place order).

    Args:
        scout_output: the ScoutOutput this decision was based on (for
            symbol/direction/underlying price/target context).
        risk_output: the RiskGuardianOutput -- the ONLY output the
            Executor is allowed to act on (per architecture decision).
        dry_run: if True (default), builds and logs the order but never
            calls the Alpaca CLI's order-submit endpoint. Set False only
            once you've reviewed a dry-run payload and are ready to place
            a real paper order on the "Fable FVG" account.
    """
    signal_id = _get(risk_output, "signal_id", default=_get(scout_output, "signal_id", default=None))
    decision = _enum_str(_get(risk_output, "decision", default="")).upper()

    if "VETO" in decision:
        veto_reason = _get(risk_output, "veto_reason", default="(no veto_reason provided)")
        log.info("Risk Guardian VETO for signal_id=%s: %s -- no order placed.", signal_id, veto_reason)
        return ExecResult(action="NO_ORDER", reason=f"vetoed: {veto_reason}", signal_id=signal_id).to_dict()

    if "APPROVE" not in decision:
        log.warning("Unrecognized RiskGuardianOutput.decision=%r; failing closed to NO_ORDER.", decision)
        return ExecResult(action="NO_ORDER", reason=f"unrecognized decision value: {decision!r}", signal_id=signal_id).to_dict()

    symbol = _get(scout_output, "symbol", default=None)
    direction = _enum_str(_get(scout_output, "direction", default="BUY"))
    underlying_price = _get(scout_output, "underlying_price", default=None)
    final_max_loss_usd = _get(risk_output, "final_max_loss_usd", default=None)
    position_size_contracts = _get(risk_output, "position_size_contracts", default=None)
    final_max_hold_hours = _get(risk_output, "final_max_hold_hours", default=None)

    missing = [name for name, val in (
        ("symbol", symbol), ("underlying_price", underlying_price),
        ("final_max_loss_usd", final_max_loss_usd), ("position_size_contracts", position_size_contracts),
    ) if val is None]
    if missing:
        reason = f"missing required fields from agent outputs: {missing}"
        log.error(reason)
        return ExecResult(action="ERROR", reason=reason, signal_id=signal_id).to_dict()

    try:
        underlying_price = float(underlying_price)
        final_max_loss_usd = float(final_max_loss_usd)
        position_size_contracts = int(position_size_contracts)
    except (TypeError, ValueError) as exc:
        reason = f"could not coerce numeric fields: {exc}"
        log.error(reason)
        return ExecResult(action="ERROR", reason=reason, signal_id=signal_id).to_dict()

    if position_size_contracts <= 0:
        reason = f"position_size_contracts={position_size_contracts} is not positive; refusing to place order"
        log.error(reason)
        return ExecResult(action="NO_ORDER", reason=reason, signal_id=signal_id).to_dict()

    price_target = _extract_price_target(scout_output)
    per_contract_cap = final_max_loss_usd / position_size_contracts

    selection = select_option_contracts(
        symbol=symbol,
        direction=direction,
        underlying_price=underlying_price,
        price_target=price_target,
        max_loss_usd=per_contract_cap,
    )
    if "_error" in selection:
        log.warning("Contract selection failed for %s: %s", symbol, selection["_error"])
        return ExecResult(action="NO_ORDER", reason=f"contract selection failed: {selection['_error']}",
                           signal_id=signal_id, final_max_hold_hours=final_max_hold_hours).to_dict()

    total_max_loss = selection["max_loss_per_contract_usd"] * position_size_contracts
    if total_max_loss > final_max_loss_usd:
        reason = (f"sized max loss ${total_max_loss:.2f} ({position_size_contracts} contracts) "
                  f"exceeds Risk Guardian cap ${final_max_loss_usd:.2f}")
        log.warning(reason)
        return ExecResult(action="NO_ORDER", reason=reason, signal_id=signal_id, order_payload=selection,
                           final_max_hold_hours=final_max_hold_hours).to_dict()

    client_order_id = f"{CLIENT_ORDER_ID_PREFIX}-{signal_id or 'nosig'}-{uuid.uuid4().hex[:8]}"
    order_payload = build_mleg_order(selection, qty=position_size_contracts, client_order_id=client_order_id)

    cli_response = submit_mleg_order(order_payload, dry_run=dry_run)
    if "_error" in cli_response:
        log.error("Order submission failed for signal_id=%s: %s", signal_id, cli_response["_error"])
        return ExecResult(action="ERROR", reason=cli_response["_error"], order_payload=order_payload,
                           cli_response=cli_response, signal_id=signal_id,
                           final_max_hold_hours=final_max_hold_hours).to_dict()

    action = "ORDER_SUBMITTED" if not cli_response.get("_dry_run") else "NO_ORDER"
    reason = "dry run only -- set dry_run=False to actually submit" if cli_response.get("_dry_run") else "order submitted"
    if action == "ORDER_SUBMITTED":
        log.info(
            "Order submitted (signal_id=%s). final_max_hold_hours=%s must still be enforced by the "
            "not-yet-built Position Monitor -- no persistence layer wires this through yet.",
            signal_id, final_max_hold_hours,
        )
    log.info("Executor result for signal_id=%s: %s (%s)", signal_id, action, reason)
    return ExecResult(action=action, reason=reason, order_payload=order_payload,
                       cli_response=cli_response, signal_id=signal_id,
                       final_max_hold_hours=final_max_hold_hours).to_dict()


# ---------------------------------------------------------------------------
# Smoke test (mocked -- no real Alpaca CLI or Groq calls). Uses the REAL
# enum classes shape (str, Enum) to specifically re-catch the str()-on-enum
# bug class if it's ever reintroduced.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from enum import Enum
    from types import SimpleNamespace
    from unittest.mock import patch

    class Direction(str, Enum):
        BUY = "BUY"
        SELL = "SELL"

    class RiskDecision(str, Enum):
        APPROVE = "APPROVE"
        APPROVE_MODIFIED = "APPROVE_MODIFIED"
        VETO = "VETO"

    print("=== Case A: VETO (real RiskDecision enum) -> executor must no-op ===")
    scout_a = SimpleNamespace(signal_id="sig-A", symbol="SPY", direction=Direction.BUY, underlying_price=600.0, thesis="target of 610")
    risk_a = SimpleNamespace(signal_id="sig-A", decision=RiskDecision.VETO, veto_reason="daily loss limit breached")
    print(run_executor(scout_a, risk_a, dry_run=True))

    print("\n=== Case B: APPROVE_MODIFIED, real Direction.BUY enum -> must select a CALL spread, not a put ===")
    fvg_context_b = SimpleNamespace(measured_move_target=608.0)
    scout_b = SimpleNamespace(
        signal_id="sig-B", symbol="SPY", direction=Direction.BUY, underlying_price=600.0,
        fvg_context=fvg_context_b, thesis="MSS confirmed, displacement strong",
    )
    risk_b = SimpleNamespace(
        signal_id="sig-B", decision=RiskDecision.APPROVE_MODIFIED,
        position_size_contracts=1, final_max_loss_usd=250.0, final_max_hold_hours=12,
    )

    fake_chain_calls = {
        "option_contracts": [
            {"symbol": "SPY260215C00600000", "type": "call", "strike_price": 600.0,
             "expiration_date": (date.today() + timedelta(days=21)).isoformat(),
             "latest_quote": {"bid_price": 2.30, "ask_price": 2.50}},
            {"symbol": "SPY260215C00608000", "type": "call", "strike_price": 608.0,
             "expiration_date": (date.today() + timedelta(days=21)).isoformat(),
             "latest_quote": {"bid_price": 0.90, "ask_price": 1.00}},
            {"symbol": "SPY260215P00600000", "type": "put", "strike_price": 600.0,
             "expiration_date": (date.today() + timedelta(days=21)).isoformat(),
             "latest_quote": {"bid_price": 2.10, "ask_price": 2.30}},
        ]
    }
    with patch("__main__.fetch_option_chain", return_value=fake_chain_calls):
        result_b = run_executor(scout_b, risk_b, dry_run=True)
    print(result_b)
    assert "C0" in result_b["order_payload"]["legs"][0]["symbol"], "Regression check failed: BUY signal did not select a call contract"
    print(">>> Regression check passed: Direction.BUY correctly selected CALL contracts (not puts).")

    print("\n=== Case C: SELL direction (real enum) -> must select a PUT spread ===")
    fvg_context_c = SimpleNamespace(measured_move_target=592.0)
    scout_c = SimpleNamespace(
        signal_id="sig-C", symbol="SPY", direction=Direction.SELL, underlying_price=600.0,
        fvg_context=fvg_context_c, thesis="bearish MSS",
    )
    risk_c = SimpleNamespace(
        signal_id="sig-C", decision=RiskDecision.APPROVE,
        position_size_contracts=1, final_max_loss_usd=250.0, final_max_hold_hours=12,
    )
    fake_chain_puts = {
        "option_contracts": [
            {"symbol": "SPY260215P00600000", "type": "put", "strike_price": 600.0,
             "expiration_date": (date.today() + timedelta(days=21)).isoformat(),
             "latest_quote": {"bid_price": 2.10, "ask_price": 2.30}},
            {"symbol": "SPY260215P00592000", "type": "put", "strike_price": 592.0,
             "expiration_date": (date.today() + timedelta(days=21)).isoformat(),
             "latest_quote": {"bid_price": 0.85, "ask_price": 0.95}},
        ]
    }
    with patch("__main__.fetch_option_chain", return_value=fake_chain_puts):
        result_c = run_executor(scout_c, risk_c, dry_run=True)
    print(result_c)
    assert "P0" in result_c["order_payload"]["legs"][0]["symbol"], "Regression check failed: SELL signal did not select a put contract"
    print(">>> Regression check passed: Direction.SELL correctly selected PUT contracts.")

    print("\n=== Case D: unparseable chain / CLI error path ===")
    scout_d = SimpleNamespace(signal_id="sig-D", symbol="QQQ", direction=Direction.SELL, underlying_price=520.0, thesis="target of 505")
    risk_d = SimpleNamespace(signal_id="sig-D", decision=RiskDecision.APPROVE, position_size_contracts=1, final_max_loss_usd=200.0, final_max_hold_hours=24)
    with patch("__main__.fetch_option_chain", return_value={"_error": "simulated CLI failure: alpaca binary not found"}):
        result_d = run_executor(scout_d, risk_d, dry_run=True)
    print(result_d)
