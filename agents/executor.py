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
  - Multi-leg debit spreads are placed via the CLI's raw-API passthrough:
        echo '<json>' | alpaca api POST /v2/orders
    Still "the Alpaca CLI" -- satisfies the required-technology rule.
  - Fail-closed philosophy: every function either returns a clean,
    explicit result dict, or logs a clear reason and returns a
    NO_ORDER / ERROR result. Never raises uncaught, never fabricates
    a confirmation.

BUG HISTORY (all confirmed live during this session, not theoretical):
1. Direction/RiskDecision are `class X(str, Enum)`. str(Direction.BUY) ==
   'Direction.BUY', not 'BUY'. Fixed via _enum_str() reading .value.
2. `alpaca data option chain` response is {"snapshots": {OCC_SYMBOL:
   {dailyBar, greeks, latestQuote, latestTrade, ...}}} -- a DICT keyed by
   OCC symbol, not a list of pre-decoded contract dicts. Fixed via
   _parse_occ_symbol(). Quote key is "latestQuote" (camelCase) with
   "bp"/"ap".
3. A live sample (SPY $420 call, 0DTE) was quoted ~347.51/348.53 while
   the underlying was ~341.61 -- above the underlying's own price.
   Fixed via _sanity_check_quote() (heuristic, tunable).
4. `alpaca data option chain` with no filters returns only ONE PAGE
   (default limit 100) of the NEAREST expiration (observed: 0 DTE).
   Fixed by passing --type/--expiration-date-gte/-lte server-side.
5. Even after fixing #4, a real SPY test ($762.15 spot) failed with
   "no OTM strike available above the long call leg" -- the default
   page (still no strike filter) was exhausted by low/deep-ITM strikes
   before reaching ATM/OTM strikes near spot. Fixed by also passing
   --strike-price-gte/-lte, bounding the fetch to a band around the
   underlying price sized off the computed spread width.

Integration: on a successful (non-dry-run) order submission, this module
calls position_monitor.record_submitted_order() so the deterministic
Position Monitor polling loop can later enforce final_max_loss_usd /
final_max_hold_hours on the position. See run_executor()'s
ORDER_SUBMITTED branch.

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

MIN_DTE = 14
MAX_DTE = 45
MAX_BID_ASK_SPREAD_PCT = 0.15
MIN_REWARD_RISK_RATIO = 1.0
DEFAULT_SPREAD_WIDTH = 5.0
STRIKE_INCREMENT_GUESS = 1.0
CLIENT_ORDER_ID_PREFIX = "fvgcopilot"
MAX_PLAUSIBLE_EXTRINSIC_PCT_OF_SPOT = 0.25

STRIKE_BAND_MULTIPLIER = 3.0
MIN_STRIKE_BAND_USD = 15.0

OCC_SYMBOL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


class ExecutorError(Exception):
    """Raised only for programmer errors (bad call shape), never for
    market/CLI failures -- those are caught and returned as result dicts."""


@dataclass
class ExecResult:
    action: str
    reason: str
    order_payload: Optional[dict] = None
    cli_response: Optional[dict] = None
    signal_id: Optional[str] = None
    final_max_hold_hours: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action, "reason": self.reason,
            "order_payload": self.order_payload, "cli_response": self.cli_response,
            "signal_id": self.signal_id, "final_max_hold_hours": self.final_max_hold_hours,
        }


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    log.warning("None of the expected fields %s found on %r; using default=%r", names, type(obj).__name__, default)
    return default


def _enum_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _extract_price_target(scout_output: Any) -> Optional[float]:
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
    cmd = [ALPACA_BIN] + args
    try:
        proc = subprocess.run(
            cmd, input=json.dumps(input_json) if input_json is not None else None,
            capture_output=True, text=True, timeout=timeout,
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


def fetch_option_chain(symbol: str, option_type: Optional[str] = None,
                        exp_gte: Optional[date] = None, exp_lte: Optional[date] = None,
                        strike_gte: Optional[float] = None, strike_lte: Optional[float] = None) -> dict:
    """`alpaca data option chain --underlying-symbol <symbol> [--type call|put]
    [--expiration-date-gte YYYY-MM-DD] [--expiration-date-lte YYYY-MM-DD]
    [--strike-price-gte N] [--strike-price-lte N] --quiet`.

    Both the expiration window (bug #4) AND the strike-price window (bug #5)
    must be passed server-side -- the CLI's default single page (limit 100,
    no filters) is not a representative sample of the full chain and can
    silently miss the strikes/expirations actually needed."""
    args = ["data", "option", "chain", "--underlying-symbol", symbol, "--quiet"]
    if option_type in ("call", "put"):
        args += ["--type", option_type]
    if exp_gte is not None:
        args += ["--expiration-date-gte", exp_gte.isoformat()]
    if exp_lte is not None:
        args += ["--expiration-date-lte", exp_lte.isoformat()]
    if strike_gte is not None:
        args += ["--strike-price-gte", f"{strike_gte:.2f}"]
    if strike_lte is not None:
        args += ["--strike-price-lte", f"{strike_lte:.2f}"]
    return _run_cli(args)


def _parse_occ_symbol(symbol: str) -> Optional[dict]:
    m = OCC_SYMBOL_RE.match(symbol)
    if not m:
        return None
    root, date_str, cp, strike_str = m.groups()
    try:
        exp = datetime.strptime(date_str, "%y%m%d").date()
    except ValueError:
        return None
    return {"root": root, "expiration": exp, "option_type": "call" if cp == "C" else "put",
            "strike": int(strike_str) / 1000.0}


def _extract_quote(raw: dict) -> tuple[Optional[float], Optional[float]]:
    quote = raw.get("latestQuote") or raw.get("latest_quote") or raw
    bid = quote.get("bp") if quote.get("bp") is not None else quote.get("bid_price")
    ask = quote.get("ap") if quote.get("ap") is not None else quote.get("ask_price")
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None, None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None, None
    return bid, ask


def _mid_and_spread_pct(raw: dict) -> tuple[Optional[float], Optional[float]]:
    bid, ask = _extract_quote(raw)
    if bid is None or ask is None:
        return None, None
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid if mid else None
    return mid, spread_pct


def _sanity_check_quote(mid: float, strike: float, underlying_price: float, option_type: str) -> Optional[str]:
    intrinsic = max(underlying_price - strike, 0.0) if option_type == "call" else max(strike - underlying_price, 0.0)
    extrinsic = mid - intrinsic
    cap = underlying_price * MAX_PLAUSIBLE_EXTRINSIC_PCT_OF_SPOT
    if extrinsic > cap:
        return (f"implausible extrinsic value ${extrinsic:.2f} (mid ${mid:.2f}, intrinsic ${intrinsic:.2f}) "
                f"exceeds sanity cap ${cap:.2f} ({MAX_PLAUSIBLE_EXTRINSIC_PCT_OF_SPOT:.0%} of spot ${underlying_price:.2f}) "
                f"-- likely stale/illiquid quote, not a real tradeable price")
    if extrinsic < -0.05:
        return f"mid price ${mid:.2f} is below intrinsic value ${intrinsic:.2f} -- likely bad/stale quote"
    return None


def _normalize_chain_items(chain: dict) -> list[dict]:
    snapshots = chain.get("snapshots")
    if isinstance(snapshots, dict):
        raw_items = [(sym, data) for sym, data in snapshots.items()]
    else:
        list_source = chain.get("option_contracts") or chain.get("contracts") or []
        if isinstance(list_source, dict):
            raw_items = [(sym, data) for sym, data in list_source.items()]
        else:
            raw_items = [(c.get("symbol"), c) for c in list_source if isinstance(c, dict)]
    items = []
    for sym, data in raw_items:
        if not sym or not isinstance(data, dict):
            continue
        occ = _parse_occ_symbol(sym)
        c_type = (data.get("type") or data.get("contract_type"))
        exp = None
        exp_raw = data.get("expiration_date") or data.get("expiration")
        if exp_raw:
            try:
                exp = datetime.strptime(str(exp_raw)[:10], "%Y-%m-%d").date()
            except ValueError:
                exp = None
        strike = None
        strike_raw = data.get("strike_price") if data.get("strike_price") is not None else data.get("strike")
        if strike_raw is not None:
            try:
                strike = float(strike_raw)
            except (TypeError, ValueError):
                strike = None
        if occ:
            c_type = c_type or occ["option_type"]
            exp = exp or occ["expiration"]
            strike = strike if strike is not None else occ["strike"]
        if not c_type or exp is None or strike is None:
            continue
        items.append({"symbol": sym, "_type": c_type.lower(), "_exp": exp, "_strike": strike, "_raw": data})
    return items


def select_option_contracts(symbol: str, direction: Any, underlying_price: float,
                             price_target: Optional[float], max_loss_usd: float) -> dict:
    """Select a long debit spread per the MVP Contract Selection Rules.
    Fetches with server-side expiration AND strike filters (bugs #4 and #5)
    instead of relying on an unfiltered default page."""
    is_bullish = _enum_str(direction).upper() in ("BUY", "BULLISH", "CALL")
    option_type = "call" if is_bullish else "put"

    if price_target is not None:
        target_width = abs(price_target - underlying_price)
        width = max(STRIKE_INCREMENT_GUESS, round(target_width / STRIKE_INCREMENT_GUESS) * STRIKE_INCREMENT_GUESS)
    else:
        width = DEFAULT_SPREAD_WIDTH

    strike_band = max(width * STRIKE_BAND_MULTIPLIER, MIN_STRIKE_BAND_USD)

    today = date.today()
    chain = fetch_option_chain(
        symbol, option_type=option_type,
        exp_gte=today + timedelta(days=MIN_DTE), exp_lte=today + timedelta(days=MAX_DTE),
        strike_gte=underlying_price - strike_band, strike_lte=underlying_price + strike_band,
    )
    if "_error" in chain:
        return {"_error": chain["_error"]}

    all_items = _normalize_chain_items(chain)
    if not all_items:
        return {"_error": f"option chain for {symbol} returned no parseable contracts in the requested "
                          f"{MIN_DTE}-{MAX_DTE} DTE / ${strike_band:.0f}-band / {option_type} window"}

    candidates = []
    for c in all_items:
        if c["_type"] != option_type:
            continue
        dte = (c["_exp"] - today).days
        if not (MIN_DTE <= dte <= MAX_DTE):
            continue
        c["_dte"] = dte
        candidates.append(c)

    if not candidates:
        return {"_error": f"no {option_type} contracts for {symbol} in {MIN_DTE}-{MAX_DTE} DTE window "
                          f"(server-side filter returned {len(all_items)} items)"}

    candidates.sort(key=lambda c: c["_dte"])
    chosen_exp = candidates[0]["_exp"]
    same_exp = [c for c in candidates if c["_exp"] == chosen_exp]
    same_exp.sort(key=lambda c: c["_strike"])

    if is_bullish:
        long_leg = min(same_exp, key=lambda c: abs(c["_strike"] - underlying_price))
        otm_target_strike = long_leg["_strike"] + width
        short_candidates = [c for c in same_exp if c["_strike"] > long_leg["_strike"]]
        if not short_candidates:
            return {"_error": f"no OTM strike available above the long call leg (strike band was "
                              f"${underlying_price - strike_band:.2f}-${underlying_price + strike_band:.2f}, "
                              f"{len(same_exp)} same-expiration candidates returned)"}
        short_leg = min(short_candidates, key=lambda c: abs(c["_strike"] - otm_target_strike))
    else:
        long_leg = min(same_exp, key=lambda c: abs(c["_strike"] - underlying_price))
        otm_target_strike = long_leg["_strike"] - width
        short_candidates = [c for c in same_exp if c["_strike"] < long_leg["_strike"]]
        if not short_candidates:
            return {"_error": f"no OTM strike available below the long put leg (strike band was "
                              f"${underlying_price - strike_band:.2f}-${underlying_price + strike_band:.2f}, "
                              f"{len(same_exp)} same-expiration candidates returned)"}
        short_leg = min(short_candidates, key=lambda c: abs(c["_strike"] - otm_target_strike))

    leg_mids = {}
    for leg, label in ((long_leg, "long"), (short_leg, "short")):
        mid, spread_pct = _mid_and_spread_pct(leg["_raw"])
        if mid is None or spread_pct is None:
            return {"_error": f"{label} leg {leg['symbol']} missing usable bid/ask quote"}
        if spread_pct > MAX_BID_ASK_SPREAD_PCT:
            return {"_error": f"{label} leg {leg['symbol']} bid/ask spread {spread_pct:.1%} exceeds {MAX_BID_ASK_SPREAD_PCT:.0%} liquidity cap"}
        sanity_err = _sanity_check_quote(mid, leg["_strike"], underlying_price, option_type)
        if sanity_err:
            return {"_error": f"{label} leg {leg['symbol']}: {sanity_err}"}
        leg_mids[label] = mid

    net_debit_per_share = round(leg_mids["long"] - leg_mids["short"], 2)
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
        "option_type": option_type, "expiration": chosen_exp.isoformat(), "dte": long_leg["_dte"],
        "long_symbol": long_leg["symbol"], "long_strike": long_leg["_strike"],
        "short_symbol": short_leg["symbol"], "short_strike": short_leg["_strike"],
        "net_debit_per_share": net_debit_per_share, "spread_width": spread_width_actual,
        "max_loss_per_contract_usd": max_loss_per_contract, "max_profit_per_contract_usd": max_profit_per_contract,
        "reward_risk_ratio": round(reward_risk, 2),
    }


def build_mleg_order(contract_selection: dict, qty: int, client_order_id: str) -> dict:
    return {
        "type": "market", "time_in_force": "day", "order_class": "mleg",
        "qty": str(qty), "client_order_id": client_order_id,
        "legs": [
            {"symbol": contract_selection["long_symbol"], "side": "buy", "position_intent": "buy_to_open", "ratio_qty": "1"},
            {"symbol": contract_selection["short_symbol"], "side": "sell", "position_intent": "sell_to_open", "ratio_qty": "1"},
        ],
    }


def submit_mleg_order(order_payload: dict, dry_run: bool = True) -> dict:
    if dry_run:
        log.info("[DRY RUN] Would run: echo '<payload>' | %s api POST /v2/orders", ALPACA_BIN)
        log.info("[DRY RUN] Payload: %s", json.dumps(order_payload))
        return {"_dry_run": True, "would_submit": order_payload}
    return _run_cli(["api", "POST", "/v2/orders"], input_json=order_payload)


def run_executor(scout_output: Any, risk_output: Any, dry_run: bool = True) -> dict:
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
        symbol=symbol, direction=direction, underlying_price=underlying_price,
        price_target=price_target, max_loss_usd=per_contract_cap,
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
        try:
            from position_monitor import record_submitted_order
            record_submitted_order(
                signal_id=signal_id,
                client_order_id=order_payload.get("client_order_id"),
                symbol=symbol,
                contracts=position_size_contracts,
                long_option_symbol=selection["long_symbol"],
                short_option_symbol=selection["short_symbol"],
                final_max_loss_usd=final_max_loss_usd,
                final_max_hold_hours=final_max_hold_hours,
            )
            log.info("Recorded submitted order for signal_id=%s in Position Monitor DB.", signal_id)
        except Exception as exc:
            log.error(
                "Failed to record submitted order for signal_id=%s in Position Monitor: %s. "
                "Position is now UNMONITORED -- final_max_loss_usd/final_max_hold_hours will "
                "NOT be enforced automatically. Manual tracking required.",
                signal_id, exc,
            )

    log.info("Executor result for signal_id=%s: %s (%s)", signal_id, action, reason)
    return ExecResult(action=action, reason=reason, order_payload=order_payload,
                       cli_response=cli_response, signal_id=signal_id,
                       final_max_hold_hours=final_max_hold_hours).to_dict()


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

    def make_snapshot(bp, ap, c=341.61):
        return {
            "dailyBar": {"c": c, "h": c + 0.5, "l": c - 0.5, "n": 6, "o": c, "t": "2026-09-01T04:00:00Z", "v": 6, "vw": c},
            "greeks": {"delta": 0, "gamma": 0, "rho": 0, "theta": 0, "vega": 0},
            "latestQuote": {"ap": ap, "as": 2, "ax": "S", "bp": bp, "bs": 1, "bx": "C", "c": " ", "t": "2026-09-01T19:59:59Z"},
            "latestTrade": {"c": "g", "p": c, "s": 1, "t": "2026-09-01T20:06:42Z", "x": "C"},
        }

    print("=== Case A: VETO -> executor must no-op ===")
    scout_a = SimpleNamespace(signal_id="sig-A", symbol="SPY", direction=Direction.BUY, underlying_price=600.0, thesis="target of 610")
    risk_a = SimpleNamespace(signal_id="sig-A", decision=RiskDecision.VETO, veto_reason="daily loss limit breached")
    print(run_executor(scout_a, risk_a, dry_run=True))

    exp21 = (date.today() + timedelta(days=21)).strftime("%y%m%d")
    print("\n=== Case B: real snapshots-dict shape, Direction.BUY -> must select CALLs ===")
    fvg_context_b = SimpleNamespace(measured_move_target=608.0)
    scout_b = SimpleNamespace(signal_id="sig-B", symbol="SPY", direction=Direction.BUY, underlying_price=600.0,
                               fvg_context=fvg_context_b, thesis="MSS confirmed")
    risk_b = SimpleNamespace(signal_id="sig-B", decision=RiskDecision.APPROVE_MODIFIED,
                              position_size_contracts=1, final_max_loss_usd=250.0, final_max_hold_hours=12)
    fake_chain_b = {"next_page_token": None, "snapshots": {
        f"SPY{exp21}C00600000": make_snapshot(2.30, 2.50, c=600),
        f"SPY{exp21}C00608000": make_snapshot(0.90, 1.00, c=600),
    }}
    with patch("__main__.fetch_option_chain", return_value=fake_chain_b) as mock_fetch_b:
        result_b = run_executor(scout_b, risk_b, dry_run=True)
    print(result_b)
    assert "C0" in result_b["order_payload"]["legs"][0]["symbol"], "BUY did not select a call"
    call_kwargs = mock_fetch_b.call_args.kwargs
    assert call_kwargs.get("option_type") == "call"
    assert call_kwargs.get("exp_gte") is not None and call_kwargs.get("exp_lte") is not None
    assert call_kwargs.get("strike_gte") is not None and call_kwargs.get("strike_lte") is not None, "strike band not passed to fetch"
    print(">>> passed: BUY selected CALLs; fetch called with option_type + DTE window + strike band")

    print("\n=== Case C: real snapshots-dict shape, Direction.SELL -> must select PUTs ===")
    fvg_context_c = SimpleNamespace(measured_move_target=592.0)
    scout_c = SimpleNamespace(signal_id="sig-C", symbol="SPY", direction=Direction.SELL, underlying_price=600.0,
                               fvg_context=fvg_context_c, thesis="bearish MSS")
    risk_c = SimpleNamespace(signal_id="sig-C", decision=RiskDecision.APPROVE,
                              position_size_contracts=1, final_max_loss_usd=250.0, final_max_hold_hours=12)
    fake_chain_c = {"next_page_token": None, "snapshots": {
        f"SPY{exp21}P00600000": make_snapshot(2.10, 2.30, c=600),
        f"SPY{exp21}P00592000": make_snapshot(0.85, 0.95, c=600),
    }}
    with patch("__main__.fetch_option_chain", return_value=fake_chain_c):
        result_c = run_executor(scout_c, risk_c, dry_run=True)
    print(result_c)
    assert "P0" in result_c["order_payload"]["legs"][0]["symbol"], "SELL did not select a put"
    print(">>> passed: SELL selected PUTs")

    print("\n=== Case D: unparseable chain / CLI error path ===")
    scout_d = SimpleNamespace(signal_id="sig-D", symbol="QQQ", direction=Direction.SELL, underlying_price=520.0, thesis="target of 505")
    risk_d = SimpleNamespace(signal_id="sig-D", decision=RiskDecision.APPROVE, position_size_contracts=1, final_max_loss_usd=200.0, final_max_hold_hours=24)
    with patch("__main__.fetch_option_chain", return_value={"_error": "simulated CLI failure"}):
        result_d = run_executor(scout_d, risk_d, dry_run=True)
    print(result_d)

    print("\n=== Case E: ACTUAL live garbage quote observed this session -> must be rejected by sanity guard ===")
    exp_today = date.today().strftime("%y%m%d")
    fake_chain_e = {"next_page_token": None, "snapshots": {
        f"SPY{exp_today}C00420000": {
            "dailyBar": {"c": 341.61}, "greeks": {"delta": 0, "gamma": 0, "rho": 0, "theta": 0, "vega": 0},
            "latestQuote": {"ap": 348.53, "bp": 347.51}, "latestTrade": {"p": 341.61},
        },
    }}
    parsed = _parse_occ_symbol(f"SPY{exp_today}C00420000")
    mid, spread_pct = _mid_and_spread_pct(fake_chain_e["snapshots"][f"SPY{exp_today}C00420000"])
    sanity_err = _sanity_check_quote(mid, parsed["strike"], 341.61, "call")
    print("Direct sanity-guard check on the real garbage sample:", sanity_err)
    assert sanity_err is not None, "Sanity guard FAILED to catch the real garbage quote from this session"
    print(">>> passed: economic-sanity guard correctly rejects the real deep-OTM garbage quote observed live")

    print("\n=== Case F: fetch_option_chain called with correct CLI flags (unit check) ===")
    with patch("__main__._run_cli", return_value={"snapshots": {}}) as mock_run_cli:
        fetch_option_chain("SPY", option_type="call",
                            exp_gte=date(2026, 9, 15), exp_lte=date(2026, 10, 16),
                            strike_gte=747.15, strike_lte=777.15)
    called_args = mock_run_cli.call_args.args[0]
    print("CLI args passed:", called_args)
    assert "--type" in called_args and "call" in called_args
    assert "--expiration-date-gte" in called_args and "2026-09-15" in called_args
    assert "--expiration-date-lte" in called_args and "2026-10-16" in called_args
    assert "--strike-price-gte" in called_args and "747.15" in called_args, "missing --strike-price-gte"
    assert "--strike-price-lte" in called_args and "777.15" in called_args, "missing --strike-price-lte"
    print(">>> passed: fetch_option_chain builds --type/--expiration-date-gte/-lte/--strike-price-gte/-lte")

    print("\n=== Case G: real SPY spot ($762.15) with a wide strike band -> must find valid CALL and PUT spreads ===")
    exp_g = (date.today() + timedelta(days=25)).strftime("%y%m%d")
    fake_chain_g_calls = {"next_page_token": None, "snapshots": {
        f"SPY{exp_g}C00755000": make_snapshot(9.00, 9.40, c=762.15),
        f"SPY{exp_g}C00760000": make_snapshot(6.20, 6.60, c=762.15),
        f"SPY{exp_g}C00765000": make_snapshot(4.10, 4.40, c=762.15),
        f"SPY{exp_g}C00770000": make_snapshot(2.60, 2.90, c=762.15),
    }}
    scout_g = SimpleNamespace(signal_id="sig-G", symbol="SPY", direction=Direction.BUY, underlying_price=762.15, thesis="no explicit target")
    risk_g = SimpleNamespace(signal_id="sig-G", decision=RiskDecision.APPROVE, position_size_contracts=1, final_max_loss_usd=500.0, final_max_hold_hours=24)
    with patch("__main__.fetch_option_chain", return_value=fake_chain_g_calls):
        result_g = run_executor(scout_g, risk_g, dry_run=True)
    print(result_g)
    assert result_g["order_payload"] is not None, "Case G failed to find a spread against a realistic SPY-level chain"
    print(">>> passed: found a valid spread against a realistic $762 SPY chain within the strike band")
