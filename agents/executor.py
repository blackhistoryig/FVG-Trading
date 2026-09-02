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
    exposes single-leg flags. Multi-leg debit spreads are placed via the
    CLI's documented raw-API escape hatch:
        echo '<json>' | alpaca api POST /v2/orders
    Still "the Alpaca CLI" -- satisfies the required-technology rule.
  - Fail-closed philosophy carried over from scout.py / risk_guardian.py:
    every function either returns a clean, explicit result dict, or logs
    a clear reason and returns a NO_ORDER / ERROR result. Never raises
    uncaught, never fabricates a confirmation.

VERIFIED AGAINST LIVE agents/agent_schemas.py (Sep 1, ~2:24 PM PDT):
  ScoutOutput / RiskGuardianOutput field names confirmed correct (see
  earlier session note). Direction/RiskDecision str(Enum) extraction bug
  found and fixed via _enum_str().

VERIFIED AGAINST LIVE `alpaca data option chain` OUTPUT (Sep 1, ~3:28 PM PDT):
  REAL BUG #2 FOUND AND FIXED: the actual response shape is
  {"next_page_token": ..., "snapshots": {<OCC_SYMBOL>: {dailyBar, greeks,
  latestQuote, latestTrade, minuteBar, prevDailyBar}}} -- `snapshots` is a
  DICT keyed by OCC option symbol, NOT a list of contract dicts with
  strike_price/expiration_date/type fields. Those fields don't exist in
  the value at all; they're encoded in the symbol string itself (e.g.
  "SPY260901C00420000" = SPY, exp 2026-09-01, Call, strike $420.00).
  The quote lives under "latestQuote" (camelCase) with "bp"/"ap" keys,
  not "latest_quote". Original code assumed a list of pre-decoded
  contract dicts and would have crashed with AttributeError/KeyError on
  the very first real call -- confirmed live: 'str' object has no
  attribute 'get' at the old c.get("type") line, because iterating a
  dict of {symbol: data} the old way iterated over symbol STRINGS, not
  dicts. Fixed by adding _parse_occ_symbol() to decode the standard OCC
  symbol format, and normalizing both the dict-of-snapshots shape and
  any hypothetical list-of-contracts shape into one candidate format
  before selection logic runs.

  SEPARATE DATA-QUALITY RED FLAG FOUND WHILE VERIFYING (not a code bug --
  a real data hazard worth guarding against): a live sample contract
  (SPY $420 call, 0 DTE, all-zero greeks) had latestQuote bp/ap of
  ~347.51/348.53 while the underlying (per dailyBar/latestTrade) was
  trading around ~341.61 -- i.e. a deep OUT-of-the-money call quoted at
  MORE than the underlying's own price. That is not a real, tradeable
  quote; it's stale/placeholder data on an illiquid, zero-volume
  contract, and the existing bid/ask-SPREAD-WIDTH liquidity filter does
  NOT catch it (the tight bp/ap spread on that garbage quote would have
  passed cleanly). Added a new economic-sanity guard,
  _sanity_check_quote(), that rejects any leg whose extrinsic value
  (mid price minus intrinsic value) exceeds a generous heuristic cap
  relative to the underlying price. This is a pragmatic, fast
  "good-enough" guard (not backtest-derived), explicitly flagged as
  tunable rather than as a validated parameter.

KNOWN OPEN ITEM: Executor does not yet persist final_max_hold_hours
anywhere -- Position Monitor (not yet built) is the piece meant to
enforce it on open positions.

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

# Economic-sanity guard (pragmatic heuristic, NOT backtest-derived -- see
# module docstring for the deep-OTM contract quoted above spot price).
MAX_PLAUSIBLE_EXTRINSIC_PCT_OF_SPOT = 0.25

OCC_SYMBOL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


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
    into its plain string form. See module docstring for the bug this fixes."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _extract_price_target(scout_output: Any) -> Optional[float]:
    """Best-effort extraction of a measured-move / target price from Scout's
    output (backtest bug-fix #1: fixed-width spreads mismatched actual price
    targets). FVGContext.measured_move_target is the confirmed real field."""
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
    Never raises -- returns {"_error": "..."} on any failure."""
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
    """`alpaca data option chain --underlying-symbol <symbol> --quiet`.
    Confirmed live shape: {"next_page_token": ..., "snapshots": {OCC_SYMBOL: {...}}}"""
    return _run_cli(["data", "option", "chain", "--underlying-symbol", symbol, "--quiet"])


# ---------------------------------------------------------------------------
# OCC symbol decoding (confirmed necessary: snapshots don't carry
# strike/expiration/type fields separately -- they're IN the symbol)
# ---------------------------------------------------------------------------
def _parse_occ_symbol(symbol: str) -> Optional[dict]:
    """Decode a standard OCC option symbol, e.g. 'SPY260901C00420000' ->
    root='SPY', expiration=date(2026,9,1), option_type='call', strike=420.0."""
    m = OCC_SYMBOL_RE.match(symbol)
    if not m:
        return None
    root, date_str, cp, strike_str = m.groups()
    try:
        exp = datetime.strptime(date_str, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "root": root,
        "expiration": exp,
        "option_type": "call" if cp == "C" else "put",
        "strike": int(strike_str) / 1000.0,
    }


def _extract_quote(raw: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract (bid, ask) from a snapshot dict. Confirmed live key is
    'latestQuote' (camelCase) with 'bp'/'ap' -- also tolerates
    'latest_quote'/'bid_price'/'ask_price' in case of a future API change
    or a differently-shaped contracts-list response."""
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
    """Pragmatic economic-sanity guard (heuristic, not backtest-derived).
    Found necessary live: a deep-OTM SPY $420 call quoted ~$347-348 while
    the underlying was ~$341 -- more than the underlying itself, on an
    illiquid all-zero-greeks 0DTE contract. A tight bid/ask SPREAD does
    NOT catch this; this checks the PRICE LEVEL itself against intrinsic
    value. Returns an error string if implausible, else None."""
    if option_type == "call":
        intrinsic = max(underlying_price - strike, 0.0)
    else:
        intrinsic = max(strike - underlying_price, 0.0)
    extrinsic = mid - intrinsic
    cap = underlying_price * MAX_PLAUSIBLE_EXTRINSIC_PCT_OF_SPOT
    if extrinsic > cap:
        return (f"implausible extrinsic value ${extrinsic:.2f} (mid ${mid:.2f}, intrinsic ${intrinsic:.2f}) "
                f"exceeds sanity cap ${cap:.2f} ({MAX_PLAUSIBLE_EXTRINSIC_PCT_OF_SPOT:.0%} of spot ${underlying_price:.2f}) "
                f"-- likely stale/illiquid quote, not a real tradeable price")
    if extrinsic < -0.05:
        return f"mid price ${mid:.2f} is below intrinsic value ${intrinsic:.2f} -- likely bad/stale quote"
    return None


# ---------------------------------------------------------------------------
# Contract selection
# ---------------------------------------------------------------------------
def _normalize_chain_items(chain: dict) -> list[dict]:
    """Normalize either shape into a flat list of dicts, each with a
    'symbol' key and enough metadata (via OCC decode if needed) to filter
    on. Confirmed live shape is snapshots-as-dict; option_contracts/
    contracts-as-list is kept as a fallback in case a different CLI
    subcommand or a future API version returns pre-decoded fields."""
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
    """Select a long debit spread (call spread if bullish/BUY, put spread if
    bearish/SELL) per the MVP Contract Selection Rules:
      - 14-45 DTE, no same-day expiry
      - buy near-ATM/one-strike-ITM, sell 1-2 strikes further OTM
      - target-aware width when a measured-move target is available,
        else DEFAULT_SPREAD_WIDTH
      - reject wide bid/ask spreads, economically implausible quotes, or an
        implied debit above max_loss_usd

    `direction` may be a plain string ("BUY"/"SELL") or a Direction enum
    instance -- routed through _enum_str() so either works correctly.
    """
    is_bullish = _enum_str(direction).upper() in ("BUY", "BULLISH", "CALL")
    option_type = "call" if is_bullish else "put"

    chain = fetch_option_chain(symbol)
    if "_error" in chain:
        return {"_error": chain["_error"]}

    all_items = _normalize_chain_items(chain)
    if not all_items:
        return {"_error": f"option chain for {symbol} returned no parseable contracts"}

    today = date.today()
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
        "option_type": option_type,
        "expiration": chosen_exp.isoformat(),
        "dte": long_leg["_dte"],
        "long_symbol": long_leg["symbol"],
        "long_strike": long_leg["_strike"],
        "short_symbol": short_leg["symbol"],
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
    return {
        "type": "market",
        "time_in_force": "day",
        "order_class": "mleg",
        "qty": str(qty),
        "client_order_id": client_order_id,
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
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
    log.info("Executor result for signal_id=%s: %s (%s)", signal_id, action, reason)
    return ExecResult(action=action, reason=reason, order_payload=order_payload,
                       cli_response=cli_response, signal_id=signal_id,
                       final_max_hold_hours=final_max_hold_hours).to_dict()


# ---------------------------------------------------------------------------
# Smoke test -- Case E uses the ACTUAL live sample data pasted during this
# session (garbage deep-OTM quote) to prove the sanity guard catches it.
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
        f"SPY{exp21}P00600000": make_snapshot(2.10, 2.30, c=600),
    }}
    with patch("__main__.fetch_option_chain", return_value=fake_chain_b):
        result_b = run_executor(scout_b, risk_b, dry_run=True)
    print(result_b)
    assert "C0" in result_b["order_payload"]["legs"][0]["symbol"], "BUY did not select a call"
    print(">>> passed: real snapshots-dict shape parsed, BUY selected CALLs")

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
    print(">>> passed: real snapshots-dict shape parsed, SELL selected PUTs")

    print("\n=== Case D: unparseable chain / CLI error path ===")
    scout_d = SimpleNamespace(signal_id="sig-D", symbol="QQQ", direction=Direction.SELL, underlying_price=520.0, thesis="target of 505")
    risk_d = SimpleNamespace(signal_id="sig-D", decision=RiskDecision.APPROVE, position_size_contracts=1, final_max_loss_usd=200.0, final_max_hold_hours=24)
    with patch("__main__.fetch_option_chain", return_value={"_error": "simulated CLI failure"}):
        result_d = run_executor(scout_d, risk_d, dry_run=True)
    print(result_d)

    print("\n=== Case E: ACTUAL live garbage quote observed this session -> must be rejected by sanity guard ===")
    exp_today = date.today().strftime("%y%m%d")
    scout_e = SimpleNamespace(signal_id="sig-E", symbol="SPY", direction=Direction.BUY, underlying_price=341.61, thesis="0DTE test")
    risk_e = SimpleNamespace(signal_id="sig-E", decision=RiskDecision.APPROVE, position_size_contracts=1, final_max_loss_usd=500.0, final_max_hold_hours=6)
    fake_chain_e = {"next_page_token": None, "snapshots": {
        f"SPY{exp_today}C00420000": {
            "dailyBar": {"c": 341.61}, "greeks": {"delta": 0, "gamma": 0, "rho": 0, "theta": 0, "vega": 0},
            "latestQuote": {"ap": 348.53, "bp": 347.51}, "latestTrade": {"p": 341.61},
        },
        f"SPY{exp_today}C00425000": {
            "dailyBar": {"c": 341.61}, "greeks": {"delta": 0, "gamma": 0, "rho": 0, "theta": 0, "vega": 0},
            "latestQuote": {"ap": 349.00, "bp": 348.00}, "latestTrade": {"p": 341.61},
        },
    }}
    parsed = _parse_occ_symbol(f"SPY{exp_today}C00420000")
    mid, spread_pct = _mid_and_spread_pct(fake_chain_e["snapshots"][f"SPY{exp_today}C00420000"])
    sanity_err = _sanity_check_quote(mid, parsed["strike"], 341.61, "call")
    print("Direct sanity-guard check on the real garbage sample:", sanity_err)
    assert sanity_err is not None, "Sanity guard FAILED to catch the real garbage quote from this session"
    print(">>> passed: economic-sanity guard correctly rejects the real deep-OTM garbage quote observed live")

    with patch("__main__.fetch_option_chain", return_value=fake_chain_e):
        result_e = run_executor(scout_e, risk_e, dry_run=True)
    print("Full pipeline result (also correctly rejected, via DTE filter for this 0DTE sample):", result_e)
