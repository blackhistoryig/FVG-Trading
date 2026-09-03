#!/usr/bin/env python3
"""
agents/signal_adapter.py — bridges live_bot.py's VALIDATED 15m FVG detection
to the hackathon pipeline's raw_signal format.

Design rule honored: WRAP the deterministic core, never reimplement it.
Everything that matters mathematically is imported from live_bot.py:
  - data_client + StockBarsRequest pattern (15m bars, 6h lookback, IEX feed)
  - compute_mss_and_displacement (ATR, volume MA, swing highs/lows, MSS)
  - has_momentum_confirmation (displacement candle >= ATR, volume >= MA,
    MSS direction within lookback)
  - the gap rules: MIN_GAP_SIZE=0.15, c1/c3 three-bar gap, gap_level as stop
  - RISK_REWARD_RATIO=2.0 measured-move target
Only the ~30-line per-symbol gap block is replicated here (verbatim logic),
plus signal normalization and its own dedup state.

Importing live_bot is safe: module level only builds clients and DB tables;
run_bot() (the thing that trades equities) only runs under __main__ and is
never called from here. TradingClient is paper=True in live_bot.

Dedup semantics mirror live_bot.is_duplicate_or_cooling_down exactly:
  - same (symbol, direction, gap_level) never re-fires
  - any signal on a symbol starts a 45-minute cooldown for that symbol
State lives in STATE_DIR/fired_signals.json (persistent disk on Render), NOT
in live_bot's SQLite — keeps equity-bot state untouched.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from live_bot import (
    data_client,
    compute_mss_and_displacement,
    has_momentum_confirmation,
    SCANNER_TIMEFRAME,
    MIN_GAP_SIZE,
    ATR_PERIOD,
    RISK_REWARD_RATIO,
    COOLDOWN_MINUTES,
    EST,
)
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed

LOG = logging.getLogger("signal_adapter")

STATE_DIR = Path(os.environ.get("STATE_DIR", Path(__file__).resolve().parent.parent))
SEEN_PATH = STATE_DIR / "fired_signals.json"
COOLDOWN_SECONDS = COOLDOWN_MINUTES * 60
LOOKBACK_HOURS = 6


# ------------------------------------------------------------- dedup state

def _load_seen() -> dict:
    if not SEEN_PATH.exists():
        return {"signals": {}}
    try:
        return json.loads(SEEN_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {"signals": {}}


def _save_seen(seen: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SEEN_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(seen, indent=2))
    tmp.replace(SEEN_PATH)


def _is_duplicate(seen: dict, symbol: str, direction: str, gap_level: float) -> bool:
    """Mirrors live_bot.is_duplicate_or_cooling_down."""
    now = datetime.now(EST)
    mem = seen["signals"].get(symbol)
    if mem is None:
        return False
    last = datetime.fromisoformat(mem["last_trade_time"])
    if (now - last).total_seconds() < COOLDOWN_SECONDS:
        return True
    if mem["direction"] == direction and mem["gap_level"] == gap_level:
        return True
    return False


def _record(seen: dict, symbol: str, direction: str, gap_level: float):
    seen["signals"][symbol] = {
        "direction": direction,
        "gap_level": gap_level,
        "last_trade_time": datetime.now(EST).isoformat(),
    }


# ----------------------------------------------------------------- helpers

def _bar_ts(bar) -> str:
    ts = getattr(bar, "name", None)
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return datetime.now(EST).isoformat()


def _fetch_bars(symbols):
    end = datetime.now(EST)
    start = end - timedelta(hours=LOOKBACK_HOURS)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=SCANNER_TIMEFRAME,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    return data_client.get_stock_bars(req).df


# ------------------------------------------------------------------ poller

def poll_signals(symbols) -> list:
    """Returns raw_signal dicts in the exact shape pipeline.run_pipeline
    accepts (confirmed against agents/pipeline.py's __main__ test signal)."""
    seen = _load_seen()
    out = []

    try:
        df = _fetch_bars(symbols)
    except Exception as e:
        LOG.error("bar fetch failed: %s", e)
        return []
    if df is None or df.empty:
        LOG.info("no bars returned")
        return []

    for symbol in symbols:
        try:
            if isinstance(df.index, pd.MultiIndex):
                if symbol not in df.index.levels[0]:
                    continue
                symbol_df = df.xs(symbol)
            else:
                symbol_df = df
            if len(symbol_df) < max(4, ATR_PERIOD + 2):
                continue

            completed_df = symbol_df.iloc[:-1]
            indexed_df = compute_mss_and_displacement(completed_df)
            i = len(indexed_df) - 1
            c1 = indexed_df.iloc[i - 2]
            c2 = indexed_df.iloc[i - 1]
            c3 = indexed_df.iloc[i]

            sig = None
            if c1["high"] < c3["low"]:
                gap_size = c3["low"] - c1["high"]
                if gap_size >= MIN_GAP_SIZE and has_momentum_confirmation(indexed_df, i, "BULLISH"):
                    gap_level = round(c1["high"], 2)
                    if not _is_duplicate(seen, symbol, "BULLISH", gap_level):
                        entry = float(c3["close"])
                        stop_loss = gap_level
                        risk_per_share = entry - stop_loss
                        if risk_per_share > 0:
                            take_profit = round(entry + risk_per_share * RISK_REWARD_RATIO, 2)
                            sig = ("BUY", "bullish", entry, stop_loss, take_profit, gap_level)

            elif c1["low"] > c3["high"]:
                gap_size = c1["low"] - c3["high"]
                if gap_size >= MIN_GAP_SIZE and has_momentum_confirmation(indexed_df, i, "BEARISH"):
                    gap_level = round(c1["low"], 2)
                    if not _is_duplicate(seen, symbol, "BEARISH", gap_level):
                        entry = float(c3["close"])
                        stop_loss = gap_level
                        risk_per_share = stop_loss - entry
                        if risk_per_share > 0:
                            take_profit = round(entry - risk_per_share * RISK_REWARD_RATIO, 2)
                            sig = ("SELL", "bearish", entry, stop_loss, take_profit, gap_level)

            if sig is None:
                continue

            direction, gap_type, entry, stop_loss, take_profit, gap_level = sig
            # Real displacement metric: displacement candle range in ATR units
            # (the same quantity has_momentum_confirmation tests against 1.0).
            atr = float(c2["atr"]) if not pd.isna(c2["atr"]) else 0.0
            displacement = round(float(c2["candle_range"]) / atr, 2) if atr > 0 else 1.0

            _record(seen, symbol, gap_type.upper(), gap_level)
            out.append({
                "signal_id": f"{symbol}-{direction}-{datetime.now(EST).strftime('%Y%m%d-%H%M')}-{gap_level}",
                "symbol": symbol,
                "direction": direction,
                "underlying_price": entry,
                "fvg_context": {
                    "gap_type": gap_type,
                    "mss_confirmed": True,
                    "displacement_strength": displacement,
                    "measured_move_target": take_profit,
                    "entry_bar_timestamp": _bar_ts(c3),
                },
                "estimated_cost_usd": None,
            })
            LOG.info("FVG signal: %s %s %s gap_level=%.2f target=%.2f displacement=%.2fx ATR",
                     symbol, direction, gap_type, gap_level, take_profit, displacement)

        except Exception as e:
            LOG.error("FVG evaluation failed for %s: %s", symbol, e)
            continue

    if out:
        _save_seen(seen)
    return out
