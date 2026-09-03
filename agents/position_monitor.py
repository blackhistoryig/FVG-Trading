"""
agents/position_monitor.py

Deterministic (non-agentic) polling loop that enforces the Risk Guardian's
final, enforceable numbers -- final_max_loss_usd and final_max_hold_hours --
on options positions placed by executor.py.

Design mirrors live_bot.py's SQLite state pattern (same DB_PATH-style env
var convention, same get_db() / CREATE TABLE IF NOT EXISTS style, same
while-True polling loop with a try/except + time.sleep per iteration) but
uses its own table (options_positions) in its own DB file so it never
touches the equity bot's tables (daily_state, symbol_pnl, open_positions,
trade_memory) or its bot_state.db.

Uses the alpaca-py SDK directly (TradingClient), NOT the Alpaca CLI --
matches the project's architecture decision logged on Aug 27: the CLI is
reserved for the Executor's order-placement step (satisfies the
hackathon's required-technology rule), while the SDK is used for the
deterministic FVG core, backtest logic, and Position Monitor (already the
pattern in live_bot.py). alpaca-py is imported lazily inside
_get_trading_client() rather than at module level, so the pure risk-gate
logic below (hold_time_exceeded, loss_limit_breached, DB read/write) can
be unit-tested without the package installed or network access.

Integration: executor.py calls record_submitted_order() below immediately
after a successful (non-dry-run) order submission -- see run_executor()'s
ORDER_SUBMITTED branch. Without that call, a placed order is invisible to
this Monitor and its risk gates will never fire on it.
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("POSITION_MONITOR_DB", "position_monitor.db")
CHECK_INTERVAL_SECONDS = int(os.environ.get("POSITION_MONITOR_INTERVAL_SECONDS", "300"))


def _get_trading_client() -> Any:
    """Lazy import so this module (and its pure logic below) can be
    unit-tested without alpaca-py installed / without network access."""
    from alpaca.trading.client import TradingClient

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not found in environment. "
            "Export them in the terminal (Codespaces) before running "
            "position_monitor.py."
        )
    return TradingClient(api_key, secret_key, paper=True)


# ---------------------------------------------------------------------------
# SQLite persistence -- mirrors live_bot.py's get_db() / CREATE TABLE IF
# NOT EXISTS / ON CONFLICT DO UPDATE pattern, in an isolated table.
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS options_positions (
        signal_id TEXT PRIMARY KEY,
        client_order_id TEXT,
        symbol TEXT,
        contracts INTEGER,
        long_option_symbol TEXT,
        short_option_symbol TEXT,
        final_max_loss_usd REAL,
        final_max_hold_hours REAL,
        submitted_at TEXT,
        status TEXT DEFAULT 'OPEN',
        closed_at TEXT,
        close_reason TEXT,
        exit_pnl_usd REAL
    )""")
    conn.commit()
    return conn


DB = get_db()


def record_submitted_order(
    signal_id: str,
    client_order_id: Optional[str],
    symbol: str,
    contracts: int,
    long_option_symbol: str,
    short_option_symbol: str,
    final_max_loss_usd: float,
    final_max_hold_hours: float,
) -> None:
    """Called by executor.py immediately after a successful (non-dry-run)
    order submission. This is the durable record Position Monitor polls
    against."""
    DB.execute(
        "INSERT INTO options_positions "
        "(signal_id, client_order_id, symbol, contracts, long_option_symbol, "
        "short_option_symbol, final_max_loss_usd, final_max_hold_hours, "
        "submitted_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN') "
        "ON CONFLICT(signal_id) DO UPDATE SET "
        "client_order_id=excluded.client_order_id, status='OPEN'",
        (
            signal_id, client_order_id, symbol, contracts,
            long_option_symbol, short_option_symbol,
            final_max_loss_usd, final_max_hold_hours,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    DB.commit()


def _get_open_positions() -> list[dict]:
    rows = DB.execute(
        "SELECT signal_id, client_order_id, symbol, contracts, "
        "long_option_symbol, short_option_symbol, final_max_loss_usd, "
        "final_max_hold_hours, submitted_at FROM options_positions "
        "WHERE status = 'OPEN'"
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "signal_id": r[0], "client_order_id": r[1], "symbol": r[2],
            "contracts": r[3], "long_option_symbol": r[4],
            "short_option_symbol": r[5], "final_max_loss_usd": r[6],
            "final_max_hold_hours": r[7],
            "submitted_at": datetime.fromisoformat(r[8]),
        })
    return out


def _close_position_record(signal_id: str, reason: str, exit_pnl_usd: Optional[float]) -> None:
    DB.execute(
        "UPDATE options_positions SET status = 'CLOSED', closed_at = ?, "
        "close_reason = ?, exit_pnl_usd = ? WHERE signal_id = ?",
        (datetime.now(timezone.utc).isoformat(), reason, exit_pnl_usd, signal_id),
    )
    DB.commit()


# ---------------------------------------------------------------------------
# Risk-gate checks -- pure functions, unit-testable without hitting Alpaca.
# ---------------------------------------------------------------------------

def hold_time_exceeded(submitted_at: datetime, final_max_hold_hours: float, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(timezone.utc)
    elapsed_hours = (now - submitted_at).total_seconds() / 3600.0
    return elapsed_hours >= final_max_hold_hours


def loss_limit_breached(unrealized_pnl_usd: float, final_max_loss_usd: float) -> bool:
    """unrealized_pnl_usd is signed (negative = losing). final_max_loss_usd
    is a positive dollar cap on how much the spread is allowed to lose."""
    return unrealized_pnl_usd <= -abs(final_max_loss_usd)


def compute_spread_unrealized_pnl(client: Any, long_symbol: str, short_symbol: str) -> Optional[float]:
    """Sums unrealized_pl across both legs of the debit spread. Returns
    None (never a fabricated 0.0) if either leg can't be found -- e.g. one
    leg already closed/expired -- so callers can distinguish "can't tell"
    from "flat"."""
    try:
        positions = {p.symbol: p for p in client.get_all_positions()}
    except Exception as exc:
        print(f"[POSITION MONITOR] Error fetching positions: {exc}")
        return None

    long_pos = positions.get(long_symbol)
    short_pos = positions.get(short_symbol)
    if long_pos is None or short_pos is None:
        return None

    return float(long_pos.unrealized_pl) + float(short_pos.unrealized_pl)


def close_spread_position(client: Any, long_symbol: str, short_symbol: str) -> bool:
    """Closes both legs. Fail-closed: returns False (not silently True) if
    either leg fails to close, so the caller can log/alert rather than
    mark the DB record CLOSED when it may not actually be closed."""
    ok = True
    for occ_symbol in (long_symbol, short_symbol):
        try:
            client.close_position(occ_symbol)
        except Exception as exc:
            print(f"[POSITION MONITOR] Error closing {occ_symbol}: {exc}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def run_once(client: Optional[Any] = None) -> None:
    """Runs a single pass over all OPEN positions, closing any that have
    breached final_max_loss_usd or final_max_hold_hours. Split out from
    run_forever() so it is directly unit-testable with a fake client."""
    client = client or _get_trading_client()
    now = datetime.now(timezone.utc)

    open_positions = _get_open_positions()
    print(f"[POSITION MONITOR] Checked {len(open_positions)} open position(s) at {now.isoformat()}.")

    for pos in open_positions:
        signal_id = pos["signal_id"]
        elapsed_hours = (now - pos["submitted_at"]).total_seconds() / 3600.0

        if hold_time_exceeded(pos["submitted_at"], pos["final_max_hold_hours"], now=now):
            print(f"[POSITION MONITOR] {signal_id}: max hold time "
                  f"({pos['final_max_hold_hours']}h) exceeded ({elapsed_hours:.1f}h elapsed). Closing.")
            pnl = compute_spread_unrealized_pnl(client, pos["long_option_symbol"], pos["short_option_symbol"])
            if close_spread_position(client, pos["long_option_symbol"], pos["short_option_symbol"]):
                _close_position_record(signal_id, "max_hold_time_exceeded", pnl)
                print(f"[POSITION MONITOR] {signal_id}: closed. Final P&L: ${pnl if pnl is not None else 'unknown'}.")
            continue

        pnl = compute_spread_unrealized_pnl(client, pos["long_option_symbol"], pos["short_option_symbol"])
        if pnl is None:
            print(f"[POSITION MONITOR] {signal_id}: could not read live position "
                  f"(already closed/expired?). Marking closed, no P&L recorded.")
            _close_position_record(signal_id, "position_not_found", None)
            continue

        if loss_limit_breached(pnl, pos["final_max_loss_usd"]):
            print(f"[POSITION MONITOR] {signal_id}: loss limit breached "
                  f"(unrealized ${pnl:.2f} vs cap -${pos['final_max_loss_usd']:.2f}). Closing.")
            if close_spread_position(client, pos["long_option_symbol"], pos["short_option_symbol"]):
                _close_position_record(signal_id, "max_loss_exceeded", pnl)
                print(f"[POSITION MONITOR] {signal_id}: closed. Final P&L: ${pnl:.2f}.")
        else:
            print(f"[POSITION MONITOR] {signal_id}: healthy "
                  f"(unrealized ${pnl:.2f} vs cap -${pos['final_max_loss_usd']:.2f}, "
                  f"{elapsed_hours:.1f}h of {pos['final_max_hold_hours']}h hold). No action.")


def run_forever() -> None:
    print("Starting Position Monitor polling loop "
          f"(interval={CHECK_INTERVAL_SECONDS}s)...")
    client = _get_trading_client()
    while True:
        try:
            run_once(client)
        except Exception as exc:
            print(f"[POSITION MONITOR] Error in polling loop: {exc}")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
