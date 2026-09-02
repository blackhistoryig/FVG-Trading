import os
import time
import threading
import json
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest, ReplaceOrderRequest, GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"OK"
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', '2')
        self.send_header('Connection', 'close')
        self.end_headers()

    def log_message(self, format, *args):
        return

def run_http_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check web server running on port {port}")
    server.serve_forever()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

SYMBOLS = ["QQQ", "NVDA", "SPY", "AAPL", "TSLA", "SMH", "IWM"]
MAX_POSITIONS = 2
RISK_REWARD_RATIO = 2.0
CHECK_INTERVAL_SECONDS = 300

SCANNER_TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)
BAR_MINUTES = 15
MIN_GAP_SIZE = 0.15

RISK_PER_TRADE_PCT = 0.01
MAX_POSITION_ALLOCATION = 0.25
DAILY_MAX_LOSS_PCT = 0.03
SYMBOL_MAX_LOSS_PCT = 0.015

MSS_SWING_WINDOW = 2
MSS_LOOKBACK_BARS = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.0
VOL_PERIOD = 20
VOL_MULTIPLIER = 1.0

STAGNATION_CHECK_BAR = 12
STAGNATION_BAND_PCT = 0.30
MAX_BARS_IN_TRADE = 20

EST = pytz.timezone("America/New_York")

STATIC_2026_FOMC_DATES = {
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16"
}
FOMC_MEETING_DATES = STATIC_2026_FOMC_DATES

DB_PATH = os.getenv("BOT_STATE_DB", "bot_state.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS daily_state (
        day TEXT PRIMARY KEY,
        start_equity REAL,
        circuit_breaker_tripped INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS symbol_pnl (
        day TEXT, symbol TEXT, realized_pnl REAL DEFAULT 0,
        suspended INTEGER DEFAULT 0,
        PRIMARY KEY (day, symbol)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS open_positions (
        symbol TEXT PRIMARY KEY,
        side TEXT, qty REAL, entry_price REAL, entry_time TEXT,
        stop_loss REAL, take_profit REAL, stop_order_id TEXT,
        already_tightened INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trade_memory (
        symbol TEXT PRIMARY KEY,
        direction TEXT, gap_level REAL, last_trade_time TEXT
    )""")
    conn.commit()
    return conn

DB = get_db()

def db_get_daily_state(today_str: str):
    row = DB.execute("SELECT start_equity, circuit_breaker_tripped FROM daily_state WHERE day = ?", (today_str,)).fetchone()
    return row

def db_set_daily_state(today_str: str, start_equity: float, tripped: bool):
    DB.execute(
        "INSERT INTO daily_state (day, start_equity, circuit_breaker_tripped) VALUES (?, ?, ?) "
        "ON CONFLICT(day) DO UPDATE SET start_equity=excluded.start_equity, circuit_breaker_tripped=excluded.circuit_breaker_tripped",
        (today_str, start_equity, int(tripped))
    )
    DB.commit()

def db_get_symbol_pnl(today_str: str, symbol: str) -> tuple[float, bool]:
    row = DB.execute("SELECT realized_pnl, suspended FROM symbol_pnl WHERE day = ? AND symbol = ?", (today_str, symbol)).fetchone()
    if row is None:
        return 0.0, False
    return row[0], bool(row[1])

def db_add_symbol_pnl(today_str: str, symbol: str, pnl_delta: float, equity: float):
    current_pnl, _ = db_get_symbol_pnl(today_str, symbol)
    new_pnl = current_pnl + pnl_delta
    suspended = new_pnl <= -abs(equity * SYMBOL_MAX_LOSS_PCT)
    DB.execute(
        "INSERT INTO symbol_pnl (day, symbol, realized_pnl, suspended) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(day, symbol) DO UPDATE SET realized_pnl=excluded.realized_pnl, suspended=excluded.suspended",
        (today_str, symbol, new_pnl, int(suspended))
    )
    DB.commit()
    if suspended:
        print(f"[SYMBOL CIRCUIT BREAKER] {symbol} realized P&L today = ${new_pnl:.2f} "
              f"(<= -{SYMBOL_MAX_LOSS_PCT*100:.1f}% of equity). New entries on {symbol} suspended for rest of day.")
    return new_pnl, suspended

def db_is_symbol_suspended(today_str: str, symbol: str) -> bool:
    _, suspended = db_get_symbol_pnl(today_str, symbol)
    return suspended

def db_save_position(symbol, side, qty, entry_price, entry_time, stop_loss, take_profit, stop_order_id):
    DB.execute(
        "INSERT INTO open_positions (symbol, side, qty, entry_price, entry_time, stop_loss, take_profit, stop_order_id, already_tightened) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0) "
        "ON CONFLICT(symbol) DO UPDATE SET side=excluded.side, qty=excluded.qty, entry_price=excluded.entry_price, "
        "entry_time=excluded.entry_time, stop_loss=excluded.stop_loss, take_profit=excluded.take_profit, stop_order_id=excluded.stop_order_id",
        (symbol, side, qty, entry_price, entry_time.isoformat(), stop_loss, take_profit, stop_order_id)
    )
    DB.commit()

def db_mark_tightened(symbol):
    DB.execute("UPDATE open_positions SET already_tightened = 1 WHERE symbol = ?", (symbol,))
    DB.commit()

def db_remove_position(symbol):
    DB.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))
    DB.commit()

def db_get_tracked_positions() -> dict:
    rows = DB.execute("SELECT symbol, side, qty, entry_price, entry_time, stop_loss, take_profit, stop_order_id, already_tightened FROM open_positions").fetchall()
    out = {}
    for r in rows:
        out[r[0]] = {
            "side": r[1], "qty": r[2], "entry_price": r[3],
            "entry_time": datetime.fromisoformat(r[4]), "stop_loss": r[5],
            "take_profit": r[6], "stop_order_id": r[7], "already_tightened": bool(r[8])
        }
    return out

def db_get_trade_memory(symbol):
    row = DB.execute("SELECT direction, gap_level, last_trade_time FROM trade_memory WHERE symbol = ?", (symbol,)).fetchone()
    if row is None:
        return None
    return {"direction": row[0], "gap_level": row[1], "last_trade_time": datetime.fromisoformat(row[2])}

def db_set_trade_memory(symbol, direction, gap_level):
    now_iso = datetime.now(EST).isoformat()
    DB.execute(
        "INSERT INTO trade_memory (symbol, direction, gap_level, last_trade_time) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(symbol) DO UPDATE SET direction=excluded.direction, gap_level=excluded.gap_level, last_trade_time=excluded.last_trade_time",
        (symbol, direction, gap_level, now_iso)
    )
    DB.commit()

CURRENT_DAY = None
DAILY_START_EQUITY = None
CIRCUIT_BREAKER_TRIPPED = False
COOLDOWN_MINUTES = 45

def check_and_reset_daily_state():
    global DAILY_START_EQUITY, CIRCUIT_BREAKER_TRIPPED, CURRENT_DAY
    today_str = datetime.now(EST).strftime("%Y-%m-%d")

    if CURRENT_DAY == today_str:
        return

    existing = db_get_daily_state(today_str)
    if existing is not None:
        DAILY_START_EQUITY, tripped_int = existing
        CIRCUIT_BREAKER_TRIPPED = bool(tripped_int)
        CURRENT_DAY = today_str
        print(f"[{datetime.now(EST).strftime('%H:%M:%S EST')}] Rehydrated existing daily state for {CURRENT_DAY}. "
              f"Baseline Equity: ${DAILY_START_EQUITY:,.2f} | Circuit Breaker Tripped: {CIRCUIT_BREAKER_TRIPPED}")
        return

    CURRENT_DAY = today_str
    CIRCUIT_BREAKER_TRIPPED = False
    try:
        account = trading_client.get_account()
        DAILY_START_EQUITY = float(account.equity)
        db_set_daily_state(today_str, DAILY_START_EQUITY, False)
        print(f"[{datetime.now(EST).strftime('%H:%M:%S EST')}] New Day Started ({CURRENT_DAY}). Baseline Equity: ${DAILY_START_EQUITY:,.2f}")
    except Exception as e:
        print(f"Error fetching account for daily state reset: {e}")

def is_fed_blackout_active() -> bool:
    now = datetime.now(EST)
    today_str = now.strftime("%Y-%m-%d")
    if today_str not in FOMC_MEETING_DATES:
        return False
    start_blackout = now.replace(hour=13, minute=55, second=0, microsecond=0)
    end_blackout = now.replace(hour=14, minute=45, second=0, microsecond=0)
    return start_blackout <= now <= end_blackout

def is_circuit_breaker_tripped() -> bool:
    global CIRCUIT_BREAKER_TRIPPED
    if CIRCUIT_BREAKER_TRIPPED:
        return True
    try:
        account = trading_client.get_account()
        current_equity = float(account.equity)
        if DAILY_START_EQUITY and DAILY_START_EQUITY > 0:
            loss_pct = (current_equity - DAILY_START_EQUITY) / DAILY_START_EQUITY
            if loss_pct <= -DAILY_MAX_LOSS_PCT:
                print(f"[CIRCUIT BREAKER TRIPPED] Loss of {loss_pct*100:.2f}% exceeds max allowed (-{DAILY_MAX_LOSS_PCT*100}%). Halting trading today.")
                close_all_positions_and_orders()
                CIRCUIT_BREAKER_TRIPPED = True
                db_set_daily_state(CURRENT_DAY, DAILY_START_EQUITY, True)
                return True
    except Exception as e:
        print(f"Error checking circuit breaker: {e}")
    return False

def calculate_dynamic_position_size(symbol: str, current_price: float, risk_per_share: float) -> int:
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        risk_budget = equity * RISK_PER_TRADE_PCT
        if risk_per_share <= 0:
            return 1
        shares_by_risk = int(risk_budget / risk_per_share)
        max_capital_allowed = equity * MAX_POSITION_ALLOCATION
        shares_by_cap = int(max_capital_allowed / current_price)
        final_shares = min(shares_by_risk, shares_by_cap)
        final_shares = max(1, final_shares)
        print(f"[SIZING] {symbol}: Risk Target = {shares_by_risk} sh | 25% Portfolio Cap = {shares_by_cap} sh | Final Order = {final_shares} sh")
        return final_shares
    except Exception as e:
        print(f"Error calculating dynamic position size: {e}")
        return 1

def is_market_close_approaching() -> bool:
    now = datetime.now(EST)
    cutoff = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return now >= cutoff

def is_eod_flatten_time() -> bool:
    now = datetime.now(EST)
    flatten_time = now.replace(hour=15, minute=55, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return flatten_time <= now <= market_close

def close_all_positions_and_orders():
    try:
        print("[FLATTEN] Closing all positions and cancelling active orders...")
        trading_client.cancel_orders()
        trading_client.close_all_positions(cancel_orders=True)
        for symbol in list(db_get_tracked_positions().keys()):
            db_remove_position(symbol)
        print("[FLATTEN] All positions and orders successfully closed!")
    except Exception as e:
        print(f"[FLATTEN] Error closing positions: {e}")

def get_open_position_count() -> int:
    try:
        positions = trading_client.get_all_positions()
        return len(positions)
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return 0

def compute_mss_and_displacement(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
    df['candle_range'] = df['high'] - df['low']
    df['vol_ma'] = df['volume'].rolling(window=VOL_PERIOD).mean()

    df['is_swing_high'] = False
    df['is_swing_low'] = False
    w = MSS_SWING_WINDOW
    for i in range(w, len(df) - w):
        if df['high'].iloc[i] == df['high'].iloc[i - w: i + w + 1].max():
            df.at[df.index[i], 'is_swing_high'] = True
        if df['low'].iloc[i] == df['low'].iloc[i - w: i + w + 1].min():
            df.at[df.index[i], 'is_swing_low'] = True

    df['mss_signal'] = None
    active_sh, active_sl = None, None
    for i in range(len(df)):
        curr_close = df.at[df.index[i], 'close']
        if df.at[df.index[i], 'is_swing_high']:
            active_sh = df.at[df.index[i], 'high']
        if df.at[df.index[i], 'is_swing_low']:
            active_sl = df.at[df.index[i], 'low']
        if active_sh is not None and curr_close > active_sh:
            df.at[df.index[i], 'mss_signal'] = 'BULLISH'
            active_sh = None
        elif active_sl is not None and curr_close < active_sl:
            df.at[df.index[i], 'mss_signal'] = 'BEARISH'
            active_sl = None
    return df

def has_momentum_confirmation(df: pd.DataFrame, i: int, direction: str, symbol: str = "?") -> bool:
    c2_idx = df.index[i - 1]
    atr = df.at[c2_idx, 'atr']
    vol_ma = df.at[c2_idx, 'vol_ma']
    if pd.isna(atr) or pd.isna(vol_ma) or atr == 0 or vol_ma == 0:
        print(f"[MOMENTUM_REJECT] {symbol} dir={direction}: NaN/zero base stats -- "
              f"atr={atr}, vol_ma={vol_ma} (insufficient warmup data in lookback window)")
        return False

    candle_range = df.at[c2_idx, 'candle_range']
    candle_volume = df.at[c2_idx, 'volume']
    range_ok = candle_range >= atr * ATR_MULTIPLIER
    vol_ok = candle_volume >= vol_ma * VOL_MULTIPLIER
    has_displacement = range_ok and vol_ok
    if not has_displacement:
        print(f"[MOMENTUM_REJECT] {symbol} dir={direction}: displacement failed -- "
              f"range={candle_range:.4f} vs atr_threshold={atr * ATR_MULTIPLIER:.4f} (ok={range_ok}) | "
              f"volume={candle_volume:.0f} vs vol_threshold={vol_ma * VOL_MULTIPLIER:.0f} (ok={vol_ok})")
        return False

    start_loc = max(0, i - MSS_LOOKBACK_BARS)
    recent_mss = df.iloc[start_loc:i + 1]['mss_signal'].dropna().tolist()
    mss_ok = direction in recent_mss
    if not mss_ok:
        print(f"[MOMENTUM_REJECT] {symbol} dir={direction}: displacement passed but no matching "
              f"MSS break in last {MSS_LOOKBACK_BARS} bars (found: {recent_mss})")
        return False

    print(f"[MOMENTUM_PASS] {symbol} dir={direction}: all conditions met")
    return True

def is_duplicate_or_cooling_down(symbol: str, direction: str, gap_level: float) -> bool:
    now = datetime.now(EST)
    mem = db_get_trade_memory(symbol)
    if mem is None:
        return False
    if (now - mem["last_trade_time"]).total_seconds() < COOLDOWN_MINUTES * 60:
        return True
    if mem["direction"] == direction and mem["gap_level"] == gap_level:
        return True
    return False

def record_trade_execution(symbol: str, direction: str, gap_level: float):
    db_set_trade_memory(symbol, direction, gap_level)

def check_for_fvg_batch(symbols: list):
    signals = []
    tracked_open = db_get_tracked_positions()
    try:
        end_time = datetime.now(EST)
        start_time = end_time - timedelta(hours=6)

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=SCANNER_TIMEFRAME,
            start=start_time,
            end=end_time,
            feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(request_params)
        df = bars.df
        if df.empty:
            return signals

        for symbol in symbols:
            try:
                if symbol in tracked_open:
                    continue

                if db_is_symbol_suspended(CURRENT_DAY, symbol):
                    continue

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
                c3 = indexed_df.iloc[i]

                if c1['high'] < c3['low']:
                    gap_size = c3['low'] - c1['high']
                    if gap_size < MIN_GAP_SIZE:
                        continue
                    print(f"[RAW_GAP] {symbol} BULLISH gap_size={gap_size:.4f} "
                          f"(min={MIN_GAP_SIZE}) at bar_time={indexed_df.index[i]}")
                    if not has_momentum_confirmation(indexed_df, i, 'BULLISH', symbol):
                        continue

                    gap_level = round(c1['high'], 2)
                    if is_duplicate_or_cooling_down(symbol, "BULLISH", gap_level):
                        continue

                    current_price = c3['close']
                    stop_loss = gap_level
                    risk_per_share = current_price - stop_loss
                    if risk_per_share > 0:
                        take_profit = round(current_price + (risk_per_share * RISK_REWARD_RATIO), 2)
                        signals.append({
                            "symbol": symbol, "type": "BULLISH_FVG (15m, MSS+displacement confirmed)",
                            "side": OrderSide.BUY, "entry": current_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "risk_per_share": risk_per_share,
                            "direction": "BULLISH", "gap_level": gap_level
                        })

                elif c1['low'] > c3['high']:
                    gap_size = c1['low'] - c3['high']
                    if gap_size < MIN_GAP_SIZE:
                        continue
                    print(f"[RAW_GAP] {symbol} BEARISH gap_size={gap_size:.4f} "
                          f"(min={MIN_GAP_SIZE}) at bar_time={indexed_df.index[i]}")
                    if not has_momentum_confirmation(indexed_df, i, 'BEARISH', symbol):
                        continue

                    gap_level = round(c1['low'], 2)
                    if is_duplicate_or_cooling_down(symbol, "BEARISH", gap_level):
                        continue

                    current_price = c3['close']
                    stop_loss = gap_level
                    risk_per_share = stop_loss - current_price
                    if risk_per_share > 0:
                        take_profit = round(current_price - (risk_per_share * RISK_REWARD_RATIO), 2)
                        signals.append({
                            "symbol": symbol, "type": "BEARISH_FVG (15m, MSS+displacement confirmed)",
                            "side": OrderSide.SELL, "entry": current_price, "stop_loss": stop_loss,
                            "take_profit": take_profit, "risk_per_share": risk_per_share,
                            "direction": "BEARISH", "gap_level": gap_level
                        })

            except Exception as inner_e:
                print(f"Error evaluating FVG for {symbol}: {inner_e}")
                continue

        return signals

    except Exception as e:
        print(f"Error during batch bar fetch: {e}")
        return []

def find_stop_order_id(symbol: str):
    try:
        request_params = GetOrdersRequest(
            symbols=[symbol], status=QueryOrderStatus.ALL, nested=True, limit=5
        )
        orders = trading_client.get_orders(filter=request_params)
        for o in orders:
            legs = getattr(o, "legs", None) or []
            for leg in legs:
                if getattr(leg, "type", None) and "stop" in str(leg.type).lower() and leg.status not in ("filled", "canceled"):
                    return str(leg.id)
    except Exception as e:
        print(f"[{symbol}] Could not resolve stop order id: {e}")
    return None

def execute_bracket_order(symbol: str, side: OrderSide, qty: int, entry_price: float, stop_loss_price: float, take_profit_price: float):
    live_price = entry_price
    try:
        latest_quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        latest_quote = data_client.get_stock_latest_quote(latest_quote_req)
        fetched_price = float(latest_quote[symbol].ask_price) if side == OrderSide.BUY else float(latest_quote[symbol].bid_price)
        if fetched_price > 0:
            live_price = fetched_price
    except Exception as e:
        print(f"[{symbol}] Dynamic quote lookup failed ({e}). Using bar entry price (${entry_price}).")

    MIN_BUFFER = 0.15
    if side == OrderSide.BUY:
        if stop_loss_price >= live_price:
            stop_loss_price = round(max(0.01, live_price - MIN_BUFFER), 2)
    elif side == OrderSide.SELL:
        if stop_loss_price <= live_price:
            stop_loss_price = round(live_price + MIN_BUFFER, 2)

    if stop_loss_price <= 0 or take_profit_price <= 0:
        print(f"[{symbol}] Aborted order placement: Invalid SL/TP calculated (SL: ${stop_loss_price}, TP: ${take_profit_price}).")
        return

    order_data = MarketOrderRequest(
        symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price)
    )

    try:
        order = trading_client.submit_order(order_data)
        print(f"Successfully placed bracket order for {symbol} ({qty} shares): ID {order.id}")
        time.sleep(2)
        stop_order_id = find_stop_order_id(symbol)
        db_save_position(
            symbol=symbol, side=("long" if side == OrderSide.BUY else "short"),
            qty=qty, entry_price=live_price, entry_time=datetime.now(EST),
            stop_loss=stop_loss_price, take_profit=take_profit_price, stop_order_id=stop_order_id
        )
    except Exception as e:
        print(f"Failed to place order for {symbol}: {e}")

def reconcile_and_manage_positions():
    tracked = db_get_tracked_positions()
    try:
        live_positions = {p.symbol: p for p in trading_client.get_all_positions()}
    except Exception as e:
        print(f"Error fetching live positions for reconciliation: {e}")
        return

    for symbol, pos in live_positions.items():
        if symbol not in tracked:
            side = "long" if float(pos.qty) > 0 else "short"
            db_save_position(
                symbol=symbol, side=side, qty=abs(float(pos.qty)),
                entry_price=float(pos.avg_entry_price), entry_time=datetime.now(EST),
                stop_loss=None, take_profit=None, stop_order_id=find_stop_order_id(symbol)
            )
            tracked = db_get_tracked_positions()

    for symbol, info in list(tracked.items()):
        if symbol not in live_positions:
            try:
                account = trading_client.get_account()
                equity = float(account.equity)
                closed_request_params = GetOrdersRequest(
                    symbols=[symbol], status=QueryOrderStatus.CLOSED, limit=3
                )
                closed_orders = trading_client.get_orders(filter=closed_request_params)
                exit_price = None
                for o in closed_orders:
                    if o.filled_avg_price and str(o.side).lower().endswith(("buy", "sell")):
                        exit_price = float(o.filled_avg_price)
                        break
                if exit_price is not None:
                    sign = 1 if info["side"] == "long" else -1
                    pnl = sign * (exit_price - info["entry_price"]) * info["qty"]
                    db_add_symbol_pnl(CURRENT_DAY, symbol, pnl, equity)
                    print(f"[RECONCILE] {symbol} closed. Realized P&L this trade: ${pnl:.2f}")
            except Exception as e:
                print(f"[RECONCILE] Could not compute realized P&L for {symbol}: {e}")
            db_remove_position(symbol)

    tracked = db_get_tracked_positions()
    for symbol, info in tracked.items():
        if info["stop_loss"] is None or info["take_profit"] is None:
            continue

        bars_since_entry = int((datetime.now(EST) - info["entry_time"]).total_seconds() // (BAR_MINUTES * 60))
        try:
            latest_quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            quote = data_client.get_stock_latest_quote(latest_quote_req)
            current_price = float(quote[symbol].bid_price if info["side"] == "long" else quote[symbol].ask_price)
        except Exception:
            continue

        sign = 1 if info["side"] == "long" else -1
        unrealized_pnl = sign * (current_price - info["entry_price"]) * info["qty"]
        initial_risk = abs(info["entry_price"] - info["stop_loss"]) * info["qty"]
        if initial_risk <= 0:
            continue

        if bars_since_entry >= MAX_BARS_IN_TRADE:
            print(f"[EXIT STACK] {symbol}: hit max bar cap ({bars_since_entry} bars). Flattening ahead of EOD.")
            try:
                trading_client.cancel_orders()
                trading_client.close_position(symbol)
            except Exception as e:
                print(f"[EXIT STACK] Error flattening {symbol}: {e}")
            db_remove_position(symbol)
            continue

        if (bars_since_entry >= STAGNATION_CHECK_BAR
                and not info["already_tightened"]
                and abs(unrealized_pnl) < STAGNATION_BAND_PCT * initial_risk
                and info["stop_order_id"]):
            new_stop = round(info["entry_price"], 2)
            print(f"[EXIT STACK] {symbol}: stagnant after {bars_since_entry} bars "
                  f"(unrealized ${unrealized_pnl:.2f} vs risk ${initial_risk:.2f}). Tightening stop to breakeven (${new_stop}).")
            try:
                trading_client.replace_order_by_id(
                    order_id=info["stop_order_id"],
                    order_data=ReplaceOrderRequest(stop_price=new_stop)
                )
                db_mark_tightened(symbol)
            except Exception as e:
                print(f"[EXIT STACK] Error tightening stop for {symbol}: {e}")

def run_bot():
    print("Starting 15-Minute FVG Engine loop...")
    while True:
        try:
            now_est = datetime.now(EST)
            check_and_reset_daily_state()

            if is_circuit_breaker_tripped():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Circuit Breaker active. Bot paused for remainder of day.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            if is_eod_flatten_time():
                close_all_positions_and_orders()
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            reconcile_and_manage_positions()

            if is_market_close_approaching():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Past 3:45 PM EST cutoff. No new positions permitted.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            if is_fed_blackout_active():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] FOMC Announcement Day: Fed Blackout Active. Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            open_positions = get_open_position_count()
            if open_positions >= MAX_POSITIONS:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Position cap reached ({open_positions}/{MAX_POSITIONS}). Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            print(f"[{now_est.strftime('%H:%M:%S EST')}] Scanning {len(SYMBOLS)} symbols on 15m timeframe (momentum-confirmed)...")
            signals = check_for_fvg_batch(SYMBOLS)

            if not signals:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Scan complete. No confirmed 15m signals.")
            else:
                for signal in signals:
                    print(f"[{now_est.strftime('%H:%M:%S EST')}] {signal['type']} detected on {signal['symbol']}!")
                    qty = calculate_dynamic_position_size(
                        symbol=signal['symbol'], current_price=signal['entry'], risk_per_share=signal['risk_per_share']
                    )
                    execute_bracket_order(
                        symbol=signal['symbol'], side=signal['side'], qty=qty,
                        entry_price=signal['entry'], stop_loss_price=signal['stop_loss'],
                        take_profit_price=signal['take_profit']
                    )
                    record_trade_execution(signal['symbol'], signal['direction'], signal['gap_level'])

            time.sleep(CHECK_INTERVAL_SECONDS)

        except Exception as e:
            print(f"Error in execution loop: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    run_bot()
