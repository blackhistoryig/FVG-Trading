import os
import time
import threading
import json
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ------------------------------------------------------------------------------
# 1. LIGHTWEIGHT HTTP SERVER (For Render & cron-job.org Pings)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 2. AUTHENTICATION
# ------------------------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ------------------------------------------------------------------------------
# 3. CONFIGURATION & RISK CONSTANTS
# ------------------------------------------------------------------------------
SYMBOLS = ["QQQ", "NVDA", "SPY", "AAPL", "TSLA", "SMH", "IWM"]
MAX_POSITIONS = 2           
RISK_REWARD_RATIO = 2.0     
CHECK_INTERVAL_SECONDS = 60 

# Professional Risk Management Controls
RISK_PER_TRADE_PCT = 0.01       # Risk 1% of account equity per trade
MAX_POSITION_ALLOCATION = 0.25  # Max 25% of account equity invested in any single stock
DAILY_MAX_LOSS_PCT = 0.03       # 3% Daily circuit breaker shutdown
MIN_GAP_SIZE = 0.15             # Minimum $0.15 gap width to avoid bid-ask noise

EST = pytz.timezone("America/New_York")

# Global State Tracking
DAILY_START_EQUITY = None
CIRCUIT_BREAKER_TRIPPED = False
CURRENT_DAY = None
FOMC_MEETING_DATES = set()

# Backup Static 2026 FOMC Announcement Dates (YYYY-MM-DD)
STATIC_2026_FOMC_DATES = {
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16"
}

# ------------------------------------------------------------------------------
# 4. TIME & RISK MANAGEMENT HELPERS
# ------------------------------------------------------------------------------
def load_fomc_calendar():
    """Fetches Fed meeting days or falls back to standard calendar schedule."""
    global FOMC_MEETING_DATES
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
            
        detected_dates = set()
        for date_str in STATIC_2026_FOMC_DATES:
            if date_str in html or datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %d") in html:
                detected_dates.add(date_str)
                
        if detected_dates:
            FOMC_MEETING_DATES = detected_dates
        else:
            FOMC_MEETING_DATES = STATIC_2026_FOMC_DATES
            
        print(f"[FOMC CALENDAR] Active Fed announcement dates loaded: {sorted(list(FOMC_MEETING_DATES))}")
    except Exception as e:
        print(f"[FOMC CALENDAR] Could not reach Fed endpoint ({e}). Utilizing fallback 2026 schedule.")
        FOMC_MEETING_DATES = STATIC_2026_FOMC_DATES

def check_and_reset_daily_state():
    """Resets equity baseline and re-verifies parameters at start of a new trading day."""
    global DAILY_START_EQUITY, CIRCUIT_BREAKER_TRIPPED, CURRENT_DAY
    today_str = datetime.now(EST).strftime("%Y-%m-%d")
    
    if CURRENT_DAY != today_str:
        CURRENT_DAY = today_str
        CIRCUIT_BREAKER_TRIPPED = False
        load_fomc_calendar()
        try:
            account = trading_client.get_account()
            DAILY_START_EQUITY = float(account.equity)
            print(f"[{datetime.now(EST).strftime('%H:%M:%S EST')}] New Day Started ({CURRENT_DAY}). Baseline Equity: ${DAILY_START_EQUITY:,.2f}")
        except Exception as e:
            print(f"Error fetching account for daily state reset: {e}")

def is_fed_blackout_active() -> bool:
    """Blocks entry ONLY on verified Fed announcement days during 1:55 PM - 2:45 PM EST."""
    now = datetime.now(EST)
    today_str = now.strftime("%Y-%m-%d")
    
    if today_str not in FOMC_MEETING_DATES:
        return False
        
    start_blackout = now.replace(hour=13, minute=55, second=0, microsecond=0)
    end_blackout = now.replace(hour=14, minute=45, second=0, microsecond=0)
    return start_blackout <= now <= end_blackout

def is_circuit_breaker_tripped() -> bool:
    """Checks if daily loss exceeds 3% threshold."""
    global CIRCUIT_BREAKER_TRIPPED
    if CIRCUIT_BREAKER_TRIPPED:
        return True
    
    try:
        account = trading_client.get_account()
        current_equity = float(account.equity)
        if DAILY_START_EQUITY and DAILY_START_EQUITY > 0:
            loss_pct = (current_equity - DAILY_START_EQUITY) / DAILY_START_EQUITY
            if loss_pct <= -DAILY_MAX_LOSS_PCT:
                print(f"[CIRCUIT BREAKER TRIPPED] Loss of {loss_pct*100:.2f}% exceeds max allowed (-{DAILY_MAX_LOSS_PCT*100}%). Halting trading for today.")
                close_all_positions_and_orders()
                CIRCUIT_BREAKER_TRIPPED = True
                return True
    except Exception as e:
        print(f"Error checking circuit breaker: {e}")
    
    return False

def calculate_dynamic_position_size(symbol: str, current_price: float, risk_per_share: float) -> int:
    """
    Institutional Sizing Model:
    1. Sizes trade to risk exactly 1% of account equity based on stop distance.
    2. Enforces a hard cap of 25% total equity allocation per trade to ensure equal portfolio balance.
    """
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        
        # 1. Calculate ideal shares for 1% account risk
        risk_budget = equity * RISK_PER_TRADE_PCT
        if risk_per_share <= 0:
            return 1
        shares_by_risk = int(risk_budget / risk_per_share)
        
        # 2. Portfolio Cap: Max 25% of account equity per position
        max_capital_allowed = equity * MAX_POSITION_ALLOCATION
        shares_by_cap = int(max_capital_allowed / current_price)
        
        # 3. Choose the stricter of the two limits
        final_shares = min(shares_by_risk, shares_by_cap)
        final_shares = max(1, final_shares)
        
        print(f"[SIZING] {symbol}: Risk Target = {shares_by_risk} sh | 25% Portfolio Cap = {shares_by_cap} sh | Final Order = {final_shares} sh")
        return final_shares

    except Exception as e:
        print(f"Error calculating dynamic position size: {e}")
        return 1

def is_market_close_approaching() -> bool:
    """Stop opening NEW positions after 3:45 PM EST."""
    now = datetime.now(EST)
    cutoff = now.replace(hour=15, minute=45, second=0, microsecond=0)
    return now >= cutoff

def is_eod_flatten_time() -> bool:
    """Triggers total position liquidation between 3:55 PM and 4:00 PM EST."""
    now = datetime.now(EST)
    flatten_time = now.replace(hour=15, minute=55, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return flatten_time <= now <= market_close

def close_all_positions_and_orders():
    """Liquidates all open positions and cancels all pending orders."""
    try:
        print("[FLATTEN] Closing all positions and cancelling active orders...")
        trading_client.cancel_orders()
        trading_client.close_all_positions(cancel_orders=True)
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

# ------------------------------------------------------------------------------
# 5. CORE BATCH FVG SCANNER (With Gap Size Filter)
# ------------------------------------------------------------------------------
def check_for_fvg_batch(symbols: list):
    signals = []
    try:
        end_time = datetime.now(EST)
        start_time = end_time - timedelta(minutes=60)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
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
                if isinstance(df.index, pd.MultiIndex):
                    if symbol not in df.index.levels[0]:
                        continue
                    symbol_df = df.xs(symbol)
                else:
                    symbol_df = df

                if len(symbol_df) < 4:
                    continue

                completed_df = symbol_df.iloc[:-1]

                c1 = completed_df.iloc[-3]
                c2 = completed_df.iloc[-2]
                c3 = completed_df.iloc[-1]

                # Bullish FVG (c3.low > c1.high)
                if c1['high'] < c3['low']:
                    gap_size = c3['low'] - c1['high']
                    if gap_size < MIN_GAP_SIZE:
                        continue
                        
                    current_price = c3['close']
                    stop_loss = round(c1['high'], 2)
                    risk_per_share = current_price - stop_loss
                    
                    if risk_per_share > 0:
                        take_profit = round(current_price + (risk_per_share * RISK_REWARD_RATIO), 2)
                        signals.append({
                            "symbol": symbol,
                            "type": "BULLISH_FVG",
                            "side": OrderSide.BUY,
                            "entry": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "risk_per_share": risk_per_share
                        })

                # Bearish FVG (c1.low > c3.high)
                elif c1['low'] > c3['high']:
                    gap_size = c1['low'] - c3['high']
                    if gap_size < MIN_GAP_SIZE:
                        continue

                    current_price = c3['close']
                    stop_loss = round(c1['low'], 2)
                    risk_per_share = stop_loss - current_price
                    
                    if risk_per_share > 0:
                        take_profit = round(current_price - (risk_per_share * RISK_REWARD_RATIO), 2)
                        signals.append({
                            "symbol": symbol,
                            "type": "BEARISH_FVG",
                            "side": OrderSide.SELL,
                            "entry": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "risk_per_share": risk_per_share
                        })

            except Exception as inner_e:
                print(f"Error evaluating FVG for {symbol}: {inner_e}")
                continue

        return signals

    except Exception as e:
        print(f"Error during batch bar fetch: {e}")
        return []

def execute_bracket_order(symbol: str, side: OrderSide, qty: int, entry_price: float, stop_loss_price: float, take_profit_price: float):
    # 1. Fetch live quote safely (falling back to entry_price if bid/ask is zero or unpopulated)
    live_price = entry_price
    try:
        latest_quote_req = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
        latest_quote = data_client.get_stock_latest_quote(latest_quote_req)
        fetched_price = float(latest_quote[symbol].ask_price) if side == OrderSide.BUY else float(latest_quote[symbol].bid_price)
        if fetched_price > 0:
            live_price = fetched_price
    except Exception as e:
        print(f"[{symbol}] Dynamic quote lookup failed ({e}). Using bar entry price (${entry_price}).")

    # 2. Safety Adjustments relative to valid live execution price
    if side == OrderSide.BUY and stop_loss_price >= live_price:
        stop_loss_price = round(max(0.01, live_price - 0.10), 2)
    elif side == OrderSide.SELL and stop_loss_price <= live_price:
        stop_loss_price = round(live_price + 0.10, 2)

    # 3. Guardrail: Hard validation check prior to submission
    if stop_loss_price <= 0 or take_profit_price <= 0:
        print(f"[{symbol}] Aborted order placement: Invalid SL/TP calculated (SL: ${stop_loss_price}, TP: ${take_profit_price}).")
        return

    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price)
    )
    
    try:
        order = trading_client.submit_order(order_data)
        print(f"Successfully placed bracket order for {symbol} ({qty} shares): ID {order.id}")
    except Exception as e:
        print(f"Failed to place order for {symbol}: {e}")

# ------------------------------------------------------------------------------
# 6. MAIN TRADING LOOP
# ------------------------------------------------------------------------------
def run_bot():
    print("Starting FVG Trading Engine loop...")
    load_fomc_calendar()
    
    while True:
        try:
            now_est = datetime.now(EST)
            check_and_reset_daily_state()

            # 1. Check Circuit Breaker
            if is_circuit_breaker_tripped():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Circuit Breaker active. Bot paused for remainder of day.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 2. End-of-Day Position Liquidation (3:55 PM EST)
            if is_eod_flatten_time():
                close_all_positions_and_orders()
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 3. Cutoff New Entries near Market Close (After 3:45 PM EST)
            if is_market_close_approaching():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Past 3:45 PM EST cutoff. No new positions permitted.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 4. Dynamic Fed Blackout Filter
            if is_fed_blackout_active():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] FOMC Announcement Day: Fed Blackout Active. Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 5. Position Limit Filter
            open_positions = get_open_position_count()
            if open_positions >= MAX_POSITIONS:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Position cap reached ({open_positions}/{MAX_POSITIONS}). Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 6. Execute Scans
            print(f"[{now_est.strftime('%H:%M:%S EST')}] Scanning {len(SYMBOLS)} symbols for FVG setups...")
            signals = check_for_fvg_batch(SYMBOLS)
            
            if not signals:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Scan complete. No active FVG signals.")
            else:
                for signal in signals:
                    print(f"[{now_est.strftime('%H:%M:%S EST')}] {signal['type']} detected on {signal['symbol']}!")
                    
                    qty = calculate_dynamic_position_size(
                        symbol=signal['symbol'],
                        current_price=signal['entry'],
                        risk_per_share=signal['risk_per_share']
                    )
                    
                    execute_bracket_order(
                        symbol=signal['symbol'],
                        side=signal['side'],
                        qty=qty,
                        entry_price=signal['entry'],
                        stop_loss_price=signal['stop_loss'],
                        take_profit_price=signal['take_profit']
                    )

            time.sleep(CHECK_INTERVAL_SECONDS)

        except Exception as e:
            print(f"Error in execution loop: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    run_bot()
