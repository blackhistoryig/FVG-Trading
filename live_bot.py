import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import pytz
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ------------------------------------------------------------------------------
# 1. LIGHTWEIGHT HTTP SERVER (For Render Free Tier Compatibility)
# ------------------------------------------------------------------------------
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"OK"
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

def run_http_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check web server running on port {port}")
    server.serve_forever()

# ------------------------------------------------------------------------------
# 2. AUTHENTICATION (Render Environment Variables)
# ------------------------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ------------------------------------------------------------------------------
# 3. CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------
SYMBOLS = ["QQQ", "NVDA", "SPY", "AAPL", "TSLA", "SMH", "IWM"]
MAX_POSITIONS = 2           
RISK_REWARD_RATIO = 2.0     
CHECK_INTERVAL_SECONDS = 60 

EST = pytz.timezone("America/New_York")

# ------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS & BATCH FVG CORE LOGIC
# ------------------------------------------------------------------------------
def is_fed_blackout_active() -> bool:
    now = datetime.now(EST)
    start_blackout = now.replace(hour=13, minute=55, second=0, microsecond=0)
    end_blackout = now.replace(hour=14, minute=45, second=0, microsecond=0)
    return start_blackout <= now <= end_blackout

def get_open_position_count() -> int:
    try:
        positions = trading_client.get_all_positions()
        return len(positions)
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return 0

def check_for_fvg_batch(symbols: list):
    signals = []
    try:
        end_time = datetime.now(EST)
        start_time = end_time - timedelta(minutes=60)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time
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

                if c1['high'] < c3['low']:
                    gap_size = c3['low'] - c1['high']
                    current_price = c3['close']
                    stop_loss = round(c1['high'], 2)
                    risk = current_price - stop_loss
                    
                    if risk > 0:
                        take_profit = round(current_price + (risk * RISK_REWARD_RATIO), 2)
                        signals.append({
                            "symbol": symbol,
                            "type": "BULLISH_FVG",
                            "side": OrderSide.BUY,
                            "entry": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        })

                elif c1['low'] > c3['high']:
                    gap_size = c1['low'] - c3['high']
                    current_price = c3['close']
                    stop_loss = round(c1['low'], 2)
                    risk = stop_loss - current_price
                    
                    if risk > 0:
                        take_profit = round(current_price - (risk * RISK_REWARD_RATIO), 2)
                        signals.append({
                            "symbol": symbol,
                            "type": "BEARISH_FVG",
                            "side": OrderSide.SELL,
                            "entry": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        })

            except Exception as inner_e:
                print(f"Error evaluating FVG for {symbol}: {inner_e}")
                continue

        return signals

    except Exception as e:
        print(f"Error during batch bar fetch: {e}")
        return []

def execute_bracket_order(symbol: str, side: OrderSide, qty: float, entry_price: float, stop_loss_price: float, take_profit_price: float):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price)
    )
    
    try:
        order = trading_client.submit_order(order_data)
        print(f"Successfully placed bracket order for {symbol}: ID {order.id}")
    except Exception as e:
        print(f"Failed to place order for {symbol}: {e}")

# ------------------------------------------------------------------------------
# 5. MAIN TRADING LOOP
# ------------------------------------------------------------------------------
def run_bot():
    print("Starting FVG Trading Engine loop...")
    
    while True:
        try:
            now_est = datetime.now(EST)
            
            if is_fed_blackout_active():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Fed Blackout Active. Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            open_positions = get_open_position_count()
            if open_positions >= MAX_POSITIONS:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Position cap reached ({open_positions}/{MAX_POSITIONS}). Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            print(f"[{now_est.strftime('%H:%M:%S EST')}] Scanning {len(SYMBOLS)} symbols for FVG setups...")
            signals = check_for_fvg_batch(SYMBOLS)
            
            if not signals:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Scan complete. No active FVG signals.")
            else:
                for signal in signals:
                    print(f"[{now_est.strftime('%H:%M:%S EST')}] {signal['type']} detected on {signal['symbol']}!")
                    
                    qty = 1 
                    
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
