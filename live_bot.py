import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import pytz
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

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
        # Silence HTTP access logs in stdout to keep terminal clean
        return

def run_http_server():
    port = int(os.getenv("PORT", 10000))  # Render automatically sets PORT
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check web server running on port {port}")
    server.serve_forever()

# ------------------------------------------------------------------------------
# 2. AUTHENTICATION (Render Environment Variables)
# ------------------------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Set paper=True for Paper Trading account
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ------------------------------------------------------------------------------
# 3. CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------
SYMBOLS = ["NVDA", "SPY"]
MAX_POSITIONS = 2           # Position cap to manage capital exposure
RISK_REWARD_RATIO = 2.0     # 2:1 Reward-to-Risk setup for FVG trades
CHECK_INTERVAL_SECONDS = 60 # Check market conditions every 1 minute

# Timezone reference
EST = pytz.timezone("America/New_York")

# ------------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def is_fed_blackout_active() -> bool:
    """
    Checks if current time is within the Fed announcement/press conference 
    blackout window (1:55 PM EST to 2:45 PM EST).
    """
    now = datetime.now(EST)
    start_blackout = now.replace(hour=13, minute=55, second=0, microsecond=0)
    end_blackout = now.replace(hour=14, minute=45, second=0, microsecond=0)

    return start_blackout <= now <= end_blackout

def get_open_position_count() -> int:
    """Returns total number of open positions."""
    try:
        positions = trading_client.get_all_positions()
        return len(positions)
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return 0

def check_for_fvg(symbol: str):
    """Placeholder logic for Fair Value Gap identification."""
    return None

def execute_bracket_order(symbol: str, side: OrderSide, qty: float, entry_price: float, stop_loss_price: float, take_profit_price: float):
    """Executes a GTC bracket order with dynamic stop-loss and take-profit targets."""
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
        take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2))
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
            
            # 1. Fed Blackout Filter
            if is_fed_blackout_active():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Fed Blackout Active (1:55 PM - 2:45 PM EST). Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 2. Risk Management Position Cap
            open_positions = get_open_position_count()
            if open_positions >= MAX_POSITIONS:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Position cap reached ({open_positions}/{MAX_POSITIONS}). Paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 3. Scan FVG Signals
            for symbol in SYMBOLS:
                signal = check_for_fvg(symbol)
                if signal:
                    print(f"[{now_est.strftime('%H:%M:%S EST')}] FVG Signal detected on {symbol}.")

            time.sleep(CHECK_INTERVAL_SECONDS)

        except Exception as e:
            print(f"Error in execution loop: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    # Start the web server on a separate thread so Render keeps the instance alive
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # Start the trading bot loop on the main thread
    run_bot()
