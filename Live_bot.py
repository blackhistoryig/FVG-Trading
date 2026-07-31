import os
import time
from datetime import datetime
import pytz
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ------------------------------------------------------------------------------
# 1. AUTHENTICATION (Render Environment Variables)
# ------------------------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Set paper=True for Paper Trading account
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# ------------------------------------------------------------------------------
# 2. CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------
SYMBOLS = ["NVDA", "SPY"]
MAX_POSITIONS = 2           # Position cap to manage capital exposure
RISK_REWARD_RATIO = 2.0     # 2:1 Reward-to-Risk setup for FVG trades
CHECK_INTERVAL_SECONDS = 60 # Check market conditions every 1 minute

# Timezone references
EST = pytz.timezone("America/New_York")

# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def is_fed_blackout_active() -> bool:
    """
    Checks if the current time falls within the Fed announcement/press conference
    blackout window (1:55 PM EST to 2:45 PM EST) to avoid elevated volatility.
    """
    now = datetime.now(EST)
    
    # 13 = 1 PM, 14 = 2 PM
    start_blackout = now.replace(hour=13, minute=55, second=0, microsecond=0)
    end_blackout = now.replace(hour=14, minute=45, second=0, microsecond=0)

    if start_blackout <= now <= end_blackout:
        return True
    return False

def get_open_position_count() -> int:
    """Returns the total number of currently open positions."""
    try:
        positions = trading_client.get_all_positions()
        return len(positions)
    except Exception as e:
        print(f"Error fetching open positions: {e}")
        return 0

def check_for_fvg(symbol: str):
    """
    Placeholder logic for detecting Fair Value Gaps (FVG).
    Integrate your exact pandas candle/gap calculation logic here.
    """
    # Example structure:
    # 1. Fetch recent minute bars using Alpaca Data API
    # 2. Identify 3-candle FVG imbalances
    # 3. Return 'BUY', 'SELL', or None along with entry, take_profit, stop_loss
    return None

def execute_bracket_order(symbol: str, side: OrderSide, qty: float, entry_price: float, stop_loss_price: float, take_profit_price: float):
    """
    Executes an institutional GTC bracket order with dynamic stop-loss and take-profit targets.
    """
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
# 4. MAIN TRADING LOOP
# ------------------------------------------------------------------------------
def run_bot():
    print("Starting FVG Trading Bot engine...")
    
    while True:
        try:
            now_est = datetime.now(EST)
            
            # 1. Fed Announcement Blackout Filter
            if is_fed_blackout_active():
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Fed Blackout Window Active (1:55 PM - 2:45 PM EST). Execution paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 2. Check Risk Management Caps
            open_positions = get_open_position_count()
            if open_positions >= MAX_POSITIONS:
                print(f"[{now_est.strftime('%H:%M:%S EST')}] Position cap reached ({open_positions}/{MAX_POSITIONS}). Scanning paused.")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 3. Scan Symbols for FVG Setups
            for symbol in SYMBOLS:
                signal = check_for_fvg(symbol)
                if signal:
                    print(f"[{now_est.strftime('%H:%M:%S EST')}] FVG Signal detected on {symbol}. Executing...")
                    # Calculate position size and execute order
                    # execute_bracket_order(...)

            time.sleep(CHECK_INTERVAL_SECONDS)

        except Exception as e:
            print(f"Error in execution loop: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_bot()
