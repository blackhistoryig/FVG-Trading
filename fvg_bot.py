import pandas as pd
import numpy as np
import yfinance as yf
import itertools
from dataclasses import dataclass
from typing import List

@dataclass
class TradeSignal:
    timestamp: pd.Timestamp
    trade_type: str           # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_units: float
    fvg_top: float
    fvg_bottom: float
    risk_usd: float

class FvgMssDisplacementStrategy:
    def __init__(
        self,
        account_balance: float = 10000.0,
        risk_per_trade_pct: float = 0.01,  
        risk_reward_ratio: float = 1.5,    # Updated winning default
        atr_period: int = 14,
        atr_multiplier: float = 1.0,        # Updated winning default
        vol_period: int = 20,
        vol_multiplier: float = 1.0,        # Updated winning default
        swing_window: int = 2,              # Updated winning default
        mss_lookback_bars: int = 5
    ):
        self.account_balance = account_balance
        self.risk_pct = risk_per_trade_pct
        self.rr_ratio = risk_reward_ratio
        self.atr_period = atr_period
        self.atr_mult = atr_multiplier
        self.vol_period = vol_period
        self.vol_mult = vol_multiplier
        self.swing_window = swing_window
        self.mss_lookback = mss_lookback_bars

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()
        df['candle_range'] = df['high'] - df['low']
        df['vol_ma'] = df['volume'].rolling(window=self.vol_period).mean()

        df['is_swing_high'] = False
        df['is_swing_low'] = False
        w = self.swing_window

        for i in range(w, len(df) - w):
            if df['high'].iloc[i] == df['high'].iloc[i - w : i + w + 1].max():
                df.at[df.index[i], 'is_swing_high'] = True
            if df['low'].iloc[i] == df['low'].iloc[i - w : i + w + 1].min():
                df.at[df.index[i], 'is_swing_low'] = True

        df['mss_signal'] = None
        active_sh, active_sl = None, None

        for i in range(len(df)):
            curr_close = df.at[df.index[i], 'close']
            if df.at[df.index[i], 'is_swing_high']: active_sh = df.at[df.index[i], 'high']
            if df.at[df.index[i], 'is_swing_low']: active_sl = df.at[df.index[i], 'low']

            if active_sh is not None and curr_close > active_sh:
                df.at[df.index[i], 'mss_signal'] = 'BULLISH'
                active_sh = None  
            elif active_sl is not None and curr_close < active_sl:
                df.at[df.index[i], 'mss_signal'] = 'BEARISH'
                active_sl = None
        return df

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> tuple[float, float]:
        risk_usd = self.account_balance * self.risk_pct
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0: return 0.0, 0.0
        return risk_usd / risk_per_unit, risk_usd

    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        df = self.calculate_indicators(df)
        signals = []

        for i in range(2, len(df)):
            c1_high, c1_low = df.at[df.index[i - 2], 'high'], df.at[df.index[i - 2], 'low']
            c2_idx = df.index[i - 1]
            c2_high, c2_low = df.at[c2_idx, 'high'], df.at[c2_idx, 'low']
            c3_high, c3_low = df.at[df.index[i], 'high'], df.at[df.index[i], 'low']

            atr = df.at[c2_idx, 'atr']
            vol_ma = df.at[c2_idx, 'vol_ma']
            if pd.isna(atr) or pd.isna(vol_ma) or atr == 0 or vol_ma == 0: continue

            has_displacement = (df.at[c2_idx, 'candle_range'] >= atr * self.atr_mult) and (df.at[c2_idx, 'volume'] >= vol_ma * self.vol_mult)
            if not has_displacement: continue

            start_loc = max(0, i - self.mss_lookback)
            recent_mss = df.iloc[start_loc : i + 1]['mss_signal'].dropna().tolist()

            if c3_low > c1_high and 'BULLISH' in recent_mss:
                entry, sl = c3_low, c2_low
                units, risk_usd = self.calculate_position_size(entry, sl)
                if units > 0:
                    tp = entry + ((entry - sl) * self.rr_ratio)
                    signals.append(TradeSignal(df.index[i], 'BUY', round(entry, 5), round(sl, 5), round(tp, 5), round(units, 4), c3_low, c1_high, round(risk_usd, 2)))

            elif c1_low > c3_high and 'BEARISH' in recent_mss:
                entry, sl = c3_high, c2_high
                units, risk_usd = self.calculate_position_size(entry, sl)
                if units > 0:
                    tp = entry - ((sl - entry) * self.rr_ratio)
                    signals.append(TradeSignal(df.index[i], 'SELL', round(entry, 5), round(sl, 5), round(tp, 5), round(units, 4), c1_low, c3_high, round(risk_usd, 2)))

        return signals


class BacktestEngine:
    def __init__(
        self, 
        strategy, 
        initial_capital: float = 10000.0, 
        max_order_lifespan_bars: int = 20,
        fee_pct: float = 0.0005,       # 0.05% commission per trade
        slippage_pct: float = 0.0002   # 0.02% price slippage per execution
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.max_order_lifespan = max_order_lifespan_bars
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def run(self, df: pd.DataFrame) -> dict:
        signals = self.strategy.generate_signals(df)
        pending_orders, closed_trades = [], []
        active_trade = None
        capital = self.initial_capital
        equity_curve = pd.Series(index=df.index, dtype=float)
        
        for idx in range(len(df)):
            curr_time = df.index[idx]
            high, low = df.at[curr_time, 'high'], df.at[curr_time, 'low']

            new_signals = [s for s in signals if s.timestamp == curr_time]
            for sig in new_signals:
                pending_orders.append({'signal': sig, 'created_idx': idx, 'expires_idx': idx + self.max_order_lifespan})

            if active_trade is not None:
                trade = active_trade['signal']
                trade_pnl = None

                # Check SL/TP triggers
                if trade.trade_type == 'BUY':
                    if low <= trade.stop_loss:
                        gross_pnl = -trade.risk_usd
                        trade_result = 'LOSS'
                        exit_price = trade.stop_loss
                    elif high >= trade.take_profit:
                        gross_pnl = trade.risk_usd * self.strategy.rr_ratio
                        trade_result = 'WIN'
                        exit_price = trade.take_profit
                    else:
                        gross_pnl = None

                elif trade.trade_type == 'SELL':
                    if high >= trade.stop_loss:
                        gross_pnl = -trade.risk_usd
                        trade_result = 'LOSS'
                        exit_price = trade.stop_loss
                    elif low <= trade.take_profit:
                        gross_pnl = trade.risk_usd * self.strategy.rr_ratio
                        trade_result = 'WIN'
                        exit_price = trade.take_profit
                    else:
                        gross_pnl = None

                if gross_pnl is not None:
                    # Calculate trading fees & slippage costs
                    entry_notional = trade.position_size_units * trade.entry_price
                    exit_notional = trade.position_size_units * exit_price
                    
                    entry_fee = entry_notional * self.fee_pct
                    exit_fee = exit_notional * self.fee_pct
                    slippage_cost = (entry_notional + exit_notional) * self.slippage_pct
                    
                    net_pnl = gross_pnl - (entry_fee + exit_fee + slippage_cost)
                    capital += net_pnl
                    
                    closed_trades.append({
                        **trade.__dict__, 
                        'result': trade_result, 
                        'net_pnl': round(net_pnl, 2),
                        'frictions_usd': round(entry_fee + exit_fee + slippage_cost, 2)
                    })
                    active_trade = None

            if active_trade is None and pending_orders:
                orders_to_remove = []
                for order in pending_orders:
                    if idx > order['expires_idx']:
                        orders_to_remove.append(order)
                        continue
                    sig = order['signal']
                    filled = (sig.trade_type == 'BUY' and low <= sig.entry_price) or (sig.trade_type == 'SELL' and high >= sig.entry_price)
                    if filled:
                        active_trade = {'signal': sig}
                        orders_to_remove.append(order)
                        pending_orders.clear()
                        break
                for o in orders_to_remove:
                    if o in pending_orders: pending_orders.remove(o)

            equity_curve.at[curr_time] = capital

        trades_df = pd.DataFrame(closed_trades)
        wins = len(trades_df[trades_df['result'] == 'WIN']) if not trades_df.empty else 0
        win_rate = (wins / len(trades_df)) * 100 if not trades_df.empty else 0
        total_frictions = trades_df['frictions_usd'].sum() if not trades_df.empty else 0.0
        
        return {
            'Total Trades': len(trades_df),
            'Win Rate (%)': round(win_rate, 2),
            'Total Fees & Slippage ($)': round(total_frictions, 2),
            'Final Equity ($)': round(capital, 2)
        }


def fetch_real_data(ticker: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    """Fetches real historical OHLCV data from Yahoo Finance."""
    print(f"\nFetching {ticker} data ({interval} timeframe, past {period})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    return df


def optimize_equities(tickers: List[str]):
    param_grid = {
        'atr_multiplier': [1.0, 1.2, 1.5],
        'vol_multiplier': [1.0, 1.2, 1.5],
        'risk_reward_ratio': [1.5, 2.0],
        'swing_window': [2, 3]
    }

    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for ticker in tickers:
        df = fetch_real_data(ticker=ticker)
        if df.empty:
            print(f"Skipping {ticker}, no data returned.")
            continue

        print(f"Testing {len(permutations)} configurations for {ticker} (including fees & slippage)...")
        results_list = []

        for params in permutations:
            strat = FvgMssDisplacementStrategy(
                account_balance=10000.0,
                risk_per_trade_pct=0.01,
                risk_reward_ratio=params['risk_reward_ratio'],
                atr_multiplier=params['atr_multiplier'],
                vol_multiplier=params['vol_multiplier'],
                swing_window=params['swing_window']
            )
            
            engine = BacktestEngine(strategy=strat, fee_pct=0.0005, slippage_pct=0.0002)
            res = engine.run(df)
            
            results_list.append({
                'Ticker': ticker,
                **params,
                'Trades': res['Total Trades'],
                'Win Rate (%)': res['Win Rate (%)'],
                'Frictions ($)': res['Total Fees & Slippage ($)'],
                'Net Profit ($)': round(res['Final Equity ($)'] - 10000.0, 2)
            })

        results_df = pd.DataFrame(results_list).sort_values(by='Net Profit ($)', ascending=False)
        print(f"\n=== TOP 3 SETUPS FOR {ticker} ===")
        print(results_df.head(3).to_string(index=False))


if __name__ == "__main__":
    # Test Stock Indices and Top Tech Equities
    STOCK_TICKERS = ["QQQ", "NVDA", "SPY", "AAPL", "TSLA", "SMH", "IWM"]
    optimize_equities(tickers=STOCK_TICKERS)
