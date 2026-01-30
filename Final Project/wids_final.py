import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# -- Configuration --
ASSETS = ['BTC-USD', '^NSEI', 'GC=F']  # Crypto, Indian Market, Gold
BENCHMARK_TICKER = '^NSEI'
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
TC = 0.001  # 0.1% transaction cost

def get_market_data(tickers, start, end):
    """Downloads close prices and cleans up missing values."""
    print(f"Downloading data for {tickers}...")
    df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    
    # Forward fill first to handle different holidays, then drop any remaining NaNs
    return df.ffill().dropna()

def get_kalman_trend(series):
    """
    Applies a Kalman Filter to estimate the 'true' price trend (slope).
    We use a Local Linear Trend model:
      State 0: Level (current price estimate)
      State 1: Slope (velocity of price)
    """
    # Initialize the Kalman Filter with some standard noise parameters
    # transition_covariance allows the trend to evolve (Process Noise)
    kf = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[series.iloc[0], 0],
        initial_state_covariance=np.eye(2),
        transition_covariance=np.eye(2) * 1e-4, 
        observation_covariance=1e-2 
    )
    
    state_means, _ = kf.filter(series.values)
    
    # Pack everything into a DataFrame
    df = pd.DataFrame(index=series.index)
    df['Price'] = series
    df['KF_Level'] = state_means[:, 0]
    df['KF_Slope'] = state_means[:, 1]
    
    # We'll need volatility later for risk-sizing (20-day rolling std)
    df['Volatility'] = series.pct_change().rolling(20).std()
    
    return df.dropna()

def calculate_allocations(kalman_results):
    """
    Determines how much to buy of each asset.
    Logic: Buy if the Kalman slope is positive. Size position inversely to volatility.
    """
    # Create an empty frame for our target weights
    tickers = list(kalman_results.keys())
    ref_index = kalman_results[tickers[0]].index
    weights = pd.DataFrame(index=ref_index, columns=tickers)
    
    for ticker, df in kalman_results.items():
        # Normalize slope by price to get a "percent trend"
        trend_strength = df['KF_Slope'] / df['KF_Level']
        
        # Binary signal: 1 if trending up, 0 if trending down
        signal = (trend_strength > 0).astype(float)
        
        # Risk Parity: divide signal by volatility so risky assets get less capital
        safe_vol = df['Volatility'].replace(0, 0.001)  # avoid div by zero
        weights[ticker] = signal / safe_vol

    # Normalize weights so we never exceed 100% leverage
    # The sum might be less than 1, which implies the remainder is held in Cash
    total_risk_weight = weights.sum(axis=1).replace(0, 1)
    
    final_allocations = weights.div(total_risk_weight, axis=0)
    return final_allocations.fillna(0)

def backtest_strategy(prices, target_weights, initial_capital=100000):
    """
    Simulates the strategy day-by-day, accounting for transaction costs.
    """
    # Make sure we're only looking at the dates where we have both price and signal
    common_dates = prices.index.intersection(target_weights.index)
    prices = prices.loc[common_dates]
    target_weights = target_weights.loc[common_dates]
    
    # Shift weights by 1 day! We calculate signals at close of Day T, execute at Open of Day T+1
    exec_weights = target_weights.shift(1).fillna(0)
    
    # Tracking variables
    portfolio_value = [initial_capital]
    current_positions = np.zeros(len(prices.columns)) # starts in cash
    returns = prices.pct_change().fillna(0)
    turnover_hist = []

    # Loop through each day
    # (Skipping the first index since we can't trade on day 0 returns)
    for i in range(1, len(prices)):
        
        # 1. Update portfolio value based on yesterday's holdings
        daily_ret = returns.iloc[i].values
        day_pnl = np.sum(current_positions * daily_ret * portfolio_value[-1])
        val_before_trade = portfolio_value[-1] + day_pnl
        
        # 2. Check what we need to rebalance today
        desired_alloc = exec_weights.iloc[i].values
        
        # 3. Calc transaction cost
        # Cost = |Target - Current| * Portfolio Value * Fee
        trade_diff = np.abs(desired_alloc - current_positions)
        cost = np.sum(trade_diff * val_before_trade * TC)
        
        turnover_hist.append(np.sum(trade_diff))
        
        # 4. Finalize value and update current positions
        portfolio_value.append(val_before_trade - cost)
        current_positions = desired_alloc

    return pd.Series(portfolio_value, index=prices.index)

# -- Main Script --
if __name__ == "__main__":
    
    # 1. Get Data
    prices = get_market_data(ASSETS, START_DATE, END_DATE)
    
    # 2. Process Trends
    kf_data = {}
    for asset in ASSETS:
        kf_data[asset] = get_kalman_trend(prices[asset])
        
    # 3. Run Strategy
    weights = calculate_allocations(kf_data)
    equity_curve = backtest_strategy(prices, weights)
    
    # 4. Compare with Benchmark
    bench_price = yf.download(BENCHMARK_TICKER, start=START_DATE, end=END_DATE, progress=False)['Close']
    bench_curve = (bench_price / bench_price.iloc[0]) * equity_curve.iloc[0]
    
    # 5. Stats & Print
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    daily_ret = equity_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    
    print("\n--- Performance Report ---")
    print(f"Total Return:     {total_ret*100:.2f}%")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Max Drawdown:     {drawdown*100:.2f}%")
    print(f"Final Capital:    ${equity_curve.iloc[-1]:,.2f}")

    # 6. Plots
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label='Kalman Strategy', color='#1f77b4')
    plt.plot(bench_curve, label='Nifty 50 (Buy & Hold)', color='gray', linestyle='--', alpha=0.6)
    plt.title("Portfolio Performance vs Benchmark")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Asset Allocation Area Chart
    plt.figure(figsize=(10, 4))
    plt.stackplot(weights.index, weights.T, labels=weights.columns, alpha=0.8)
    plt.title("Portfolio Allocation Over Time")
    plt.ylabel("Weight")
    plt.legend(loc='upper left')
    plt.show()