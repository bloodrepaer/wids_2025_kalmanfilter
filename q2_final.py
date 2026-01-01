import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_processed_data(ticker='MSFT'):
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31', progress=False)
    
    # this flattens the multi-indexed data
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['RSI'] = ta.rsi(df['Close'], length=14)#helps us know the speed and change of price movements
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)#this gives us an idea of the true volatility of the stock in a given period usually take a 14-day period
    df['SMA_50'] = ta.sma(df['Close'], length=50)# this gives us the average closing price over the last 50 days
    
    # Robust Bollinger Band extraction to avoid KeyError
    bb = ta.bbands(df['Close'], length=20, std=2)#bollinger bands are defined using a 20-day moving average and 2 standard deviations
    pct_b_col = [c for c in bb.columns if c.startswith('BBP')][0]
    df['BB_PctB'] = bb[pct_b_col]#this helps us understand the relative position of the price within the Bollinger Bands
    
    # Target: Sign of next day's return
    df['Next_Ret'] = df['Close'].pct_change().shift(-1)#percentage change in closing price to predict next day movement
    df['Target'] = (df['Next_Ret'] > 0).astype(int)#help us classify whether the stock will go up (1) or down (0)
    
    return df.dropna()#cleaning the data by removing any rows with missing values

def apply_kalman_regression(df):
    
    # We model: Price_t = Beta_t * SMA_50_t + Alpha_t
    obs_mat = np.vstack([df['SMA_50'], np.ones(len(df))]).T[:, np.newaxis, :]#setting up the observation matrix for the Kalman Filter
    
    kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=[1, 0],
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2.0,
        transition_covariance=np.eye(2) * 1e-4
    )
    
    state_means, _ = kf.filter(df['Close'].values)
    df['KF_Beta'] = state_means[:, 0]
    df['KF_Alpha'] = state_means[:, 1]
    
    # Calculate Residual (Spread)
    df['Fair_Value'] = df['KF_Beta'] * df['SMA_50'] + df['KF_Alpha']
    df['KF_Resid'] = df['Close'] - df['Fair_Value']
    
    # NEW: Rolling Window normalization of the spread
    window = 20
    df['Resid_Mean'] = df['KF_Resid'].rolling(window=window).mean()
    df['Resid_Std'] = df['KF_Resid'].rolling(window=window).std()
    df['KF_Zscore'] = (df['KF_Resid'] - df['Resid_Mean']) / df['Resid_Std']
    
    return df.dropna()

def run_hybrid_strategy(df):
    # Added KF_Zscore to the feature set
    features = ['RSI', 'BB_PctB', 'KF_Beta', 'KF_Zscore']
    
    split_idx = int(len(df) * 0.7)
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(train[features], train['Target'])
    
    # Predict
    test['ML_Signal'] = model.predict(test[features])
    capital = 100000
    cost_bps = 10
    
    # Logic: Long if ML says UP and Beta is positive
    test['Signal'] = np.where((test['ML_Signal'] == 1) & (test['KF_Beta'] > 0), 1, 0)
    
    test['Position'] = test['Signal'].shift(1).fillna(0)
    test['Trades'] = test['Position'].diff().abs().fillna(0)
    
    test['Strat_Ret'] = (test['Position'] * test['Next_Ret']) - (test['Trades'] * (cost_bps / 10000))
    
    test['Strategy_Value'] = capital * (1 + test['Strat_Ret']).cumprod()
    test['Benchmark_Value'] = capital * (1 + test['Next_Ret']).cumprod()
    
    return test
#for analysis to look clean and nice i used AI to help me format it
def show_analysis(res):
    strat_final = res['Strategy_Value'].iloc[-1]
    bench_final = res['Benchmark_Value'].iloc[-1]
    
    s_rets = res['Strat_Ret'].dropna()
    strat_sharpe = (s_rets.mean() / s_rets.std()) * np.sqrt(252) if s_rets.std() != 0 else 0
    
    print("\n" + "="*45)
    print(f"{'Performance Metric':<20} | {'Strategy':<10} | {'Benchmark':<10}")
    print("-" * 45)
    print(f"{'Total Return (%)':<20} | {(strat_final/100000-1)*100:>9.2f}% | {(bench_final/100000-1)*100:>9.2f}%")
    print(f"{'Sharpe Ratio':<20} | {strat_sharpe:>10.2f} | ---")
    print(f"{'Final Capital ($)':<20} | ${strat_final:>9.0f} | ${bench_final:>9.0f}")
    print("="*45)

if __name__ == "__main__":
    data = get_processed_data()
    data = apply_kalman_regression(data)
    results = run_hybrid_strategy(data)
    show_analysis(results)
#plot using AI  
import matplotlib.pyplot as plt

def plot_performance(res):
    """
    Plots the equity curves for the Strategy and the Benchmark.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Strategy Equity Curve
    plt.plot(res.index, res['Strategy_Value'], label='Kalman-ML Strategy Return', color='blue', linewidth=2)
    
    # Plot Benchmark Equity Curve
    plt.plot(res.index, res['Benchmark_Value'], label='MSFT Benchmark (Buy & Hold)', color='gray', linestyle='--', alpha=0.7)
    
    # Formatting
    plt.title('Performance Comparison: Hybrid Strategy vs. MSFT Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Run the plotting function using your results DataFrame
plot_performance(results)