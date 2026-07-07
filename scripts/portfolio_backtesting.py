import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure local src module is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import load_portfolio_data

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

print("--- Step 1: Loading Asset Return Data for Backtesting ---")
try:
    df_returns = load_portfolio_data("data/processed/assets_returns.csv")
    print("Successfully loaded return matrices.")
except Exception as e:
    print(f"Data Load Error: {str(e)}")
    sys.exit(1)

# Define Allocation Strategies
# Note: Ensure asset ordering aligns perfectly with DataFrame columns
# df_returns columns are ['BND', 'SPY', 'TSLA'] based on prior script output
assets = list(df_returns.columns)

# Task 4 Optimized Portfolio Weights
opt_weights = np.array([0.5781, 0.3230, 0.0989])

# Institutional Benchmark Weights (60% SPY / 40% BND / 0% TSLA)
bench_weights = np.array([0.40, 0.60, 0.00])

print("\n--- Step 2: Running Historical Performance Backtest ---")
# Compute daily portfolio returns
df_returns['Optimized_Portfolio'] = df_returns[assets].dot(opt_weights)
df_returns['Benchmark_Portfolio'] = df_returns[assets].dot(bench_weights)

# Calculate Cumulative Growth (growth of $1 investment)
cum_optimized = (1 + df_returns['Optimized_Portfolio']).cumprod()
cum_benchmark = (1 + df_returns['Benchmark_Portfolio']).cumprod()

# Calculate Performance Metrics
trading_days = 252

# Optimized Portfolio Metrics
opt_mean = df_returns['Optimized_Portfolio'].mean() * trading_days
opt_vol = df_returns['Optimized_Portfolio'].std() * np.sqrt(trading_days)
opt_sharpe = opt_mean / opt_vol

# Benchmark Portfolio Metrics
bench_mean = df_returns['Benchmark_Portfolio'].mean() * trading_days
bench_vol = df_returns['Benchmark_Portfolio'].std() * np.sqrt(trading_days)
bench_sharpe = bench_mean / bench_vol

# Calculate Maximum Drawdowns
def calculate_max_drawdown(cum_returns_series):
    rolling_max = cum_returns_series.cummax()
    drawdowns = (cum_returns_series - rolling_max) / rolling_max
    return drawdowns.min()

opt_max_dd = calculate_max_drawdown(cum_optimized)
bench_max_dd = calculate_max_drawdown(cum_benchmark)

print("\n📊 HISTORICAL BACKTEST PERFORMANCE SUMMARY (2015 - 2026):")
print("====================================================================")
print(f"METRIC                 OPTIMIZED MPT PORTFOLIO     60/40 BENCHMARK")
print("====================================================================")
print(f"Annualized Return:     {opt_mean:.2%}{'':<24}{bench_mean:.2%}")
print(f"Annualized Volatility: {opt_vol:.2%}{'':<24}{bench_vol:.2%}")
print(f"Sharpe Ratio:          {opt_sharpe:.4f}{'':<24}{bench_sharpe:.4f}")
print(f"Maximum Drawdown:      {opt_max_dd:.2%}{'':<24}{bench_max_dd:.2%}")
print(f"Final Value of $1:     ${cum_optimized.iloc[-1]:.2f}{'':<23}${cum_benchmark.iloc[-1]:.2f}")
print("====================================================================")

print("\n--- Step 3: Generating Cumulative Growth Plot ---")
plt.figure(figsize=(11, 5))
plt.plot(cum_optimized, label=f"Optimized MPT Portfolio (Sharpe: {opt_sharpe:.2f})", color='emerald' if 'emerald' in plt.cm.datad else 'green', lw=2)
plt.plot(cum_benchmark, label=f"Institutional Benchmark 60/40 (Sharpe: {bench_sharpe:.2f})", color='navy', linestyle='--', lw=1.5)
plt.title("Portfolio Cumulative Growth Comparison ($1 Base Investment)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "backtest_performance.png"))
plt.close()

print(f"Performance plotting complete. Artifact saved to '{output_dir}/backtest_performance.png'.")