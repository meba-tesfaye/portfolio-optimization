import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import load_portfolio_data

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

print("--- Step 1: Isolating Final-Year Backtesting Window (July 2025 - June 2026) ---")
df_returns = load_portfolio_data("data/processed/assets_returns.csv")
assets = ["BND", "SPY", "TSLA"]

# Enforce clean isolation of the final year dataset window
backtest_window = df_returns[df_returns.index >= "2025-07-01"].copy()
print(f"Isolated backtesting sessions: {len(backtest_window)}")

opt_weights = np.array([0.5781, 0.3230, 0.0989])
bench_weights = np.array([0.40, 0.60, 0.00])

print("\n--- Step 2: Simulating Allocation Performance Path ---")
backtest_window['Optimized_Portfolio'] = backtest_window[assets].dot(opt_weights)
backtest_window['Benchmark_Portfolio'] = backtest_window[assets].dot(bench_weights)

cum_optimized = (1 + backtest_window['Optimized_Portfolio']).cumprod()
cum_benchmark = (1 + backtest_window['Benchmark_Portfolio']).cumprod()

trading_days = 252
opt_mean = backtest_window['Optimized_Portfolio'].mean() * trading_days
opt_vol = backtest_window['Optimized_Portfolio'].std() * np.sqrt(trading_days)
opt_sharpe = opt_mean / opt_vol

bench_mean = backtest_window['Benchmark_Portfolio'].mean() * trading_days
bench_vol = backtest_window['Benchmark_Portfolio'].std() * np.sqrt(trading_days)
bench_sharpe = bench_mean / bench_vol

def get_max_drawdown(series):
    return ((series - series.cummax()) / series.cummax()).min()

print("\n====================================================================")
print("FINAL ISOLATED OUT-OF-SAMPLE PERFORMANCE EVALUATION SUMMARY MATRIX")
print("====================================================================")
print(f"METRIC                 OPTIMIZED MPT PORTFOLIO     60/40 BENCHMARK")
print("====================================================================")
print(f"Annualized Return:     {opt_mean:.2%}{'':<24}{bench_mean:.2%}")
print(f"Annualized Volatility: {opt_vol:.2%}{'':<24}{bench_vol:.2%}")
print(f"Sharpe Ratio:          {opt_sharpe:.4f}{'':<24}{bench_sharpe:.4f}")
print(f"Maximum Drawdown:      {get_max_drawdown(cum_optimized):.2%}{'':<24}{get_max_drawdown(cum_benchmark):.2%}")
print(f"Total Capital Return:  {((cum_optimized.iloc[-1] - 1) * 100):.2f}%{'':<21}{((cum_benchmark.iloc[-1] - 1) * 100):.2f}%")
print("====================================================================")

plt.figure(figsize=(11, 5))
plt.plot(cum_optimized, label="GMF Optimized MPT Strategy Model Portfolio", color='green', lw=2)
plt.plot(cum_benchmark, label="Static Institutional Balanced 60/40 Benchmark", color='navy', linestyle='--', lw=1.5)
plt.title("Isolated Final Year Strategic Performance Validation Curve ($1 Base Investment)")
plt.xlabel("Date")
plt.ylabel("Portfolio Accumulated Value ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "backtest_performance.png"))
plt.close()
print(f"Artifact completely saved to '{output_dir}/backtest_performance.png'.")