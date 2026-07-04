import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Inject root paths so Python recognizes our local src development module cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import detect_return_outliers, run_stationarity_check

os.makedirs("data/processed", exist_ok=True)

tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-01-01"
end_date = "2026-06-30"

print("--- Step 1: Secure Data Extraction ---")
try:
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    if raw_data.empty:
        raise ConnectionError("YFinance API returned an empty dataset frame. Verify connectivity.")
    
    df_close = raw_data['Close'].copy() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data[['Close']].copy()
    print(f"Data acquired successfully. Structural shape: {df_close.shape}")
except Exception as e:
    print(f"CRITICAL NETWORK/API FAILURE: {str(e)}")
    sys.exit(1)

print("\n--- Step 2: Modularity, Imputation & Cleaning ---")
df_close = df_close.ffill().bfill()
df_returns = df_close.pct_change().dropna()

print("\n--- Step 3: Explicit Outlier Anomaly Detection ---")
# Call our new core modular function to isolate market shock events (> 3 Standard Deviations)
outlier_results = detect_return_outliers(df_returns, threshold_sigmas=3.0)
for asset, metrics in outlier_results.items():
    print(f"Asset: {asset} | Isolated Extreme Outlier Days Detected: {metrics['count']}")
    # Sort and slice out the top 3 most extreme historical shock days for detailed review
    sorted_shocks = sorted(metrics['extreme_days'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for date, val in sorted_shocks:
        print(f"  -> Date: {date.strftime('%Y-%m-%d')} | Return Shock: {val:.2%}")

print("\n--- Step 4: System Stationary Appraisals ---")
for asset in tickers:
    raw_status = run_stationarity_check(df_close[asset])
    ret_status = run_stationarity_check(df_returns[asset])
    print(f"{asset} Raw Close P-Value: {raw_status.get('p_value', 1):.4e} | Stationary: {raw_status.get('is_stationary')}")
    print(f"{asset} Returns P-Value: {ret_status.get('p_value', 0):.4e} | Stationary: {ret_status.get('is_stationary')}")

# Save state output files
df_close.to_csv("data/processed/assets_close.csv")
df_returns.to_csv("data/processed/assets_returns.csv")
print("\nData pipeline execution concluded successfully.")