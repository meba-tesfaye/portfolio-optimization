import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import detect_return_outliers, run_stationarity_check, calculate_risk_metrics

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-01-01"
end_date = "2026-06-30"

print("--- Step 1: Secure Data Extraction ---")
try:
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    df_close = raw_data['Close'].copy() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data[['Close']].copy()
    df_close = df_close.ffill().bfill()
    df_returns = df_close.pct_change().dropna()
    print("Data ingestion and cleaning successfully executed.")
except Exception as e:
    print(f"Extraction Error: {str(e)}")
    sys.exit(1)

print("\n--- Step 2: Calculate Risk Metrics ---")
risk_results = calculate_risk_metrics(df_returns)
for asset, metrics in risk_results.items():
    print(f"{asset} -> Sharpe Ratio: {metrics['Sharpe Ratio']:.4f} | 95% Historical VaR: {metrics['95% Historical VaR']:.2%}")

print("\n--- Step 3: Outlier Detection ---")
outlier_results = detect_return_outliers(df_returns)
for asset, metrics in outlier_results.items():
    print(f"Asset: {asset} | Outlier Days (Z > 3): {metrics['count']}")

print("\n--- Step 4: Generating and Saving Visual EDA Plots ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Visualization 1: Closing Prices Over Time
plt.figure(figsize=(11, 5))
for asset in tickers:
    plt.plot(df_close[asset], label=asset)
plt.title("Asset Historical Normalized Closing Prices (2015 - 2026)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "price_history.png"))
plt.close()

# Visualization 2: Daily Percentage Change
plt.figure(figsize=(11, 5))
for asset in tickers:
    plt.plot(df_returns[asset], label=asset, alpha=0.6)
plt.title("Asset Daily Percentage Changes / Asset Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "daily_returns.png"))
plt.close()

# Visualization 3: Rolling Mean & Volatility (Standard Deviation)
plt.figure(figsize=(11, 5))
for asset in tickers:
    rolling_vol = df_returns[asset].rolling(window=21).std() * np.sqrt(252) # 21-day rolling window annualized
    plt.plot(rolling_vol, label=f"{asset} 21-Day Vol")
plt.title("Annualized Rolling Market Volatility (Standard Deviation)")
plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rolling_volatility.png"))
plt.close()

print(f"All 3 mandatory visual EDA plots saved successfully to '{output_dir}/'.")

# Save files
df_close.to_csv(os.path.join(output_dir, "assets_close.csv"))
df_returns.to_csv(os.path.join(output_dir, "assets_returns.csv"))
print("Data pipeline executed to completion.")