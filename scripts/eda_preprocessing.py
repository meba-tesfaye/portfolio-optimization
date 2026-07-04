import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Ensure paths exist for data outputs
os.makedirs("data/processed", exist_ok=True)

# 1. Fetching Historical Data via yfinance
tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-01-01"
end_date = "2026-06-30"

print("--- Step 1: Extracting Financial Records ---")
# auto_adjust=True scales all historical closing metrics automatically to account for dividends/splits
raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

# Unpack the adjusted close from the downloaded structure
if isinstance(raw_data.columns, pd.MultiIndex):
    df_close = raw_data['Close'].copy()
else:
    df_close = raw_data[['Close']].copy()

print(f"Data acquired successfully. Structural shape: {df_close.shape}")

# 2. Data Cleaning & Integrity Check
print("\n--- Step 2: Data Cleaning & Metrics ---")
print(df_close.info())
print("\nMissing Values Found Initially:")
print(df_close.isnull().sum())

# Handle missing values using forward fill then backward fill to smooth out holiday discrepancies
df_close = df_close.ffill().bfill()

# Calculate Daily Returns
df_returns = df_close.pct_change().dropna()

# 3. Exploratory Data Analysis Plots
print("\n--- Step 3: Rendering System Exploratory Visualizations ---")
sns.set_theme(style="whitegrid")

# Visual 1: Absolute Performance Trajectory
plt.figure(figsize=(12, 6))
for col in df_close.columns:
    plt.plot(df_close[col], label=col)
plt.title("Asset Adjusted Close Price History (2015 - 2026)", fontsize=12)
plt.xlabel("Timeline")
plt.ylabel("Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/price_history.png")
plt.close()

# Visual 2: Volatility Fluctuations
plt.figure(figsize=(12, 6))
for col in df_returns.columns:
    plt.plot(df_returns[col], label=col, alpha=0.5)
plt.title("Daily Distribution Returns Tracking", fontsize=12)
plt.xlabel("Timeline")
plt.ylabel("Return %")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/daily_returns.png")
plt.close()

# Visual 3: Rolling Risk Exposure (30-Day Window)
rolling_vol = df_returns.rolling(window=30).std() * np.sqrt(252)
plt.figure(figsize=(12, 6))
for col in rolling_vol.columns:
    plt.plot(rolling_vol[col], label=col)
plt.title("30-Day Rolling Annualized Volatility Profile", fontsize=12)
plt.xlabel("Timeline")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/rolling_volatility.png")
plt.close()
print("Plots generated and saved directly to 'data/processed/'.")

# 4. Stationarity Dynamics (Augmented Dickey-Fuller Test)
print("\n--- Step 4: Stationary Evaluations ---")
def run_adf_test(series, name):
    res = adfuller(series)
    print(f"Asset: {name} | ADF Statistic: {res[0]:.4f} | p-value: {res[1]:.4e}")
    if res[1] <= 0.05:
        print("  Status: Stationary (Null Hypothesis Rejected)")
    else:
        print("  Status: Non-Stationary (Requires Differencing)")

for asset in tickers:
    run_adf_test(df_close[asset], f"{asset} Raw Close")
for asset in tickers:
    run_adf_test(df_returns[asset], f"{asset} Daily Returns")

# 5. Core Mathematical Risk Profiling
print("\n--- Step 5: Portfolio Foundations Metrics Calculation ---")
risk_metrics = {}
for asset in tickers:
    ann_return = df_returns[asset].mean() * 252
    ann_volatility = df_returns[asset].std() * np.sqrt(252)
    sharpe = ann_return / ann_volatility if ann_volatility != 0 else 0
    var_95 = np.percentile(df_returns[asset], 5)
    
    risk_metrics[asset] = {
        "Annual Return": f"{ann_return:.2%}",
        "Annual Volatility": f"{ann_volatility:.2%}",
        "Sharpe Ratio": f"{sharpe:.4f}",
        "95% Historical VaR (Daily)": f"{var_95:.2%}"
    }

metrics_summary = pd.DataFrame(risk_metrics).T
print(metrics_summary)

# 6. Save State Output Files
df_close.to_csv("data/processed/assets_close.csv")
df_returns.to_csv("data/processed/assets_returns.csv")
print("\nData state captured successfully inside 'data/processed/' framework.")