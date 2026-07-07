import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import load_portfolio_data

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

df_returns = load_portfolio_data("data/processed/assets_returns.csv")
tickers = list(df_returns.columns)

print("--- Step 1: Ingesting Dynamic Forward View Assumptions ---")
# Reading the 12-month future trajectory view constructed in Task 3
df_forecast = pd.read_csv(os.path.join(output_dir, "tsla_future_forecast.csv"), index_col=0, parse_dates=True)
tsla_predicted_annual_return = ((df_forecast["Forecast"].iloc[-1] - df_forecast["Forecast"].iloc[0]) / df_forecast["Forecast"].iloc[0])

annual_returns = df_returns.mean() * 252
# Inject the specific forecast return view for TSLA
annual_returns["TSLA"] = tsla_predicted_annual_return
cov_matrix = df_returns.cov() * 252

print("\nAnnualized Allocation Reference Returns:")
for asset, ret in annual_returns.items():
    print(f"  {asset}: {ret:.4f}")

print("\n--- Step 2: Generating Covariance Heatmap Visualization ---")
plt.figure(figsize=(6, 5))
plt.imshow(cov_matrix.values, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Annualized Covariance Scale')
plt.xticks(np.arange(len(tickers)), tickers)
plt.yticks(np.arange(len(tickers)), tickers)
for i in range(len(tickers)):
    for j in range(len(tickers)):
        plt.text(j, i, f"{cov_matrix.iloc[i, j]:.4f}", ha='center', va='center', color='black', weight='bold')
plt.title("Asset Annualized Covariance Heatmap Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "covariance_heatmap.png"))
plt.close()
print(f"Covariance matrix saved to '{output_dir}/covariance_heatmap.png'.")

print("\n--- Step 3: Executing Modern Portfolio Theory Optimization Loop ---")
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []
np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)
    p_return = np.dot(weights, annual_returns)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    results[0,i] = p_volatility
    results[1,i] = p_return
    results[2,i] = p_return / p_volatility

max_sharpe_idx = np.argmax(results[2])
sd_max_sharpe, rt_max_sharpe = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
best_weights = weights_record[max_sharpe_idx]

min_vol_idx = np.argmin(results[0])
sd_min_vol, rt_min_vol = results[0, min_vol_idx], results[1, min_vol_idx]

print("\n📈 FINAL RECOMMENDED MODEL ALLOCATION SPECTRUM:")
for asset, weight in zip(tickers, best_weights):
    print(f"  {asset} Allocation Weight: {weight:.2%}")
print(f"  Expected Return Matrix: {rt_max_sharpe:.2%} | Expected Volatility Matrix: {sd_max_sharpe:.2%}")
print(f"  Optimized Sharpe Ratio Target: {results[2, max_sharpe_idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(sd_max_sharpe, rt_max_sharpe, color='red', marker='*', s=200, label='Maximum Sharpe Ratio Portfolio')
plt.scatter(sd_min_vol, rt_min_vol, color='blue', marker='X', s=150, label='Minimum Volatility Portfolio')
plt.title("GMF Investments: Markowitz Efficient Frontier Spectrum Analysis")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Expected Return")
plt.legend(labelspacing=0.8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "efficient_frontier.png"))
plt.close()