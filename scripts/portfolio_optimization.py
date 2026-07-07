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

print("--- Step 1: Loading Asset Return Data ---")
try:
    df_returns = load_portfolio_data("data/processed/assets_returns.csv")
    tickers = list(df_returns.columns)
    print(f"Successfully loaded returns for: {tickers}")
except Exception as e:
    print(f"Initialization Error: {str(e)}")
    sys.exit(1)

print("\n--- Step 2: Calculating Portfolio Core Statistics ---")
# Annualize mean returns and covariance matrix (assuming 252 trading days)
num_assets = len(tickers)
annual_returns = df_returns.mean() * 252
cov_matrix = df_returns.cov() * 252

print("Annualized Expected Returns:")
for asset, ret in annual_returns.items():
    print(f"  {asset}: {ret:.4f}")

print("\n--- Step 3: Running Efficient Frontier Monte Carlo Simulation ---")
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []

# Set random seed for reproducibility
np.random.seed(42)

for i in range(num_portfolios):
    # Generate random weights and normalize to sum to 1.0
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    # Portfolio expected return and volatility calculation
    p_return = np.dot(weights, annual_returns)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe Ratio calculation (assuming risk-free rate of 0%)
    p_sharpe = p_return / p_volatility
    
    results[0,i] = p_volatility
    results[1,i] = p_return
    results[2,i] = p_sharpe

# Find key portfolio boundary matrices
max_sharpe_idx = np.argmax(results[2])
sd_max_sharpe, rt_max_sharpe = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
best_weights = weights_record[max_sharpe_idx]

min_vol_idx = np.argmin(results[0])
sd_min_vol, rt_min_vol = results[0, min_vol_idx], results[1, min_vol_idx]

print("\n📈 MAXIMUM SHARPE RATIO ALLOCATION PROFILE:")
for asset, weight in zip(tickers, best_weights):
    print(f"  Optimal Weight Allocation for {asset}: {weight:.2%}")
print(f"  Expected Portfolio Annual Return: {rt_max_sharpe:.2%}")
print(f"  Expected Portfolio Annual Volatility: {sd_max_sharpe:.2%}")
print(f"  Max Optimized Sharpe Ratio: {results[2, max_sharpe_idx]:.4f}")

print("\n--- Step 4: Generating Efficient Frontier Visualization ---")
plt.figure(figsize=(10, 6))
# Plot the simulated allocation spectrum colored by Sharpe Ratio
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')

# Highlight the Maximum Sharpe Ratio Portfolio
plt.scatter(sd_max_sharpe, rt_max_sharpe, color='red', marker='*', s=200, label='Maximum Sharpe Ratio')
# Highlight the Minimum Variance Portfolio
plt.scatter(sd_min_vol, rt_min_vol, color='blue', marker='X', s=150, label='Minimum Variance')

plt.title("Markowitz Efficient Frontier Simulation Portfolio")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Expected Returns")
plt.legend(labelspacing=0.8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "efficient_frontier.png"))
plt.close()

print(f"Efficient Frontier plotting complete. Artifact saved to '{output_dir}/efficient_frontier.png'.")