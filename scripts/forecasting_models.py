import os
import sys
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ensure local library is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import load_portfolio_data, split_data_chronologically

print("--- Loading Preprocessed Datasets ---")
try:
    df_close = load_portfolio_data("data/processed/assets_close.csv")
    # Isolate Tesla stock price targets
    tsla_prices = df_close["TSLA"]
except Exception as e:
    print(f"Data loading failed: {str(e)}")
    sys.exit(1)

# Chronological split using our reusable library component
train_data, test_data = split_data_chronologically(tsla_prices, split_date="2025-01-01")
print(f"Training observations (2015-2024): {len(train_data)}")
print(f"Testing observations (2025-2026): {len(test_data)}")

print("\n--- Fitting Baseline ARIMA Model ---")
try:
    # Explicitly use documented parameters (p=1, d=1, q=1)
    model = SARIMAX(train_data, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    model_fitted = model.fit(disp=False)
    
    # Generate dynamic forecasts over the out-of-sample testing window
    forecast = model_fitted.forecast(steps=len(test_data))
    print("Baseline ARIMA model training and forecast loop complete.")
except Exception as e:
    print(f"Mathematical modeling error encountered: {str(e)}")