import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def load_portfolio_data(file_path: str) -> pd.DataFrame:
    """Loads asset data securely with explicit error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical Data Error: Target asset file not found at '{file_path}'")
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if df.empty:
            raise ValueError(f"Data Integrity Error: File at '{file_path}' is empty.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to parse target CSV structure: {str(e)}")

def detect_return_outliers(df_returns: pd.DataFrame, threshold_sigmas: float = 3.0) -> dict:
    """Identifies extreme anomaly trading days using statistical Z-scores."""
    outlier_map = {}
    for asset in df_returns.columns:
        series = df_returns[asset]
        z_scores = (series - series.mean()) / series.std()
        outliers = series[np.abs(z_scores) > threshold_sigmas]
        outlier_map[asset] = {
            "count": len(outliers),
            "extreme_days": outliers.to_dict()
        }
    return outlier_map

def run_stationarity_check(series: pd.Series) -> dict:
    """Executes an Augmented Dickey-Fuller test with structured results output."""
    try:
        res = adfuller(series.dropna())
        return {
            "adf_statistic": float(res[0]),
            "p_value": float(res[1]),
            "is_stationary": bool(res[1] <= 0.05)
        }
    except Exception as e:
        return {"error": f"ADF mathematical calculation failure: {str(e)}"}

def split_data_chronologically(df: pd.DataFrame, split_date: str = "2025-01-01"):
    """Splits financial datasets sequentially to preserve temporal order."""
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    if train.empty or test.empty:
        raise RuntimeError(f"Split date '{split_date}' resulted in empty subsets.")
    return train, test

def calculate_risk_metrics(df_returns: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Calculates Annualized Sharpe Ratio and 95% Historical Value at Risk (VaR)."""
    metrics = {}
    trading_days = 252
    for asset in df_returns.columns:
        series = df_returns[asset].dropna()
        mean_return = series.mean()
        std_dev = series.std()
        annualized_return = mean_return * trading_days
        annualized_vol = std_dev * np.sqrt(trading_days)
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0
        var_95 = np.percentile(series, 5)
        metrics[asset] = {
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe,
            "95% Historical VaR": var_95
        }
    return metrics

def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates performance comparison metrics: MAE, RMSE, and MAPE."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Filter zeros to safeguard division operations in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def create_windowed_sequences(data: np.ndarray, window_size: int = 60):
    """Prepares sequential matrix data structures matching window sizes for LSTM input."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def scale_minmax(series_data: np.ndarray):
    """Normalizes an array-like series strictly within a [0, 1] range for deep learning models."""
    min_val = np.min(series_data)
    max_val = np.max(series_data)
    scaled = (series_data - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else series_data
    return scaled, min_val, max_val

def inverse_scale_minmax(scaled_data: np.ndarray, min_val: float, max_val: float):
    """Reverts normalized arrays back to their original nominal financial price scales."""
    return scaled_data * (max_val - min_val) + min_val