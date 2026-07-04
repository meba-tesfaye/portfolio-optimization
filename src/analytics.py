import os
import pandas as pd
import numpy as np
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
        raise RuntimeError(f"Failed to parse target CSV structure. Internal details: {str(e)}")

def detect_return_outliers(df_returns: pd.DataFrame, threshold_sigmas: float = 3.0) -> dict:
    """
    Identifies extreme anomaly trading days using statistical Z-scores.
    Fulfills the 'Explicit Outlier Detection' EDA feedback requirement.
    """
    outlier_map = {}
    for asset in df_returns.columns:
        series = df_returns[asset]
        mean = series.mean()
        std = series.std()
        
        # Calculate statistical Z-scores for each trading day
        z_scores = (series - mean) / std
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