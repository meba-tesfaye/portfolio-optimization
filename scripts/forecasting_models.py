import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analytics import load_portfolio_data, split_data_chronologically, calculate_forecast_metrics, scale_minmax, inverse_scale_minmax, create_windowed_sequences

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Enforce strict reproducibility
np.random.seed(42)

print("--- Step 1: Loading Ingested Closing Price Context ---")
df_close = load_portfolio_data("data/processed/assets_close.csv")
tsla_series = df_close["TSLA"].values

train, test = split_data_chronologically(df_close, "2025-01-01")
tsla_train_raw = train["TSLA"].values
tsla_test_raw = test["TSLA"].values

print("\n--- Step 2: Preparing Windowed Data Matrices (Window Size = 60) ---")
window_size = 60

# Normalize arrays to safeguard mathematical scaling boundaries [0, 1]
scaled_train, min_t, max_t = scale_minmax(tsla_train_raw)

full_test_arr = np.concatenate([tsla_train_raw[-window_size:], tsla_test_raw])
scaled_test = (full_test_arr - min_t) / (max_t - min_t)

X_train, y_train = create_windowed_sequences(scaled_train, window_size)
X_test, y_test = create_windowed_sequences(scaled_test, window_size)

print(f"Train shapes: X={X_train.shape}, y={y_train.shape} | Test shapes: X={X_test.shape}, y={y_test.shape}")

print("\n--- Step 3: Initializing NumPy Recurrent Network Parameters ---")
hidden_dim = 16
learning_rate = 0.01
epochs = 5

# Weight matrices initialization (Xavier/Glorot style)
Wx = np.random.randn(hidden_dim, 1) * np.sqrt(1.0 / hidden_dim)
Wh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
Wy = np.random.randn(1, hidden_dim) * np.sqrt(1.0 / hidden_dim)

bh = np.zeros((hidden_dim, 1))
by = np.zeros((1, 1))

print(f"Training Recurrent Model from scratch for {epochs} epochs...")
for epoch in range(epochs):
    epoch_loss = 0
    # Stochastic gradient descent loop over training samples
    for i in range(len(X_train)):
        x_seq = X_train[i]  # shape: (60,)
        y_true = y_train[i] # scalar
        
        # Forward pass tracking hidden states over time window steps
        h_states = {}
        h_states[-1] = np.zeros((hidden_dim, 1))
        
        for t in range(window_size):
            xt = np.array([[x_seq[t]]]) # (1, 1)
            # Hidden state recurrence calculation: h_t = tanh(Wx*x_t + Wh*h_{t-1} + bh)
            h_states[t] = np.tanh(np.dot(Wx, xt) + np.dot(Wh, h_states[t-1]) + bh)
            
        # Compute dense output prediction layer
        pred_y = np.dot(Wy, h_states[window_size - 1]) + by
        loss = 0.5 * (pred_y[0, 0] - y_true) ** 2
        epoch_loss += loss
        
        # Backpropagation Through Time (BPTT) derivative chains
        dy = pred_y - y_true
        dWy = np.dot(dy, h_states[window_size - 1].T)
        dby = dy
        
        # Initialize gradient backprop state
        dh = np.dot(Wy.T, dy)
        dWx, dWh, dbh = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(bh)
        
        for t in reversed(range(window_size)):
            # Tanh activation derivative node
            dtanh = (1 - h_states[t] ** 2) * dh
            dbh += dtanh
            dWx += np.dot(dtanh, np.array([[x_seq[t]]]))
            dWh += np.dot(dtanh, h_states[t-1].T)
            dh = np.dot(Wh.T, dtanh)
            
        # Parameter updates via gradient updates
        Wy -= learning_rate * dWy
        by -= learning_rate * dby
        Wx -= learning_rate * dWx
        Wh -= learning_rate * dWh
        bh -= learning_rate * dbh
        
    print(f"  Epoch {epoch + 1}/{epochs} Completed - Normalized Tracking Error: {epoch_loss / len(X_train):.6f}")

print("\n--- Step 4: Out-of-Sample Sequence Inference Evaluation ---")
rnn_predictions_scaled = []
for i in range(len(X_test)):
    x_seq = X_test[i]
    h_current = np.zeros((hidden_dim, 1))
    for t in range(window_size):
        xt = np.array([[x_seq[t]]])
        h_current = np.tanh(np.dot(Wx, xt) + np.dot(Wh, h_current) + bh)
    pred = np.dot(Wy, h_current) + by
    rnn_predictions_scaled.append(pred[0, 0])

rnn_pred = inverse_scale_minmax(np.array(rnn_predictions_scaled), min_t, max_t)

# Fit Baseline ARIMA Model for benchmarking comparison table
arima_model = SARIMAX(tsla_train_raw, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
arima_fit = arima_model.fit(disp=False)
arima_pred = arima_fit.predict(start=len(tsla_train_raw), end=len(tsla_train_raw) + len(tsla_test_raw) - 1)

# Generate baseline comparison metrics matrices
arima_metrics = calculate_forecast_metrics(tsla_test_raw, arima_pred)
rnn_metrics = calculate_forecast_metrics(tsla_test_raw, rnn_pred)

print("\n====================================================================")
print("PRODUCTION MODEL EVALUATION PERFORMANCE COMPARISON SUMMARY MATRIX")
print("====================================================================")
print(f"METRIC      CLASSICAL ARIMA(1,1,1)     PRODUCTION RECURRENT NETWORK")
print("====================================================================")
print(f"MAE:        {arima_metrics['MAE']:.4f}{'':<23}{rnn_metrics['MAE']:.4f}")
print(f"RMSE:       {arima_metrics['RMSE']:.4f}{'':<23}{rnn_metrics['RMSE']:.4f}")
print(f"MAPE:       {arima_metrics['MAPE']:.2f}% ashes{'':<20}{rnn_metrics['MAPE']:.2f}%")
print("====================================================================")

print("\n--- Step 5: Multi-Step Recursive 12-Month Future Forecast Loop ---")
current_window = list(scaled_test[-window_size:])
future_forecast_scaled = []
future_horizon = 252

for _ in range(future_horizon):
    h_current = np.zeros((hidden_dim, 1))
    for t in range(window_size):
        xt = np.array([[current_window[t]]])
        h_current = np.tanh(np.dot(Wx, xt) + np.dot(Wh, h_current) + bh)
    pred_step = np.dot(Wy, h_current) + by
    val = pred_step[0, 0]
    future_forecast_scaled.append(val)
    current_window.append(val)
    current_window.pop(0)

future_forecast = inverse_scale_minmax(np.array(future_forecast_scaled), min_t, max_t)
future_dates = pd.date_range(start=test.index[-1], periods=future_horizon + 1, freq='B')[1:]

df_future = pd.DataFrame({"Forecast": future_forecast}, index=future_dates)
df_future.to_csv(os.path.join(output_dir, "tsla_future_forecast.csv"))

print("\n--- Step 6: Archiving Visual Trajectory Outputs ---")
plt.figure(figsize=(11, 5))
plt.plot(df_close.index[-400:], tsla_series[-400:], label="Historical / Actual Price Paths", color='black')
plt.plot(test.index, rnn_pred, label="RNN Test Window Predictions", color='orange', linestyle='--')
plt.plot(df_future.index, df_future["Forecast"], label="12-Month Multi-Step Future Forecast Path", color='blue', lw=2)
plt.title("TSLA Structural Sequence Multi-Horizon Trajectory Engine Profile")
plt.xlabel("Date")
plt.ylabel("Asset Valuation Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "forecast_trajectory.png"))
plt.close()

print(f"All structural processing completed. Projections saved to '{output_dir}/forecast_trajectory.png'.")