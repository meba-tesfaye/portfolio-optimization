# Asset Portfolio Optimization & Price Forecasting Engine

A high-performance data engineering and predictive modeling pipeline built to track, forecast, and optimize asset allocations for **Tesla (TSLA)**, **Vanguard Total Bond Market (BND)**, and **SPDR S&P 500 ETF (SPY)** using historical records spanning January 2015 to June 2026.

---

## 📊 Interim Milestone Progress Summary

### 🛠️ Task 1: Data Preprocessing & EDA (100% Complete)
* **Data Ingestion:** Extracted 2,888 trading days of data via `yfinance` with automated MultiIndex handling.
* **Data Cleaning:** Implemented forward-fill (`ffill()`) and backward-fill (`bfill()`) mechanics to handle non-trading holiday gaps.
* **Stationarity Dynamics (ADF Test):** Proven that raw asset prices are non-stationary, while daily returns are strictly stationary ($p \approx 0.00$), establishing the analytical integration parameter $d = 1$ for baseline forecasting.
* **Risk Profiles:** Programmatically isolated daily 95% Historical Value at Risk (VaR) and Annualized Sharpe Ratios:
  * **TSLA:** Annual Return: 45.42% | Volatility: 57.18% | Sharpe: 0.7944 | Daily 95% VaR: -5.17%
  * **SPY:** Annual Return: 14.43% | Volatility: 17.65% | Sharpe: 0.8175 | Daily 95% VaR: -1.67%
  * **BND:** Annual Return: 2.00% | Volatility: 5.31% | Sharpe: 0.3756 | Daily 95% VaR: -0.48%

### 📈 Task 2: Time Series Forecasting Pipeline (100% Complete)
* **Chronological Split:** Data partitioned sequentially (Train: 2015–2024; Test: 2025–2026) to prevent forward data leakage.
* **Architectures Evaluated:** Evaluated a classical statistical **ARIMA (1, 1, 1)** framework against a high-frequency **ML Sliding Window Momentum Engine** (60-day historical window matrix).
* **Backtest Evaluation Leaderboard:**
  * **ARIMA (1, 1, 1):** MAE: \$54.4586 | RMSE: \$70.5762 | MAPE: 17.2485%
  * **ML Sliding Window:** MAE: \$10.3570 | RMSE: \$13.0550 | MAPE: 2.9130% *(Winner)*

---

## 📂 Project Repository Directory Tree

```text
portfolio-optimization/
├── data/
│   └── processed/          # Saved CSV state artifacts and PNG visualizations
├── scripts/
│   ├── eda_preprocessing.py   # Task 1 Pipeline Engine
│   ├── forecasting_models.py  # Task 2 Predictive Pipeline
│   └── portfolio_optimization.py # Task 3 Optimization Framework
├── requirements.txt         # Production Dependency Management
└── README.md                # System Documentation Homepage