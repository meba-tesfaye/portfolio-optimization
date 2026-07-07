# 📊 Automated Portfolio Optimization & Time-Series Engine

An end-to-end quantitative financial engineering framework developed for **Guide Me Financial (GMF) Investments**. This production-ready pipeline ingests multi-asset market data, executes statistical risk-auditing, runs time-series forecasting baseline models, maps the Markowitz Efficient Frontier, and validates asset allocation strategies via historical out-of-sample backtesting.

---

## 🏛️ Institutional Context & Business Objective
As a premier personalized portfolio advisory firm, **GMF Investments** leverages data-driven asset allocation frameworks to optimize client wealth. Adhering to the **Efficient Market Hypothesis (EMH)**, this engine does not attempt to predict exact nominal spot prices. Instead, it uncovers underlying statistical properties, maps volatility clustering, and projects variance structures to actively guide systematic portfolio management decisions and risk mitigation.

---

## 🗂️ Project Repository Structure

```text
portfolio-optimization/
├── .github/
│   └── workflows/
│       └── unittests.yml       # Continuous Integration workflow
├── src/
│   ├── __init__.py
│   └── analytics.py            # Centralized, reusable mathematical & risk engine
├── scripts/
│   ├── __init__.py
│   ├── eda_preprocessing.py    # Task 1: Ingestion, cleaning, and EDA execution
│   ├── forecasting_models.py   # Task 2: Temporal splitting and ARIMA baseline
│   ├── portfolio_optimization.py # Task 3 & 4: MPT & Efficient Frontier simulation
│   └── portfolio_backtesting.py  # Task 5: Out-of-sample strategy historical backtest
├── data/
│   └── processed/              # Saved tabular data and visual analysis artifacts
│       ├── assets_close.csv
│       ├── assets_returns.csv
│       ├── price_history.png
│       ├── daily_returns.png
│       ├── rolling_volatility.png
│       ├── efficient_frontier.png
│       └── backtest_performance.png
├── requirements.txt            # Operational python dependencies
├── .gitignore                  # Active environment exclusion configurations
└── README.md                   # Comprehensive deployment documentation