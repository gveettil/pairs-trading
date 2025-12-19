**Project Overview**

This workspace contains small tools for identifying and backtesting statistical arbitrage (pairs trading) opportunities.

**Files**
- `engle_granger_pairs.py.py`: Utilities to download price series, estimate OLS hedge ratios, and run the Engle–Granger residual ADF test over ordered ticker pairs to find candidate cointegrated relationships.
- `main.py`: A simple pairs trading backtest harness. Defines `PairConfig` objects, builds per-pair spread time series, runs an entry/exit trading simulation, and computes basic performance and trade statistics. Also includes a small example that downloads prices, runs the backtest, and plots NAV.

**Key Functions & Classes**
- `engle_granger_pairs.py.py`
  - `download_prices(tickers, start, end, ...)` : flexible price downloader using `yfinance` that returns log or level prices and handles different yfinance column layouts.
  - `ols_hedge_ratio(y, x, include_const=True)` : runs OLS of `y` on `x` and returns `(alpha, beta, residuals)`.
  - `adf_on_residuals(residuals, ...)` : runs the Augmented Dickey–Fuller test on residuals and computes an AR(1) phi and half-life estimate.
  - `engle_granger_pairs(tickers, ...)` : performs Engle–Granger step 1 (OLS) and step 2 (ADF on residuals) across ordered pairs and returns a DataFrame ranked by ADF statistic.

- `main.py`
  - `PairConfig` : dataclass holding per-pair parameters (tickers, hedge ratio, entry/exit z thresholds, lookback, adf stat). Provides `confidence`, `bet_size`, and `name` properties.
  - `build_pair_frame(prices, cfg)` : builds spread, rolling mean/std, and z-score series for a pair.
  - `run_pairs_backtest(prices, pair_configs, initial_capital, start_date, end_date)` : simulates daily trading over all configured pairs using simple entry/exit rules. Returns `(portfolio, pair_states, trades)`.
  - `compute_performance_metrics(portfolio, ...)` : computes cumulative return, annualized return/volatility, Sharpe, Sortino, drawdowns, VaR/CVaR, etc.
  - `compute_relative_metrics(portfolio, benchmark, ...)` : computes beta, correlation, tracking error and information ratio vs a benchmark.
  - `compute_trade_stats(trades, portfolio)` : simple aggregation of trade-level PnL, hit rate, profit factor, and PnL by pair.

**Dependencies**
- Python 3.8+ (tested with 3.10+ recommended)
- Required packages (install with pip):
  - `pandas`
  - `numpy`
  - `yfinance`
  - `statsmodels`
  - `matplotlib`

Quick install (recommended inside a venv):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy yfinance statsmodels matplotlib
```

**How to run**
- The repository contains an example run inside `main.py`. From the project root run:
```bash
python main.py
```

- What `main.py` does by default:
  - Downloads `Close` prices for the tickers listed near the bottom of `main.py` using `yfinance` (date range: `2020-06-01` → `2025-12-05`).
  - Builds `PairConfig` objects (six example pairs are provided).
  - Runs the pairs backtest from `backtest_start` (default `2025-06-01`) to the last date.
  - Prints portfolio tail, performance metrics, trade stats and (if available) compares vs `SPY`.
  - Plots NAV over time using `matplotlib`.

**Using `engle_granger_pairs.py.py` to find candidates**
- Example (interactive / Python REPL):
```python
from engle_granger_pairs import engle_granger_pairs

candidates = engle_granger_pairs([
    "AAPL", "MSFT", "GOOG", "AMZN"
], start="2018-01-01", pretest_unit_root=True)
print(candidates.head())
```

Notes & Recommendations
- `engle_granger_pairs.py.py` returns log prices by default (`log_prices=True`). If you prefer levels, call `download_prices(..., log_prices=False)`.
- The ADF test and cointegration checks are sensitive to sample period, lookback and the inclusion of constants/trends. Use `pretest_unit_root` to eliminate obvious I(0) series when searching for cointegration.
- `main.py` implements a simple notional-based sizing: `dollar_notional = NAV * bet_size`, where `bet_size = confidence / 4` and `confidence = adf_stat / -5`. Adjust sizing and cash handling for realistic trading constraints.
- The simple PnL accounting in `main.py` assumes instantaneous execution at daily prices and uses `shares_x * spread` to update cash on entry/exit; treat this as an educational/backtest prototype, not production-ready execution code.

If you want, I can:
- add a `requirements.txt` or `pyproject.toml` for reproducible installs,
- run the example and capture the printed output (I can run it locally if you want), or
- extend the backtest to simulate transaction costs, slippage, or position limits.

---
File created by GitHub Copilot assistant.
