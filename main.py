from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from collections import defaultdict

pd.set_option("display.precision", 4)
np.set_printoptions(precision=4, suppress=True)

@dataclass
class PairConfig:
    # "left box" from your sketch
    ticker_x: str
    ticker_y: str
    hedge_ratio: float          # hedge_ratio_i
    entry_z: float              # entry_z_i
    exit_z: float               # exit_z_i
    lookback: int               # lookback_period_i
    adf_stat: float             # adf_stat_i (negative)

    @property
    def confidence(self) -> float:
        # confidence_i = adf_stat_i / -5.0   (from your middle box)
        return self.adf_stat / -5.0

    @property
    def bet_size(self) -> float:
        # bet_size_i = confidence_i / 4
        return self.confidence / 4.0

    @property
    def name(self) -> str:
        return f"{self.ticker_x}-{self.ticker_y}"


def build_pair_frame(prices: pd.DataFrame, cfg: PairConfig) -> pd.DataFrame:
    """
    For one pair, build a DataFrame indexed by date with:
        price_x, price_y, spread, spread_mean, spread_std, spread_z
    Everything is exactly as in your 'pair i every day has' box.
    """
    df = pd.DataFrame(index=prices.index).copy()

    df["price_x"] = prices[cfg.ticker_x]
    df["price_y"] = prices[cfg.ticker_y]

    # spread_i = price_x_i - hedge_ratio_i * price_y_i
    df["spread"] = df["price_x"] - cfg.hedge_ratio * df["price_y"]

    # rolling mean / std over lookback_period_i
    roll = df["spread"].rolling(cfg.lookback, min_periods=cfg.lookback)
    df["spread_mean"] = roll.mean()
    df["spread_std"] = roll.std()

    # spread_z_i = (spread_i - spread_mean_i) / spread_std_i
    df["spread_z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

    return df


@dataclass
class TradeRecord:
    date: pd.Timestamp
    pair: str
    action: str          # 'ENTER' or 'EXIT'
    z: float
    spread: float
    shares_x: float
    shares_y: float
    nav_before: float
    cash_after: float


def run_pairs_backtest(
    prices: pd.DataFrame,
    pair_configs: List[PairConfig],
    initial_capital: float = 1_000_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
):
    # --- build per-pair market data on FULL history ---
    pair_data: Dict[str, pd.DataFrame] = {}
    for cfg in pair_configs:
        pair_data[cfg.name] = build_pair_frame(prices, cfg)

    # full date index from prices
    all_dates = prices.index

    # restrict to backtest window *only for trading*
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        all_dates = all_dates[all_dates >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        all_dates = all_dates[all_dates <= end_date]

    dates = all_dates

    # --- state containers ---
    cash = float(initial_capital)

    # per-pair state
    states = {
        cfg.name: {
            "in_position": False,
            "shares_x": 0.0,
            "shares_y": 0.0,
        }
        for cfg in pair_configs
    }

    # history for outputs
    portfolio_rows = []
    pair_state_hist: Dict[str, list] = {cfg.name: [] for cfg in pair_configs}
    trades: List[TradeRecord] = []

    def current_nav(current_date: pd.Timestamp) -> float:
        """Compute NAV = cash + sum_i (shares_x * price_x + shares_y * price_y)."""
        positions_value = 0.0
        for cfg in pair_configs:
            name = cfg.name
            st = states[name]
            if not st["in_position"]:
                continue
            row = pair_data[name].loc[current_date]
            px = row["price_x"]
            py = row["price_y"]
            positions_value += st["shares_x"] * px + st["shares_y"] * py
        return cash + positions_value

    # --- main simulation loop ---
    for date in dates:
        # 1) compute NAV at start of day
        nav_start = current_nav(date)
        positions_value = nav_start - cash

        # 2) loop over pairs: apply exit then entry logic
        for cfg in pair_configs:
            name = cfg.name
            st = states[name]
            row = pair_data[name].loc[date]

            z = row["spread_z"]
            spread = row["spread"]

            # if we don't have enough lookback yet, skip
            if np.isnan(z) or np.isnan(spread):
                pair_state_hist[name].append(
                    {
                        "date": date,
                        "in_position": st["in_position"],
                        "shares_x": st["shares_x"],
                        "shares_y": st["shares_y"],
                        "spread": spread,
                        "spread_z": z,
                    }
                )
                continue

            # ---- EXIT LOGIC ----
            # when |spread_z_i| <= exit_z_i:
            if st["in_position"] and abs(z) <= cfg.exit_z:
                nav_before = current_nav(date)
                # Cash += shares_x_i * spread_i
                cash += st["shares_x"] * spread

                trades.append(
                    TradeRecord(
                        date=date,
                        pair=name,
                        action="EXIT",
                        z=z,
                        spread=spread,
                        shares_x=st["shares_x"],
                        shares_y=st["shares_y"],
                        nav_before=nav_before,
                        cash_after=cash,
                    )
                )

                # reset position
                st["in_position"] = False
                st["shares_x"] = 0.0
                st["shares_y"] = 0.0

            # ---- ENTRY LOGIC ----
            # when |spread_z_i| >= entry_z_i:
            if abs(z) >= cfg.entry_z:
                if st["in_position"]:
                    # 'if (in_position_i) { return }' -> do nothing
                    pass
                else:
                    nav_before = current_nav(date)
                    bet_size = cfg.bet_size

                    # shares_x_i = (NAV * bet_size_i) / price_x_i
                    px = row["price_x"]
                    dollar_notional = nav_before * bet_size
                    shares_x = dollar_notional / px

                    # if (spread_z_i > 0) { shares_x_i *= -1 }
                    if z > 0:
                        shares_x *= -1.0

                    # shares_y_i = (-shares_x_i / hedge_ratio_i)
                    shares_y = -cfg.hedge_ratio * shares_x


                    # Cash -= shares_x_i * spread_i
                    cash -= shares_x * spread

                    # update state
                    st["in_position"] = True
                    st["shares_x"] = shares_x
                    st["shares_y"] = shares_y

                    trades.append(
                        TradeRecord(
                            date=date,
                            pair=name,
                            action="ENTER",
                            z=z,
                            spread=spread,
                            shares_x=shares_x,
                            shares_y=shares_y,
                            nav_before=nav_before,
                            cash_after=cash,
                        )
                    )

            # store per-day per-pair state
            pair_state_hist[name].append(
                {
                    "date": date,
                    "in_position": st["in_position"],
                    "shares_x": st["shares_x"],
                    "shares_y": st["shares_y"],
                    "spread": spread,
                    "spread_z": z,
                }
            )

        # 3) portfolio NAV at end of day (after all trades)
        nav_end = current_nav(date)
        positions_value = nav_end - cash

        portfolio_rows.append(
            {
                "date": date,
                "cash": cash,
                "positions_value": positions_value,
                "nav": nav_end,
            }
        )

    # --- build nice output structures ---
    portfolio = pd.DataFrame(portfolio_rows).set_index("date")

    pair_states = {
        name: pd.DataFrame(rows).set_index("date")
        for name, rows in pair_state_hist.items()
    }

    return portfolio, pair_states, trades



pairs = [
    PairConfig(
        ticker_x="UUUU",
        ticker_y="USAR",
        hedge_ratio=0.462336,
        entry_z=1.4,
        exit_z=0.7,
        lookback=90,
        adf_stat=-4.77,
    ),
    PairConfig(
        ticker_x="OKLO",
        ticker_y="NNE",
        hedge_ratio=0.5747,
        entry_z=1.9,
        exit_z=0.55,
        lookback=40,
        adf_stat=-3.53,
    ),
    PairConfig(
        ticker_x="UMAC",
        ticker_y="RCAT",
        hedge_ratio=0.902581,
        entry_z=1.7,
        exit_z=0.45,
        lookback=50,
        adf_stat=-3.41,
    ),
    PairConfig(
        ticker_x="SPY",
        ticker_y="VOO",
        hedge_ratio=1.00,
        entry_z=2.2,
        exit_z=0.5,
        lookback=50,
        adf_stat=-14,
    ),
    PairConfig(
        ticker_x="AAPL",
        ticker_y="LOW",
        hedge_ratio=0.6019,
        entry_z=2.3,
        exit_z=0.85,
        lookback=20,
        adf_stat=-4.54,
    ),
    PairConfig(
        ticker_x="V",
        ticker_y="MA",
        hedge_ratio=1.138495,
        entry_z=2.0,
        exit_z=1.0,
        lookback=80,
        adf_stat=-4.51,
    ),
]

prices = yf.download(
    sorted({cfg.ticker_x for cfg in pairs} | {cfg.ticker_y for cfg in pairs}),
    start="2020-06-01",
    end="2025-12-05",
)["Close"]

last_date = "2025-12-05"
backtest_start = "2025-06-01"


portfolio, pair_states, trades = run_pairs_backtest(
    prices,
    pairs,
    initial_capital=1_000_000,
    start_date=backtest_start,  # <-- only trade here
)


print(portfolio.tail())


import matplotlib.pyplot as plt
import matplotlib.dates as mdates


fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(portfolio.index, portfolio["nav"])

ax.set_xlabel("Date")
ax.set_ylabel("NAV")
ax.set_title("Value over time")

# Nice automatic date ticks
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
plt.show()


def compute_performance_metrics(
    portfolio: pd.DataFrame,
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
    var_alpha: float = 0.05,
) -> dict:
    nav = portfolio["nav"].astype(float)
    rets = nav.pct_change().dropna()

    if rets.empty:
        raise ValueError("No returns available to compute metrics.")

    # risk-free per period
    rf_periodic = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = rets - rf_periodic

    # basic return metrics
    cumulative_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    num_periods = len(rets)
    years = num_periods / periods_per_year
    annualized_return = (1.0 + cumulative_return) ** (1.0 / years) - 1.0

    # volatility, Sharpe
    vol_annual = excess.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = excess.mean() / excess.std(ddof=0) * np.sqrt(periods_per_year)

    # downside volatility & Sortino
    downside = excess[excess < 0]
    if len(downside) > 0:
        downside_vol_annual = downside.std(ddof=0) * np.sqrt(periods_per_year)
        sortino = excess.mean() / downside.std(ddof=0) * np.sqrt(periods_per_year)
    else:
        downside_vol_annual = 0.0
        sortino = np.nan

    # drawdowns
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = drawdown.min()

    if max_drawdown < 0:
        calmar = annualized_return / abs(max_drawdown)
    else:
        calmar = np.nan

    # distributional risk: VaR & CVaR on 1-period returns
    var = np.quantile(rets, var_alpha)          # e.g. 5% one-day VaR
    cvar = rets[rets <= var].mean() if (rets <= var).any() else np.nan

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": vol_annual,
        "sharpe_ratio": sharpe,
        "downside_vol_annual": downside_vol_annual,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "var_1p": var,
        "cvar_1p": cvar,
        "num_periods": num_periods,
        "start_date": nav.index[0],
        "end_date": nav.index[-1],
    }

def compute_relative_metrics(portfolio, benchmark, periods_per_year=252):
    nav = portfolio["nav"]
    r = nav.pct_change().dropna()
    rb = benchmark.pct_change().reindex_like(r).dropna()

    # align
    r, rb = r.align(rb, join="inner")
    excess = r - rb

    beta = np.cov(r, rb)[0, 1] / np.var(rb)
    corr = np.corrcoef(r, rb)[0, 1]
    tracking_error = excess.std(ddof=0) * np.sqrt(periods_per_year)
    info_ratio = excess.mean() / excess.std(ddof=0) * np.sqrt(periods_per_year)

    return {
        "beta": beta,
        "corr": corr,
        "tracking_error_annual": tracking_error,
        "information_ratio": info_ratio,
    }



def compute_trade_stats(trades: list, portfolio: pd.DataFrame) -> dict:
    """
    Very simple trade stats: assumes one position at a time per pair.
    Uses nav on exit vs nav on entry as a proxy for PnL per trade.
    """
    nav = portfolio["nav"]

    # group ENTER/EXIT by pair
    stack = defaultdict(list)   # pair -> list of ENTER trades waiting for EXIT
    completed = []

    for tr in trades:
        if tr.action == "ENTER":
            stack[tr.pair].append(tr)
        elif tr.action == "EXIT":
            if stack[tr.pair]:
                enter = stack[tr.pair].pop()
                pnl = nav.loc[tr.date] - nav.loc[enter.date]
                holding_period = (tr.date - enter.date).days
                completed.append((tr.pair, enter.date, tr.date, pnl, holding_period))

    if not completed:
        return {}

    trade_df = pd.DataFrame(
        completed,
        columns=["pair", "entry_date", "exit_date", "pnl", "holding_days"],
    )

    wins = trade_df["pnl"] > 0
    losses = trade_df["pnl"] < 0

    profit_factor = (
        trade_df.loc[wins, "pnl"].sum() / abs(trade_df.loc[losses, "pnl"].sum())
        if losses.any() else np.inf
    )

    stats = {
        "num_trades": len(trade_df),
        "hit_rate": wins.mean(),
        "avg_pnl": trade_df["pnl"].mean(),
        "median_pnl": trade_df["pnl"].median(),
        "avg_win": trade_df.loc[wins, "pnl"].mean() if wins.any() else 0.0,
        "avg_loss": trade_df.loc[losses, "pnl"].mean() if losses.any() else 0.0,
        "profit_factor": profit_factor,
        "avg_holding_days": trade_df["holding_days"].mean(),
        "pnl_by_pair": trade_df.groupby("pair")["pnl"].sum().to_dict(),
    }
    return stats


perf = compute_performance_metrics(portfolio, periods_per_year=252, rf_annual=0.0)

print("=== Portfolio performance metrics ===")
for k, v in perf.items():
    # Format numeric values with 4 decimals; fall back to plain printing for others (e.g. timestamps)
    try:
        if isinstance(v, (int, float, np.floating, np.integer)):
            print(f"{k:25s}: {v:.4f}")
        else:
            # pandas Timestamp and other objects -> print as-is
            print(f"{k:25s}: {v}")
    except Exception:
        print(f"{k:25s}: {v}")

# 6) compute trade-level stats
trade_stats = compute_trade_stats(trades, portfolio)
print("\n=== Trade stats ===")
for k, v in trade_stats.items():
    print(f"{k:25s}: {v}")

# 7) optional: relative metrics vs SPY as a benchmark
if "SPY" in prices.columns:
    rel = compute_relative_metrics(portfolio, prices["SPY"])
    print("\n=== Relative to SPY ===")
    for k, v in rel.items():
        print(f"{k:25s}: {v}")