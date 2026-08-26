"""Transaction-cost-aware backtest.

A backtest without costs is not evidence of anything. This module reports three
numbers that separate a professional result from a student one:

* **turnover** - how much trading the signal demands
* **break-even cost** - the round-trip cost in basis points at which the edge
  vanishes
* **deflated Sharpe** - the Sharpe ratio after discounting for the number of
  variants tried

If the break-even cost sits below realistic execution costs for the asset class,
the strategy does not work, regardless of how good the gross curve looks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sp500bench.evaluation.stat_tests import deflated_sharpe_ratio


def positions_from_forecast(
    forecast: pd.Series,
    mode: str = "long_flat",
    threshold: float = 0.0,
) -> pd.Series:
    """Map a return forecast to a position in {-1, 0, 1}.

    ``long_flat`` is the honest default for an equity study without a borrow
    model; ``long_short`` assumes shorting is frictionless, which it is not.
    """
    if mode == "long_flat":
        return (forecast > threshold).astype(float)
    if mode == "long_short":
        return np.sign(forecast).astype(float)
    raise ValueError(f"unknown mode: {mode}")


def run_backtest(
    positions: pd.Series,
    realised_returns: pd.Series,
    cost_bps: float = 5.0,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Simulate the strategy and return its performance summary.

    Positions are applied to the *next* period's return - the signal formed at
    the close of ``t`` earns ``r[t+1]``. Getting this off by one turns a null
    result into a spectacular one, so it is done here, once, explicitly.
    """
    pos = positions.reindex(realised_returns.index).fillna(0.0)
    gross = pos.shift(1).fillna(0.0) * realised_returns

    traded = pos.diff().abs().fillna(pos.abs())
    cost = traded * (cost_bps / 10_000.0)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    ann_factor = periods_per_year
    turnover = float(traded.mean() * ann_factor)

    gross_mean = float(gross.mean())
    trade_rate = float(traded.mean())
    break_even_bps = (
        float(gross_mean / trade_rate * 10_000.0) if trade_rate > 0 else float("inf")
    )

    dsr = deflated_sharpe_ratio(net.to_numpy(), n_trials=n_trials,
                               periods_per_year=periods_per_year)

    return {
        "gross_return_ann": float(gross.mean() * ann_factor),
        "net_return_ann": float(net.mean() * ann_factor),
        "volatility_ann": float(net.std(ddof=1) * np.sqrt(ann_factor)),
        "sharpe_net": dsr["sharpe"],
        "sharpe_hurdle": dsr["sr_hurdle"],
        "deflated_sharpe_prob": dsr["dsr"],
        "max_drawdown": float(drawdown.min()),
        "turnover_ann": turnover,
        "break_even_cost_bps": break_even_bps,
        "cost_bps_assumed": float(cost_bps),
        "hit_rate": float((gross[gross != 0] > 0).mean()) if (gross != 0).any() else float("nan"),
    }
