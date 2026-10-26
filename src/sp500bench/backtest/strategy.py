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
    realised_is_forward: bool = False,
) -> dict[str, float | str]:
    """Simulate the strategy and return its performance summary.

    Positions are applied to the *next* period's return - the signal formed at
    the close of ``t`` earns ``r[t+1]``. Getting this off by one turns a null
    result into a spectacular one, so it is done here, once, explicitly.

    By default ``realised_returns`` is indexed by the session in which the
    return ends, so ``r[t+1]`` sits one row after the signal and positions are
    shifted by one row. The study's target is indexed the other way: ``y[t]`` is
    the return from the close of ``t`` to the close of ``t+1``. Pass
    ``realised_is_forward=True`` for such a series; the position is then applied
    to the same row and not shifted a second time. Feeding the forward target
    through the default path delays every position by one session, which is the
    defect corrected on 25 October 2026.

    Cost convention - round trip
    ----------------------------
    ``cost_bps`` is a **round-trip** cost. That is the unit used in
    ``configs/eval.yaml``, in the README, and in the H1 falsification criterion
    ("break-even above 5 bps round trip"), so it is the unit the public
    interface speaks.

    A round trip is two legs, one in and one out, and ``traded`` counts legs -
    it is ``|delta position|``, which registers 1 on entry and 1 again on exit.
    Each leg is therefore charged ``cost_bps / 2``. Charging the full figure per
    leg would bill every round trip twice and halve the reported break-even,
    which is the number H1 turns on.

    ``break_even_round_trip_bps`` is returned in the same unit, and
    ``cost_convention`` states it in the output so no downstream table has to
    infer it.
    """
    pos = positions.reindex(realised_returns.index).fillna(0.0)
    held = pos if realised_is_forward else pos.shift(1).fillna(0.0)
    gross = held * realised_returns

    traded = pos.diff().abs().fillna(pos.abs())
    cost = traded * (cost_bps / 2.0 / 10_000.0)
    net = gross - cost

    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    ann_factor = periods_per_year
    turnover = float(traded.mean() * ann_factor)

    gross_mean = float(gross.mean())
    trade_rate = float(traded.mean())
    # Round trip = two legs, so the break-even round-trip cost is twice the
    # per-leg figure the gross edge can absorb.
    break_even_round_trip_bps = (
        float(2.0 * gross_mean / trade_rate * 10_000.0)
        if trade_rate > 0
        else float("inf")
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
        "break_even_round_trip_bps": break_even_round_trip_bps,
        "cost_bps_assumed": float(cost_bps),
        "cost_convention": "round_trip",
        "hit_rate": float((gross[gross != 0] > 0).mean()) if (gross != 0).any() else float("nan"),
    }
