"""Tier 0 baselines: the benchmark that actually matters on daily returns.

Three models, and the difference between the first two is the whole point.

``ZeroReturn`` is the random walk **on prices**. A random walk in the price level
implies an expected return of zero, so the forecast is zero at every session for
every ticker. Its RMSE is the sample standard deviation of the test returns, by
construction. **This is the H1 reference.**

``LastReturn`` is ``statsforecast``'s ``Naive`` applied to the return series:
``y[t+1] = y[t]``. It is a perfectly respectable model and a completely different
one. The target is already a first difference of log prices, so predicting the
previous return differences it a second time, and its error variance is

    Var(y_t - y_{t-1}) = 2 * sigma^2 * (1 - rho)

against ``sigma^2`` for the zero forecast. On this panel the lag-1
autocorrelation of returns is about -0.06, so ``LastReturn``'s RMSE is
``sqrt(2(1-rho))`` = 1.45 times ``ZeroReturn``'s. Any model scored against
``LastReturn`` therefore looks about 30% better than it is, purely from the
choice of comparator. That mistake was live here until 2 September 2026: it made
AutoARIMA appear to cut RMSE 31% against "the random walk", when in fact
AutoARIMA's 0.01982 is marginally *worse* than the 0.01973 a constant zero
scores. ``LastReturn`` is kept because that gap is worth reporting, and is named
for what it does so it cannot be mistaken for the benchmark again.

``RandomWalkWithDrift`` is ``LastReturn`` plus the mean per-session change and
inherits the same double-differencing penalty.

Predictive quantiles
--------------------
See :mod:`sp500bench.models.statsforecast_driver` for why the spread is
estimated from in-sample residuals rather than repeated nine times. ZeroReturn is
the clean case: its residual *is* the return, so its predictive distribution is
the empirical distribution of training returns for that ticker - which is the
honest statement of what a random walk knows, namely nothing about the
conditional mean and a great deal about the unconditional spread.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from sp500bench.config import quantile_column
from sp500bench.models.statsforecast_driver import (MIN_TRAIN_RESIDUALS,  # noqa: F401
                                                    residual_quantiles,
                                                    statsforecast_fold,
                                                    warmup_statsforecast)

__all__ = ["zero_return_forecast_fold", "last_return_forecast_fold",
           "rwd_forecast_fold", "residual_quantiles", "statsforecast_fold",
           "warmup_statsforecast", "MIN_TRAIN_RESIDUALS"]


def zero_return_forecast_fold(
    panel: pd.DataFrame,
    train: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    quantiles: Sequence[float],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    **_,
) -> pd.DataFrame:
    """The random walk on prices: forecast zero, every session, every ticker.

    The point forecast is identically zero, so the quantile band is the empirical
    distribution of the fold's training returns for that ticker - the residual of
    a zero forecast is the return itself. Nothing is fitted beyond those
    quantiles, and they come from the purged training window alone.

    The band is **centred on zero**: the empirical quantiles are shifted by the
    training median before being written out. Two things have to hold at once
    and this is what reconciles them. The point forecast must be exactly zero,
    or the model is not the random walk and the identity that makes it checkable
    from outside - ``RMSE == sqrt(mean(y**2))`` on the test returns - stops
    holding. And ``point_forecast_from: median`` in configs/eval.yaml means the
    point forecast is read from ``q0.5``, so ``q0.5`` must *be* that zero rather
    than the empirical median, which on real data sits a few basis points off it.
    Centring keeps the shape of the empirical distribution - its width, its skew,
    its fat tails - while putting its median where the model's claim actually is.
    """
    levels = list(quantiles)
    columns = [quantile_column(q) for q in levels]

    bands, thin = {}, []
    for ticker, group in train.groupby(id_col, sort=True):
        returns = group[target_col].to_numpy(dtype=float)
        returns = returns[np.isfinite(returns)]
        if returns.size < MIN_TRAIN_RESIDUALS:
            thin.append((ticker, int(returns.size)))
        if returns.size == 0:
            raise ValueError(f"{ticker}: no training returns to fit quantiles on")
        # Centred on zero: shape from the data, median from the model's claim.
        bands[ticker] = np.quantile(returns, levels) - np.median(returns)
    if thin:
        raise ValueError(
            f"{len(thin)} ticker(s) have fewer than {MIN_TRAIN_RESIDUALS} training "
            f"returns, e.g. {thin[:3]}.")

    test = panel[(panel[time_col] >= test_start) & (panel[time_col] <= test_end)]
    test = test.dropna(subset=[target_col])

    out = test[[id_col, time_col]].rename(
        columns={id_col: "unique_id", time_col: "ds"}).reset_index(drop=True)
    offsets = np.vstack([bands[t] for t in out["unique_id"]])
    for j, column in enumerate(columns):
        out[column] = offsets[:, j]
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def last_return_forecast_fold(**kwargs):
    """statsforecast's Naive on the return series: y[t+1] = y[t].

    Not the random-walk benchmark - see the module docstring. Kept to quantify
    what a spurious extra lag costs.
    """
    from statsforecast.models import Naive
    return statsforecast_fold(model=Naive(), refit=True, **kwargs)


def rwd_forecast_fold(**kwargs):
    """Random walk with drift: LastReturn plus the average per-session change.

    On daily log returns the drift is order 1e-6, so this sits almost exactly on
    top of LastReturn - and inherits its double-differencing penalty.
    """
    from statsforecast.models import RandomWalkWithDrift
    return statsforecast_fold(model=RandomWalkWithDrift(), refit=True, **kwargs)
