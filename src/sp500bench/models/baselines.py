"""Tier 0 baselines: the benchmark that actually matters on daily returns.

``Naive`` and ``RandomWalkWithDrift``, both via ``statsforecast``. These are the
models everything else in the roster has to beat, and H1 is written against
them, so they are worth more care than their names suggest.

The shared machinery - the trading-calendar handling, the empirical residual
quantiles that give these models an honest predictive spread, and the fold
discipline - lives in :mod:`sp500bench.models.statsforecast_driver`, which
documents why the spread is estimated rather than repeated. Both baselines refit
per origin because refitting them is free.
"""

from __future__ import annotations

from sp500bench.models.statsforecast_driver import (MIN_TRAIN_RESIDUALS,  # noqa: F401
                                                    residual_quantiles,
                                                    statsforecast_fold,
                                                    warmup_statsforecast)

__all__ = ["naive_forecast_fold", "rwd_forecast_fold", "residual_quantiles",
           "statsforecast_fold", "warmup_statsforecast", "MIN_TRAIN_RESIDUALS"]


def naive_forecast_fold(**kwargs):
    """Random walk: the forecast for session t is the return observed at t-1."""
    from statsforecast.models import Naive
    return statsforecast_fold(model=Naive(), refit=True, **kwargs)


def rwd_forecast_fold(**kwargs):
    """Random walk with drift: Naive plus the average per-session change.

    On daily log returns the drift is tiny - the mean change between consecutive
    returns over a decade is order 1e-6 - so this is expected to sit almost on
    top of Naive. That is the point of including it: if the two separate
    materially, something is wrong with the target rather than interesting.
    """
    from statsforecast.models import RandomWalkWithDrift
    return statsforecast_fold(model=RandomWalkWithDrift(), refit=True, **kwargs)
