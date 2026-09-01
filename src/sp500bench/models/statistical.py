"""Tier 1: classical statistical forecasters, one model per ticker.

``AutoARIMA`` and ``AutoETS`` via ``statsforecast``, sharing the driver in
:mod:`sp500bench.models.statsforecast_driver` with the Tier 0 baselines so that
the only difference between the tiers is the model object and the refit policy.

Refit once per fold, not per session
------------------------------------
Both are called with ``refit=False``, so statsforecast fits once per fold - at
the first cross-validation cutoff, which the driver anchors on the purged
``train_end`` - and reaches every later origin by applying those fixed parameters
forward. The alternative is not a slower version of the same experiment: with
``refit=True`` AutoARIMA re-runs order selection at every origin, which is 913
model selections per ticker in fold 3 alone, 50 tickers deep, and it would also
mean the model being scored on 31 December is not the model that was selected in
January. Annual refits are the design, stated in configs/models.yaml as
``refit: per_fold`` and in the README's frozen decisions table.

The cost is stated plainly: parameters drift stale across a fold, most visibly
in fold 3, which spans three and a half years. That is the trade the compute
budget buys, and it applies equally to every model in the tier.
"""

from __future__ import annotations

from sp500bench.models.statsforecast_driver import statsforecast_fold

__all__ = ["autoarima_forecast_fold", "autoets_forecast_fold"]


def autoarima_forecast_fold(**kwargs):
    """AutoARIMA: order selected once per fold, per ticker, on training data.

    Order selection runs inside the fold's training window, so the (p, d, q)
    chosen for a ticker is a function of purged training data alone.
    """
    from statsforecast.models import AutoARIMA
    return statsforecast_fold(model=AutoARIMA(), refit=False, **kwargs)


def autoets_forecast_fold(**kwargs):
    """AutoETS: error/trend/season form selected once per fold, per ticker.

    On daily log returns the selected form is expected to collapse to something
    close to simple exponential smoothing with no trend or season - there is no
    daily seasonality in a return series and no persistent drift. A model that
    selects a strong trend here is a signal to check the target, not a finding.
    """
    from statsforecast.models import AutoETS
    return statsforecast_fold(model=AutoETS(), refit=False, **kwargs)
