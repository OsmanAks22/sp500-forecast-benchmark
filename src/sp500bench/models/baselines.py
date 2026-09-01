"""Tier 0 baselines: the benchmark that actually matters on daily returns.

``Naive`` and ``RandomWalkWithDrift``, both via ``statsforecast``. These are the
models everything else in the roster has to beat, and H1 is written against
them, so they are worth more care than their names suggest.

Why the quantiles are empirical residual quantiles
--------------------------------------------------
The point forecast is a one-step-ahead extrapolation of the target series: for
Naive, the prediction for session ``t`` is the return observed at ``t-1``; for
RandomWalkWithDrift, that plus the average per-session change over the history
available at the cutoff. Either way it is a single number, and the obvious way
to fill a nine-column quantile grid with it is to repeat it nine times.

That would be wrong, and wrong in a direction that flatters every other model.
Nine identical columns describe a predictive distribution with no spread at all
- a point mass. Pinball loss at level ``q`` against a point mass degenerates to
``|y - yhat|`` scaled by ``q`` or ``1-q``, so CRPS collapses to roughly the mean
absolute error. Every probabilistic model in the study would then beat Naive on
CRPS regardless of whether it was better calibrated, because it would be
competing against a baseline that declined to express uncertainty at all. H2 is
a calibration hypothesis; a baseline that is uncalibrated *by construction*
makes H2 unfalsifiable.

So the spread is estimated instead. For each ticker the model's own in-sample
one-step-ahead residuals, ``r_t = y_t - fitted_t``, are collected over the fold's
training window, and their empirical quantiles are added to the point forecast:

    qhat(tau) = yhat + empirical_quantile(r_train, tau)

The residuals come from ``statsforecast``'s own fitted values rather than from a
hand-rolled ``y.diff()``, so the band belongs to the model that produced the
point forecast. For Naive the two happen to coincide; for RandomWalkWithDrift
they do not, since its residuals are centred by the estimated drift.

This is the standard empirical-residual predictive distribution. It is honest
about what a random walk knows - nothing about the conditional mean, a great
deal about the unconditional spread - and it gives the rest of the roster
something real to beat. It also gives Naive a fat-tailed predictive
distribution for free, which matters at the 0.01 and 0.99 levels the grid was
densified to reach.

Fold discipline
---------------
The residual quantiles are fitted on the fold's training window alone, which
already has the purge and the embargo subtracted. The *point* forecast for
session ``t`` uses ``y_{t-1}``, which inside a test window is test-period data -
that is not leakage but the definition of a one-step-ahead forecast. The
distinction is the one that matters here: parameters come from the training
window, conditioning information comes from whatever was observable at forecast
time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from sp500bench.config import quantile_column

__all__ = ["residual_quantiles", "statsforecast_fold", "naive_forecast_fold",
           "rwd_forecast_fold", "warmup_statsforecast", "MIN_TRAIN_RESIDUALS"]

# Below this many training residuals the tail levels are being read off a
# handful of points and the 0.01/0.99 columns are noise dressed as a forecast.
MIN_TRAIN_RESIDUALS = 250


def _to_sf_frame(frame, position, id_col, time_col, target_col):
    """Rename to statsforecast's schema and put ds on integer session positions.

    The panel's calendar is trading sessions. A ``freq`` of ``'D'`` or ``'B'``
    would have statsforecast invent dates across holidays and long weekends, so
    sessions are numbered instead and mapped back after the call.
    """
    out = frame[[id_col, time_col, target_col]].rename(
        columns={id_col: "unique_id", time_col: "ds", target_col: "y"}
    ).copy()
    out["ds"] = out["ds"].map(position).astype("int64")
    return out


def residual_quantiles(fitted, quantiles, alias):
    """Per-ticker empirical quantiles of a model's in-sample residuals.

    ``fitted`` is what ``StatsForecast.forecast_fitted_values()`` returns for the
    training window: one row per ticker-session with the outcome and the model's
    own one-step-ahead fitted value. The first session of each series has no
    fitted value and drops out.
    """
    levels = list(quantiles)
    rows, thin = {}, []
    for ticker, group in fitted.groupby("unique_id", sort=True):
        residual = (group["y"] - group[alias]).to_numpy(dtype=float)
        residual = residual[np.isfinite(residual)]
        if residual.size == 0:
            raise ValueError(f"{ticker}: no training residuals to fit quantiles on")
        if residual.size < MIN_TRAIN_RESIDUALS:
            thin.append((ticker, int(residual.size)))
        rows[ticker] = np.quantile(residual, levels)

    if thin:
        raise ValueError(
            f"{len(thin)} ticker(s) have fewer than {MIN_TRAIN_RESIDUALS} training "
            f"residuals, e.g. {thin[:3]}. The 0.01/0.99 columns would be read off "
            "too few points to mean anything."
        )

    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=[quantile_column(q) for q in levels])
    out.index.name = "unique_id"
    return out


def statsforecast_fold(
    panel,
    train,
    test_start,
    test_end,
    quantiles,
    model,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
):
    """One fold of a statsforecast baseline: point forecasts plus residual bands.

    ``panel`` is the full history up to and including the test window; ``train``
    is the fold's training frame, already purged and embargoed by
    :class:`~sp500bench.evaluation.splits.PurgedWalkForward`.

    Two calls into statsforecast. The first fits on the training window with
    ``fitted=True`` and yields the in-sample residuals the quantile bands are
    read from - training data only, which is what keeps the purge meaningful.
    The second is ``cross_validation`` with ``h=1, step_size=1``, which produces a
    genuine one-step-ahead forecast at every session of the test window rather
    than one origin extrapolated forward.

    Returns ``[unique_id, ds, q0.01 ... q0.99]``, one row per ticker-session.
    """
    from statsforecast import StatsForecast

    levels = list(quantiles)
    alias = model.alias

    history = panel.loc[panel[time_col] <= test_end].dropna(subset=[target_col])
    calendar = pd.DatetimeIndex(np.sort(history[time_col].unique()))
    position = pd.Series(np.arange(len(calendar)), index=calendar)

    in_window = (calendar >= test_start) & (calendar <= test_end)
    n_windows = int(in_window.sum())
    if n_windows == 0:
        raise ValueError(f"no evaluable sessions in [{test_start.date()}, {test_end.date()}]")

    # Bands: fitted on the purged training window alone.
    sf_train = StatsForecast(models=[model], freq=1, n_jobs=1)
    sf_train.forecast(df=_to_sf_frame(train.dropna(subset=[target_col]), position,
                                      id_col, time_col, target_col),
                      h=1, fitted=True)
    bands = residual_quantiles(sf_train.forecast_fitted_values(), levels, alias)

    # Points: one-step-ahead across the test window.
    sf = StatsForecast(models=[model], freq=1, n_jobs=1)
    cv = sf.cross_validation(
        df=_to_sf_frame(history, position, id_col, time_col, target_col),
        h=1, step_size=1, n_windows=n_windows,
    )
    cv = cv.reset_index() if cv.index.name == "unique_id" else cv

    out = pd.DataFrame({
        "unique_id": cv["unique_id"].to_numpy(),
        "ds": calendar[cv["ds"].to_numpy()],
    })
    point = cv[alias].to_numpy(dtype=float)
    offsets = bands.reindex(out["unique_id"]).to_numpy(dtype=float)
    for j, q in enumerate(levels):
        out[quantile_column(q)] = point + offsets[:, j]

    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def naive_forecast_fold(**kwargs):
    """Random walk: the forecast for session t is the return observed at t-1."""
    from statsforecast.models import Naive
    return statsforecast_fold(model=Naive(), **kwargs)


def rwd_forecast_fold(**kwargs):
    """Random walk with drift: Naive plus the average per-session change.

    On daily log returns the drift is tiny - the mean change between consecutive
    returns over a decade is order 1e-6 - so this is expected to sit almost on
    top of Naive. That is the point of including it: if the two separate
    materially, something is wrong with the target rather than interesting.
    """
    from statsforecast.models import RandomWalkWithDrift
    return statsforecast_fold(model=RandomWalkWithDrift(), **kwargs)


def warmup_statsforecast() -> None:
    """Trigger statsforecast's numba compilation on a throwaway series.

    statsforecast JIT-compiles on first use, and the cost lands on whichever fold
    happens to run first - 20 seconds against 0.2 seconds for every fold after
    it. Left inside the measured region it makes a baseline look slower than
    AutoARIMA, which is the opposite of true and would corrupt the compute axis.
    Called once before timing starts, and charged separately.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import Naive, RandomWalkWithDrift

    rng = np.random.default_rng(20260826)
    frame = pd.DataFrame({
        "unique_id": np.repeat(["warmup_a", "warmup_b"], 64),
        "ds": np.tile(np.arange(64), 2),
        "y": rng.normal(0, 0.01, 128),
    })
    sf = StatsForecast(models=[Naive(), RandomWalkWithDrift()], freq=1, n_jobs=1)
    sf.forecast(df=frame, h=1, fitted=True)
    sf.forecast_fitted_values()
    sf.cross_validation(df=frame, h=1, step_size=1, n_windows=2)
