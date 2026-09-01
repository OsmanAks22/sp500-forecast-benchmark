"""Tier 0 baselines: the benchmark that actually matters on daily returns.

Only ``Naive`` is implemented here. It is the model everything else in the
roster has to beat, and H1 is written against it, so it is worth more care than
its name suggests.

Why the quantiles are empirical residual quantiles
--------------------------------------------------
The point forecast is the one-step-ahead random walk on the target series: the
prediction for session ``t`` is the return observed at ``t-1``. That is a single
number, and the obvious way to fill a nine-column quantile grid with it is to
repeat it nine times.

That would be wrong, and wrong in a direction that flatters every other model.
Nine identical columns describe a predictive distribution with no spread at all
- a point mass. Pinball loss at level ``q`` against a point mass degenerates to
``|y - yhat|`` scaled by ``q`` or ``1-q``, so CRPS collapses to roughly the mean
absolute error. Every probabilistic model in the study would then beat Naive on
CRPS regardless of whether it was better calibrated, because it would be
competing against a baseline that declined to express uncertainty at all. H2 is
a calibration hypothesis; a baseline that is uncalibrated *by construction*
makes H2 unfalsifiable.

So the spread is estimated instead. For each ticker the in-sample residual of
the naive forecast, ``r_t = y_t - y_{t-1}``, is collected over the fold's
training window, and its empirical quantiles are added to the point forecast:

    qhat(tau) = yhat + empirical_quantile(r_train, tau)

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

__all__ = ["naive_residual_quantiles", "naive_forecast_fold", "MIN_TRAIN_RESIDUALS"]

# Below this many training residuals the tail levels are being read off a
# handful of points and the 0.01/0.99 columns are noise dressed as a forecast.
MIN_TRAIN_RESIDUALS = 250


def naive_residual_quantiles(
    train: pd.DataFrame,
    quantiles: Sequence[float],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """Per-ticker empirical quantiles of the in-sample naive residual.

    Returns a frame indexed by ``unique_id`` with one column per level. Fitted
    on ``train`` only - this is the whole of what the model estimates.
    """
    levels = list(quantiles)
    rows: dict[str, np.ndarray] = {}
    thin: list[tuple[str, int]] = []

    for ticker, group in train.sort_values(time_col).groupby(id_col, sort=True):
        residual = group[target_col].to_numpy(dtype=float)
        residual = np.diff(residual)                       # y_t - y_{t-1}
        residual = residual[np.isfinite(residual)]
        if residual.size < MIN_TRAIN_RESIDUALS:
            thin.append((ticker, int(residual.size)))
        if residual.size == 0:
            raise ValueError(f"{ticker}: no training residuals to fit quantiles on")
        rows[ticker] = np.quantile(residual, levels)

    if thin:
        raise ValueError(
            f"{len(thin)} ticker(s) have fewer than {MIN_TRAIN_RESIDUALS} training "
            f"residuals, e.g. {thin[:3]}. The 0.01/0.99 columns would be read off "
            "too few points to mean anything."
        )

    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=[quantile_column(q) for q in levels])
    out.index.name = id_col
    return out


def naive_forecast_fold(
    panel: pd.DataFrame,
    train: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    quantiles: Sequence[float],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """One fold of Naive: statsforecast point forecasts plus residual bands.

    ``panel`` is the full history up to and including the test window; ``train``
    is the fold's training frame, already purged and embargoed by
    :class:`~sp500bench.evaluation.splits.PurgedWalkForward`.

    The point forecasts come from ``statsforecast``'s :class:`Naive` via
    ``cross_validation`` with ``h=1, step_size=1``, which is what produces a
    genuine one-step-ahead forecast at every session of the window rather than a
    single origin extrapolated forward. Session dates are mapped to integer
    positions on the trading calendar before the call, because the panel's
    calendar is trading sessions - a ``freq`` of ``'D'`` or ``'B'`` would invent
    dates across holidays.

    Returns ``[unique_id, ds, q0.01 ... q0.99]``, one row per ticker-session in
    the test window.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    levels = list(quantiles)
    bands = naive_residual_quantiles(train, levels, id_col, time_col, target_col)

    history = panel.loc[panel[time_col] <= test_end].dropna(subset=[target_col])
    calendar = pd.DatetimeIndex(np.sort(history[time_col].unique()))
    position = pd.Series(np.arange(len(calendar)), index=calendar)

    in_window = (calendar >= test_start) & (calendar <= test_end)
    n_windows = int(in_window.sum())
    if n_windows == 0:
        raise ValueError(f"no evaluable sessions in [{test_start.date()}, {test_end.date()}]")

    frame = history[[id_col, time_col, target_col]].rename(
        columns={id_col: "unique_id", time_col: "ds", target_col: "y"}
    ).copy()
    frame["ds"] = frame["ds"].map(position).astype("int64")

    sf = StatsForecast(models=[Naive()], freq=1, n_jobs=1)
    cv = sf.cross_validation(df=frame, h=1, step_size=1, n_windows=n_windows)
    cv = cv.reset_index() if cv.index.name == "unique_id" else cv

    out = pd.DataFrame({
        "unique_id": cv["unique_id"].to_numpy(),
        "ds": calendar[cv["ds"].to_numpy()],
    })
    point = cv["Naive"].to_numpy(dtype=float)
    offsets = bands.reindex(out["unique_id"]).to_numpy(dtype=float)
    for j, q in enumerate(levels):
        out[quantile_column(q)] = point + offsets[:, j]

    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)
