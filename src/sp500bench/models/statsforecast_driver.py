"""Shared driver for every ``statsforecast`` model in the roster.

Tier 0 (:mod:`sp500bench.models.baselines`) and Tier 1
(:mod:`sp500bench.models.statistical`) differ only in which model object they
hand to :func:`statsforecast_fold` and whether it is refitted inside a fold.
Everything else - how the trading calendar is handled, where the predictive
spread comes from, which window the parameters are estimated on - is identical
by construction, which is what makes the tiers comparable.

Why the predictive quantiles are empirical residual quantiles
--------------------------------------------------------------
The point forecast is one number. The obvious way to fill a nine-column quantile
grid with it is to repeat it nine times, and that would be wrong in a direction
that flatters every other model: nine identical columns describe a point mass,
pinball loss against a point mass degenerates to a scaled absolute error, and
CRPS collapses to roughly the MAE. Every probabilistic model in the study would
then beat these models on CRPS regardless of calibration. H2 is a calibration
hypothesis, so a comparator that is uncalibrated *by construction* makes H2
unfalsifiable rather than merely hard to pass.

So the spread is estimated. For each ticker the model's own in-sample
one-step-ahead residuals, ``r_t = y_t - fitted_t``, are collected over the fold's
training window and their empirical quantiles are added to the point forecast:

    qhat(tau) = yhat + empirical_quantile(r_train, tau)

The residuals come from statsforecast's own fitted values, so the band always
belongs to the model that produced the point forecast: AutoARIMA is scored
against AutoARIMA's residuals, not against a hand-rolled difference.

Note the alternative not taken. statsforecast can emit native prediction
intervals for AutoARIMA and AutoETS via ``level=``, derived from the models'
Gaussian innovation variance. Those were not used, for two reasons: they are
Gaussian, and daily equity returns are not, so they would understate exactly the
0.01 and 0.99 levels the grid was densified to reach; and applying them to two
models while the rest of the roster uses empirical residuals would make part of
the CRPS comparison a comparison of interval methodologies. One method, applied
uniformly, is the comparable choice. This is a design decision worth stating in
the write-up rather than a detail.

Fold discipline
---------------
Parameters are estimated on the fold's training window alone, which already has
the purge and the embargo subtracted. The *conditioning* data for a one-step-
ahead forecast at session ``t`` is whatever was observable at ``t-1``, which
inside a test window is test-period data - that is the definition of a
one-step-ahead forecast, not a leak. The two are kept apart deliberately:
``refit=False`` anchors the single fit at ``train_end`` and every later origin is
reached by applying those fixed parameters forward.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from sp500bench.config import quantile_column

__all__ = ["residual_quantiles", "statsforecast_fold", "warmup_statsforecast",
           "MIN_TRAIN_RESIDUALS"]

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


def residual_quantiles(fitted: pd.DataFrame, quantiles: Sequence[float],
                       alias: str) -> pd.DataFrame:
    """Per-ticker empirical quantiles of a model's in-sample residuals.

    ``fitted`` is what ``StatsForecast.forecast_fitted_values()`` returns for the
    training window: one row per ticker-session with the outcome and the model's
    own one-step-ahead fitted value. Sessions with no fitted value - the start of
    each series, and any the model could not fit - drop out.
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
    panel: pd.DataFrame,
    train: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    quantiles: Sequence[float],
    model,
    refit: bool = True,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
) -> pd.DataFrame:
    """One fold of a statsforecast model: point forecasts plus residual bands.

    ``panel`` is the full history; ``train`` is the fold's training frame, already
    purged and embargoed by
    :class:`~sp500bench.evaluation.splits.PurgedWalkForward`.

    Two calls into statsforecast. The first fits on the training window with
    ``fitted=True`` and yields the in-sample residuals the quantile bands are read
    from. The second is ``cross_validation`` with ``h=1, step_size=1``, which gives
    a genuine one-step-ahead forecast at every origin rather than one origin
    extrapolated forward.

    The cross-validation window is anchored so that its *first cutoff lands
    exactly on* ``train.ds.max()``. That matters when ``refit=False``: statsforecast
    then fits once, at the first cutoff, and the fit therefore sees the purged
    training window and nothing else. Forecasts for the purge-and-embargo gap
    fall out of that window too and are discarded here; only the test window is
    returned.

    ``refit=False`` is what "refit once per fold, not per session" means in
    practice. With ``refit=True`` AutoARIMA would re-run order selection at all
    913 origins of fold 3, on every one of 50 tickers.

    Returns ``[unique_id, ds, q0.01 ... q0.99]``, one row per ticker-session in
    the test window.
    """
    from statsforecast import StatsForecast

    levels = list(quantiles)
    alias = model.alias

    history = panel.loc[panel[time_col] <= test_end].dropna(subset=[target_col])
    calendar = pd.DatetimeIndex(np.sort(history[time_col].unique()))
    position = pd.Series(np.arange(len(calendar)), index=calendar)

    train_clean = train.dropna(subset=[target_col])
    train_end_pos = int(position[train_clean[time_col].max()])
    n_windows = len(calendar) - 1 - train_end_pos
    if n_windows <= 0:
        raise ValueError(f"no sessions after train_end for [{test_start.date()}, "
                         f"{test_end.date()}]")

    # Bands: fitted on the purged training window alone.
    sf_train = StatsForecast(models=[model], freq=1, n_jobs=1)
    sf_train.forecast(
        df=_to_sf_frame(train_clean, position, id_col, time_col, target_col),
        h=1, fitted=True,
    )
    bands = residual_quantiles(sf_train.forecast_fitted_values(), levels, alias)

    # Points: one-step-ahead from train_end onwards; the gap is trimmed after.
    sf = StatsForecast(models=[model], freq=1, n_jobs=1)
    cv = sf.cross_validation(
        df=_to_sf_frame(history, position, id_col, time_col, target_col),
        h=1, step_size=1, n_windows=n_windows, refit=refit,
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

    out = out[(out["ds"] >= test_start) & (out["ds"] <= test_end)]
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def warmup_statsforecast() -> None:
    """Import statsforecast and run a throwaway fit before any clock starts.

    The lazy ``from statsforecast import StatsForecast`` above costs 25 seconds on
    a cold bytecode cache - a 135MB dependency tree - and under a second warm.
    Left inside the measured region it lands entirely on whichever fold runs
    first, reading as 20 seconds of model cost against 0.2 seconds for every fold
    after it. That would rank a random walk slower than AutoARIMA on the compute
    axis, which is the opposite of true. Called once per process, charged
    separately, and excluded from fold times.
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
