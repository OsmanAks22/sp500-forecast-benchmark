"""Fold executors for the volatility arm: HAR-RV, Log-HAR and GJR-GARCH.

A separate arm on a separate target. The rest of the roster forecasts the 1-day
forward log return; these three forecast 21-day realised volatility, so they
share the fold structure and the quantile grid with the return arm but nothing
else. They are never ranked against a return model.

The target and its proxy
------------------------
Realised volatility proper is the sum of squared *intraday* returns, and this
project has daily bars, so the target is a range-based proxy: the 21-session
Yang-Zhang estimator from :func:`sp500bench.models.volatility.yang_zhang`.
Yang-Zhang is the choice because it is the only estimator in that module that
accounts for the overnight gap, and on daily equity bars a real share of variance
arrives between the close and the next open. Parkinson and Garman-Klass discard
it and understate equity volatility as a result; Rogers-Satchell handles drift
but not gaps.

Forecast timing, and a caveat worth stating plainly
----------------------------------------------------
The target at session ``t`` is the Yang-Zhang estimate over the 21 sessions
ending at ``t``, and the HAR features at ``t`` are lagged one session, so this is
a one-step-ahead forecast of a trailing window - the convention already
implemented and tested in ``volatility.py`` and ``tests/test_mcs_volatility.py``.
It is leak-free under the frozen purge of one session: a label at ``t`` looks
*backward* 21 sessions and consumes nothing after ``t``.

The caveat is that a trailing 21-session window overlaps its own predecessor in
20 of 21 sessions. Only one session of the target is genuinely unknown at
forecast time, so every model on this arm will post a very low error and the
differences between them will be small in absolute terms. Read the arm as a
ranking, not as evidence that volatility is easy to forecast.

The alternative - a *forward* 21-session window, which is what "forecasting
volatility" usually means - is not available under the frozen configuration. A
forward label at ``t`` would consume data through ``t+21``, so it would need
``cross_validation.horizon: 21`` in configs/eval.yaml to be purged correctly;
under the frozen ``horizon: 1`` it would leak fifteen sessions of test data into
every training fold, and ``assert_no_overlap`` would not catch it because it
would be checking the wrong horizon. That is a config change, and the config is
frozen.

Predictive quantiles
--------------------
Volatility is positive and right-skewed, so the additive empirical-residual
bands used on the return arm are the wrong shape here - they would put negative
volatility at the 0.01 level. HAR-RV and Log-HAR use ``HARModel.quantiles``
directly, which is Gaussian on whatever scale the model was fitted on; for
Log-HAR that is the log scale, so the band is log-normal and stays positive.
GJR-GARCH is given the same log-scale treatment, from its own training
residuals. HAR-RV's band is Gaussian on the raw scale and *can* go negative at
the extreme levels - that is a real property of the specification, not a bug
here, and it is one of the reasons Log-HAR is the stronger comparator.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from sp500bench.config import quantile_column
from sp500bench.models.volatility import HARModel, har_features, yang_zhang

__all__ = ["build_volatility_panel", "har_forecast_fold", "log_har_forecast_fold",
           "gjr_garch_forecast_fold", "RV_WINDOW"]

RV_WINDOW = 21
HAR_COLUMNS = ["rv_d", "rv_w", "rv_m"]


def build_volatility_panel(bars: pd.DataFrame) -> pd.DataFrame:
    """Yang-Zhang target plus the HAR cascade, per ticker.

    Returns ``[unique_id, ds, y, ret, rv_d, rv_w, rv_m]`` where ``y`` is the
    21-session Yang-Zhang estimate at that session and the ``rv_*`` columns are
    the daily, weekly and monthly components of the cascade, each already lagged
    one session by :func:`~sp500bench.models.volatility.har_features`.

    Every series is built per ticker and never across the panel, so no rolling
    window can straddle two names.
    """
    blocks = []
    for ticker, group in bars.groupby("unique_id", sort=True):
        bar = group.set_index("ds").sort_index()
        rv = yang_zhang(bar["open"], bar["high"], bar["low"], bar["close"],
                        window=RV_WINDOW)
        block = har_features(rv)
        block["y"] = rv
        block["ret"] = np.log(bar["close"]).diff()
        block.insert(0, "unique_id", ticker)
        blocks.append(block.reset_index())
    out = pd.concat(blocks, ignore_index=True)
    return out[["unique_id", "ds", "y", "ret", *HAR_COLUMNS]].sort_values(
        ["unique_id", "ds"]).reset_index(drop=True)


def _har_fold(panel, train, test_start, test_end, quantiles, log_target, **_):
    """Shared body for HAR-RV and Log-HAR: fit per ticker on the purged window."""
    levels = list(quantiles)
    columns = [quantile_column(q) for q in levels]
    test = panel[(panel["ds"] >= test_start) & (panel["ds"] <= test_end)]

    out = []
    for ticker, test_group in test.groupby("unique_id", sort=True):
        train_group = train[train["unique_id"] == ticker]
        fit_frame = train_group[HAR_COLUMNS + ["y"]].dropna()
        if fit_frame.empty:
            continue

        model = HARModel(log_target=log_target, smearing=True).fit(
            fit_frame[HAR_COLUMNS], fit_frame["y"])

        features = test_group.set_index("ds")[HAR_COLUMNS]
        bands = model.quantiles(features, levels)
        bands.columns = columns
        bands.insert(0, "unique_id", ticker)
        out.append(bands.reset_index())

    if not out:
        raise ValueError("no ticker produced a HAR fit for this fold")
    frame = pd.concat(out, ignore_index=True)
    frame = frame.dropna(subset=columns)
    return frame[["unique_id", "ds", *columns]].sort_values(
        ["unique_id", "ds"]).reset_index(drop=True)


def har_forecast_fold(**kwargs):
    """HAR-RV: Corsi (2009) cascade on the raw volatility scale."""
    return _har_fold(log_target=False, **kwargs)


def log_har_forecast_fold(**kwargs):
    """Log-HAR: the cascade on log(RV), back-transformed with a smearing factor.

    ``HARModel(log_target=True, smearing=True)`` multiplies the exponentiated
    prediction by ``mean(exp(residual))`` estimated on the training window.
    Without it every forecast is biased low, because ``exp(E[log X]) < E[X]`` by
    Jensen's inequality - the correction is not optional and the module asserts
    the factor exceeds 1. This is the benchmark most foundation models fail to
    beat (Brini, 2026), so it has to be implemented at its best.
    """
    return _har_fold(log_target=True, **kwargs)


# ---------------------------------------------------------------------------
# GJR-GARCH
# ---------------------------------------------------------------------------

def _gjr_variance_path(returns: np.ndarray, omega: float, alpha: float,
                       gamma: float, beta: float, mu: float) -> np.ndarray:
    """One-step-ahead conditional variance at every session, from fixed parameters.

    The GJR recursion (Glosten, Jagannathan and Runkle, 1993)::

        sigma2_t = omega + (alpha + gamma * 1[e_{t-1} < 0]) * e_{t-1}^2
                   + beta * sigma2_{t-1}

    The indicator is the leverage effect: a negative shock raises next-session
    variance by ``alpha + gamma`` where a positive shock of the same size raises
    it by ``alpha`` alone. This is exactly what the HAR cascade cannot represent,
    since HAR regresses on unsigned realised variance and never sees the sign.

    Run forward over the whole series with parameters fitted on training data
    only. ``sigma2_t`` depends on returns through ``t-1`` and nothing later, so
    the path is a genuine one-step-ahead forecast at every session, including
    inside the test window.
    """
    n = len(returns)
    sigma2 = np.empty(n, dtype=float)
    resid = returns - mu
    denom = 1.0 - alpha - 0.5 * gamma - beta
    sigma2[0] = omega / denom if denom > 1e-8 else np.nanvar(resid)
    for t in range(1, n):
        e = resid[t - 1]
        shock = (alpha + (gamma if e < 0 else 0.0)) * e * e
        sigma2[t] = omega + shock + beta * sigma2[t - 1]
    return sigma2


def gjr_garch_forecast_fold(panel, train, test_start, test_end, quantiles, **_):
    """GJR-GARCH(1,1) with skew-t innovations, mapped onto the RV target.

    Fitted once per fold per ticker on the purged training window, by maximum
    likelihood through ``arch``::

        arch_model(r, p=1, o=1, q=1, dist='skewt')

    Returns are scaled by 100 before fitting, which is what ``arch`` expects for
    numerical stability, and scaled back afterwards.

    Two mappings are needed to score a daily-variance model against a 21-session
    range-based proxy, and both are stated rather than buried:

    *Aggregation.* The model's prediction of the 21-session realised volatility
    at ``t`` is the root mean of its own conditional variance path over the 21
    sessions ending at ``t``. Every term in that mean is a one-step-ahead
    conditional variance, so the whole quantity is a function of returns through
    ``t-1``.

    *Calibration.* GJR-GARCH is fitted on close-to-close returns; Yang-Zhang also
    counts the overnight gap and uses the daily range, so the two sit on
    systematically different scales. A single multiplicative factor - the ratio
    of mean target to mean prediction over the training window - puts them on the
    same scale. It is one number per ticker per fold, estimated on training data
    alone. Without it the arm would be measuring the gap between two volatility
    proxies rather than the models' forecasting skill.

    Bands are log-normal, from the model's own training residuals on the log
    scale, which keeps the predictive distribution positive and right-skewed as
    Log-HAR's is.
    """
    from arch import arch_model
    from scipy import stats

    levels = list(quantiles)
    columns = [quantile_column(q) for q in levels]
    train_end = train["ds"].max()
    history = panel[panel["ds"] <= test_end]

    out = []
    for ticker, group in history.groupby("unique_id", sort=True):
        series = group.sort_values("ds").reset_index(drop=True)
        returns = series["ret"].to_numpy(dtype=float)
        usable = np.isfinite(returns)
        if usable.sum() < 500:
            continue
        returns = np.where(usable, returns, 0.0)

        in_train = (series["ds"] <= train_end).to_numpy()
        fitted = arch_model(returns[in_train] * 100.0, p=1, o=1, q=1,
                            dist="skewt", mean="Constant").fit(disp="off",
                                                               show_warning=False)
        params = fitted.params
        sigma2 = _gjr_variance_path(
            returns * 100.0,
            omega=float(params["omega"]), alpha=float(params["alpha[1]"]),
            gamma=float(params["gamma[1]"]), beta=float(params["beta[1]"]),
            mu=float(params["mu"]),
        ) / 10_000.0

        # 21-session aggregation: root mean of the conditional variance path.
        predicted = np.sqrt(pd.Series(sigma2).rolling(RV_WINDOW,
                                                      min_periods=RV_WINDOW).mean()
                            ).to_numpy()

        target = series["y"].to_numpy(dtype=float)
        calibratable = in_train & np.isfinite(target) & np.isfinite(predicted) & (predicted > 0)
        if calibratable.sum() < 250:
            continue
        scale = float(np.mean(target[calibratable]) / np.mean(predicted[calibratable]))
        predicted = predicted * scale

        # Log-scale residual spread, training window only.
        positive = calibratable & (target > 0)
        log_resid = np.log(target[positive]) - np.log(predicted[positive])
        sigma_log = float(np.std(log_resid, ddof=1))
        centre_shift = float(np.mean(log_resid))

        in_test = ((series["ds"] >= test_start) & (series["ds"] <= test_end)).to_numpy()
        keep = in_test & np.isfinite(predicted) & (predicted > 0)
        if not keep.any():
            continue

        block = pd.DataFrame({"unique_id": ticker, "ds": series["ds"].to_numpy()[keep]})
        centre = np.log(predicted[keep]) + centre_shift
        for q, column in zip(levels, columns):
            block[column] = np.exp(centre + stats.norm.ppf(q) * sigma_log)
        out.append(block)

    if not out:
        raise ValueError("no ticker produced a GJR-GARCH fit for this fold")
    return pd.concat(out, ignore_index=True).sort_values(
        ["unique_id", "ds"]).reset_index(drop=True)
