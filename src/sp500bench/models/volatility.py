"""Volatility estimation and the HAR family.

Two jobs:

1. Build a defensible volatility *proxy* from daily OHLC bars.
2. Provide the HAR-RV and Log-HAR benchmarks, which are the field-standard
   comparators for any volatility forecasting claim. Omitting them makes a
   volatility result indefensible.

On the proxy
------------
True realised volatility is the sum of squared *intraday* returns. This project
has daily bars only, so what follows are range-based estimators - proxies, and
they must be labelled as such in the write-up.

They are not a poor substitute. Close-to-close volatility throws away the high
and low, and range-based estimators are several times more efficient for the
same sample. Yang-Zhang additionally handles overnight gaps, which matter for
equities because a large share of variance arrives between the close and the
next open.

On Log-HAR
----------
Volatility is right-skewed and roughly log-normal; log(RV) is far closer to
Gaussian, so OLS assumptions hold better and forecasts are more stable. Brini
(2026) finds a well-specified Log-HAR is the benchmark most foundation models
fail to beat. Fit both and report both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "yang_zhang",
    "realised_vol_close",
    "har_features",
    "HARModel",
]

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Range-based volatility estimators
# ---------------------------------------------------------------------------

def parkinson(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson (1980) high-low range estimator.

    Roughly five times more efficient than close-to-close, but assumes no drift
    and no overnight gap, so it understates equity volatility.
    """
    factor = 1.0 / (4.0 * np.log(2.0))
    daily = factor * np.log(high / low) ** 2
    return np.sqrt(daily.rolling(window, min_periods=window).mean())


def garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Garman-Klass (1980) OHLC estimator.

    Uses the full bar and is more efficient still than Parkinson, at the cost of
    the same no-gap assumption.
    """
    hl = 0.5 * np.log(high / low) ** 2
    co = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_) ** 2
    daily = hl - co
    return np.sqrt(daily.rolling(window, min_periods=window).mean())


def rogers_satchell(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Rogers-Satchell (1991) estimator, valid under non-zero drift.

    Matters over long samples where a trending stock would otherwise inflate
    Parkinson and Garman-Klass.
    """
    daily = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )
    return np.sqrt(daily.rolling(window, min_periods=window).mean())


def yang_zhang(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Yang-Zhang (2000) estimator - the recommended default here.

    Combines overnight, open-to-close and Rogers-Satchell components. It is
    drift-independent *and* gap-aware, which is what daily equity bars require:
    a meaningful share of variance arrives between the close and the next open,
    and every other estimator on this page discards it.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    log_oc_prev = np.log(open_ / close.shift(1))     # overnight
    log_co = np.log(close / open_)                   # open to close

    n = window
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))

    var_overnight = log_oc_prev.rolling(n, min_periods=n).var(ddof=1)
    var_open_close = log_co.rolling(n, min_periods=n).var(ddof=1)
    rs = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    ).rolling(n, min_periods=n).mean()

    var = var_overnight + k * var_open_close + (1.0 - k) * rs
    return np.sqrt(var.clip(lower=0.0))


def realised_vol_close(close: pd.Series, window: int = 21) -> pd.Series:
    """Plain close-to-close rolling volatility.

    Kept as the naive proxy so the write-up can quantify how much the
    range-based estimators actually buy.
    """
    return np.log(close / close.shift(1)).rolling(window, min_periods=window).std(ddof=1)


# ---------------------------------------------------------------------------
# HAR family
# ---------------------------------------------------------------------------

def har_features(rv: pd.Series) -> pd.DataFrame:
    """Daily, weekly and monthly components of the HAR cascade (Corsi, 2009).

    All three lags are trailing and shifted by one session, so the feature block
    at ``t`` contains nothing observed after the close of ``t``.
    """
    rv = rv.astype(float)
    return pd.DataFrame(
        {
            "rv_d": rv.shift(1),
            "rv_w": rv.rolling(5, min_periods=5).mean().shift(1),
            "rv_m": rv.rolling(22, min_periods=22).mean().shift(1),
        }
    )


class HARModel:
    """Heterogeneous Autoregressive model of realised volatility.

    ``log_target=True`` gives Log-HAR: the regression is run on ``log(RV)`` and
    predictions are transformed back with a smearing correction, since
    ``exp(E[log X]) != E[X]``. Skipping that correction biases every forecast
    downward - a common and avoidable error.

    Deliberately implemented with plain least squares rather than a modelling
    library: HAR is three regressors, and a transparent implementation is easier
    to defend in a viva than a wrapper.
    """

    def __init__(self, log_target: bool = False, smearing: bool = True) -> None:
        self.log_target = log_target
        self.smearing = smearing
        self.coef_: np.ndarray | None = None
        self.smearing_factor_: float = 1.0
        self.resid_scale_: float = float("nan")

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X[["rv_d", "rv_w", "rv_m"]].to_numpy()])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HARModel":
        frame = X.join(y.rename("_target")).dropna()
        if len(frame) < 50:
            raise ValueError(f"only {len(frame)} usable rows; HAR needs a longer history")

        target = frame["_target"].to_numpy(dtype=float)
        features = frame[["rv_d", "rv_w", "rv_m"]]

        if self.log_target:
            if (target <= 0).any() or (features.to_numpy() <= 0).any():
                raise ValueError("Log-HAR requires strictly positive volatility")
            target = np.log(target)
            features = np.log(features)

        A = self._design(features)
        self.coef_, *_ = np.linalg.lstsq(A, target, rcond=None)

        resid = target - A @ self.coef_
        self.resid_scale_ = float(np.std(resid, ddof=4))
        self.smearing_factor_ = (
            float(np.mean(np.exp(resid))) if (self.log_target and self.smearing) else 1.0
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("call fit() before predict()")
        features = X[["rv_d", "rv_w", "rv_m"]]
        usable = features.notna().all(axis=1)
        if self.log_target:
            usable &= (features > 0).all(axis=1)
            features = features.where(usable).apply(np.log)

        out = pd.Series(np.nan, index=X.index, dtype=float)
        if usable.any():
            A = self._design(features.loc[usable])
            pred = A @ self.coef_
            if self.log_target:
                pred = np.exp(pred) * self.smearing_factor_
            out.loc[usable] = pred
        return out

    def quantiles(self, X: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
        """Gaussian predictive quantiles, so HAR competes on CRPS too.

        On the log scale the residuals are close to normal, which is precisely
        why Log-HAR is the stronger of the two specifications.
        """
        from scipy import stats

        if self.coef_ is None:
            raise RuntimeError("call fit() before quantiles()")
        features = X[["rv_d", "rv_w", "rv_m"]]
        usable = features.notna().all(axis=1)
        if self.log_target:
            usable &= (features > 0).all(axis=1)

        out = pd.DataFrame(index=X.index, columns=[f"q{q}" for q in levels], dtype=float)
        if not usable.any():
            return out

        feats = features.loc[usable]
        A = self._design(np.log(feats) if self.log_target else feats)
        centre = A @ self.coef_
        for q in levels:
            z = stats.norm.ppf(q)
            vals = centre + z * self.resid_scale_
            if self.log_target:
                vals = np.exp(vals) * self.smearing_factor_
            out.loc[usable, f"q{q}"] = vals
        return out
