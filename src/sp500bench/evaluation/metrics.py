"""Forecast accuracy metrics: point, directional and probabilistic.

All functions take flat numpy arrays and return scalars. Aggregation across
tickers and regimes happens one layer up, in the reporting code, so that the
weighting scheme is explicit rather than buried here.

A note on scale
---------------
Never average raw RMSE across tickers whose price levels differ by an order of
magnitude - the result is dominated by the expensive names. Either work on log
returns (as this project does) or use :func:`mase`, which divides through by an
in-sample naive benchmark and is therefore unit-free.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "directional_accuracy_ci",
    "directional_accuracy_with_ci",
    "directional_hits",
    "rmse",
    "mae",
    "mase",
    "directional_accuracy",
    "pinball_loss",
    "crps_from_quantiles",
    "picp",
    "mean_interval_score",
]


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[keep], y_pred[keep]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean absolute scaled error (Hyndman & Koehler, 2006).

    The denominator is the in-sample MAE of a seasonal naive forecast, computed
    on ``y_train`` only. Values below 1 beat that benchmark.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_train = y_train[np.isfinite(y_train)]
    if y_train.size <= seasonality:
        raise ValueError("training series too short to scale against")
    denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def directional_hits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: "pd.Series | np.ndarray | None" = None,
) -> tuple[np.ndarray, "np.ndarray | None"]:
    """Per-observation hit indicator, with the zero-outcome convention applied once.

    Zero-valued outcomes are *dropped* rather than counted as agreement. On daily
    equity returns exact zeros are rare but real - halted names, and low-priced
    names that close unchanged - and ``np.sign(0) == np.sign(anything)`` is false,
    so scoring them would silently count every flat day as a miss and drag the
    statistic below its true value.

    This function is the single definition of that convention. Both
    :func:`directional_accuracy` and :func:`directional_accuracy_with_ci` go
    through it, so a point estimate and its confidence interval are computed from
    the same vector by construction and cannot drift apart. They did drift once:
    the point estimate dropped flat days while a hand-built hits vector passed to
    :func:`directional_accuracy_ci` scored them as misses, and the two reported
    49.79% and 49.62% for the same forecasts.

    Returns ``(hits, dates)``, where ``dates`` is ``None`` if none was supplied
    and otherwise the matching subset - filtered by the same mask, in the same
    order, so the caller cannot misalign them either.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    keep = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    hits = (np.sign(y_true[keep]) == np.sign(y_pred[keep])).astype(float)

    if dates is None:
        return hits, None
    dates = np.asarray(dates).ravel()
    if dates.shape != keep.shape:
        raise ValueError(f"dates length {dates.shape} does not match {keep.shape}")
    return hits, dates[keep]


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of periods where the sign of the forecast matches the outcome.

    Zero-valued outcomes are dropped rather than counted as agreement; see
    :func:`directional_hits`, which owns that convention.
    """
    hits, _ = directional_hits(y_true, y_pred)
    if hits.size == 0:
        return float("nan")
    return float(np.mean(hits))


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss at level ``q``."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly inside (0, 1)")
    y_true, y_pred_q = _clean(y_true, y_pred_q)
    delta = y_true - y_pred_q
    return float(np.mean(np.maximum(q * delta, (q - 1.0) * delta)))


def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """Approximate CRPS by averaging pinball loss over a quantile grid.

    The average pinball loss over levels drawn uniformly from (0, 1) converges to
    CRPS / 2 as the grid densifies; the factor of two is applied here so the
    return value is on the CRPS scale. With a five-point grid the approximation
    is coarse but consistent *across models*, which is what a benchmark needs.
    Report the grid alongside the number.
    """
    if not quantile_forecasts:
        raise ValueError("no quantile forecasts supplied")
    levels = sorted(quantile_forecasts)
    losses = [pinball_loss(y_true, quantile_forecasts[q], q) for q in levels]
    return float(2.0 * np.mean(losses))


def picp(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Prediction interval coverage probability.

    Compare against the nominal level: a 90% interval that covers 74% of
    outcomes is overconfident, one that covers 99% is uselessly wide.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    keep = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if keep.sum() == 0:
        return float("nan")
    inside = (y_true[keep] >= lower[keep]) & (y_true[keep] <= upper[keep])
    return float(np.mean(inside))


def mean_interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """Winkler interval score for a central (1 - alpha) interval.

    Penalises width and misses jointly, so it cannot be gamed by widening the
    band the way PICP alone can.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    keep = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    y_true, lower, upper = y_true[keep], lower[keep], upper[keep]
    width = upper - lower
    below = (2.0 / alpha) * np.maximum(lower - y_true, 0.0)
    above = (2.0 / alpha) * np.maximum(y_true - upper, 0.0)
    return float(np.mean(width + below + above))


# ---------------------------------------------------------------------------
# Cross-sectionally robust inference
# ---------------------------------------------------------------------------

def directional_accuracy_ci(
    hits: "pd.Series",
    dates: "pd.Series",
    n_boot: int = 5_000,
    block_size: int | None = None,
    alpha: float = 0.05,
    seed: int = 20260826,
) -> dict[str, float]:
    """Confidence interval for directional accuracy under cross-sectional dependence.

    The naive standard error treats 50 tickers x 1,650 days as 82,500 independent
    trials, giving se ~ 0.17pp and making 50.4% look decisive. That is wrong: on
    any given day the 50 stocks move together through the market factor, so the
    effective sample is closer to the number of *days* than the number of
    ticker-days.

    The fix here is a moving-block bootstrap over dates. Whole days are resampled
    as units, which preserves both the within-day cross-sectional correlation and
    short-range serial dependence. Report the resulting interval, and report the
    effective sample size alongside it - the gap between the naive and corrected
    figures is itself worth showing.
    """
    import pandas as pd

    frame = pd.DataFrame({"hit": hits.astype(float).to_numpy(),
                          "date": pd.to_datetime(dates).to_numpy()}).dropna()
    if frame.empty:
        return {k: float("nan") for k in
                ("estimate", "ci_low", "ci_high", "se_naive", "se_clustered", "n_days")}

    daily = frame.groupby("date")["hit"].mean().sort_index()
    n_days = len(daily)
    n_obs = len(frame)
    point = float(frame["hit"].mean())

    se_naive = float(np.sqrt(point * (1 - point) / n_obs))

    if block_size is None:
        block_size = max(1, int(np.ceil(n_days ** (1 / 3))))

    from sp500bench.evaluation.mcs import moving_block_bootstrap_indices

    rng = np.random.default_rng(seed)
    idx = moving_block_bootstrap_indices(n_days, block_size, n_boot, rng)
    boot = daily.to_numpy()[idx].mean(axis=1)

    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    se_clustered = float(boot.std(ddof=1))

    return {
        "estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se_naive": se_naive,
        "se_clustered": se_clustered,
        "variance_inflation": float(se_clustered / se_naive) if se_naive > 0 else float("nan"),
        "n_days": float(n_days),
        "n_obs": float(n_obs),
        "min_detectable_effect": float(1.96 * se_clustered),
    }


def directional_accuracy_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: "pd.Series | np.ndarray",
    **kwargs,
) -> dict[str, float]:
    """Directional accuracy and its date-clustered interval, from one hit vector.

    The reason to prefer this over calling :func:`directional_accuracy` and
    :func:`directional_accuracy_ci` separately: those two take different
    arguments - outcomes and forecasts on one side, a pre-built hit vector on the
    other - and the caller is left to apply the zero-outcome convention twice,
    consistently. Here it is applied once, in :func:`directional_hits`, so
    ``estimate`` is the same number :func:`directional_accuracy` returns.

    ``kwargs`` are forwarded to :func:`directional_accuracy_ci` (``n_boot``,
    ``block_size``, ``alpha``, ``seed``).
    """
    import pandas as pd

    hits, kept_dates = directional_hits(y_true, y_pred, dates)
    if hits.size == 0:
        return {k: float("nan") for k in
                ("estimate", "ci_low", "ci_high", "se_naive", "se_clustered",
                 "variance_inflation", "n_days", "n_obs", "min_detectable_effect")}
    return directional_accuracy_ci(pd.Series(hits), pd.Series(kept_dates), **kwargs)
