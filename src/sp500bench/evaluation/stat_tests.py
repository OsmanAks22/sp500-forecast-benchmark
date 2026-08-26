"""Statistical significance machinery for the benchmark.

Three pieces, in the order they are used:

1. :func:`diebold_mariano` - is model A's loss genuinely lower than model B's, or
   is the gap inside sampling noise?
2. :func:`benjamini_hochberg` - across 13 models x 50 tickers x 3 regimes there
   are hundreds of comparisons. Some will clear p < 0.05 by luck. Control the
   false discovery rate before claiming anything.
3. :func:`deflated_sharpe_ratio` - the backtest analogue. A Sharpe ratio selected
   as the best of ``n_trials`` is biased upward; DSR discounts it.

Skipping step 2 is the single most common way a forecasting study overstates its
findings.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "diebold_mariano",
    "benjamini_hochberg",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
]


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    horizon: int = 1,
    harvey_correction: bool = True,
) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Parameters
    ----------
    loss_a, loss_b:
        Per-observation losses from the two models, aligned on the same
        timestamps. Squared error and absolute error are both admissible.
    horizon:
        Forecast horizon. Sets the truncation lag of the Newey-West style
        variance estimator, since ``h``-step errors are MA(h-1) by construction.
    harvey_correction:
        Apply the Harvey, Leybourne & Newbold (1997) small-sample adjustment.
        Leave on - daily folds are not large enough to ignore it.

    Returns
    -------
    (statistic, p_value)
        A negative statistic favours model A. Two-sided p-value.
    """
    loss_a = np.asarray(loss_a, dtype=float).ravel()
    loss_b = np.asarray(loss_b, dtype=float).ravel()
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss arrays must be aligned and equal length")
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    n = d.size
    if n < 10:
        return float("nan"), float("nan")

    d_bar = d.mean()
    gamma0 = np.mean((d - d_bar) ** 2)
    acc = gamma0
    for lag in range(1, horizon):
        gamma = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        acc += 2.0 * gamma
    if acc <= 0:
        return float("nan"), float("nan")

    dm = d_bar / np.sqrt(acc / n)

    if harvey_correction:
        adj = np.sqrt(
            (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n
        )
        dm *= adj
        p = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1))
    else:
        p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))

    return float(dm), float(p)


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg false discovery rate control.

    Returns
    -------
    (reject, q_values)
        ``reject`` is a boolean mask of hypotheses surviving FDR control at
        ``alpha``; ``q_values`` are the adjusted p-values, safe to report in a
        table alongside the raw ones.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    finite = np.isfinite(p)
    m = int(finite.sum())
    reject = np.zeros(p.shape, dtype=bool)
    q = np.full(p.shape, np.nan)
    if m == 0:
        return reject, q

    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[order] = adjusted
    reject[order] = adjusted <= alpha
    return reject, q


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio under the null of zero true skill.

    Uses the Gumbel approximation from Bailey & Lopez de Prado (2014). This is
    the hurdle a backtested Sharpe must clear before it means anything.
    """
    if n_trials < 2:
        return 0.0
    euler = 0.5772156649015329
    a = (1.0 - euler) * stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = euler * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * (a + b))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe for (a) the number of strategy variants tried,
    (b) non-normality of the return distribution, and (c) sample length. The
    returned probability is the confidence that true skill exceeds zero.

    Report this instead of a bare Sharpe ratio. A backtest that survives
    deflation is worth discussing; one that does not is a fitting artefact.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30:
        return {"sharpe": float("nan"), "dsr": float("nan"), "sr_hurdle": float("nan")}

    sd = r.std(ddof=1)
    if sd == 0:
        return {"sharpe": float("nan"), "dsr": float("nan"), "sr_hurdle": float("nan")}

    sr = r.mean() / sd
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    sr_var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1.0)
    hurdle = expected_max_sharpe(n_trials, sr_var)

    denom = np.sqrt(max(sr_var, 1e-12))
    dsr = float(stats.norm.cdf((sr - hurdle) / denom))

    ann = np.sqrt(periods_per_year)
    return {
        "sharpe": float(sr * ann),
        "sharpe_per_period": float(sr),
        "sr_hurdle": float(hurdle * ann),
        "dsr": dsr,
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": float(n),
        "n_trials": float(n_trials),
    }
