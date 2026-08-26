"""Model Confidence Set (Hansen, Lunde & Nason, 2011).

Why this exists
---------------
Pairwise Diebold-Mariano tests, even with Benjamini-Hochberg correction, assume
the individual comparisons are informative in isolation. They are not, here: 50
S&P 500 constituents share a common market factor, so per-ticker loss
differentials are heavily cross-sectionally dependent. Applying BH across them
is anti-conservative - it understates the joint error rate and manufactures
significance.

The MCS solves the right problem directly. Rather than asking "is model A better
than model B?" it asks "which models cannot be distinguished from the best one?"
and returns that set. Data snooping across many models is handled natively,
because the elimination sequence is bootstrapped jointly.

Interpretation
--------------
A large surviving set is a *finding*, not a failure. It says the data cannot
separate these models - which, on daily equity returns, is the expected outcome
and the honest one to report.

Algorithm (T-max variant)
-------------------------
1. Compute each model's loss deviation from the cross-model average.
2. Studentise using a moving-block bootstrap, which preserves serial dependence.
3. Test the null of equal predictive ability; if rejected, eliminate the worst
   model and repeat.
4. A model's MCS p-value is the running maximum of the p-values observed up to
   its elimination, which makes the sequence monotone.

Reference
---------
Hansen, P.R., Lunde, A. and Nason, J.M. (2011) 'The model confidence set',
*Econometrica*, 79(2), pp. 453-497.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["MCSResult", "model_confidence_set", "moving_block_bootstrap_indices"]


@dataclass
class MCSResult:
    """Outcome of the elimination procedure."""

    included: list[str]
    excluded: list[str]
    p_values: dict[str, float]
    alpha: float
    n_obs: int
    block_size: int

    @property
    def size(self) -> int:
        return len(self.included)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"model": m, "mcs_p_value": self.p_values[m], "in_mcs": m in self.included}
            for m in sorted(self.p_values, key=lambda k: -self.p_values[k])
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MCSResult(alpha={self.alpha}, surviving={self.size}, "
            f"eliminated={len(self.excluded)}, n_obs={self.n_obs})"
        )


def moving_block_bootstrap_indices(
    n_obs: int,
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw moving-block bootstrap index matrices of shape (n_boot, n_obs).

    Blocks of consecutive observations are resampled with replacement, which
    preserves short-range serial dependence. An i.i.d. bootstrap would destroy
    the volatility clustering that dominates daily return losses and would
    understate the variance badly.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    n_blocks = int(np.ceil(n_obs / block_size))
    max_start = max(n_obs - block_size, 0)
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n_obs] % n_obs


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    n_boot: int = 5_000,
    block_size: int | None = None,
    seed: int = 20260826,
) -> MCSResult:
    """Compute the Model Confidence Set at confidence level ``1 - alpha``.

    Parameters
    ----------
    losses:
        ``(T, m)`` frame of per-period losses, one column per model. Rows must be
        aligned in time across models - typically the daily cross-sectional mean
        squared error, so that each row is one trading day and the panel's
        cross-sectional dependence is already collapsed into the row.
    alpha:
        Significance level for elimination. The MCS literature conventionally
        uses 0.10 rather than 0.05; report whichever you use.
    n_boot:
        Bootstrap replications. 5,000 is adequate; below 1,000 the p-values are
        visibly granular.
    block_size:
        Moving-block length. Defaults to ``ceil(T ** (1/3))``, the standard rule
        of thumb for weakly dependent series.

    Returns
    -------
    MCSResult

    Notes
    -----
    Feed this *daily aggregated* losses, not the stacked ticker-day panel. Rows
    must be independent-ish draws; stacking tickers would treat 50 correlated
    observations as 50 independent ones, reintroducing the exact problem the MCS
    is here to solve.
    """
    if losses.shape[1] < 2:
        raise ValueError("need at least two models to compare")
    clean = losses.dropna(axis=0, how="any")
    n_obs, n_models = clean.shape
    if n_obs < 30:
        raise ValueError(f"only {n_obs} usable periods; MCS is unreliable below ~30")

    if block_size is None:
        block_size = max(1, int(np.ceil(n_obs ** (1 / 3))))

    rng = np.random.default_rng(seed)
    boot_idx = moving_block_bootstrap_indices(n_obs, block_size, n_boot, rng)

    alive = list(clean.columns)
    p_values: dict[str, float] = {}
    excluded: list[str] = []
    running_max = 0.0

    while len(alive) > 1:
        L = clean[alive].to_numpy()

        # Deviation of each model's loss from the cross-model average.
        d = L - L.mean(axis=1, keepdims=True)
        d_bar = d.mean(axis=0)

        boot_means = d[boot_idx, :].mean(axis=1)          # (n_boot, k)
        var = ((boot_means - d_bar) ** 2).mean(axis=0)
        se = np.sqrt(np.maximum(var, 1e-300))

        t_obs = d_bar / se
        t_boot = (boot_means - d_bar) / se

        stat = float(np.max(t_obs))
        stat_boot = np.max(t_boot, axis=1)
        p = float(np.mean(stat_boot >= stat))

        running_max = max(running_max, p)

        if p >= alpha:
            for m in alive:
                p_values[m] = running_max
            break

        worst = alive[int(np.argmax(t_obs))]
        p_values[worst] = running_max
        excluded.append(worst)
        alive.remove(worst)
    else:
        p_values[alive[0]] = 1.0

    return MCSResult(
        included=list(alive),
        excluded=excluded,
        p_values=p_values,
        alpha=alpha,
        n_obs=n_obs,
        block_size=block_size,
    )
