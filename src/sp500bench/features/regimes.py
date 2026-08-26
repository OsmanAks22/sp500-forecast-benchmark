"""Market regime labelling for stratified evaluation.

Regime labels exist to *slice results*, not to feed models. An aggregate RMSE
hides the case that matters: a model that is fine in calm markets and useless in
a crisis. Stratifying by regime surfaces it.

Two independent axes, deliberately kept separate rather than collapsed into one
label:

* **Volatility** - VIX terciles, thresholds fixed on the training period only.
* **Direction** - drawdown from the index's trailing 252-session peak.

Leakage warning
---------------
Tercile thresholds are computed on training data and then applied unchanged to
validation and test. Recomputing them over the full sample would let the test
period define its own boundaries - a subtle but real leak. ``fit_thresholds``
and ``apply_thresholds`` are split for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BEAR_DRAWDOWN = -0.20
CORRECTION_DRAWDOWN = -0.10


@dataclass(frozen=True)
class VolThresholds:
    low: float
    high: float


def fit_thresholds(vix_train: pd.Series) -> VolThresholds:
    """Fix the volatility tercile cuts on training data alone."""
    clean = vix_train.dropna()
    if clean.empty:
        raise ValueError("no training VIX observations")
    return VolThresholds(
        low=float(clean.quantile(1 / 3)),
        high=float(clean.quantile(2 / 3)),
    )


def apply_thresholds(vix: pd.Series, thresholds: VolThresholds) -> pd.Series:
    """Label each session low / medium / high volatility."""
    out = pd.Series(pd.NA, index=vix.index, dtype="object")
    out[vix <= thresholds.low] = "low_vol"
    out[(vix > thresholds.low) & (vix <= thresholds.high)] = "mid_vol"
    out[vix > thresholds.high] = "high_vol"
    return out


def drawdown_regime(index_close: pd.Series, window: int = 252) -> pd.DataFrame:
    """Trailing-peak drawdown and its discrete regime label.

    The rolling maximum is trailing, so the label at ``t`` never consults a peak
    that has not happened yet.
    """
    peak = index_close.rolling(window, min_periods=window).max()
    dd = index_close / peak - 1.0
    label = pd.Series("bull", index=index_close.index, dtype="object")
    label[dd <= CORRECTION_DRAWDOWN] = "correction"
    label[dd <= BEAR_DRAWDOWN] = "bear"
    label[dd.isna()] = pd.NA
    return pd.DataFrame({"drawdown": dd, "direction_regime": label})


def build_regime_panel(
    vix: pd.Series,
    index_close: pd.Series,
    train_end: str,
    vol_window: int = 252,
) -> pd.DataFrame:
    """Assemble the full regime table used by the reporting layer."""
    thresholds = fit_thresholds(vix.loc[:train_end])
    panel = drawdown_regime(index_close, window=vol_window)
    panel["vol_regime"] = apply_thresholds(vix.reindex(panel.index), thresholds)
    panel.attrs["vol_thresholds"] = {"low": thresholds.low, "high": thresholds.high}
    return panel
