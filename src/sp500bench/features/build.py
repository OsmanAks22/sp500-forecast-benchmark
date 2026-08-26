"""Causal feature construction.

Every feature in this module obeys one rule: the value stamped at session ``t``
uses information available at the close of ``t`` and nothing later. Rolling
windows are trailing, never centred. No interpolation runs backwards. The rule
is enforced by ``tests/test_leakage.py``, which perturbs future values and
asserts that no feature moves.

The target is the *forward* log return, which by construction does use future
data - that is what makes it a target. It is created by a separate function,
:func:`make_target`, so the distinction is visible in the call site rather than
hidden inside a generic pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["make_target", "build_features", "FEATURE_COLUMNS"]

FEATURE_COLUMNS: list[str] = [
    "ret_1d", "ret_5d", "ret_21d",
    "vol_21d", "vol_63d", "vol_ratio",
    "rsi_14", "dist_ma50", "dist_ma200",
    "volume_z", "hl_range", "overnight_gap",
    "dow", "month",
]


def make_target(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward log return over ``horizon`` sessions.

    ``target[t] = log(P[t + h] / P[t])``

    The last ``horizon`` observations become NaN, which is correct - their labels
    have not happened yet. Never fill them.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return np.log(prices.shift(-horizon) / prices)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the causal feature block for a single ticker.

    ``df`` must be sorted by date and carry columns
    ``open, high, low, close, volume``.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending before feature construction")

    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # Trailing returns.
    log_close = np.log(close)
    out["ret_1d"] = log_close.diff(1)
    out["ret_5d"] = log_close.diff(5)
    out["ret_21d"] = log_close.diff(21)

    # Realised volatility and its term structure.
    out["vol_21d"] = out["ret_1d"].rolling(21, min_periods=21).std()
    out["vol_63d"] = out["ret_1d"].rolling(63, min_periods=63).std()
    out["vol_ratio"] = out["vol_21d"] / out["vol_63d"]

    # Momentum and trend position.
    out["rsi_14"] = _rsi(close, 14)
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    out["dist_ma50"] = np.log(close / ma50)
    out["dist_ma200"] = np.log(close / ma200)

    # Activity and intraday structure.
    vol = df["volume"].replace(0.0, np.nan)
    log_vol = np.log(vol)
    vol_mean = log_vol.rolling(63, min_periods=63).mean()
    vol_std = log_vol.rolling(63, min_periods=63).std()
    out["volume_z"] = (log_vol - vol_mean) / vol_std
    out["hl_range"] = np.log(df["high"] / df["low"])
    out["overnight_gap"] = np.log(df["open"] / close.shift(1))

    # Calendar effects.
    out["dow"] = df.index.dayofweek
    out["month"] = df.index.month

    return out[FEATURE_COLUMNS]
