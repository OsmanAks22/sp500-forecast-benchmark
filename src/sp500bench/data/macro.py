"""FRED macro series, with publication lags applied.

The lag is the whole point of this module. January CPI is published in mid
February. A model that reads the January value on a January date is reading the
future. Every series below carries an explicit release lag in calendar days,
applied before the series ever touches the feature matrix.

Daily market series (VIX, Treasury yields) are observable same-day and carry a
lag of one session only, to respect the close-to-close convention used for
returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MacroSeries:
    fred_id: str
    name: str
    release_lag_days: int
    description: str


MACRO_SERIES: list[MacroSeries] = [
    MacroSeries("VIXCLS",  "vix",         1,  "CBOE volatility index, daily close"),
    MacroSeries("DGS10",   "yield_10y",   1,  "10-year Treasury constant maturity"),
    MacroSeries("DGS2",    "yield_2y",    1,  "2-year Treasury constant maturity"),
    MacroSeries("T10Y2Y",  "term_spread", 1,  "10y minus 2y term spread"),
    MacroSeries("DFF",     "fed_funds",   1,  "Effective federal funds rate"),
    MacroSeries("BAMLH0A0HYM2", "hy_oas", 1,  "High-yield option-adjusted spread"),
    MacroSeries("CPIAUCSL","cpi",        45,  "CPI, all urban consumers (monthly)"),
    MacroSeries("UNRATE",  "unemployment",30, "Unemployment rate (monthly)"),
    MacroSeries("INDPRO",  "indpro",     45,  "Industrial production (monthly)"),
]


def apply_release_lag(df: pd.DataFrame, spec: MacroSeries) -> pd.DataFrame:
    """Shift a series forward by its publication lag.

    After this call the value stamped at date ``d`` is the most recent figure a
    market participant could actually have known on ``d``.
    """
    out = df.copy()
    out.index = out.index + pd.Timedelta(days=spec.release_lag_days)
    return out


def align_to_sessions(
    macro: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex lagged macro data onto the trading calendar.

    Forward fill is safe *here and only here*, because the lag has already been
    applied - filling forward propagates the last known value, never a future
    one. Filling before lagging would leak.
    """
    return macro.reindex(macro.index.union(sessions)).ffill().reindex(sessions)
