"""Adversarial leakage tests.

These are not coverage tests. Each one constructs a situation in which a leak
*would* show up and asserts that it does not. The pattern throughout: corrupt
data that lies in the future relative to some reference point, recompute, and
require the output to be bit-identical.

If a reviewer reads one file in this repository, it should be this one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500bench.data.macro import MacroSeries, align_to_sessions, apply_release_lag
from sp500bench.evaluation.splits import PurgedWalkForward, assert_no_overlap
from sp500bench.features.build import FEATURE_COLUMNS, build_features, make_target
from sp500bench.features.regimes import apply_thresholds, drawdown_regime, fit_thresholds


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(20260826)
    dates = pd.bdate_range("2015-01-01", periods=900)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, len(dates))),
            "high": close * (1 + np.abs(rng.normal(0, 0.005, len(dates)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, len(dates)))),
            "close": close,
            "volume": rng.lognormal(15, 0.4, len(dates)),
        },
        index=dates,
    )


# --------------------------------------------------------------------------
# Features must not react to the future
# --------------------------------------------------------------------------

def test_features_are_causal(ohlcv: pd.DataFrame) -> None:
    """Corrupt the last third of the sample; earlier features must not move."""
    cut = 600
    baseline = build_features(ohlcv)

    corrupted = ohlcv.copy()
    corrupted.iloc[cut:] *= 3.7
    perturbed = build_features(corrupted)

    pd.testing.assert_frame_equal(
        baseline.iloc[:cut],
        perturbed.iloc[:cut],
        check_exact=False,
        rtol=1e-12,
        obj="features before the corruption point",
    )


@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_each_feature_individually_causal(ohlcv: pd.DataFrame, column: str) -> None:
    """Per-column version, so a failure names the offending feature."""
    cut = 500
    corrupted = ohlcv.copy()
    corrupted.iloc[cut:] *= 2.5

    a = build_features(ohlcv)[column].iloc[:cut]
    b = build_features(corrupted)[column].iloc[:cut]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), (
        f"feature '{column}' changed when only future data was altered"
    )


def test_target_is_forward_looking_by_design(ohlcv: pd.DataFrame) -> None:
    """The target *must* see the future - confirm it actually does."""
    target = make_target(ohlcv["close"], horizon=1)
    expected = np.log(ohlcv["close"].iloc[1] / ohlcv["close"].iloc[0])
    assert np.isclose(target.iloc[0], expected)
    assert target.iloc[-1:].isna().all(), "final labels must stay NaN, never filled"


def test_target_horizon_shifts_correctly(ohlcv: pd.DataFrame) -> None:
    t5 = make_target(ohlcv["close"], horizon=5)
    assert t5.iloc[-5:].isna().all()
    assert np.isclose(t5.iloc[0], np.log(ohlcv["close"].iloc[5] / ohlcv["close"].iloc[0]))


# --------------------------------------------------------------------------
# Purge and embargo
# --------------------------------------------------------------------------

def test_purge_and_embargo_open_a_real_gap() -> None:
    calendar = pd.bdate_range("2005-01-03", "2026-08-01")
    cv = PurgedWalkForward(
        test_windows=[("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31")],
        horizon=1,
        embargo=5,
        train_start="2005-01-03",
    )
    for split in cv.splits(calendar):
        gap_sessions = calendar.searchsorted(split.test_start) - calendar.searchsorted(
            split.train_end
        )
        assert gap_sessions >= cv.gap, (
            f"fold {split.fold}: gap of {gap_sessions} sessions, {cv.gap} required"
        )


def test_train_and_test_indices_never_intersect() -> None:
    panel = pd.DataFrame(
        {
            "unique_id": np.repeat(["AAPL", "MSFT"], 1500),
            "ds": np.tile(pd.bdate_range("2015-01-01", periods=1500), 2),
            "y": np.random.default_rng(0).normal(0, 0.01, 3000),
        }
    )
    cv = PurgedWalkForward(
        test_windows=[("2020-01-01", "2020-06-30")],
        horizon=1,
        embargo=5,
        train_start="2015-01-01",
    )
    for split, train, test in cv.iter_frames(panel):
        assert set(train["ds"]).isdisjoint(set(test["ds"]))
        assert_no_overlap(train["ds"], test["ds"], horizon=1, embargo=5)


def test_overlap_assertion_actually_fires() -> None:
    """A guard that never fails is not a guard."""
    train = pd.Series(pd.bdate_range("2019-01-01", "2019-12-31"))
    test = pd.Series(pd.bdate_range("2020-01-01", "2020-06-30"))
    with pytest.raises(AssertionError):
        assert_no_overlap(train, test, horizon=1, embargo=90)


def test_overlapping_test_windows_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PurgedWalkForward(
            test_windows=[("2020-01-01", "2020-12-31"), ("2020-06-01", "2021-12-31")],
            horizon=1,
            embargo=5,
            train_start="2005-01-03",
        )


# --------------------------------------------------------------------------
# Macro publication lag
# --------------------------------------------------------------------------

def test_monthly_macro_is_lagged_past_its_release() -> None:
    spec = MacroSeries("CPIAUCSL", "cpi", 45, "CPI")
    raw = pd.DataFrame(
        {"cpi": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    )
    lagged = apply_release_lag(raw, spec)
    assert lagged.index[0] == pd.Timestamp("2020-02-15"), "January CPI must land mid-February"

    sessions = pd.bdate_range("2020-01-01", "2020-04-30")
    aligned = align_to_sessions(lagged, sessions)
    january_window = aligned.loc["2020-01-01":"2020-02-14", "cpi"]
    assert january_window.isna().all(), "January CPI leaked into January sessions"


def test_forward_fill_never_moves_backwards() -> None:
    spec = MacroSeries("UNRATE", "unemployment", 30, "Unemployment")
    raw = pd.DataFrame(
        {"unemployment": [3.5, 14.7]},
        index=pd.to_datetime(["2020-02-01", "2020-04-01"]),
    )
    sessions = pd.bdate_range("2020-03-01", "2020-06-01")
    aligned = align_to_sessions(apply_release_lag(raw, spec), sessions)
    pre_release = aligned.loc["2020-03-01":"2020-04-30", "unemployment"].dropna()
    assert (pre_release == 3.5).all(), "the April spike leaked into March/April sessions"


# --------------------------------------------------------------------------
# Regime labels
# --------------------------------------------------------------------------

def test_vol_thresholds_fitted_on_train_only() -> None:
    dates = pd.bdate_range("2015-01-01", periods=1500)
    vix = pd.Series(np.linspace(10, 30, 1500), index=dates)

    thresholds = fit_thresholds(vix.loc[:"2019-12-31"])
    calm = apply_thresholds(vix, thresholds)

    spiked = vix.copy()
    spiked.loc["2020-01-01":] = 80.0
    thresholds_again = fit_thresholds(spiked.loc[:"2019-12-31"])
    assert thresholds == thresholds_again, "test-period volatility moved the thresholds"

    stressed = apply_thresholds(spiked, thresholds_again)
    pd.testing.assert_series_equal(
        calm.loc[:"2019-12-31"], stressed.loc[:"2019-12-31"], obj="training-period labels"
    )


def test_drawdown_uses_trailing_peak_only() -> None:
    dates = pd.bdate_range("2018-01-01", periods=800)
    close = pd.Series(np.linspace(100, 150, 800), index=dates)

    base = drawdown_regime(close)
    spiked = close.copy()
    spiked.iloc[700:] = 400.0
    after = drawdown_regime(spiked)

    pd.testing.assert_series_equal(
        base["drawdown"].iloc[:700],
        after["drawdown"].iloc[:700],
        obj="drawdown before the future spike",
    )
