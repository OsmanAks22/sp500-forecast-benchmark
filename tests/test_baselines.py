"""The H1 reference has to be the random walk, and has to stay that way.

Regression tests for a benchmark bug. What the registry called ``Naive`` was
``statsforecast``'s Naive applied to a return series - ``y[t+1] = y[t]`` - which
is not the random walk on prices. The random walk on prices forecasts *zero*
return. The two differ by a factor of ``sqrt(2(1-rho))`` in RMSE, about 1.45 on
this panel, because predicting the previous return differences an
already-differenced series a second time.

The consequence was not cosmetic. Scored against the wrong benchmark, AutoARIMA
appeared to cut RMSE by 31%; against the zero forecast it is marginally worse.
H1 is written against the random walk, so the whole primary result turned on
which model wore that name.

The tests below pin the identity that makes ``ZeroReturn`` checkable from
outside: its RMSE *is* the root mean square of the outcomes, exactly, because
its forecast is zero. Nothing about the implementation can drift without one of
these failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500bench.models.baselines import zero_return_forecast_fold
from sp500bench.models.registry import BY_NAME
from sp500bench.evaluation.metrics import rmse

GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two tickers of iid returns, so the population quantities are known."""
    rng = np.random.default_rng(20260902)
    dates = pd.bdate_range("2015-01-01", periods=1_500)
    frames = []
    for ticker, sigma in (("AAA", 0.02), ("BBB", 0.01)):
        frames.append(pd.DataFrame({
            "unique_id": ticker,
            "ds": dates,
            "y": rng.normal(0.0, sigma, len(dates)),
        }))
    return pd.concat(frames, ignore_index=True)


def _split(panel: pd.DataFrame):
    cut = panel["ds"].iloc[len(panel) // 4]
    train = panel[panel["ds"] <= cut]
    test = panel[panel["ds"] > cut]
    return train, test["ds"].min(), test["ds"].max()


def test_zero_return_rmse_equals_the_sample_std_of_test_returns(panel):
    """The identity that defines the benchmark.

    A constant-zero forecast has error equal to the outcome, so its RMSE is
    ``sqrt(mean(y**2))`` - the sample standard deviation of the test returns
    about zero. If this ever stops holding, the model has acquired a non-zero
    point forecast and is no longer the random walk.
    """
    train, start, end = _split(panel)
    forecasts = zero_return_forecast_fold(
        panel=panel, train=train, test_start=start, test_end=end, quantiles=GRID)

    scored = forecasts.merge(panel, on=["unique_id", "ds"], validate="one_to_one")
    observed = rmse(scored["y"].to_numpy(), scored["q0.5"].to_numpy())
    expected = float(np.sqrt(np.mean(scored["y"].to_numpy() ** 2)))

    assert observed == pytest.approx(expected, rel=1e-3), (
        f"ZeroReturn RMSE {observed:.6f} != sample std {expected:.6f}; "
        "the point forecast is no longer zero"
    )


def test_zero_return_point_forecast_is_negligible_against_the_spread(panel):
    """q0.5 is the empirical median of training returns, which sits near zero."""
    train, start, end = _split(panel)
    forecasts = zero_return_forecast_fold(
        panel=panel, train=train, test_start=start, test_end=end, quantiles=GRID)

    for ticker, group in forecasts.groupby("unique_id"):
        spread = float(group["q0.9"].iloc[0] - group["q0.1"].iloc[0])
        assert abs(float(group["q0.5"].iloc[0])) < 0.05 * spread, (
            f"{ticker}: median forecast is not negligible against the band"
        )


def test_zero_return_bands_come_from_the_training_window_only(panel):
    """Corrupting the test period must not move a single quantile."""
    train, start, end = _split(panel)
    baseline = zero_return_forecast_fold(
        panel=panel, train=train, test_start=start, test_end=end, quantiles=GRID)

    corrupted = panel.copy()
    future = corrupted["ds"] > train["ds"].max()
    corrupted.loc[future, "y"] = corrupted.loc[future, "y"] * 1_000.0 + 7.0
    after = zero_return_forecast_fold(
        panel=corrupted, train=train, test_start=start, test_end=end, quantiles=GRID)

    columns = [c for c in baseline.columns if c.startswith("q")]
    pd.testing.assert_frame_equal(baseline[columns], after[columns], check_exact=True)


def test_zero_return_quantiles_are_ordered_and_ticker_specific(panel):
    train, start, end = _split(panel)
    forecasts = zero_return_forecast_fold(
        panel=panel, train=train, test_start=start, test_end=end, quantiles=GRID)

    columns = [f"q{q:g}" for q in GRID]
    values = forecasts[columns].to_numpy()
    assert np.all(np.diff(values, axis=1) >= -1e-15), "quantiles must be monotone"

    widths = forecasts.groupby("unique_id").apply(
        lambda g: g["q0.9"].iloc[0] - g["q0.1"].iloc[0], include_groups=False)
    assert widths["AAA"] > 1.5 * widths["BBB"], (
        "bands must be fitted per ticker; AAA has twice BBB's volatility"
    )


def test_the_registry_names_zero_return_as_the_h1_reference():
    """LastReturn must not be marked as coming from the proposal's benchmark."""
    assert BY_NAME["ZeroReturn"].in_proposal, "ZeroReturn is the proposal's benchmark"
    assert BY_NAME["ZeroReturn"].tier == "baseline"
    assert not BY_NAME["LastReturn"].in_proposal, (
        "LastReturn is an added diagnostic, not the H1 reference"
    )
    assert "Naive" not in BY_NAME, (
        "the ambiguous name must not come back; it is what caused the bug"
    )
