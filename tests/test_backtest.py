"""Transaction-cost accounting in the backtest.

The unit matters more than the arithmetic here. ``cost_bps`` is a *round-trip*
figure everywhere it is written down - ``configs/eval.yaml``, the README, the
H1 falsification criterion - while the simulator charges per leg. These tests
pin the factor of two between the two conventions in both directions: what a
known number of round trips actually costs, and whether the reported break-even
is quoted in the unit it was charged in.

The H1 criterion is "break-even above 5 bps round trip". A per-leg break-even
reported as though it were round-trip is off by exactly the factor that decides
the hypothesis, which is why this file exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500bench.backtest.strategy import positions_from_forecast, run_backtest

PERIODS = 252


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _round_trip_positions(n: int, n_round_trips: int) -> pd.Series:
    """Flat, then in for one session, then flat - repeated ``n_round_trips`` times.

    Each isolated holding contributes exactly two legs, one entry and one exit,
    so the series contains ``n_round_trips`` complete round trips and nothing
    else. It starts and ends flat, so no leg is left dangling at either edge.
    """
    dates = _dates(n)
    pattern = np.zeros(n)
    pattern[1 : 2 * n_round_trips : 2] = 1.0
    assert pattern[-1] == 0.0, "the series must close flat"
    return pd.Series(pattern, index=dates)


def _total_cost_charged(result: dict, n_periods: int) -> float:
    """Recover total cost from the annualised gross and net figures."""
    return (result["gross_return_ann"] - result["net_return_ann"]) / PERIODS * n_periods


# --------------------------------------------------------------------------
# Round-trip cost accounting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n_round_trips", [1, 7, 25])
@pytest.mark.parametrize("cost_bps", [2.0, 5.0, 10.0])
def test_n_round_trips_cost_exactly_n_times_the_round_trip_charge(
    n_round_trips: int, cost_bps: float
) -> None:
    """N round trips at c bps round-trip must cost N*c bps, not 2*N*c."""
    n = 200
    rng = np.random.default_rng(41)
    returns = pd.Series(rng.normal(0.0, 0.01, n), index=_dates(n))
    positions = _round_trip_positions(n, n_round_trips)

    result = run_backtest(positions, returns, cost_bps=cost_bps)

    expected = n_round_trips * cost_bps / 10_000.0
    assert _total_cost_charged(result, n) == pytest.approx(expected, rel=1e-9), (
        f"{n_round_trips} round trips at {cost_bps} bps round-trip should cost "
        f"{expected:.6f}, charged per leg it would cost {2 * expected:.6f}"
    )


def test_cost_scales_linearly_with_the_quoted_rate() -> None:
    n = 200
    returns = pd.Series(0.0, index=_dates(n))
    positions = _round_trip_positions(n, 10)

    cheap = _total_cost_charged(run_backtest(positions, returns, cost_bps=2.0), n)
    dear = _total_cost_charged(run_backtest(positions, returns, cost_bps=10.0), n)
    assert dear == pytest.approx(5.0 * cheap, rel=1e-9)


def test_zero_turnover_incurs_zero_cost() -> None:
    """A strategy that never trades pays nothing, whatever the cost assumption."""
    n = 300
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0003, 0.01, n), index=_dates(n))
    flat = pd.Series(0.0, index=_dates(n))

    result = run_backtest(flat, returns, cost_bps=10.0)

    assert result["turnover_ann"] == 0.0
    assert _total_cost_charged(result, n) == pytest.approx(0.0, abs=1e-15)
    assert result["net_return_ann"] == pytest.approx(result["gross_return_ann"])
    assert result["break_even_round_trip_bps"] == float("inf")


def test_a_held_position_pays_only_for_the_legs_it_trades() -> None:
    """Holding costs nothing; only the entry and exit are charged."""
    n = 250
    returns = pd.Series(0.0, index=_dates(n))
    held = pd.Series(0.0, index=_dates(n))
    held.iloc[10:200] = 1.0                       # one entry, one exit

    result = run_backtest(held, returns, cost_bps=8.0)
    assert _total_cost_charged(result, n) == pytest.approx(8.0 / 10_000.0, rel=1e-9)


# --------------------------------------------------------------------------
# Break-even is reported in the unit it was charged in
# --------------------------------------------------------------------------

def _profitable_signal(n: int = 1500, seed: int = 31):
    """A forecast with a small genuine edge, so break-even is finite and positive."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    returns = pd.Series(rng.normal(0.0002, 0.01, n), index=dates)
    peek = returns.shift(-1).fillna(0.0).to_numpy()
    forecast = pd.Series(peek + rng.normal(0.0, 0.02, n), index=dates)
    return forecast, returns


def test_break_even_is_the_round_trip_cost_that_zeroes_the_net_edge() -> None:
    """Charge exactly the reported break-even and the net edge must vanish.

    This is the test that ties the two conventions together. If break-even were
    reported per leg while costs were charged round-trip (or vice versa), the
    net return at the break-even rate would come out at plus or minus half the
    gross edge instead of zero.
    """
    forecast, returns = _profitable_signal()
    positions = positions_from_forecast(forecast)

    gross_run = run_backtest(positions, returns, cost_bps=0.0)
    break_even = gross_run["break_even_round_trip_bps"]
    assert 0.0 < break_even < float("inf"), "fixture must have a real edge"

    at_break_even = run_backtest(positions, returns, cost_bps=break_even)
    assert at_break_even["net_return_ann"] == pytest.approx(0.0, abs=1e-12)

    # And the sign flips either side of it, so the root is genuinely crossed.
    assert run_backtest(positions, returns, cost_bps=break_even * 0.5)[
        "net_return_ann"
    ] > 0
    assert run_backtest(positions, returns, cost_bps=break_even * 1.5)[
        "net_return_ann"
    ] < 0


def test_output_states_its_cost_convention() -> None:
    forecast, returns = _profitable_signal(400)
    result = run_backtest(positions_from_forecast(forecast), returns, cost_bps=5.0)

    assert result["cost_convention"] == "round_trip"
    assert "break_even_round_trip_bps" in result
    assert "break_even_cost_bps" not in result, "the ambiguous key must be gone"
    assert result["cost_bps_assumed"] == 5.0


# --------------------------------------------------------------------------
# Position construction and signal timing
# --------------------------------------------------------------------------

def test_positions_are_applied_to_the_next_period_return() -> None:
    """The signal formed at the close of t earns r[t+1], never r[t]."""
    dates = _dates(5)
    returns = pd.Series([0.0, 0.10, 0.0, 0.0, 0.0], index=dates)
    positions = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=dates)

    result = run_backtest(positions, returns, cost_bps=0.0)
    # Position held at t0 collects the day-1 return; nothing else trades.
    assert result["gross_return_ann"] == pytest.approx(0.10 / len(dates) * PERIODS)


def test_long_flat_never_goes_short() -> None:
    forecast = pd.Series([-1.0, 0.0, 1.0, -0.5])
    assert positions_from_forecast(forecast, "long_flat").tolist() == [0.0, 0.0, 1.0, 0.0]


def test_long_short_takes_both_sides() -> None:
    forecast = pd.Series([-1.0, 0.0, 1.0, -0.5])
    assert positions_from_forecast(forecast, "long_short").tolist() == [-1.0, 0.0, 1.0, -1.0]


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        positions_from_forecast(pd.Series([1.0]), "market_neutral")
