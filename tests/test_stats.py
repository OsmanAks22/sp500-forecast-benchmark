"""Numerical checks on the metrics and significance machinery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500bench.evaluation.metrics import (crps_from_quantiles, directional_accuracy,
                                           directional_accuracy_with_ci, directional_hits,
                                           mase, picp, pinball_loss, rmse)
from sp500bench.evaluation.stat_tests import (benjamini_hochberg, deflated_sharpe_ratio,
                                              diebold_mariano, expected_max_sharpe)


def test_dm_finds_no_difference_between_identical_models():
    rng = np.random.default_rng(1)
    loss = rng.chisquare(3, 500)
    stat, p = diebold_mariano(loss, loss.copy())
    assert np.isnan(stat) or p > 0.99


def test_dm_detects_a_genuinely_better_model():
    rng = np.random.default_rng(2)
    good = rng.chisquare(3, 1000)
    bad = good + rng.chisquare(2, 1000)
    stat, p = diebold_mariano(good, bad)
    assert stat < 0, "negative statistic should favour the first model"
    assert p < 0.01


def test_dm_size_is_close_to_nominal_under_the_null():
    """Repeated tests on equal-accuracy models should reject about 5% of the time."""
    rng = np.random.default_rng(3)
    rejections = 0
    trials = 400
    for _ in range(trials):
        a = rng.normal(0, 1, 250) ** 2
        b = rng.normal(0, 1, 250) ** 2
        _, p = diebold_mariano(a, b)
        if np.isfinite(p) and p < 0.05:
            rejections += 1
    rate = rejections / trials
    assert 0.02 < rate < 0.09, f"empirical size {rate:.3f} far from nominal 0.05"


def test_bh_is_never_more_liberal_than_raw_p_values():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.7])
    reject, q = benjamini_hochberg(p, alpha=0.05)
    assert np.all(q >= p - 1e-12), "adjusted p-values must not shrink"
    assert reject.sum() <= (p < 0.05).sum(), "FDR control cannot add discoveries"


def test_bh_controls_fdr_on_pure_noise():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 500)
    reject, _ = benjamini_hochberg(p, alpha=0.05)
    assert reject.sum() <= 5, f"{reject.sum()} false discoveries from pure noise"


def test_sharpe_hurdle_rises_with_the_number_of_trials():
    var = 0.002
    hurdles = [expected_max_sharpe(n, var) for n in (2, 10, 100, 1000)]
    assert hurdles == sorted(hurdles)
    assert hurdles[-1] > hurdles[0]


def test_deflated_sharpe_punishes_data_mining():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0004, 0.01, 1200)
    honest = deflated_sharpe_ratio(returns, n_trials=1)
    mined = deflated_sharpe_ratio(returns, n_trials=5000)
    assert honest["sharpe"] == pytest.approx(mined["sharpe"])
    assert mined["dsr"] < honest["dsr"], "trying 5000 variants must lower confidence"
    assert mined["sr_hurdle"] > honest["sr_hurdle"]


def test_crps_rewards_a_sharper_correct_forecast():
    rng = np.random.default_rng(6)
    y = rng.normal(0, 1, 2000)
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    from scipy import stats as sstats
    tight = {q: np.full(2000, sstats.norm.ppf(q, 0, 1)) for q in levels}
    loose = {q: np.full(2000, sstats.norm.ppf(q, 0, 4)) for q in levels}
    assert crps_from_quantiles(y, tight) < crps_from_quantiles(y, loose)


def test_picp_matches_nominal_for_a_calibrated_interval():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, 20000)
    from scipy import stats as sstats
    lo = np.full(20000, sstats.norm.ppf(0.05))
    hi = np.full(20000, sstats.norm.ppf(0.95))
    assert picp(y, lo, hi) == pytest.approx(0.90, abs=0.01)


def test_pinball_is_minimised_at_the_true_quantile():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, 50000)
    from scipy import stats as sstats
    true_q90 = sstats.norm.ppf(0.9)
    at_truth = pinball_loss(y, np.full(50000, true_q90), 0.9)
    for offset in (-0.5, -0.2, 0.2, 0.5):
        assert pinball_loss(y, np.full(50000, true_q90 + offset), 0.9) > at_truth


def test_mase_equals_one_for_the_naive_forecast():
    rng = np.random.default_rng(9)
    series = np.cumsum(rng.normal(0, 1, 600))
    train, test = series[:400], series[400:]
    naive_pred = np.concatenate([[train[-1]], test[:-1]])
    assert mase(test, naive_pred, train) == pytest.approx(1.0, rel=0.25)


def test_directional_accuracy_ignores_flat_days():
    y = np.array([0.01, -0.01, 0.0, 0.02])
    pred = np.array([0.5, -0.5, 99.0, 0.5])
    assert directional_accuracy(y, pred) == pytest.approx(1.0)


def test_rmse_is_zero_for_a_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


# ---------------------------------------------------------------------------
# The zero-outcome convention has to be applied in exactly one place
# ---------------------------------------------------------------------------

def test_point_estimate_and_ci_agree_when_outcomes_contain_exact_zeros():
    """The headline number and its interval must come from the same hit vector.

    Regression test. ``directional_accuracy`` drops flat days; a hand-built hit
    vector passed to ``directional_accuracy_ci`` scored them as misses, because
    ``np.sign(0) == np.sign(x)`` is false. The two disagreed by 0.17pp on the
    Naive baseline - small, but it is the difference between a statistic and two
    statistics wearing the same name.
    """
    rng = np.random.default_rng(20260901)
    n = 6_000
    y = rng.normal(0, 0.02, n)
    y[rng.choice(n, 400, replace=False)] = 0.0          # flat days, ~7%
    pred = rng.normal(0, 0.02, n)
    dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=n // 10), 10))

    point = directional_accuracy(y, pred)
    with_ci = directional_accuracy_with_ci(y, pred, dates, n_boot=400)

    assert with_ci["estimate"] == point, (
        f"CI estimate {with_ci['estimate']!r} != point estimate {point!r}"
    )
    assert with_ci["n_obs"] == n - 400, "flat days must be dropped, not scored"
    assert with_ci["ci_low"] <= point <= with_ci["ci_high"]


def test_directional_hits_drops_zeros_and_keeps_dates_aligned():
    y = np.array([0.01, 0.0, -0.02, 0.03, 0.0])
    pred = np.array([0.5, 0.5, 0.5, -0.5, -0.5])
    dates = pd.Series(pd.date_range("2020-01-01", periods=5))

    hits, kept = directional_hits(y, pred, dates)
    assert hits.tolist() == [1.0, 0.0, 0.0], "only the three non-zero outcomes score"
    assert list(kept) == list(dates[[0, 2, 3]]), "dates must be filtered by the same mask"

    bare, none_dates = directional_hits(y, pred)
    assert none_dates is None and bare.tolist() == hits.tolist()


def test_directional_hits_rejects_misaligned_dates():
    y = np.array([0.01, -0.02, 0.03])
    pred = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        directional_hits(y, pred, pd.Series(pd.date_range("2020-01-01", periods=2)))


def test_a_forecast_that_is_always_flat_scores_no_hits():
    """sign(0) matches nothing, so a constant-zero forecast cannot score."""
    y = np.array([0.01, -0.02, 0.03, -0.01])
    assert directional_accuracy(y, np.zeros(4)) == 0.0
