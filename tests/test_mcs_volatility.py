"""Tests for the Model Confidence Set, HAR family and robust inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500bench.evaluation.mcs import model_confidence_set, moving_block_bootstrap_indices
from sp500bench.evaluation.metrics import directional_accuracy_ci
from sp500bench.models.volatility import (HARModel, garman_klass, har_features,
                                          parkinson, realised_vol_close,
                                          rogers_satchell, yang_zhang)


# --------------------------------------------------------------------------
# Model Confidence Set
# --------------------------------------------------------------------------

def test_mcs_keeps_everything_when_models_are_equivalent():
    rng = np.random.default_rng(11)
    common = rng.chisquare(3, 800)
    losses = pd.DataFrame({f"m{i}": common + rng.normal(0, 0.05, 800) for i in range(5)})
    res = model_confidence_set(losses, alpha=0.10, n_boot=1500)
    assert res.size == 5, "indistinguishable models must all survive"


def test_mcs_eliminates_a_clearly_worse_model():
    rng = np.random.default_rng(12)
    base = rng.chisquare(3, 1200)
    losses = pd.DataFrame({
        "good_a": base + rng.normal(0, 0.1, 1200),
        "good_b": base + rng.normal(0, 0.1, 1200),
        "terrible": base + 4.0 + rng.normal(0, 0.1, 1200),
    })
    res = model_confidence_set(losses, alpha=0.10, n_boot=1500)
    assert "terrible" in res.excluded
    assert {"good_a", "good_b"} <= set(res.included)


def test_mcs_p_values_are_monotone_along_the_elimination_path():
    rng = np.random.default_rng(13)
    base = rng.chisquare(3, 1000)
    losses = pd.DataFrame({
        "best": base,
        "mid": base + 1.5 + rng.normal(0, 0.1, 1000),
        "worst": base + 5.0 + rng.normal(0, 0.1, 1000),
    })
    res = model_confidence_set(losses, alpha=0.10, n_boot=1500)
    assert res.p_values["worst"] <= res.p_values["mid"] + 1e-12


def test_mcs_survivor_count_shrinks_as_alpha_rises():
    rng = np.random.default_rng(14)
    base = rng.chisquare(3, 1500)
    losses = pd.DataFrame({
        f"m{i}": base + i * 0.35 + rng.normal(0, 0.4, 1500) for i in range(6)
    })
    lenient = model_confidence_set(losses, alpha=0.01, n_boot=1500).size
    strict = model_confidence_set(losses, alpha=0.25, n_boot=1500).size
    assert strict <= lenient


def test_block_bootstrap_preserves_shape_and_stays_in_range():
    rng = np.random.default_rng(15)
    idx = moving_block_bootstrap_indices(500, block_size=8, n_boot=100, rng=rng)
    assert idx.shape == (100, 500)
    assert idx.min() >= 0 and idx.max() < 500


def test_mcs_rejects_a_single_model():
    with pytest.raises(ValueError, match="at least two"):
        model_confidence_set(pd.DataFrame({"only": np.random.rand(200)}))


# --------------------------------------------------------------------------
# Volatility estimators
# --------------------------------------------------------------------------

@pytest.fixture
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(16)
    n = 600
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=dates)
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


@pytest.mark.parametrize("name", ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"])
def test_estimators_are_positive_and_finite(bars: pd.DataFrame, name: str):
    fns = {
        "parkinson": lambda: parkinson(bars["high"], bars["low"]),
        "garman_klass": lambda: garman_klass(bars["open"], bars["high"], bars["low"], bars["close"]),
        "rogers_satchell": lambda: rogers_satchell(bars["open"], bars["high"], bars["low"], bars["close"]),
        "yang_zhang": lambda: yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"]),
    }
    vol = fns[name]().dropna()
    assert len(vol) > 400
    assert (vol > 0).all(), f"{name} produced a non-positive volatility"
    assert np.isfinite(vol).all()


def test_range_estimators_are_more_efficient_than_close_to_close(bars: pd.DataFrame):
    """Range-based estimators should be visibly less noisy for the same window."""
    cc = realised_vol_close(bars["close"]).dropna()
    gk = garman_klass(bars["open"], bars["high"], bars["low"], bars["close"]).dropna()
    common = cc.index.intersection(gk.index)
    assert gk.loc[common].std() < cc.loc[common].std(), (
        "Garman-Klass should be smoother than close-to-close"
    )


def test_volatility_estimators_are_causal(bars: pd.DataFrame):
    cut = 400
    corrupted = bars.copy()
    corrupted.iloc[cut:] *= 2.0
    a = yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"]).iloc[:cut]
    b = yang_zhang(corrupted["open"], corrupted["high"], corrupted["low"], corrupted["close"]).iloc[:cut]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------
# HAR family
# --------------------------------------------------------------------------

def test_har_features_are_lagged(bars: pd.DataFrame):
    rv = realised_vol_close(bars["close"])
    feats = har_features(rv)
    aligned = feats["rv_d"].dropna()
    assert np.allclose(aligned.to_numpy(), rv.shift(1).loc[aligned.index].to_numpy())


def test_har_beats_a_constant_forecast_on_persistent_volatility(bars: pd.DataFrame):
    """Volatility clusters, so HAR should exploit persistence a constant cannot."""
    rv = yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"])
    X, y = har_features(rv), rv
    frame = X.join(y.rename("t")).dropna()
    split = int(len(frame) * 0.7)
    tr, te = frame.iloc[:split], frame.iloc[split:]

    har = HARModel().fit(tr[["rv_d", "rv_w", "rv_m"]], tr["t"])
    pred = har.predict(te[["rv_d", "rv_w", "rv_m"]])

    har_mse = np.mean((te["t"] - pred) ** 2)
    const_mse = np.mean((te["t"] - tr["t"].mean()) ** 2)
    assert har_mse < const_mse


def test_log_har_applies_the_smearing_correction(bars: pd.DataFrame):
    rv = yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"])
    frame = har_features(rv).join(rv.rename("t")).dropna()
    X, y = frame[["rv_d", "rv_w", "rv_m"]], frame["t"]

    corrected = HARModel(log_target=True, smearing=True).fit(X, y)
    naive = HARModel(log_target=True, smearing=False).fit(X, y)

    assert corrected.smearing_factor_ > 1.0, "exp(E[log X]) understates E[X]"
    assert naive.smearing_factor_ == 1.0
    assert (corrected.predict(X).dropna() >= naive.predict(X).dropna()).all()


def test_log_har_rejects_non_positive_volatility(bars: pd.DataFrame):
    rv = realised_vol_close(bars["close"])
    frame = har_features(rv).join(rv.rename("t")).dropna()
    frame.iloc[0, 0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        HARModel(log_target=True).fit(frame[["rv_d", "rv_w", "rv_m"]], frame["t"])


def test_har_quantiles_are_ordered(bars: pd.DataFrame):
    rv = yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"])
    frame = har_features(rv).join(rv.rename("t")).dropna()
    X, y = frame[["rv_d", "rv_w", "rv_m"]], frame["t"]
    model = HARModel(log_target=True).fit(X, y)
    q = model.quantiles(X, [0.1, 0.5, 0.9]).dropna()
    assert (q["q0.1"] <= q["q0.5"]).all() and (q["q0.5"] <= q["q0.9"]).all()


# --------------------------------------------------------------------------
# Cross-sectionally robust directional accuracy
# --------------------------------------------------------------------------

def _simulate_comoving_hits(n_days: int, n_tickers: int, factor_strength: float, seed: int):
    """Hits driven by a common daily market factor, as real stocks are."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 1, n_days)
    p = 0.5 + factor_strength * np.tanh(market)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n_tickers)
    draws = rng.random((n_days, n_tickers)) < p[:, None]
    return pd.Series(draws.ravel().astype(float)), pd.Series(dates), p


def test_clustered_se_matches_theoretical_variance_inflation():
    """Correlated stocks must inflate the standard error by roughly sqrt(1 + (m-1)rho).

    The naive binomial standard error treats 50 x 800 ticker-days as 40,000
    independent trials. They are not: a common market factor makes the 50 stocks
    on any given day close to one observation. The clustered bootstrap must
    recover the textbook design effect.
    """
    n_days, n_tickers = 800, 50
    hits, dates, p = _simulate_comoving_hits(n_days, n_tickers, 0.25, seed=17)

    rho = p.var() / ((p * (1 - p)).mean() + p.var())
    expected_vif = np.sqrt(1 + (n_tickers - 1) * rho)

    out = directional_accuracy_ci(hits, dates, n_boot=800)
    assert out["variance_inflation"] == pytest.approx(expected_vif, rel=0.30), (
        f"inflation {out['variance_inflation']:.2f}x vs theory {expected_vif:.2f}x"
    )
    assert out["variance_inflation"] > 2.0
    assert out["ci_low"] < out["estimate"] < out["ci_high"]
    assert out["n_days"] == n_days


def test_variance_inflation_grows_with_factor_strength():
    """Stronger comovement, larger design effect - as in a crisis regime."""
    weak, dates_w, _ = _simulate_comoving_hits(600, 50, 0.05, seed=21)
    strong, dates_s, _ = _simulate_comoving_hits(600, 50, 0.30, seed=21)

    vif_weak = directional_accuracy_ci(weak, dates_w, n_boot=600)["variance_inflation"]
    vif_strong = directional_accuracy_ci(strong, dates_s, n_boot=600)["variance_inflation"]
    assert vif_strong > vif_weak * 1.5


def test_independent_hits_show_almost_no_inflation():
    """Control case: without a common factor the correction should do nothing."""
    rng = np.random.default_rng(22)
    n_days, n_tickers = 600, 50
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n_tickers)
    hits = pd.Series((rng.random(n_days * n_tickers) < 0.52).astype(float))
    out = directional_accuracy_ci(hits, pd.Series(dates), n_boot=600)
    assert out["variance_inflation"] < 1.5, "no dependence, so no design effect"


def test_minimum_detectable_effect_is_reported():
    rng = np.random.default_rng(18)
    n_days, n_tickers = 600, 50
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n_tickers)
    hits = pd.Series(rng.random(n_days * n_tickers) < 0.51).astype(float)
    out = directional_accuracy_ci(hits, pd.Series(dates), n_boot=800)
    assert out["min_detectable_effect"] > 0
    assert 0.4 < out["estimate"] < 0.6
