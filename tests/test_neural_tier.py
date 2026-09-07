"""Tier 2 - the seven supervised deep-learning models.

None of these tests needs neuralforecast, torch or a GPU. What they cover is
everything that can be wrong before a single training step runs: which models
are registered, what grid reaches the loss, how the library's output is mapped
back onto the frozen columns, and whether the predictive distribution that comes
out is the right width.

The last one matters most. Every silent failure in this project produced
plausible numbers, and a model trained on the wrong quantile grid produces
exactly that - present columns, monotone ordering, finite values, wrong width.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500bench.config import load_configs, quantile_column
from sp500bench.evaluation.scale import (SIGMA_BANDS, SigmaOutOfBand,
                                         check_sigma_band, implied_daily_sigma)
from sp500bench.models import neural
from sp500bench.models.registry import REGISTRY, runnable

ROOT = Path(__file__).resolve().parents[1]
FROZEN = list(load_configs().quantiles)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_experiment_under_test", ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_runner()


# --------------------------------------------------------------------------
# 1. the tier exists at all
# --------------------------------------------------------------------------

def test_every_neuralforecast_spec_has_an_executor():
    """Tier 2 was skipped for two months with 'no executor implemented yet'."""
    specs = [m.name for m in REGISTRY if m.library == "neuralforecast"]
    assert len(specs) == 7
    for name in specs:
        assert name in experiment.EXECUTORS, f"{name} has no executor"


def test_every_runnable_spec_now_has_an_executor():
    """Nineteen runnable models, nineteen executors - the scope is not narrowed."""
    missing = [m.name for m in runnable() if m.name not in experiment.EXECUTORS
               and m.library not in ("chronos", "timesfm", "granite-tsfm", "kronos")]
    assert missing == [], f"no executor for {missing}"


def test_the_neural_executors_are_distinct_per_model():
    executors = neural.executors()
    assert len(executors) == 7
    assert len({id(fn) for fn in executors.values()}) == 7
    assert executors["PatchTST"].__name__ == "patchtst_forecast_fold"


# --------------------------------------------------------------------------
# 2. the grid that reaches MQLoss
# --------------------------------------------------------------------------

def test_the_frozen_grid_is_accepted():
    assert neural.assert_training_grid(FROZEN, FROZEN) == FROZEN


def test_a_subset_of_the_grid_is_rejected():
    """MQLoss emits only what it was trained on; the rest arrive as NaN."""
    with pytest.raises(ValueError, match="scored but not trained"):
        neural.assert_training_grid([0.1, 0.5, 0.9], FROZEN)


def test_a_superset_of_the_grid_is_rejected():
    with pytest.raises(ValueError, match="trained but not scored"):
        neural.assert_training_grid(FROZEN + [0.999], FROZEN)


def test_the_same_levels_in_a_different_order_are_rejected():
    """MQLoss emits them in the order given, so order is part of the contract."""
    with pytest.raises(ValueError, match="different order"):
        neural.assert_training_grid(list(reversed(FROZEN)), FROZEN)


def test_the_error_names_both_grids():
    with pytest.raises(ValueError) as excinfo:
        neural.assert_training_grid([0.5], FROZEN)
    message = str(excinfo.value)
    assert "passed to the loss" in message and "configs/eval.yaml" in message


# --------------------------------------------------------------------------
# 3. mapping the library's output back onto the frozen columns
# --------------------------------------------------------------------------

def test_a_level_is_located_across_output_spellings():
    for column in ("DLinear-ql0.05", "DLinear-q-0.05", "DLinear-q0.05"):
        assert neural._find_level_column([ "unique_id", "ds", column],
                                         "DLinear", 0.05) == column


def test_a_missing_level_raises_rather_than_producing_a_nan_column():
    """The whole point of the grid gate: a gap must not reach the parquet."""
    with pytest.raises(KeyError, match="no output column for quantile level"):
        neural._find_level_column(["unique_id", "ds", "DLinear-ql0.5"],
                                  "DLinear", 0.01)


def test_the_output_is_mapped_onto_the_frozen_column_names():
    prediction = pd.DataFrame({"unique_id": ["AAA", "BBB"]})
    for level in FROZEN:
        prediction[f"DLinear-ql{level:g}"] = [level, level * 2]
    out = neural._to_grid(prediction, "DLinear", pd.Timestamp("2020-03-02"),
                          FROZEN, [quantile_column(q) for q in FROZEN])
    assert list(out.columns) == ["unique_id", "ds"] + [quantile_column(q) for q in FROZEN]
    assert out["ds"].nunique() == 1
    assert out[quantile_column(0.5)].tolist() == [0.5, 1.0]


def test_a_forecast_window_that_misses_sessions_is_rejected():
    """A misaligned roll would score each forecast against the wrong outcome."""
    origins = pd.DatetimeIndex(["2020-03-02", "2020-03-03", "2020-03-04"])
    out = pd.DataFrame({"unique_id": ["AAA"] * 2,
                        "ds": pd.DatetimeIndex(["2020-03-02", "2020-03-03"])})
    with pytest.raises(AssertionError, match="do not match the test window"):
        neural._assert_origins_match(out, origins, "DLinear")

    complete = pd.DataFrame({"unique_id": ["AAA"] * 3, "ds": origins})
    neural._assert_origins_match(complete, origins, "DLinear")


# --------------------------------------------------------------------------
# 4. the runtime: device, workers and the dataloader trap
# --------------------------------------------------------------------------

def test_dataloader_settings_stay_out_of_the_trainer_kwargs():
    """num_workers_loader in trainer_kwargs cost a Kaggle session."""
    runtime = neural.NeuralRuntime(device="cuda", num_workers=4)
    trainer = runtime.trainer_kwargs()
    assert "num_workers" not in trainer
    assert "num_workers_loader" not in trainer
    assert trainer["accelerator"] == "gpu"
    assert trainer["devices"] == 1


def test_the_cpu_and_auto_accelerators_map_through():
    assert neural.NeuralRuntime(device="cpu").accelerator == "cpu"
    assert neural.NeuralRuntime(device="auto").accelerator == "auto"
    assert neural.NeuralRuntime(device="gpu").accelerator == "gpu"
    # devices is only meaningful on a gpu run.
    assert "devices" not in neural.NeuralRuntime(device="cpu").trainer_kwargs()


def test_the_worker_count_is_configurable_and_not_zero_by_default():
    """The smoke test's 0 was a workaround for cell ordering, not a setting."""
    assert neural.NeuralRuntime().num_workers > 0
    assert neural.configure(num_workers=0).num_workers == 0
    neural.configure()                                   # restore the default


def test_the_runtime_describes_itself_for_the_run_log():
    text = neural.NeuralRuntime(device="cuda", num_workers=2).describe()
    assert "device=cuda" in text and "num_workers=2" in text


# --------------------------------------------------------------------------
# 5. the scale guardrail, shared with the foundation tier
# --------------------------------------------------------------------------

def test_a_realistic_return_width_passes():
    sigma = 0.0157
    half = sigma * 2.5631 / 2
    report = check_sigma_band(np.full(64, -half), np.full(64, half), "return")
    assert "inside the return band" in report


def test_a_width_off_by_a_factor_is_rejected():
    """The TimesFM failure: 0.0906/day, 144% annualised, plausible-looking."""
    half = 0.0906 * 2.5631 / 2
    with pytest.raises(SigmaOutOfBand, match="outside the return-arm band"):
        check_sigma_band(np.full(64, -half), np.full(64, half), "return")


def test_the_two_arms_do_not_share_a_band():
    assert SIGMA_BANDS["return"] != SIGMA_BANDS["volatility"]
    half = 0.001 * 2.5631 / 2
    check_sigma_band(np.full(8, -half), np.full(8, half), "volatility")
    with pytest.raises(SigmaOutOfBand):
        check_sigma_band(np.full(8, -half), np.full(8, half), "return")


def test_a_point_only_model_skips_the_band_rather_than_failing():
    nan = np.full(8, np.nan)
    assert "skipped" in check_sigma_band(nan, nan, "return")


def test_implied_sigma_matches_the_normal_80_percent_span():
    assert implied_daily_sigma(np.array([-1.28155]),
                               np.array([1.28155])) == pytest.approx(1.0, abs=1e-4)


def test_an_unknown_arm_is_an_error_not_a_default():
    with pytest.raises(KeyError):
        check_sigma_band(np.zeros(4), np.ones(4), "equities")


# --------------------------------------------------------------------------
# 6. leakage: the fit must not reach the test window
# --------------------------------------------------------------------------

def test_a_training_frame_reaching_the_test_window_is_refused():
    dates = pd.bdate_range("2019-01-01", periods=300)
    train = pd.DataFrame({"unique_id": "AAA", "ds": dates, "y": 0.0})
    with pytest.raises(AssertionError, match="on or after the test window opens"):
        neural.neural_forecast_fold(
            "DLinear", panel=train, train=train,
            test_start=dates[-1], test_end=dates[-1],
            quantiles=FROZEN, frozen_grid=FROZEN)
