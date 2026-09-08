"""A forecast that is identically zero makes no directional call.

``sign(0)`` matches nothing, so a constant-zero forecast scores 0.00% on
directional accuracy with a [0.00%, 0.00%] bootstrap interval. Every number
there is arithmetically correct and the cell says the opposite of the truth: a
random walk in prices is not directionally catastrophic, it declines to make a
directional prediction, and that is the point of it.

The cell matters more than most because ``ZeroReturn`` is H1's reference. A
reader comparing a model at 51% against a benchmark at 0% would read a 51-point
directional edge that does not exist.

The README fixed the convention when ZeroReturn was identified as the benchmark
- "reported as n/a rather than 0%" - and the scoring code did not implement it
for some time afterwards, which is the ordinary way a frozen specification and
its implementation come apart: nothing downstream can question the spec, because
everything downstream is the spec.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500bench.config import load_configs, quantile_column

ROOT = Path(__file__).resolve().parents[1]


def _load_evaluate():
    spec = importlib.util.spec_from_file_location(
        "evaluate_zero_dir_under_test", ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_evaluate()


@pytest.fixture(scope="module")
def cfg():
    return load_configs()


def _cell(cfg, point_values: np.ndarray, seed: int = 5) -> pd.DataFrame:
    """One scoreable cell whose median column is supplied by the caller."""
    rng = np.random.default_rng(seed)
    n = len(point_values)
    frame = pd.DataFrame({
        "unique_id": np.repeat(["AAA", "BBB"], n // 2),
        "ds": pd.bdate_range("2021-01-04", periods=n // 2).tolist() * 2,
        "fold": 0,
        "y": rng.normal(0.0, 0.02, n),
    })
    for q in cfg.quantiles:
        frame[quantile_column(q)] = point_values + (q - 0.5) * 0.04
    frame[cfg.median_column] = point_values
    return frame


def _denominators(frame):
    return {(0, t): (0.01, 0.01) for t in frame["unique_id"].unique()}


def test_a_constant_zero_forecast_has_no_directional_accuracy(cfg):
    frame = _cell(cfg, np.zeros(600))
    cells = evaluate.cell_metrics(frame, cfg, _denominators(frame), n_boot=50)

    assert np.isnan(cells["directional_accuracy"]), (
        "a zero forecast scored 0.00% - the statistic is undefined, not zero")
    assert np.isnan(cells["da_ci_low"]) and np.isnan(cells["da_ci_high"])


def test_the_point_metrics_of_a_zero_forecast_are_untouched(cfg):
    """It is a real forecast on every other axis and must still be scored."""
    frame = _cell(cfg, np.zeros(600))
    cells = evaluate.cell_metrics(frame, cfg, _denominators(frame), n_boot=50)

    assert np.isfinite(cells["rmse"]) and cells["rmse"] > 0
    assert np.isfinite(cells["mae"]) and np.isfinite(cells["mase"])
    assert np.isfinite(cells["crps"])


def test_a_forecast_that_is_merely_small_still_scores(cfg):
    """Only an identically-zero forecast is exempt, not a nearly-flat one.

    A model that predicts 1e-9 every day is making a directional call - a silly
    one - and must be scored on it. Widening the exemption to 'close to zero'
    would hand any sufficiently timid model an escape from the axis.
    """
    frame = _cell(cfg, np.full(600, 1e-9))
    cells = evaluate.cell_metrics(frame, cfg, _denominators(frame), n_boot=50)
    assert np.isfinite(cells["directional_accuracy"])


def test_a_mixed_forecast_with_some_zeros_still_scores(cfg):
    """One flat session does not make the model directionally silent."""
    point = np.tile([0.0, 0.01, -0.01, 0.0, 0.02, -0.02], 100)
    frame = _cell(cfg, point)
    cells = evaluate.cell_metrics(frame, cfg, _denominators(frame), n_boot=50)
    assert np.isfinite(cells["directional_accuracy"])


def test_the_real_zeroreturn_cell_is_na_not_zero(cfg):
    """Against the artefact on disk, not a synthetic stand-in.

    Skipped rather than failed when the study has not been scored yet, so a
    fresh checkout does not report a defect it cannot have.
    """
    path = ROOT / "results" / "metrics.parquet"
    if not path.exists():
        pytest.skip("results/metrics.parquet has not been produced yet")
    table = pd.read_parquet(path)
    cell = table[(table.model == "ZeroReturn") & (table.regime_kind == "all")
                 & (table.fold.isna())]
    if cell.empty:
        pytest.skip("ZeroReturn has not been scored in this run")
    assert cell["directional_accuracy"].isna().all(), (
        "ZeroReturn's directional accuracy is a number in metrics.parquet; the "
        "README requires n/a and Table 4.1 reads this column")
