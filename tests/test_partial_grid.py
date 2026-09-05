"""A model that cannot reach the whole quantile grid must not be scored as if it did.

Two of the foundation checkpoints emit fewer levels than the frozen nine:
TimesFM-2.5 has a fixed ``[0.1 .. 0.9]`` head, and TTM has no quantile head at
all. The failure these tests exist to prevent is not a crash - it is a CRPS that
is quietly computed over whatever happened to be finite, and then printed in the
same column as everybody else's.

The distinction that matters, and the reason the runtime guard is not redundant
with NaN arithmetic:

* an **entirely** NaN column propagates NaN through ``crps_from_quantiles`` and
  produces the right answer on its own;
* a **partially** NaN column does not. ``pinball_loss`` drops non-finite pairs,
  so that level is averaged over a different set of observations than its
  neighbours and the CRPS built from them is a mixture of sample sizes. It comes
  out finite and looks ordinary.

The second case is asserted here as the behaviour it actually has, not the
behaviour it ought to have, together with the guard that catches it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500bench.config import load_configs, quantile_column
from sp500bench.evaluation.metrics import crps_from_quantiles

ROOT = Path(__file__).resolve().parents[1]


def _load_evaluate():
    """``scripts/`` is not a package, so load the module by path.

    Registered in ``sys.modules`` before execution because ``@dataclass``
    resolves annotations through the defining module and cannot find one that is
    not there yet.
    """
    spec = importlib.util.spec_from_file_location(
        "evaluate_under_test", ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_evaluate()


@pytest.fixture(scope="module")
def cfg():
    return load_configs()


@pytest.fixture
def cell(cfg) -> pd.DataFrame:
    """One scoreable cell: two tickers, thirty sessions, the full frozen grid."""
    rng = np.random.default_rng(20260905)
    dates = pd.bdate_range("2020-02-03", periods=30)
    frames = []
    for ticker in ("AAA", "BBB"):
        y = rng.normal(0.0, 0.012, len(dates))
        frame = pd.DataFrame({"unique_id": ticker, "ds": dates, "y": y, "fold": 0})
        for q in cfg.quantiles:
            # A plausible, correctly ordered band centred near zero.
            frame[quantile_column(q)] = np.quantile(y, q) + rng.normal(0, 1e-5, len(dates))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def denominators() -> dict:
    return {(0, "AAA"): (0.010, 0.008), (0, "BBB"): (0.010, 0.008)}


# --------------------------------------------------------------------------
# the requirement: NaN tails must not become a CRPS over the survivors
# --------------------------------------------------------------------------

def test_nan_tail_columns_give_nan_crps_not_a_score_over_the_rest(cfg, cell):
    """The four tail levels NaN must yield NaN, never an average of the other five."""
    y = cell["y"].to_numpy(dtype=float)
    full = {q: cell[quantile_column(q)].to_numpy(dtype=float) for q in cfg.quantiles}

    complete = crps_from_quantiles(y, full)
    assert np.isfinite(complete), "the control case must produce a real number"

    tails_gone = dict(full)
    for q in (0.01, 0.05, 0.95, 0.99):
        tails_gone[q] = np.full(len(cell), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        partial_grid = crps_from_quantiles(y, tails_gone)

    assert np.isnan(partial_grid), (
        "a model missing the tail levels must not be handed a CRPS computed over "
        "the five levels that survived - that number would be a different "
        "quantity printed in the same column")

    # And specifically not equal to the score over the surviving five, which is
    # the wrong answer this test exists to exclude.
    survivors = {q: full[q] for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    assert np.isfinite(crps_from_quantiles(y, survivors))


def test_a_partially_nan_column_stays_finite_and_the_guard_is_what_catches_it(cfg, cell):
    """Documented as it behaves, not as it ought to.

    Half a column of NaN does not propagate: ``pinball_loss`` drops the
    non-finite pairs and returns a mean over the rest. Nothing raises, so the
    protection has to come from the grid guard rather than from arithmetic.
    """
    y = cell["y"].to_numpy(dtype=float)
    full = {q: cell[quantile_column(q)].to_numpy(dtype=float) for q in cfg.quantiles}

    holed = dict(full)
    values = holed[0.01].copy()
    values[: len(values) // 2] = np.nan
    holed[0.01] = values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = crps_from_quantiles(y, holed)
    assert np.isfinite(score), (
        "if this ever starts returning NaN the guard below can be relaxed - "
        "until then a half-empty level is scored silently")

    frame = cell.copy()
    frame[quantile_column(0.01)] = values
    grid = evaluate.CrpsGrid("frozen-9", tuple(cfg.quantiles), True)
    with pytest.raises(SystemExit, match="declared reachable but is NaN"):
        evaluate.assert_grid_matches_data("Holed", frame, grid, cfg)


# --------------------------------------------------------------------------
# grid resolution
# --------------------------------------------------------------------------

def test_a_model_without_grid_json_is_scored_on_the_frozen_grid(cfg, tmp_path):
    grid = evaluate.load_crps_grid(tmp_path, "AutoARIMA", cfg)
    assert grid.levels == tuple(cfg.quantiles)
    assert grid.comparable is True
    assert grid.has_band


def test_timesfm_style_grid_uses_its_own_levels_and_is_marked(cfg, tmp_path):
    native = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model_dir = tmp_path / "TimesFM-2.5"
    model_dir.mkdir()
    (model_dir / "grid.json").write_text(json.dumps({
        "model": "TimesFM-2.5", "scored_on_frozen_grid": False,
        "native_levels": native, "notes": "fixed nine-decile head"}))

    grid = evaluate.load_crps_grid(tmp_path, "TimesFM-2.5", cfg)
    assert grid.levels == tuple(native)
    assert grid.comparable is False, "must be marked, never ranked against the frozen nine"
    assert grid.has_band, "0.1 and 0.9 are in its head, so PICP80 is still defined"
    assert grid.levels != tuple(cfg.quantiles)
    assert evaluate._display_name("TimesFM-2.5", {"TimesFM-2.5": grid}).endswith("*")


def test_point_only_grid_is_absent_from_crps_and_carries_no_band(cfg, tmp_path):
    model_dir = tmp_path / "TTM"
    model_dir.mkdir()
    (model_dir / "grid.json").write_text(json.dumps({
        "model": "TTM", "scored_on_frozen_grid": False,
        "native_levels": [], "notes": "no quantile head"}))

    grid = evaluate.load_crps_grid(tmp_path, "TTM", cfg)
    assert grid.levels == ()
    assert grid.comparable is False
    assert not grid.has_band
    # Every level but the point forecast must be empty in the file.
    assert set(grid.unavailable) == {q for q in cfg.quantiles if q != 0.5}


def test_interpolated_levels_inside_the_support_are_not_treated_as_unavailable(
        cfg, tmp_path):
    """The distinction the guard turns on.

    TimesFM's q0.25 and q0.75 are interpolated *inside* [0.1, 0.9] and are real
    values the runner writes. Only the four levels outside that support are
    unavailable. A guard that required every non-CRPS-grid level to be NaN would
    reject a perfectly valid file.
    """
    model_dir = tmp_path / "TimesFM-2.5"
    model_dir.mkdir()
    (model_dir / "grid.json").write_text(json.dumps({
        "model": "TimesFM-2.5", "scored_on_frozen_grid": False,
        "native_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "frozen_levels_interpolated": [0.25, 0.75],
        "frozen_levels_unavailable": [0.01, 0.05, 0.95, 0.99],
        "notes": "fixed nine-decile head"}))

    grid = evaluate.load_crps_grid(tmp_path, "TimesFM-2.5", cfg)
    assert grid.unavailable == (0.01, 0.05, 0.95, 0.99)
    assert 0.25 not in grid.unavailable and 0.75 not in grid.unavailable


# --------------------------------------------------------------------------
# the absences, asserted rather than inferred
# --------------------------------------------------------------------------

def test_point_only_model_is_absent_from_crps_picp_and_interval_score(
        cfg, cell, denominators):
    """TTM's shape: a median and eight NaN columns."""
    frame = cell.copy()
    for q in cfg.quantiles:
        if q != 0.5:
            frame[quantile_column(q)] = np.nan

    grid = evaluate.CrpsGrid("none", (), False, "no quantile head",
                             tuple(q for q in cfg.quantiles if q != 0.5))
    evaluate.assert_grid_matches_data("TTM", frame, grid, cfg)   # must not raise

    row = evaluate.cell_metrics(frame, cfg, denominators, n_boot=50,
                                arm="return", grid=grid)
    assert np.isnan(row["crps"])
    assert np.isnan(row["picp_80"])
    assert np.isnan(row["mean_interval_score_80"])
    # Still present on the point-accuracy axis - absent from CRPS is not absent
    # from the study.
    assert np.isfinite(row["rmse"])
    assert np.isfinite(row["mae"])
    assert np.isfinite(row["directional_accuracy"])


def test_guard_rejects_a_point_only_model_that_starts_emitting_a_band(
        cfg, cell):
    """If the absence stops being true, it must fail loudly rather than score."""
    grid = evaluate.CrpsGrid("none", (), False, "no quantile head",
                             tuple(q for q in cfg.quantiles if q != 0.5))
    with pytest.raises(SystemExit, match="declared unavailable"):
        evaluate.assert_grid_matches_data("TTM", cell, grid, cfg)


def test_partial_grid_model_is_scored_on_its_own_levels(cfg, cell, denominators):
    """TimesFM's shape: its nine deciles present, the four frozen tails NaN."""
    native = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    frame = cell.copy()
    for q in native:
        if quantile_column(q) not in frame.columns:
            frame[quantile_column(q)] = np.quantile(
                frame["y"].to_numpy(dtype=float), q)
    for q in (0.01, 0.05, 0.95, 0.99):
        frame[quantile_column(q)] = np.nan

    grid = evaluate.CrpsGrid("timesfm-9", native, False, "fixed nine-decile head",
                             (0.01, 0.05, 0.95, 0.99))
    evaluate.assert_grid_matches_data("TimesFM-2.5", frame, grid, cfg)

    row = evaluate.cell_metrics(frame, cfg, denominators, n_boot=50,
                                arm="return", grid=grid)
    assert np.isfinite(row["crps"]), "its own nine levels are all present"
    assert np.isfinite(row["picp_80"])

    # The same cell scored on the frozen grid is NaN, which is precisely why the
    # two numbers must never share a column without a mark.
    frozen = evaluate.CrpsGrid("frozen-9", tuple(cfg.quantiles), True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on_frozen = evaluate.cell_metrics(frame, cfg, denominators, n_boot=50,
                                          arm="return", grid=frozen)
    assert np.isnan(on_frozen["crps"])
    assert row["crps"] != pytest.approx(on_frozen["crps"], nan_ok=True)
