"""Two ways a model's cells can form and mean nothing, and what is done about each.

Both defects here produce finite numbers in the right columns. Neither raises,
neither is visible in a table, and both are answered by *removing* something -
which is the kind of decision that has to be tested, because a suppression that
fires on the wrong model is as damaging as one that never fires.

**Not the model's output at all.** NLinear's median carries a constant shift of
about -6.2% per day, invariant across every fold and every ticker. Nothing it
writes is a forecast, so it is excluded from scoring outright rather than scored
and marked - a marked row still invites a reader to compare it, and there is
nothing here to compare. What is tested is that the exclusion is *visible*: the
model is named, the reason is printed, and the roster denominator still counts
it, so a reader cannot mistake nineteen runnable models for eighteen.

**A quantile grid that is not ordered.** DLinear's levels arrived scrambled, so
its CRPS, PICP and interval score are averages over an unordered sample. The
point metrics read the median alone and are unaffected, so they stand. The hard
part is the threshold: TimesNet crosses on 0.1% of rows, which is an ordinary
property of quantile regression without a monotonicity constraint and must not
be suppressed. Both populations are reproduced here and the threshold is
asserted to separate them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500bench.config import load_configs, quantile_column
from sp500bench.models.broken import BROKEN_MODELS
from sp500bench.models.registry import runnable

ROOT = Path(__file__).resolve().parents[1]


def _load_evaluate():
    spec = importlib.util.spec_from_file_location(
        "evaluate_broken_under_test", ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load_evaluate()


@pytest.fixture(scope="module")
def cfg():
    return load_configs()


def _frame(cfg, n_rows: int = 2000, seed: int = 3) -> pd.DataFrame:
    """An ordered quantile panel with a median that is a real forecast."""
    rng = np.random.default_rng(seed)
    values = np.sort(rng.normal(0.0, 0.02, (n_rows, len(cfg.quantiles))), axis=1)
    frame = pd.DataFrame(values, columns=list(cfg.quantile_columns))
    frame["unique_id"] = np.repeat([f"T{i:02d}" for i in range(50)], n_rows // 50)
    frame["ds"] = pd.bdate_range("2020-01-02", periods=n_rows // 50).tolist() * 50
    frame["fold"] = 0
    frame["y"] = frame[quantile_column(0.5)] + rng.normal(0, 0.01, n_rows)
    return frame


def _scramble(frame: pd.DataFrame, cfg, share: float, seed: int = 11):
    """Shuffle the level columns on ``share`` of the rows, as a mapping fault does."""
    rng = np.random.default_rng(seed)
    columns = list(cfg.quantile_columns)
    values = frame[columns].to_numpy().copy()
    hit = rng.permutation(len(frame))[:int(round(share * len(frame)))]
    for row in hit:
        rng.shuffle(values[row])
    out = frame.copy()
    out[columns] = values
    return out


def _frozen(cfg):
    return evaluate.CrpsGrid(f"frozen-{len(cfg.quantiles)}",
                             tuple(cfg.quantiles), True)


# --------------------------------------------------------------- unordered ---

def test_a_fully_scrambled_grid_is_suppressed(cfg):
    """The DLinear population: every row inverted somewhere."""
    frame = _scramble(_frame(cfg), cfg, share=1.0)
    grid, share = evaluate.downgrade_unordered("DLinearLike", frame,
                                               _frozen(cfg), cfg)
    assert share < 0.05
    assert grid.label == "unordered"
    assert grid.levels == ()
    assert grid.comparable is False


def test_a_sparse_crossing_is_reported_not_suppressed(cfg):
    """The TimesNet population: 0.1% of rows, a model property rather than a fault.

    Suppressing this would remove a working model's CRPS from the table over
    87 rows in 83,450, which is a larger error than reporting the crossings.
    """
    frame = _frame(cfg)
    columns = list(cfg.quantile_columns)
    values = frame[columns].to_numpy().copy()
    n_crossed = 2                                        # 0.1% of 2,000 rows
    values[:n_crossed, [3, 4]] = values[:n_crossed, [4, 3]]
    frame[columns] = values

    grid, share = evaluate.downgrade_unordered("TimesNetLike", frame,
                                               _frozen(cfg), cfg)
    assert share == pytest.approx(1 - n_crossed / len(frame))
    assert grid.levels == tuple(cfg.quantiles)
    assert grid.comparable is True


def test_the_threshold_separates_the_two_measured_populations(cfg):
    """DLinear at 100% and TimesNet at 0.1% must land on opposite sides."""
    assert 0.001 < evaluate.NON_MONOTONE_TOLERANCE < 0.5


def test_a_clean_model_is_untouched(cfg):
    grid, share = evaluate.downgrade_unordered("Clean", _frame(cfg),
                                               _frozen(cfg), cfg)
    assert share == 1.0
    assert grid.levels == tuple(cfg.quantiles)


def test_suppression_leaves_the_point_metrics_standing(cfg):
    """RMSE, MAE and directional accuracy read the median, which is mapped right.

    Discarding four real measurements to punish a fault in three others would
    be the wrong trade, so this asserts they survive - and that the three that
    should go are actually gone.
    """
    frame = _scramble(_frame(cfg), cfg, share=1.0)
    grid, _ = evaluate.downgrade_unordered("DLinearLike", frame,
                                           _frozen(cfg), cfg)
    denominators = {(0, t): (0.01, 0.01) for t in frame["unique_id"].unique()}
    cells = evaluate.cell_metrics(frame, cfg, denominators, n_boot=50,
                                  arm="return", grid=grid)

    assert np.isfinite(cells["rmse"]) and cells["rmse"] > 0
    assert np.isfinite(cells["mae"]) and np.isfinite(cells["mase"])
    assert np.isfinite(cells["directional_accuracy"])
    assert np.isnan(cells["crps"])
    assert np.isnan(cells["picp_80"])
    assert np.isnan(cells["mean_interval_score_80"])


def test_a_suppressed_model_contributes_no_pinball_rows(cfg):
    """Otherwise a CRPS-based DM test or MCS downstream picks them up.

    ``build_losses`` feeds ``analyse.py``, which knows nothing about grids. If
    the pinball rows were written, the suppression would hold in the tables and
    silently fail in the inference.
    """
    frame = _scramble(_frame(cfg), cfg, share=1.0)
    grid, _ = evaluate.downgrade_unordered("DLinearLike", frame,
                                           _frozen(cfg), cfg)
    losses = evaluate.build_losses("DLinear", frame, cfg, grid)
    kinds = set(losses["loss_type"].unique())

    assert not [k for k in kinds if k.startswith("pinball_")]
    assert {"squared_error", "absolute_error", "hit"} <= kinds


def test_an_ordered_model_still_contributes_every_pinball_level(cfg):
    losses = evaluate.build_losses("Clean", _frame(cfg), cfg, _frozen(cfg))
    kinds = {k for k in losses["loss_type"].unique() if k.startswith("pinball_")}
    assert kinds == {f"pinball_{c}" for c in cfg.quantile_columns}


# ------------------------------------------------------------------ broken ---

def test_a_broken_model_is_excluded_but_still_counted(tmp_path, cfg):
    """Excluded from scoring, named in the roster, and still in the denominator.

    "16 of 18" would say the roster has eighteen runnable models. It has
    nineteen, and the nineteenth was dropped for a stated reason - which is a
    different claim, and the one a reader has to be able to check.
    """
    assert "NLinear" in evaluate.BROKEN_MODELS
    present, pending, broken = evaluate.roster_status(tmp_path, [0])

    assert broken == ["NLinear"]
    assert "NLinear" not in present and "NLinear" not in pending
    assert len(present) + len(pending) + len(broken) == len(list(runnable()))


def test_the_broken_flag_puts_it_back(tmp_path):
    """The exclusion is inspectable rather than final; nothing is deleted."""
    _, pending, broken = evaluate.roster_status(tmp_path, [0], score_broken=True)
    assert broken == []
    assert "NLinear" in pending          # no forecasts under tmp_path


def test_the_broken_registry_has_one_home(tmp_path):
    """evaluate.py and backtest.py must not be able to disagree.

    backtest.py excluding a different set from evaluate.py would not produce a
    wrong table so much as an incoherent one - and the deflated Sharpe hurdle is
    parameterised by the number of strategies tried, so a broken model left in
    backtest's count silently raises the bar for every other model.
    """
    import importlib.util as u
    spec = u.spec_from_file_location("backtest_under_test",
                                     ROOT / "scripts" / "backtest.py")
    module = u.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.BROKEN_MODELS is BROKEN_MODELS
    assert evaluate.BROKEN_MODELS is BROKEN_MODELS


def test_every_broken_model_states_its_evidence():
    """A reason that is a verdict rather than a measurement is not a reason.

    Dropping a model after seeing it score badly and dropping one after
    establishing its output is not the model's look identical in a results
    table. The difference lives entirely in what is written here.
    """
    for model, reason in evaluate.BROKEN_MODELS.items():
        assert len(reason) > 80, f"{model}: reason is too thin to check"
        assert any(char.isdigit() for char in reason), (
            f"{model}: the reason cites no measured figure")
