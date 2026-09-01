"""The configuration gate: the two frozen specifications must agree.

These are regression tests for a bug that actually happened. The evaluation grid
was densified from five quantile levels to nine, and ``training.quantiles`` in
configs/models.yaml was left behind at five. Nothing failed - the configs still
parsed, the roster still resolved - but the seven neuralforecast models would
have emitted only the five levels they were trained on while the other twelve
were scored on nine. CRPS averaged over one grid is not comparable to CRPS
averaged over another, so the headline probabilistic result for H2 would have
been uninterpretable, and nothing in the pipeline would have said so.

Each test below constructs one way the two files can drift apart and requires
that loading them raises. Checked once by hand is not the same as enforced.
"""

from __future__ import annotations

import pytest
import yaml

from sp500bench.config import (ConfigError, DEFAULT_EVAL_CONFIG, DEFAULT_MODELS_CONFIG,
                               load_configs, quantile_column, quantile_columns)

GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


@pytest.fixture
def configs(tmp_path):
    """Writer for a pair of minimal configs, defaulting to a matched grid."""

    def write(training=GRID, evaluation=GRID, point_forecast_from="median"):
        models = tmp_path / "models.yaml"
        evaluation_path = tmp_path / "eval.yaml"
        models.write_text(yaml.safe_dump(
            {"version": 4, "training": {"loss": "MQLoss", "quantiles": list(training)}}
        ))
        evaluation_path.write_text(yaml.safe_dump({
            "version": 2,
            "quantiles": list(evaluation),
            "point_forecast_from": point_forecast_from,
            "cross_validation": {"horizon": 1, "embargo": 5,
                                 "test_windows": [["2020-01-01", "2020-12-31"]]},
        }))
        return str(models), str(evaluation_path)

    return write


# ---------------------------------------------------------------------------
# The four ways the grids drift apart
# ---------------------------------------------------------------------------

def test_training_grid_narrower_than_evaluation_grid_is_rejected(configs):
    """The exact bug: eval densified to nine levels, training left at five.

    This is the dangerous direction. An MQLoss model cannot produce a level it
    has no head for, so the tail levels would be missing for precisely the seven
    models the densification was meant to measure.
    """
    models, evaluation = configs(training=[0.1, 0.25, 0.5, 0.75, 0.9])
    with pytest.raises(ConfigError) as excinfo:
        load_configs(models, evaluation)
    message = str(excinfo.value)
    assert "0.01" in message and "0.99" in message, "must name the missing levels"
    assert "scored but never trained" in message


def test_training_grid_wider_than_evaluation_grid_is_rejected(configs):
    """The other direction. Not unsafe, but it means the files disagree."""
    models, evaluation = configs(training=sorted(GRID + [0.02]))
    with pytest.raises(ConfigError) as excinfo:
        load_configs(models, evaluation)
    assert "trained but never scored" in str(excinfo.value)
    assert "0.02" in str(excinfo.value)


def test_same_levels_in_a_different_order_is_rejected(configs):
    """Set equality is not enough.

    The grid is the axis CRPS is averaged over and the order is what pairs a
    level with its column. Two files holding the same levels in different orders
    do not describe the same object, and a set comparison would wave this
    through.
    """
    models, evaluation = configs(training=list(reversed(GRID)))
    with pytest.raises(ConfigError) as excinfo:
        load_configs(models, evaluation)
    assert "different order" in str(excinfo.value)


def test_median_missing_from_the_grid_is_rejected(configs):
    """Grids can agree perfectly and still be unusable.

    ``point_forecast_from: median`` means every point metric and the entire
    backtest read the q0.5 column. If 0.5 is not in the grid that column is
    never written, and the failure would surface far downstream.
    """
    grid = [q for q in GRID if q != 0.5]
    models, evaluation = configs(training=grid, evaluation=grid)
    with pytest.raises(ConfigError) as excinfo:
        load_configs(models, evaluation)
    assert "0.5 is not in the quantile grid" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The gate must not be so strict it rejects the real thing
# ---------------------------------------------------------------------------

def test_matched_grids_load(configs):
    models, evaluation = configs()
    cfg = load_configs(models, evaluation)
    assert list(cfg.quantiles) == GRID
    assert cfg.median_column == "q0.5"


def test_an_unrecognised_point_forecast_source_is_rejected(configs):
    models, evaluation = configs(point_forecast_from="mean")
    with pytest.raises(ConfigError):
        load_configs(models, evaluation)


def test_the_repositorys_own_configs_pass_the_gate():
    """The frozen specification in configs/ must satisfy its own checks."""
    cfg = load_configs(DEFAULT_MODELS_CONFIG, DEFAULT_EVAL_CONFIG)
    assert list(cfg.quantiles) == GRID
    assert cfg.models["training"]["quantiles"] == cfg.evaluation["quantiles"]
    assert cfg.horizon == 1 and cfg.embargo == 5
    assert len(cfg.test_windows) == 4


def test_quantile_column_names_are_stable():
    """Writer and reader must spell a level the same way, or forecasts read NaN."""
    assert quantile_column(0.5) == "q0.5"
    assert quantile_column(0.01) == "q0.01"
    assert quantile_column(0.10) == "q0.1"
    assert quantile_columns(GRID) == [
        "q0.01", "q0.05", "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q0.95", "q0.99"
    ]
    assert len(set(quantile_columns(GRID))) == len(GRID), "levels must not collide"
