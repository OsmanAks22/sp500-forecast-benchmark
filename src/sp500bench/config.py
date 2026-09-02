"""Loading and validation of the frozen experiment specification.

``configs/models.yaml`` and ``configs/eval.yaml`` are two halves of one object.
Every entry point that touches forecasts - ``run_experiment.py`` today,
``evaluate.py`` and ``backtest.py`` when they exist - has to agree on the
quantile grid, on which column carries the point forecast, and on how a quantile
level is spelled as a column name. Keeping that agreement in the package rather
than in whichever script happened to need it first is the only way it can be
imported by all three and tested once.

Why the quantile grids are checked for equality
-----------------------------------------------
``MQLoss`` trains one output head per quantile level, so a neuralforecast model
emits exactly the levels named in ``training.quantiles`` and no others. CRPS is
approximated in :func:`sp500bench.evaluation.metrics.crps_from_quantiles` by
averaging pinball loss over the evaluation grid. If the two grids differ, one of
two things happens, and both are silent:

* the evaluation grid is *wider*, and the seven neuralforecast models cannot
  supply the missing levels - so either the run crashes deep into the GPU
  budget, or the metric quietly falls back to whatever levels are present;
* the evaluation grid is *narrower*, and those models are scored on a subset of
  what they optimised, while the twelve models that can emit arbitrary levels
  are scored on the full grid.

Either way CRPS stops being comparable across the roster, because an average of
pinball losses over one grid is not commensurable with an average over another.
H2 is a calibration hypothesis, so that lands directly on the headline result.
The two grids diverged once already, between eval.yaml v2 and models.yaml v3;
these checks exist so it cannot happen again without someone being told.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "ExperimentConfig",
    "load_configs",
    "quantile_column",
    "quantile_columns",
    "assert_quantile_grids_agree",
    "assert_point_forecast_is_available",
    "DEFAULT_MODELS_CONFIG",
    "DEFAULT_EVAL_CONFIG",
    "DEFAULT_DATA_CONFIG",
]

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DEFAULT_MODELS_CONFIG = CONFIG_DIR / "models.yaml"
DEFAULT_EVAL_CONFIG = CONFIG_DIR / "eval.yaml"
DEFAULT_DATA_CONFIG = CONFIG_DIR / "data.yaml"


class ConfigError(RuntimeError):
    """A frozen specification is internally inconsistent.

    Raised rather than ``SystemExit`` so the checks are testable and so a caller
    that wants to catch and re-raise as a CLI error can. The entry points turn
    it into a ``SystemExit`` at the boundary.
    """


def quantile_column(q: float) -> str:
    """Column name for quantile level ``q``: ``0.05 -> 'q0.05'``.

    One definition, imported by whatever writes forecasts and whatever reads
    them back. Two spellings of the same level is the sort of mismatch that
    surfaces as a column of NaNs rather than as an error.
    """
    return f"q{q:g}"


def quantile_columns(grid) -> list[str]:
    return [quantile_column(q) for q in grid]


@dataclass(frozen=True)
class ExperimentConfig:
    """The two frozen specifications, loaded and checked against each other."""

    models: dict
    evaluation: dict
    quantiles: tuple[float, ...]

    @property
    def quantile_columns(self) -> list[str]:
        return quantile_columns(self.quantiles)

    @property
    def median_column(self) -> str:
        return quantile_column(0.5)

    @property
    def horizon(self) -> int:
        return int(self.evaluation["cross_validation"]["horizon"])

    @property
    def embargo(self) -> int:
        return int(self.evaluation["cross_validation"]["embargo"])

    def horizon_for(self, arm: str = "return") -> int:
        """Purge width for one arm, in sessions.

        The two arms forecast different targets and so consume different amounts
        of future data per label. The return arm's label at ``t`` is the 1-day
        forward log return and consumes one session; the volatility arm's is the
        Yang-Zhang estimate over ``t+1 .. t+21`` and consumes twenty-one.

        Purging both at the return arm's horizon would leave fifteen sessions of
        test data inside every volatility training fold - and
        :func:`~sp500bench.evaluation.splits.assert_no_overlap` would not catch
        it, because it would be measuring the gap it was told to measure. A guard
        given the wrong parameter is worse than no guard; it reads as assurance.
        """
        if arm == "volatility":
            return int(self.evaluation["volatility_arm"]["horizon"])
        return self.horizon

    @property
    def test_windows(self) -> list[tuple[str, str]]:
        return [tuple(w) for w in self.evaluation["cross_validation"]["test_windows"]]


def assert_quantile_grids_agree(models_cfg: dict, eval_cfg: dict) -> list[float]:
    """The training grid and the evaluation grid must be the same list.

    Equality is required element for element, in order - not as sets. The grid is
    the axis CRPS is averaged over, and a reordering would mean the two files no
    longer describe the same object even where they hold the same levels.
    """
    training = models_cfg["training"]["quantiles"]
    evaluation = eval_cfg["quantiles"]

    if training != evaluation:
        missing = sorted(set(evaluation) - set(training))
        extra = sorted(set(training) - set(evaluation))
        detail = ""
        if missing:
            detail += (
                f"\n  levels scored but never trained: {missing}\n"
                "    MQLoss emits only the levels it was trained on, so every\n"
                "    neuralforecast model would be unable to supply these."
            )
        if extra:
            detail += f"\n  levels trained but never scored: {extra}"
        if not missing and not extra:
            detail = "\n  same levels, different order - the grids must match in order too."
        raise ConfigError(
            "\nquantile grids disagree.\n"
            f"  configs/models.yaml  training.quantiles : {training}\n"
            f"  configs/eval.yaml    quantiles          : {evaluation}"
            f"{detail}\n\n"
            "  CRPS is the pinball loss averaged over this grid. Averages taken\n"
            "  over different grids are not comparable, so a roster split across\n"
            "  two grids cannot be ranked on CRPS at all. Fix the configs; do not\n"
            "  work around this at evaluation time."
        )
    return list(training)


def assert_point_forecast_is_available(grid: list[float], eval_cfg: dict) -> None:
    """The level the point forecast is read from has to be in the grid."""
    source = eval_cfg["point_forecast_from"]
    if source != "median":
        raise ConfigError(
            f"eval.yaml sets point_forecast_from to {source!r}; the point "
            "forecast is read from the median."
        )
    if 0.5 not in grid:
        raise ConfigError(
            f"point_forecast_from is 'median' but 0.5 is not in the quantile "
            f"grid {grid}. Every point metric - RMSE, MAE, MASE, directional "
            "accuracy - and the whole backtest read that column."
        )


def load_configs(
    models_path: str | Path = DEFAULT_MODELS_CONFIG,
    eval_path: str | Path = DEFAULT_EVAL_CONFIG,
) -> ExperimentConfig:
    """Load both frozen specifications and refuse to return if they disagree."""
    with open(models_path) as fh:
        models_cfg = yaml.safe_load(fh)
    with open(eval_path) as fh:
        eval_cfg = yaml.safe_load(fh)

    grid = assert_quantile_grids_agree(models_cfg, eval_cfg)
    assert_point_forecast_is_available(grid, eval_cfg)
    return ExperimentConfig(models=models_cfg, evaluation=eval_cfg,
                            quantiles=tuple(grid))


def load_data_config(path: str | Path = DEFAULT_DATA_CONFIG) -> dict:
    """The data specification, for the train start and the split boundaries."""
    with open(path) as fh:
        return yaml.safe_load(fh)
