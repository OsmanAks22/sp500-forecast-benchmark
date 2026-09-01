#!/usr/bin/env python
"""Fit and forecast one or more tiers of the roster.

Step three of the pipeline. It reads the frozen specification in
``configs/models.yaml`` and ``configs/eval.yaml``, resolves the tiers named on
the command line against
:data:`sp500bench.models.registry.REGISTRY` - the executable source of truth for
the roster - and runs them over the purged walk-forward folds.

The model execution layer is not written yet. What *is* written, and what runs
today, is the configuration gate below: the checks that must pass before any
model is allowed to consume compute. They are cheap, they run first, and they
fail loudly.

Why the quantile grids are checked for equality
-----------------------------------------------
``MQLoss`` trains one output head per quantile level, so a neuralforecast model
emits exactly the levels named in ``training.quantiles`` and no others. CRPS is
approximated here by averaging pinball loss over the evaluation grid
(:func:`sp500bench.evaluation.metrics.crps_from_quantiles`). If the two grids
differ, one of two things happens, and both are silent:

* the evaluation grid is *wider*, and the seven neuralforecast models cannot
  supply the missing levels - so either the run crashes deep into the GPU
  budget, or the metric quietly falls back to whatever levels are present;
* the evaluation grid is *narrower*, and those models are scored on a subset of
  what they optimised, while the twelve models that can emit arbitrary levels
  are scored on the full grid.

Either way CRPS stops being comparable across the roster, because an average of
pinball losses over one grid is not commensurable with an average over another.
H2 is a calibration hypothesis, so that would land directly on the headline
result. The two grids diverged once already, between eval.yaml v2 and
models.yaml v3; this check is here so it cannot happen a second time without
someone being told.

Usage
-----
    python scripts/run_experiment.py --tier baseline --tier statistical
    python scripts/run_experiment.py --tier foundation --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.models.registry import REGISTRY, ModelSpec  # noqa: E402

TIERS = ["baseline", "statistical", "linear", "recurrent", "transformer", "foundation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more tiers of the frozen model roster."
    )
    parser.add_argument(
        "--tier", action="append", dest="tiers", choices=TIERS, metavar="TIER",
        help=f"tier to run; repeatable. One of: {', '.join(TIERS)}",
    )
    parser.add_argument(
        "--models-config", default=str(ROOT / "configs" / "models.yaml"),
        help="frozen roster specification (default: configs/models.yaml)",
    )
    parser.add_argument(
        "--eval-config", default=str(ROOT / "configs" / "eval.yaml"),
        help="frozen evaluation specification (default: configs/eval.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run the configuration gate, print the resolved roster, and stop",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration gate
# ---------------------------------------------------------------------------

def assert_quantile_grids_agree(models_cfg: dict, eval_cfg: dict) -> list[float]:
    """The training grid and the evaluation grid must be the same list.

    Equality is required element for element, in order - not as sets. The grid
    is the axis CRPS is averaged over, and a reordering would mean the two files
    no longer describe the same object even where they hold the same levels.
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
        raise SystemExit(
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
        raise SystemExit(
            f"eval.yaml sets point_forecast_from to {source!r}; this script "
            "reads the point forecast from the median."
        )
    if 0.5 not in grid:
        raise SystemExit(
            f"point_forecast_from is 'median' but 0.5 is not in the quantile "
            f"grid {grid}. Every point metric - RMSE, MAE, MASE, directional "
            "accuracy - and the whole backtest read that column."
        )


def load_configs(models_path: str, eval_path: str) -> tuple[dict, dict, list[float]]:
    """Load both frozen specifications and refuse to proceed if they disagree."""
    with open(models_path) as fh:
        models_cfg = yaml.safe_load(fh)
    with open(eval_path) as fh:
        eval_cfg = yaml.safe_load(fh)

    grid = assert_quantile_grids_agree(models_cfg, eval_cfg)
    assert_point_forecast_is_available(grid, eval_cfg)
    return models_cfg, eval_cfg, grid


def resolve_roster(tiers: list[str] | None) -> list[ModelSpec]:
    """Select from the registry, which is the source of truth for the roster."""
    if not tiers:
        return list(REGISTRY)
    wanted = set(tiers)
    return [m for m in REGISTRY if m.tier in wanted]


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    models_cfg, eval_cfg, grid = load_configs(args.models_config, args.eval_config)

    print(f"models.yaml v{models_cfg['version']}  eval.yaml v{eval_cfg['version']}")
    print(f"quantile grid ({len(grid)} levels, training == evaluation): {grid}")
    print(f"point forecast: {eval_cfg['point_forecast_from']} (q0.5)")

    roster = resolve_roster(args.tiers)
    scope = ", ".join(args.tiers) if args.tiers else "all tiers"
    print(f"\nroster: {len(roster)} of {len(REGISTRY)} models ({scope})")
    print(f"\n{'model':<22}{'tier':<14}{'library':<16}{'scope'}")
    print("-" * 62)
    for m in roster:
        print(f"{m.name:<22}{m.tier:<14}{m.library:<16}{m.scope}")

    if not roster:
        raise SystemExit(f"\nno models match {scope}")

    if args.dry_run:
        print("\nDry run - configuration gate passed, nothing executed.")
        return

    raise SystemExit(
        "\nThe model execution layer is not implemented yet.\n"
        "  The configuration gate above passed. Re-run with --dry-run to check\n"
        "  the specification without this error."
    )


if __name__ == "__main__":
    main()
