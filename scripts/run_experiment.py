#!/usr/bin/env python
"""Fit and forecast one or more tiers of the roster.

Step three of the pipeline. It reads the frozen specification through
:func:`sp500bench.config.load_configs` - which refuses to return if the training
and evaluation quantile grids disagree - resolves the tiers named on the command
line against :data:`sp500bench.models.registry.REGISTRY`, and runs them over the
purged walk-forward folds from ``configs/eval.yaml``.

Only Tier 0 and Tier 1 have executors so far. Every other model in a selected tier is
reported and skipped, rather than silently omitted: a run that quietly produces
fewer forecast files than the roster has models is how a missing model becomes a
missing row in a results table nobody questions.

Outputs
-------
``forecasts/{model}/fold_{k}.parquet``  unique_id, ds, one column per quantile
``results/timings.json``              per-model wall-clock, for the compute axis

Usage
-----
    python scripts/run_experiment.py --tier baseline
    python scripts/run_experiment.py --tier baseline --tier statistical --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def show(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.config import (ConfigError, load_configs,  # noqa: E402
                               load_data_config)
from sp500bench.evaluation.splits import PurgedWalkForward, assert_no_overlap  # noqa: E402
from sp500bench.models.baselines import (last_return_forecast_fold,  # noqa: E402
                                         rwd_forecast_fold, warmup_statsforecast,
                                         zero_return_forecast_fold)
from sp500bench.models.statistical import (autoarima_forecast_fold,  # noqa: E402
                                           autoets_forecast_fold)
from sp500bench.models.volatility_arm import (build_volatility_panel,  # noqa: E402
                                              gjr_garch_forecast_fold,
                                              har_forecast_fold,
                                              log_har_forecast_fold)
from sp500bench.models.registry import (REGISTRY, ModelSpec,  # noqa: E402
                                        excluded)
from sp500bench.models.volatility_arm import RV_WINDOW  # noqa: E402

TIERS = ["baseline", "statistical", "linear", "recurrent", "transformer", "foundation"]

# Import name -> what to type when it is missing. The two differ often enough
# ("pip install granite-tsfm" for `tsfm_public`) that guessing is no help.
LIBRARIES = {
    "statsforecast":  ("statsforecast", "pip install statsforecast"),
    "neuralforecast": ("neuralforecast", "pip install neuralforecast"),
    "arch":           ("arch", "pip install arch"),
    "chronos":        ("chronos", "pip install chronos-forecasting"),
    "timesfm":        ("timesfm", "pip install timesfm"),
    "uni2ts":         ("uni2ts", "pip install uni2ts"),
    "kronos":         ("kronos", "see the Kronos repository - not on PyPI"),
    "granite-tsfm":   ("tsfm_public", "pip install granite-tsfm"),
    "sp500bench":     ("sp500bench", "pip install -e ."),
}

# Models with a working executor. Everything else in the registry is reported
# as pending rather than skipped in silence.
EXECUTORS = {
    "ZeroReturn": zero_return_forecast_fold,
    "LastReturn": last_return_forecast_fold,
    "RandomWalkWithDrift": rwd_forecast_fold,
    "AutoARIMA": autoarima_forecast_fold,
    "AutoETS": autoets_forecast_fold,
    # Volatility arm - a different target, so a different panel; see load_panels.
    "HAR-RV": har_forecast_fold,
    "Log-HAR": log_har_forecast_fold,
    "GJR-GARCH": gjr_garch_forecast_fold,
}

# Libraries that JIT-compile on first use. Warming one up before the clock
# starts keeps compilation out of fold 0's time; see warmup_statsforecast.
WARMUPS = {"statsforecast": warmup_statsforecast}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more tiers of the frozen model roster."
    )
    parser.add_argument("--tier", action="append", dest="tiers", choices=TIERS,
                        metavar="TIER",
                        help=f"tier to run; repeatable. One of: {', '.join(TIERS)}")
    parser.add_argument("--model", action="append", dest="models", metavar="NAME",
                        help="restrict to these registry names; repeatable. Combined "
                             "with --tier as an intersection.")
    parser.add_argument("--models-config", default=None,
                        help="frozen roster specification (default: configs/models.yaml)")
    parser.add_argument("--eval-config", default=None,
                        help="frozen evaluation specification (default: configs/eval.yaml)")
    parser.add_argument("--features", default=str(ROOT / "data" / "features.parquet"),
                        help="panel written by scripts/build_features.py")
    parser.add_argument("--bars", default=str(ROOT / "data" / "processed" / "panel.parquet"),
                        help="OHLCV panel, for the volatility arm's Yang-Zhang target")
    parser.add_argument("--out", default=str(ROOT / "forecasts"),
                        help="forecast root (default: forecasts/)")
    parser.add_argument("--timings", default=str(ROOT / "results" / "timings.json"),
                        help="wall-clock log (default: results/timings.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the configuration gate, print the roster, and stop")
    return parser.parse_args()


def is_volatility_model(spec: ModelSpec) -> bool:
    """Volatility-arm models forecast 21-day realised volatility, not returns.

    They take the same folds and the same quantile grid but a different panel,
    and they are never ranked against a return model.
    """
    return spec.params.get("target") == "volatility"


def library_available(library: str) -> tuple[bool, str]:
    module, how = LIBRARIES.get(library, (library, f"pip install {library}"))
    return importlib.util.find_spec(module) is not None, how


def resolve_roster(tiers: list[str] | None,
                   names: list[str] | None = None) -> list[ModelSpec]:
    """Select from the registry, which is the source of truth for the roster."""
    roster = REGISTRY if not tiers else [m for m in REGISTRY if m.tier in set(tiers)]
    if names:
        known = {m.name for m in REGISTRY}
        unknown = sorted(set(names) - known)
        if unknown:
            raise SystemExit(f"not in the registry: {unknown}\n  known: {sorted(known)}")
        roster = [m for m in roster if m.name in set(names)]
    return list(roster)


def partition(roster: list[ModelSpec]) -> tuple[list[ModelSpec], list[tuple[ModelSpec, str]]]:
    """Split the roster into what can run now and what cannot, with reasons."""
    runnable, skipped = [], []
    for spec in roster:
        available, how = library_available(spec.library)
        if excluded(spec):
            # Pre-registered, attempted, and excluded for a stated reason. It is
            # reported every run rather than dropped from the roster, so the
            # exclusion stays visible instead of becoming a silent absence.
            skipped.append((spec, f"EXCLUDED - {spec.params['excluded_reason']}"))
        elif not available:
            skipped.append((spec, f"library {spec.library!r} not installed - {how}"))
        elif spec.name not in EXECUTORS:
            skipped.append((spec, "no executor implemented yet"))
        else:
            runnable.append(spec)
    return runnable, skipped


def record_timing(path: Path, model: str, payload: dict) -> None:
    """Merge one model's wall-clock into results/timings.json.

    Merged rather than overwritten so tiers run on different days, on different
    machines, accumulate into one file. Axis 5 of the evaluation is compute cost,
    and it cannot be reconstructed after the fact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! {path} is not valid JSON; starting a fresh record")
    existing[model] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


def run_model(
    spec: ModelSpec,
    panel: pd.DataFrame,
    walk_forward: PurgedWalkForward,
    quantiles: tuple[float, ...],
    out_root: Path,
    timings_path: Path,
    warmed: set[str],
    horizon: int = 1,
) -> None:
    executor = EXECUTORS[spec.name]
    model_dir = out_root / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{spec.name}  ({spec.library}, {spec.scope})")

    # Warm-up first, and outside the fold clock. statsforecast compiles on first
    # use; folded into fold 0 it reads as 20s of model cost against 0.2s for
    # every fold after it, which would rank a random walk slower than AutoARIMA
    # on the compute axis. Charged once, to whichever model runs first in the
    # process, and reported on its own line.
    warmup_seconds = 0.0
    warmup_charged = spec.library not in warmed
    if warmup_charged and spec.library in WARMUPS:
        t0 = time.perf_counter()
        WARMUPS[spec.library]()
        warmup_seconds = time.perf_counter() - t0
        warmed.add(spec.library)
        print(f"  warm-up  {spec.library} JIT compile  {warmup_seconds:6.2f}s "
              "(excluded from fold times)")

    fold_times: dict[str, float] = {}
    rows_written = 0
    started = time.perf_counter()

    for split, train, test in walk_forward.iter_frames(panel, time_col="ds"):
        # iter_frames already runs this before yielding. Repeated here because
        # the guard is the point of the fold structure, and a future refactor
        # that loosens iter_frames should not be able to loosen the runner too.
        assert_no_overlap(
            train["ds"], test["ds"],
            horizon=walk_forward.horizon,
            embargo=walk_forward.embargo,
            calendar=pd.DatetimeIndex(panel["ds"].unique()).sort_values(),
        )

        t0 = time.perf_counter()
        forecasts = executor(
            panel=panel, train=train,
            test_start=split.test_start, test_end=split.test_end,
            quantiles=quantiles, target_col="y", horizon=horizon,
        )
        elapsed = time.perf_counter() - t0

        path = model_dir / f"fold_{split.fold}.parquet"
        forecasts.to_parquet(path, index=False)
        fold_times[str(split.fold)] = round(elapsed, 3)
        rows_written += len(forecasts)

        print(f"  fold {split.fold}  train<=|{split.train_end.date()}| "
              f"test=[{split.test_start.date()} .. {split.test_end.date()}]  "
              f"{len(train):>7,} train rows -> {len(forecasts):>6,} forecasts  "
              f"{elapsed:6.2f}s  {show(path)}")

    steady = sum(fold_times.values())
    record_timing(timings_path, spec.name, {
        "library": spec.library,
        "tier": spec.tier,
        "scope": spec.scope,
        "target": spec.params.get("target", "return"),
        "seconds_warmup": round(warmup_seconds, 3),
        "warmup_charged_here": warmup_charged,
        "seconds_per_fold": fold_times,
        "seconds_steady_state_total": round(steady, 3),
        "seconds_steady_state_mean": round(steady / len(fold_times), 3) if fold_times else None,
        "seconds_wall_clock_total": round(time.perf_counter() - started + warmup_seconds, 3),
        "n_forecasts": rows_written,
        "n_folds": len(fold_times),
        "warmup_note": (
            "seconds_warmup is one-off library JIT compilation, not model cost. It "
            "is charged to the first model that uses the library in a process; a "
            "later model in the same process shows 0.0 with warmup_charged_here "
            "false. Compare models on seconds_steady_state_total."
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"  {rows_written:,} forecasts | steady state {steady:.2f}s "
          f"({steady / len(fold_times):.2f}s/fold) | warm-up {warmup_seconds:.2f}s "
          f"-> {show(timings_path)}")


def main() -> None:
    args = parse_args()
    try:
        cfg = load_configs(
            args.models_config or ROOT / "configs" / "models.yaml",
            args.eval_config or ROOT / "configs" / "eval.yaml",
        )
    except ConfigError as exc:
        raise SystemExit(str(exc))

    print(f"models.yaml v{cfg.models['version']}  eval.yaml v{cfg.evaluation['version']}")
    print(f"quantile grid ({len(cfg.quantiles)} levels, training == evaluation): "
          f"{list(cfg.quantiles)}")
    print(f"point forecast: {cfg.evaluation['point_forecast_from']} ({cfg.median_column})")

    roster = resolve_roster(args.tiers, args.models)
    scope = ", ".join(args.tiers) if args.tiers else "all tiers"
    if args.models:
        scope += f" | models: {', '.join(args.models)}"
    if not roster:
        raise SystemExit(f"no models match {scope}")
    runnable, skipped = partition(roster)

    print(f"\nroster: {len(roster)} of {len(REGISTRY)} models ({scope})")
    print(f"  runnable now : {', '.join(m.name for m in runnable) or 'none'}")
    for spec, reason in skipped:
        print(f"  skip {spec.name:<20} {reason}")

    if args.dry_run:
        print("\nDry run - configuration gate passed, nothing executed.")
        return
    if not runnable:
        raise SystemExit("\nnothing to run")

    features = Path(args.features)
    if not features.exists():
        raise SystemExit(
            f"missing panel: {features}\n"
            "  run: python scripts/build_features.py --config configs/data.yaml"
        )
    panel = pd.read_parquet(features, columns=["unique_id", "ds", "target"])
    panel = panel.rename(columns={"target": "y"}).sort_values(["unique_id", "ds"])

    volatility_panel = None
    if any(is_volatility_model(spec) for spec in runnable):
        bars_path = Path(args.bars)
        if not bars_path.exists():
            raise SystemExit(
                f"missing OHLCV panel: {bars_path}\n"
                "  the volatility arm needs bars for the Yang-Zhang estimator\n"
                "  run: python scripts/build_dataset.py --config configs/data.yaml")
        bars = pd.read_parquet(bars_path,
                               columns=["unique_id", "ds", "open", "high", "low", "close"])
        volatility_panel = build_volatility_panel(bars)
        usable = volatility_panel["y"].notna().sum()
        print(f"volatility panel: {len(volatility_panel):,} rows, "
              f"{usable:,} with a {RV_WINDOW}-session Yang-Zhang target "
              f"[{volatility_panel['ds'].min().date()} .. "
              f"{volatility_panel['ds'].max().date()}]")

    data_cfg = load_data_config()
    # One scheme per arm. The folds, the embargo and the grid are shared; the
    # purge width is not, because the two arms' labels consume different amounts
    # of future data. See ExperimentConfig.horizon_for.
    walk_forwards = {
        arm: PurgedWalkForward(
            test_windows=cfg.test_windows,
            horizon=cfg.horizon_for(arm),
            embargo=cfg.embargo,
            train_start=data_cfg["splits"]["train"]["start"],
        )
        for arm in ("return", "volatility")
    }
    walk_forward = walk_forwards["return"]
    print(f"\npanel: {len(panel):,} rows, {panel['unique_id'].nunique()} tickers, "
          f"{panel['ds'].nunique():,} sessions "
          f"[{panel['ds'].min().date()} .. {panel['ds'].max().date()}]")
    for arm, wf in walk_forwards.items():
        print(f"folds ({arm:<10}): {len(cfg.test_windows)}, purge "
              f"{wf.horizon} + embargo {wf.embargo} = {wf.gap} sessions excised per fold")

    out_root, timings_path = Path(args.out), Path(args.timings)
    warmed: set[str] = set()
    for spec in runnable:
        arm = "volatility" if is_volatility_model(spec) else "return"
        target_panel = volatility_panel if arm == "volatility" else panel
        run_model(spec, target_panel, walk_forwards[arm], cfg.quantiles, out_root,
                  timings_path, warmed, horizon=cfg.horizon_for(arm))

    print("\nDone.")


if __name__ == "__main__":
    main()
