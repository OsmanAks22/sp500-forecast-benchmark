#!/usr/bin/env python
"""Score the forecasts and write the results table the tests read from.

Step four of the pipeline. It reads ``forecasts/{model}/fold_{k}.parquet``, joins
the outcome and the regime labels, and writes two tables:

``results/metrics.parquet``  one row per model x fold x regime cell
``results/losses.parquet``   one row per model x ticker x session x loss type

Why a table rather than numbers on a terminal
---------------------------------------------
The Diebold-Mariano tests, the Benjamini-Hochberg correction across the roster
and the Model Confidence Set all need the same cells, and they need them to be
the *same* cells - computed once, from one join, with one convention for flat
days and one weighting for MASE. Recomputing them per consumer is how a table in
a write-up ends up disagreeing with the test that supposedly supports it. This
script is the only place a metric is computed; everything downstream reads the
parquet.

The cells
---------
``regime_kind`` is ``all``, ``volatility`` or ``direction``; ``regime`` is the
label inside that stratification. ``fold`` is the fold index, or null for the
pooled test period. Cells with no sessions are written out with ``n_obs = 0``
rather than omitted - an empty cell is a fact about the design (H4's clean slice
has no bear days at all), and a missing row is easy to mistake for an oversight.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --n-boot 2000 --models Naive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.config import ConfigError, load_configs, load_data_config  # noqa: E402
from sp500bench.evaluation.metrics import (crps_from_quantiles,  # noqa: E402
                                           directional_accuracy,
                                           directional_accuracy_with_ci, mae,
                                           mean_interval_score, picp, rmse)
from sp500bench.evaluation.splits import PurgedWalkForward  # noqa: E402
from sp500bench.models.registry import BY_NAME, REGISTRY  # noqa: E402
from sp500bench.models.volatility_arm import build_volatility_panel  # noqa: E402

# Below this many distinct sessions the block bootstrap is resampling too few
# blocks to mean anything, so the interval is written as null rather than as a
# number that looks like evidence. The cell's point estimates still stand.
MIN_DAYS_FOR_CI = 10

STRATIFICATIONS = {"volatility": "vol_regime", "direction": "direction_regime"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score forecasts into results/metrics.parquet.")
    p.add_argument("--forecasts", default=str(ROOT / "forecasts"))
    p.add_argument("--features", default=str(ROOT / "data" / "features.parquet"))
    p.add_argument("--regimes", default=str(ROOT / "data" / "regimes.parquet"))
    p.add_argument("--bars", default=str(ROOT / "data" / "processed" / "panel.parquet"),
                   help="OHLCV panel, for the volatility arm's Yang-Zhang target")
    p.add_argument("--out", default=str(ROOT / "results" / "metrics.parquet"))
    p.add_argument("--losses-out", default=str(ROOT / "results" / "losses.parquet"))
    p.add_argument("--models", action="append", default=None,
                   help="restrict to these models; repeatable (default: all found)")
    p.add_argument("--n-boot", type=int, default=5_000,
                   help="bootstrap replications for the directional-accuracy CI")
    return p.parse_args()


def arm_of(model: str) -> str:
    """Which target a model forecasts: the return arm or the volatility arm.

    The two arms are scored against different panels and are never ranked
    against each other - an RMSE on 1-day log returns and an RMSE on 21-day
    realised volatility are not the same quantity and do not belong in one
    ordering. The ``arm`` column exists so a downstream consumer cannot pool
    them by accident.
    """
    spec = BY_NAME.get(model)
    if spec is not None and spec.params.get("target") == "volatility":
        return "volatility"
    return "return"


def load_forecasts(root: Path, model: str, folds: list[int]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        path = root / model / f"fold_{fold}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["fold"] = fold
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def scale_denominators(panel: pd.DataFrame, walk_forward: PurgedWalkForward,
                       splits) -> dict[tuple[int, str], tuple[float, float]]:
    """Two in-sample scaling denominators per (fold, ticker), from training only.

    ``(mase_denominator, mad_denominator)``:

    * ``mase_denominator`` is ``mean |y_t - y_{t-1}|``, the in-sample error of a
      seasonal-naive forecast, which is what standard MASE divides by.
    * ``mad_denominator`` is ``mean |y_t|``, the in-sample mean absolute level.

    Both come from the same purged window the model was fitted on. Using one
    global denominator would rescale every fold by the same number and hide the
    regime effect H3 asks about.

    Why two. On a *return* target MASE is easy to misread. The target is already
    the first difference of log prices, so the MASE denominator differences it a
    second time and is inflated by ``sqrt(2(1-rho))`` relative to the level of
    the series. The zero forecast - the actual random-walk benchmark - therefore
    scores a MASE near ``1/sqrt(2) ~ 0.707`` rather than 1.0, and a model at 0.77
    is *not* beating the benchmark by 23%. The MAD-scaled error has no-skill 1.0
    on this target, so it can be read directly. Report both.
    """
    out: dict[tuple[int, str], tuple[float, float]] = {}
    for split in splits:
        train = panel[(panel["ds"] >= walk_forward.train_start)
                      & (panel["ds"] <= split.train_end)]
        for ticker, group in train.groupby("unique_id", sort=True):
            values = group.sort_values("ds")["y"].to_numpy()
            values = values[np.isfinite(values)]
            diff = np.abs(np.diff(values))
            out[(split.fold, ticker)] = (
                float(np.mean(diff)) if diff.size else float("nan"),
                float(np.mean(np.abs(values))) if values.size else float("nan"),
            )
    return out


def cell_metrics(cell: pd.DataFrame, cfg, denominators, n_boot: int,
                 arm: str = "return") -> dict:
    """Every metric named in configs/eval.yaml, for one cell."""
    empty = {k: float("nan") for k in
             ("rmse", "mae", "mase", "scaled_error_mad",
              "directional_accuracy", "da_ci_low", "da_ci_high",
              "da_se_naive", "da_se_clustered", "da_variance_inflation",
              "da_min_detectable_effect", "da_n_obs", "crps", "picp_80",
              "mean_interval_score_80")}
    if cell.empty:
        return {**empty, "n_obs": 0, "n_days": 0, "n_tickers": 0,
                "window_start": pd.NaT, "window_end": pd.NaT}

    y = cell["y"].to_numpy(dtype=float)
    point = cell[cfg.median_column].to_numpy(dtype=float)

    # Both scaled errors: per (fold, ticker), then observation-weighted.
    weights, scaled, scaled_mad = [], [], []
    for (fold, ticker), group in cell.groupby(["fold", "unique_id"], sort=True):
        naive_denom, mad_denom = denominators.get(
            (int(fold), ticker), (float("nan"), float("nan")))
        if not np.isfinite(naive_denom) or naive_denom == 0:
            continue
        group_mae = mae(group["y"].to_numpy(), group[cfg.median_column].to_numpy())
        weights.append(len(group))
        scaled.append(group_mae / naive_denom)
        scaled_mad.append(group_mae / mad_denom
                          if np.isfinite(mad_denom) and mad_denom else float("nan"))
    mase_value = (float(np.average(scaled, weights=weights)) if scaled else float("nan"))
    mad_value = (float(np.average(scaled_mad, weights=weights)) if scaled_mad
                 else float("nan"))

    n_days = int(cell["ds"].nunique())
    if arm == "volatility":
        # Realised volatility is positive by construction, so the sign of the
        # forecast always matches the sign of the outcome and directional
        # accuracy is identically 1. Left null rather than written as a perfect
        # score that means nothing.
        da = {}
    elif n_days >= MIN_DAYS_FOR_CI:
        da = directional_accuracy_with_ci(y, point, cell["ds"], n_boot=n_boot)
    else:
        # Too few sessions for the block bootstrap to say anything; the point
        # estimate stands and the interval columns stay null.
        da = {"estimate": directional_accuracy(y, point)}

    quantile_forecasts = {q: cell[c].to_numpy(dtype=float)
                          for q, c in zip(cfg.quantiles, cfg.quantile_columns)}
    lower, upper = cell["q0.1"].to_numpy(), cell["q0.9"].to_numpy()

    return {
        "n_obs": len(cell),
        "n_days": n_days,
        "n_tickers": int(cell["unique_id"].nunique()),
        "window_start": cell["ds"].min(),
        "window_end": cell["ds"].max(),
        "rmse": rmse(y, point),
        "mae": mae(y, point),
        "mase": mase_value,
        "scaled_error_mad": mad_value,
        "directional_accuracy": da.get("estimate", float("nan")),
        "da_ci_low": da.get("ci_low", float("nan")),
        "da_ci_high": da.get("ci_high", float("nan")),
        "da_se_naive": da.get("se_naive", float("nan")),
        "da_se_clustered": da.get("se_clustered", float("nan")),
        "da_variance_inflation": da.get("variance_inflation", float("nan")),
        "da_min_detectable_effect": da.get("min_detectable_effect", float("nan")),
        "da_n_obs": da.get("n_obs", float("nan")),
        "crps": crps_from_quantiles(y, quantile_forecasts),
        "picp_80": picp(y, lower, upper),
        "mean_interval_score_80": mean_interval_score(y, lower, upper, alpha=0.2),
    }


def build_losses(model: str, scored: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per-observation losses in long form, for the DM tests and the MCS.

    One row per model x ticker x session x loss type. ``squared_error`` and
    ``absolute_error`` feed the point-forecast comparisons; the ``pinball_q*``
    rows feed CRPS, whose cell value is twice their mean over the grid; ``hit``
    feeds directional accuracy.

    ``hit`` is null on sessions whose outcome is exactly zero. That is the same
    convention :func:`~sp500bench.evaluation.metrics.directional_hits` applies,
    and writing null rather than 0.0 is what keeps a downstream mean from
    silently scoring flat days as misses.
    """
    y = scored["y"].to_numpy(dtype=float)
    point = scored[cfg.median_column].to_numpy(dtype=float)

    losses = {
        "squared_error": (y - point) ** 2,
        "absolute_error": np.abs(y - point),
    }
    for q, column in zip(cfg.quantiles, cfg.quantile_columns):
        delta = y - scored[column].to_numpy(dtype=float)
        losses[f"pinball_{column}"] = np.maximum(q * delta, (q - 1.0) * delta)

    hit = np.where(np.sign(y) == np.sign(point), 1.0, 0.0)
    hit[y == 0] = np.nan
    losses["hit"] = hit

    keys = scored[["unique_id", "ds", "fold"]]
    frames = []
    for loss_type, values in losses.items():
        frame = keys.copy()
        frame["loss_type"] = loss_type
        frame["value"] = values
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "arm", arm_of(model))
    out.insert(0, "model", model)
    out["loss_type"] = out["loss_type"].astype("category")
    return out


def build_rows(model: str, scored: pd.DataFrame, cfg, denominators, folds,
               n_boot: int) -> list[dict]:
    arm = arm_of(model)
    rows = []
    scopes = [(None, scored)] + [(f, scored[scored["fold"] == f]) for f in folds]

    for fold, frame in scopes:
        cells = [("all", "all", frame)]
        for kind, column in STRATIFICATIONS.items():
            for label in sorted(scored[column].dropna().unique()):
                cells.append((kind, label, frame[frame[column] == label]))
        for kind, label, cell in cells:
            rows.append({
                "model": model,
                "arm": arm,
                "fold": pd.NA if fold is None else int(fold),
                "regime_kind": kind,
                "regime": label,
                **cell_metrics(cell, cfg, denominators, n_boot, arm),
            })
    return rows


def _window_label(row) -> str:
    if pd.isna(row["fold"]):
        return "TEST PERIOD (pooled)"
    return f"fold {int(row['fold'])}  {row['window_start'].date()}"


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Folds in order, pooled last."""
    frame = frame.copy()
    frame["_order"] = frame["fold"].astype("float").fillna(np.inf)
    return frame.sort_values(["_order", "model"]).drop(columns="_order")


def print_volatility_arm(table: pd.DataFrame, reference: str = "Log-HAR") -> None:
    """The volatility arm, read against Log-HAR.

    Log-HAR is the reference because it is the benchmark the literature says
    foundation models fail to beat (Brini, 2026), so it is the number every
    later entrant on this arm has to clear. Ratios below 1.00 on RMSE and CRPS
    beat it; PICP80 is read against the nominal 80%, not against the reference.
    """
    arm = _ordered(table[(table.arm == "volatility") & (table.regime_kind == "all")])
    if arm.empty:
        return

    print("\n" + "=" * 96)
    print(f"VOLATILITY ARM - 21-session Yang-Zhang realised volatility, "
          f"read against {reference}")
    print("=" * 96)
    head = (f"{'window':<26}{'model':<12}{'n':>8}{'MASE':>8}{'RMSE':>10}{'vs ref':>8}"
            f"{'CRPS':>10}{'vs ref':>8}{'PICP80':>9}")
    print(head)
    print("-" * len(head))

    order = arm["fold"].astype("float").fillna(np.inf)
    for position, (_, block) in enumerate(arm.groupby(order, sort=True)):
        if position:
            print()
        label = _window_label(block.iloc[0])
        ref = block[block.model == reference]
        ref_rmse = float(ref["rmse"].iloc[0]) if len(ref) else float("nan")
        ref_crps = float(ref["crps"].iloc[0]) if len(ref) else float("nan")
        for _, r in block.sort_values("model").iterrows():
            marker = "  <- ref" if r["model"] == reference else ""
            print(f"{label:<26}{r['model']:<12}{r['n_obs']:>8,}{r['mase']:>8.3f}"
                  f"{r['rmse']:>10.5f}{r['rmse'] / ref_rmse:>8.2f}"
                  f"{r['crps']:>10.5f}{r['crps'] / ref_crps:>8.2f}"
                  f"{r['picp_80'] * 100:>8.1f}%{marker}")
            label = ""

    print("\nRatios are model / reference; below 1.00 beats the reference.")
    print("Directional accuracy is omitted: realised volatility is positive, so the")
    print("sign always matches and the statistic is identically 1.")
    print("The target is a TRAILING 21-session window, so 20 of its 21 sessions are")
    print("already observed at forecast time. Errors are correspondingly small -")
    print("read the arm as a ranking, not as evidence that volatility is easy.")


def print_comparison(table: pd.DataFrame) -> None:
    table = table[table.arm == "return"]
    if table.empty:
        return
    overall = _ordered(table[table.regime_kind == "all"])

    print("\n" + "=" * 92)
    print("BASELINE COMPARISON - test period, aggregate and per fold")
    print("=" * 92)
    head = (f"{'window':<26}{'model':<22}{'n':>8}{'MASE':>8}{'DA':>8}"
            f"{'RMSE':>9}{'CRPS':>10}{'PICP80':>8}")
    print(head)
    print("-" * len(head))
    previous = object()
    for _, r in overall.iterrows():
        label = _window_label(r)
        if label != previous and previous is not object():
            print()
        print(f"{(label if label != previous else ''):<26}{r['model']:<22}"
              f"{r['n_obs']:>8,}{r['mase']:>8.3f}{r['directional_accuracy'] * 100:>7.2f}%"
              f"{r['rmse']:>9.5f}{r['crps']:>10.5f}{r['picp_80'] * 100:>7.1f}%")
        previous = label
    print("\nPICP80 is the share of outcomes inside the q0.1-q0.9 band; 80% is nominal.")

    print("\n" + "=" * 92)
    print("DIRECTIONAL ACCURACY - date-clustered moving-block bootstrap")
    print("=" * 92)
    h2 = (f"{'window':<26}{'model':<22}{'DA':>8}{'95% CI (clustered)':>22}"
          f"{'se_clust':>10}{'infl':>7}")
    print(h2)
    print("-" * len(h2))
    previous = object()
    for _, r in overall.iterrows():
        label = _window_label(r)
        if label != previous and previous is not object():
            print()
        ci = f"[{r['da_ci_low'] * 100:.2f}%, {r['da_ci_high'] * 100:.2f}%]"
        print(f"{(label if label != previous else ''):<26}{r['model']:<22}"
              f"{r['directional_accuracy'] * 100:>7.2f}%{ci:>22}"
              f"{r['da_se_clustered'] * 100:>9.3f}%{r['da_variance_inflation']:>7.1f}")
        previous = label
    print("\n50% is the no-skill line. infl is the ratio of the clustered standard")
    print("error to the naive one - how much the naive SE overstates the evidence.")

    print("\n" + "=" * 92)
    print("REGIME STRATIFICATION - pooled test period")
    print("=" * 92)
    strat = table[(table.fold.isna()) & (table.regime_kind != "all")]
    h3 = (f"{'stratum':<12}{'regime':<13}{'model':<22}{'n':>8}{'days':>7}"
          f"{'MASE':>8}{'DA':>8}{'RMSE':>9}{'CRPS':>10}")
    print(h3)
    print("-" * len(h3))
    previous = None
    for _, r in strat.sort_values(["regime_kind", "regime", "model"]).iterrows():
        key = (r["regime_kind"], r["regime"])
        if previous is not None and key != previous:
            print()
        da = ("    n/a" if not np.isfinite(r["directional_accuracy"])
              else f"{r['directional_accuracy'] * 100:6.2f}%")
        print(f"{(r['regime_kind'] if key != previous else ''):<12}"
              f"{(r['regime'] if key != previous else ''):<13}{r['model']:<22}"
              f"{r['n_obs']:>8,}{r['n_days']:>7,}{r['mase']:>8.3f}{da:>8}"
              f"{r['rmse']:>9.5f}{r['crps']:>10.5f}")
        previous = key


def main() -> None:
    args = parse_args()
    try:
        cfg = load_configs()
    except ConfigError as exc:
        raise SystemExit(str(exc))

    panel = (pd.read_parquet(args.features, columns=["unique_id", "ds", "target"])
             .rename(columns={"target": "y"}).sort_values(["unique_id", "ds"]))
    regimes = pd.read_parquet(args.regimes)[["vol_regime", "direction_regime"]]

    walk_forward = PurgedWalkForward(
        test_windows=cfg.test_windows, horizon=cfg.horizon, embargo=cfg.embargo,
        train_start=load_data_config()["splits"]["train"]["start"])
    splits = walk_forward.splits(pd.DatetimeIndex(panel["ds"].unique()).sort_values())
    folds = [s.fold for s in splits]

    forecast_root = Path(args.forecasts)
    wanted = args.models or [m.name for m in REGISTRY]
    present = [m for m in wanted if not load_forecasts(forecast_root, m, folds).empty]

    # Two arms, two targets, two panels. Built only if that arm has forecasts.
    panels = {"return": panel}
    denominators = {"return": scale_denominators(panel, walk_forward, splits)}
    if any(arm_of(m) == "volatility" for m in present):
        bars = pd.read_parquet(args.bars,
                               columns=["unique_id", "ds", "open", "high", "low", "close"])
        vol_panel = build_volatility_panel(bars)[["unique_id", "ds", "y"]]
        panels["volatility"] = vol_panel
        denominators["volatility"] = scale_denominators(vol_panel, walk_forward, splits)

    print(f"quantile grid: {list(cfg.quantiles)}  point forecast: {cfg.median_column}")
    print(f"bootstrap: {args.n_boot:,} replications, clustered on dates\n")

    rows, loss_frames = [], []
    for model in present:
        arm = arm_of(model)
        forecasts = load_forecasts(forecast_root, model, folds)
        scored = (forecasts
                  .merge(panels[arm], on=["unique_id", "ds"], how="left",
                         validate="one_to_one")
                  .merge(regimes, left_on="ds", right_index=True, how="left"))
        missing = int(scored["y"].isna().sum())
        if missing:
            raise SystemExit(f"{model}: {missing} forecasts have no outcome to score against")
        print(f"  {model:<22} {arm:<11} {len(scored):>7,} forecasts over {len(folds)} folds")
        rows.extend(build_rows(model, scored, cfg, denominators[arm], folds, args.n_boot))
        loss_frames.append(build_losses(model, scored, cfg))

    if not rows:
        raise SystemExit(
            f"no forecasts found under {forecast_root}\n"
            "  run: python scripts/run_experiment.py --tier baseline")

    table = pd.DataFrame(rows)
    table["fold"] = table["fold"].astype("Int64")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    losses = pd.concat(loss_frames, ignore_index=True)
    losses["fold"] = losses["fold"].astype("int16")
    losses_out = Path(args.losses_out)
    losses.to_parquet(losses_out, index=False)

    print_comparison(table)
    print_volatility_arm(table)
    print(f"\n{len(table)} cells -> {out.relative_to(ROOT)}")
    print(f"{len(losses):,} per-observation losses "
          f"({losses['loss_type'].nunique()} types x {losses['model'].nunique()} models) "
          f"-> {losses_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
