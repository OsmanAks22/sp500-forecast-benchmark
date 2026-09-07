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

Models that cannot reach the whole grid
---------------------------------------
CRPS here is the pinball average over a grid of levels, so it is only comparable
between two models when both were scored on the *same* grid. Most of the roster
emits the frozen nine and the question does not arise. Two of the foundation
checkpoints cannot:

* **TimesFM-2.5** has a fixed nine-decile head, ``[0.1 .. 0.9]``, and the level
  set is not settable. Four frozen levels - 0.01, 0.05, 0.95, 0.99 - lie outside
  it. Its CRPS is computed over its own nine levels and every table marks it, so
  the number is never read against a model scored on the frozen nine.
* **TTM** has no quantile head at all and is absent from CRPS, PICP and interval
  score entirely.

Each model's reachable levels come from ``forecasts/{model}/grid.json``, written
by ``run_foundation.py``. A model with no such file is scored on the frozen grid,
which is what every existing tier does.

The absences are *asserted*, not inferred from arithmetic. An all-NaN column does
propagate NaN through ``crps_from_quantiles`` and would produce the right answer
on its own - but a *partially* NaN column does not: ``pinball_loss`` drops
non-finite pairs, so a model that failed on some origins would get a CRPS
averaged over a different set of observations at each level, silently and with
nothing raised. ``assert_grid_matches_data`` therefore checks that every level a
model declares reachable is finite everywhere and every level it declares
unreachable is NaN everywhere, and fails the run otherwise.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --n-boot 2000 --models Naive
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.config import (ConfigError, load_configs,  # noqa: E402
                               load_data_config, quantile_column)
from sp500bench.evaluation.metrics import (crps_from_quantiles,  # noqa: E402
                                           directional_accuracy,
                                           directional_accuracy_with_ci, mae,
                                           mean_interval_score, picp, rmse)
from sp500bench.evaluation.splits import PurgedWalkForward  # noqa: E402
from sp500bench.models.registry import BY_NAME, REGISTRY, runnable  # noqa: E402
from sp500bench.models.volatility_arm import build_volatility_panel  # noqa: E402

# Below this many distinct sessions the block bootstrap is resampling too few
# blocks to mean anything, so the interval is written as null rather than as a
# number that looks like evidence. The cell's point estimates still stand.
MIN_DAYS_FOR_CI = 10

STRATIFICATIONS = {"volatility": "vol_regime", "direction": "direction_regime"}

# The two levels PICP80 and the interval score are read off. A model that cannot
# reach both is absent from those columns for the same reason it is absent from
# CRPS.
BAND_LEVELS = (0.1, 0.9)


@dataclass(frozen=True)
class CrpsGrid:
    """The levels one model's CRPS is averaged over, and whether that is comparable.

    ``levels`` empty means the model emits no quantiles at all and is absent from
    every probabilistic column. ``comparable`` is false whenever ``levels`` is not
    the frozen grid, because a pinball average over a different set of levels is a
    different quantity and must never be ranked against one that is not.
    """

    label: str
    levels: tuple[float, ...]
    comparable: bool
    note: str = ""
    # Frozen levels the model *cannot* emit, which must therefore be NaN in the
    # file. Distinct from "not in ``levels``": a level can be absent from the
    # CRPS grid and still be legitimately present in the forecasts. TimesFM's
    # q0.25 and q0.75 are interpolated *inside* its [0.1, 0.9] support, so they
    # are real numbers and are written; only the four levels outside that
    # support are unavailable. Collapsing the two would either reject valid
    # interpolation or accept invented tails.
    unavailable: tuple[float, ...] = ()

    @property
    def columns(self) -> list[str]:
        return [quantile_column(q) for q in self.levels]

    @property
    def unavailable_columns(self) -> list[str]:
        return [quantile_column(q) for q in self.unavailable]

    @property
    def has_band(self) -> bool:
        return all(any(np.isclose(q, level, atol=1e-12) for level in self.levels)
                   for q in BAND_LEVELS)


def load_crps_grid(root: Path, model: str, cfg) -> CrpsGrid:
    """The model's reachable levels, from its own ``grid.json`` where it has one.

    Absent - which is every tier written by ``run_experiment.py`` - means the
    model emits the frozen grid, which is the whole point of the frozen grid.
    """
    frozen = tuple(cfg.quantiles)
    path = root / model / "grid.json"
    if not path.exists():
        return CrpsGrid(f"frozen-{len(frozen)}", frozen, True)

    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")

    if spec.get("scored_on_frozen_grid", True):
        return CrpsGrid(f"frozen-{len(frozen)}", frozen, True, spec.get("notes", ""))

    note = spec.get("notes", "")
    levels = tuple(float(q) for q in spec.get("native_levels", []))
    if not levels:
        # No quantile head: every level but the point forecast must be empty.
        unavailable = tuple(q for q in frozen if q != 0.5)
        return CrpsGrid("none", (), False, note, unavailable)

    declared = spec.get("frozen_levels_unavailable")
    if declared is None:
        # Older or hand-written grid.json: anything outside the native support
        # could only have been reached by extrapolation, so it must be absent.
        low, high = min(levels), max(levels)
        declared = [q for q in frozen if q < low or q > high]
    label = f"{model.lower().split('-')[0]}-{len(levels)}"
    return CrpsGrid(label, levels, False, note,
                    tuple(float(q) for q in declared))


def assert_grid_matches_data(model: str, scored: pd.DataFrame, grid: CrpsGrid,
                             cfg) -> None:
    """The declared grid must be what is actually in the file.

    Two directions, both of which have silently wrong answers if unchecked.

    A level declared *reachable* must be finite on every row. ``pinball_loss``
    drops non-finite pairs, so a column that is NaN on some origins - a failed
    batch, a truncated context - yields a pinball average over a different set of
    observations than its neighbours, and the CRPS built from it is a mixture of
    sample sizes rather than a score. Nothing raises on its own; this does.

    A level declared *unavailable* must be NaN on every row. If a model that is
    supposed to be absent from CRPS ever starts emitting a band, the absence stops
    being a property of the model and becomes a bug in this file, and it should
    fail loudly rather than quietly produce a number.

    A level that is neither - present in the file but not part of this model's
    CRPS grid, which is what TimesFM's interpolated q0.25 and q0.75 are - is held
    only to the all-or-nothing rule, because ``build_losses`` still writes a
    pinball row for it.
    """
    n_rows = len(scored)
    reachable = set(grid.columns)
    unavailable = set(grid.unavailable_columns)
    present = [c for c in cfg.quantile_columns if c in scored.columns]

    for column in grid.columns:
        if column not in scored.columns:
            raise SystemExit(
                f"{model}: grid.json declares {column} reachable but the column "
                f"is not in the forecasts")
        n_missing = int((~np.isfinite(scored[column].to_numpy(dtype=float))).sum())
        if n_missing:
            raise SystemExit(
                f"{model}: {column} is declared reachable but is NaN on "
                f"{n_missing:,} of {n_rows:,} rows. pinball_loss drops "
                "non-finite pairs, so this would produce a CRPS averaged over a "
                "different observation set at each level rather than an error.")

    for column in present:
        if column not in unavailable:
            continue
        finite = int(np.isfinite(scored[column].to_numpy(dtype=float)).sum())
        if finite:
            raise SystemExit(
                f"{model}: {column} is declared unavailable (grid '{grid.label}') "
                f"but is finite on {finite:,} rows. Either the checkpoint now "
                "emits this level and grid.json is stale, or the runner "
                "extrapolated a level it should have left empty.")

    for column in present:
        if column in reachable or column in unavailable:
            continue
        finite = int(np.isfinite(scored[column].to_numpy(dtype=float)).sum())
        if finite not in (0, n_rows):
            raise SystemExit(
                f"{model}: {column} is neither in the CRPS grid nor declared "
                f"unavailable, and is finite on {finite:,} of {n_rows:,} rows. "
                "A level must be present everywhere or nowhere; a partial column "
                "produces a pinball loss over a shifting observation set.")


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
                 arm: str = "return", grid: CrpsGrid | None = None) -> dict:
    """Every metric named in configs/eval.yaml, for one cell.

    ``grid`` names the levels this model's CRPS is averaged over. It defaults to
    the frozen grid, which is what every model that emits it gets.
    """
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

    # CRPS over this model's own reachable levels, and only if it has any. A
    # model with no quantile head is absent from the probabilistic columns by
    # decision, written NaN here rather than left to NaN arithmetic further down.
    if grid is None:
        grid = CrpsGrid(f"frozen-{len(cfg.quantiles)}", tuple(cfg.quantiles), True)
    if grid.levels:
        quantile_forecasts = {q: cell[c].to_numpy(dtype=float)
                              for q, c in zip(grid.levels, grid.columns)}
        crps_value = crps_from_quantiles(y, quantile_forecasts)
    else:
        crps_value = float("nan")

    if grid.has_band:
        lower = cell[quantile_column(BAND_LEVELS[0])].to_numpy()
        upper = cell[quantile_column(BAND_LEVELS[1])].to_numpy()
        picp_value = picp(y, lower, upper)
        interval_value = mean_interval_score(y, lower, upper, alpha=0.2)
    else:
        picp_value = float("nan")
        interval_value = float("nan")

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
        "crps": crps_value,
        "picp_80": picp_value,
        "mean_interval_score_80": interval_value,
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
               n_boot: int, grid: CrpsGrid) -> list[dict]:
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
                # Carried per row so a consumer reading metrics.parquet alone
                # cannot pool a partial-grid CRPS with a frozen-grid one.
                "crps_grid": grid.label,
                "crps_comparable": grid.comparable,
                **cell_metrics(cell, cfg, denominators, n_boot, arm, grid),
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


def _show(path: Path) -> str:
    """Repo-relative where possible. ``relative_to`` raises on a path outside the
    repo or on a relative one, which crashed the run *after* both files had been
    written - the work was done and the summary line killed it."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


PENDING = "[PENDING]"


def roster_status(root: Path, folds: list[int]) -> tuple[list[str], list[str]]:
    """Which runnable specs have forecasts on disk, and which do not."""
    expected = [spec.name for spec in runnable()]
    present, pending = [], []
    for name in expected:
        (present if not load_forecasts(root, name, folds).empty
         else pending).append(name)
    return present, pending


def print_roster_caption(present: list[str], pending: list[str]) -> None:
    """The roster every table above was computed on.

    A table is only meaningful beside the set of models it ranks, and a table
    copied into a chapter loses that context unless it travels with it. Missing
    models are named rather than omitted, because a reader cannot tell an absent
    model from one that was never in the study.
    """
    total = len(present) + len(pending)
    print(f"\nROSTER: computed on {len(present)} of {total} runnable models.")
    print(f"  present ({len(present)}): {', '.join(present)}")
    if not pending:
        print("  no models pending - this is the complete runnable roster.")
        return
    print(f"  {PENDING} ({len(pending)}): {', '.join(pending)}")
    print("  These have no forecasts on disk, so no cell was computed for them.")
    print(f"  Their rows read {PENDING} and never a number. Rankings, set")
    print("  membership and any 'best model' claim are provisional until they land.")


def print_pending_rows(pending: list[str], arm_of_model, arm: str,
                       label_width: int, name_width: int,
                       n_metrics: int, metric_width: int = 9) -> None:
    """One row per absent model, carrying a token that cannot be read as data."""
    rows = [m for m in pending if arm_of_model(m) == arm]
    if not rows:
        return
    print()
    for model in sorted(rows):
        cells = "".join(f"{PENDING:>{metric_width}}" for _ in range(n_metrics))
        print(f"{'':<{label_width}}{model:<{name_width}}{cells}")


def _display_name(model: str, grids: dict[str, CrpsGrid]) -> str:
    """A model scored on a partial grid carries a mark everywhere it is printed.

    The mark is on the *model*, not on the CRPS cell, because the reason it is
    not comparable is a property of the checkpoint rather than of one number.
    """
    grid = grids.get(model)
    return f"{model} *" if grid is not None and not grid.comparable else model


def print_grid_notes(grids: dict[str, CrpsGrid], cfg) -> None:
    """Why each marked model is not directly comparable on CRPS.

    The reasoning is carried verbatim from the model's own ``grid.json`` rather
    than restated here, so the write-up, the table footnote and the runner that
    produced the forecasts cannot drift into three different explanations.
    """
    marked = {m: g for m, g in grids.items() if not g.comparable}
    if not marked:
        return

    print("\n" + "=" * 92)
    print("* NOT DIRECTLY COMPARABLE ON CRPS")
    print("=" * 92)
    print(f"CRPS is a pinball average over a grid, so it is comparable only "
          f"between models scored\non the same levels. Unmarked models use the "
          f"frozen grid: {list(cfg.quantiles)}.\n")
    for model, grid in sorted(marked.items()):
        levels = list(grid.levels) if grid.levels else "none - no quantile head"
        print(f"  {model}")
        print(f"    grid   {grid.label}: {levels}")
        if not grid.levels:
            print("    CRPS, PICP80 and the interval score are absent for this model -")
            print("    not zero, and not imputed.")
        for line in _wrap(grid.note, width=84):
            print(f"    {line}")
        print()


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return []
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def print_volatility_arm(table: pd.DataFrame, grids: dict[str, CrpsGrid],
                         reference: str = "Log-HAR",
                         pending: list[str] | None = None) -> None:
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
            print(f"{label:<26}{_display_name(r['model'], grids):<12}"
                  f"{r['n_obs']:>8,}{r['mase']:>8.3f}"
                  f"{r['rmse']:>10.5f}{r['rmse'] / ref_rmse:>8.2f}"
                  f"{r['crps']:>10.5f}{r['crps'] / ref_crps:>8.2f}"
                  f"{r['picp_80'] * 100:>8.1f}%{marker}")
            label = ""

    print_pending_rows(pending or [], arm_of, "volatility", 26, 12, 7, 9)
    print("\nRatios are model / reference; below 1.00 beats the reference.")
    print("Directional accuracy is omitted: realised volatility is positive, so the")
    print("sign always matches and the statistic is identically 1.")
    print("The target is the FORWARD window: Yang-Zhang over sessions t+1 .. t+21,")
    print("forecast from information through t. This arm is purged at horizon 21,")
    print("not the return arm's 1, so its training windows end 26 sessions before")
    print("each test window rather than 6.")


def print_comparison(table: pd.DataFrame, grids: dict[str, CrpsGrid],
                     pending: list[str] | None = None) -> None:
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
        print(f"{(label if label != previous else ''):<26}"
              f"{_display_name(r['model'], grids):<22}"
              f"{r['n_obs']:>8,}{r['mase']:>8.3f}{r['directional_accuracy'] * 100:>7.2f}%"
              f"{r['rmse']:>9.5f}{r['crps']:>10.5f}{r['picp_80'] * 100:>7.1f}%")
        previous = label
    print_pending_rows(pending or [], arm_of, "return", 26, 22, 6, 9)
    print("\nPICP80 is the share of outcomes inside the q0.1-q0.9 band; 80% is nominal.")
    print("* marks a model not scored on the frozen grid; its CRPS is not directly")
    print("  comparable to the unmarked rows. See the footnote below. nan means the")
    print("  model emits no quantiles and is absent from the column.")

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
              f"{(r['regime'] if key != previous else ''):<13}"
              f"{_display_name(r['model'], grids):<22}"
              f"{r['n_obs']:>8,}{r['n_days']:>7,}{r['mase']:>8.3f}{da:>8}"
              f"{r['rmse']:>9.5f}{r['crps']:>10.5f}")
        previous = key
    print("\n* not scored on the frozen grid; CRPS not directly comparable.")


def main() -> None:
    args = parse_args()
    try:
        cfg = load_configs()
    except ConfigError as exc:
        raise SystemExit(str(exc))

    panel = (pd.read_parquet(args.features, columns=["unique_id", "ds", "target"])
             .rename(columns={"target": "y"}).sort_values(["unique_id", "ds"]))
    regimes = pd.read_parquet(args.regimes)[["vol_regime", "direction_regime"]]

    # One scheme per arm: the volatility arm's label consumes 21 sessions of
    # future data against the return arm's 1, so its training windows end
    # earlier and its MASE denominators come from a different span.
    calendar = pd.DatetimeIndex(panel["ds"].unique()).sort_values()
    walk_forwards, splits_by_arm = {}, {}
    for arm in ("return", "volatility"):
        wf = PurgedWalkForward(
            test_windows=cfg.test_windows, horizon=cfg.horizon_for(arm),
            embargo=cfg.embargo,
            train_start=load_data_config()["splits"]["train"]["start"])
        walk_forwards[arm] = wf
        splits_by_arm[arm] = wf.splits(calendar)
    walk_forward = walk_forwards["return"]
    splits = splits_by_arm["return"]
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
        panels["volatility"] = vol_panel.dropna(subset=["y"])
        denominators["volatility"] = scale_denominators(
            panels["volatility"], walk_forwards["volatility"],
            splits_by_arm["volatility"])

    print(f"quantile grid: {list(cfg.quantiles)}  point forecast: {cfg.median_column}")
    print(f"bootstrap: {args.n_boot:,} replications, clustered on dates\n")

    rows, loss_frames, grids = [], [], {}
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

        grid = load_crps_grid(forecast_root, model, cfg)
        assert_grid_matches_data(model, scored, grid, cfg)
        grids[model] = grid

        suffix = "" if grid.comparable else f"  [CRPS grid '{grid.label}', marked *]"
        print(f"  {model:<22} {arm:<11} {len(scored):>7,} forecasts over "
              f"{len(folds)} folds{suffix}")
        rows.extend(build_rows(model, scored, cfg, denominators[arm], folds,
                               args.n_boot, grid))
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

    present, pending = roster_status(forecast_root, folds)
    print_comparison(table, grids, pending)
    print_volatility_arm(table, grids, pending=pending)
    print_grid_notes(grids, cfg)
    print_roster_caption(present, pending)
    print(f"\n{len(table)} cells -> {_show(out)}")
    print(f"{len(losses):,} per-observation losses "
          f"({losses['loss_type'].nunique()} types x {losses['model'].nunique()} models) "
          f"-> {_show(losses_out)}")


if __name__ == "__main__":
    main()
