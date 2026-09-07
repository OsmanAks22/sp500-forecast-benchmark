"""Check a model's written forecasts before they are scored.

Two defects have reached the parquet stage in this study and neither announced
itself. DLinear's quantile columns arrived in scrambled order - 51.6% of
adjacent pairs inverted - and were scored into a CRPS six times everyone else's
before anyone looked at the ordering. Kronos can produce a non-finite predicted
close, because the decoder output is not clipped and denormalisation can put a
sampled price at or below zero; ``np.log`` then yields NaN, and ``np.quantile``
propagates rather than ignoring it, so a single bad path out of a hundred turns
that origin's entire row into nine NaNs.

Both are cheap to find here and expensive to find in a CRPS that looks odd.

    python scripts/validate_forecasts.py --model Kronos-small
    python scripts/validate_forecasts.py                 # every model on disk

What it reports, per model:

* non-finite quantiles - how many rows, which folds, which tickers, which
  dates, and whether they cluster or are scattered;
* non-monotone rows - a quantile grid must be non-decreasing across levels, and
  a row that is not has had its levels mixed up somewhere;
* the levels a model is *expected* not to emit, read from its ``grid.json``, so
  TTM's eight empty columns and TimesFM's four tail levels are not reported as
  defects.

Exit status is non-zero if anything is found, so it can gate a scoring run.
It reads only; nothing on disk is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.config import load_configs, quantile_column  # noqa: E402

TOP_N = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--forecasts", default=str(ROOT / "forecasts"))
    p.add_argument("--model", action="append", dest="models", default=None,
                   help="restrict to these models; repeatable")
    p.add_argument("--top", type=int, default=TOP_N,
                   help=f"how many offending tickers/dates to list (default {TOP_N})")
    return p.parse_args()


def expected_missing(model_dir: Path, grid: list[float]) -> set[str]:
    """Levels this model declares it cannot emit, from its own grid.json.

    Without this, TTM - which has no quantile head at all - would be reported as
    eight columns of defects, and TimesFM's four tail levels likewise. A model
    that declares an absence is not defective for having it.
    """
    path = model_dir / "grid.json"
    if not path.exists():
        return set()
    try:
        note = json.loads(path.read_text())
    except json.JSONDecodeError:
        return set()
    if note.get("scored_on_frozen_grid", True):
        declared = note.get("frozen_levels_unavailable") or []
    else:
        native = [float(q) for q in note.get("native_levels") or []]
        declared = ([q for q in grid if not any(abs(q - n) < 1e-12 for n in native)]
                    if native else [q for q in grid if abs(q - 0.5) > 1e-12])
        declared = note.get("frozen_levels_unavailable", declared)
    return {quantile_column(float(q)) for q in declared}


def load_model(root: Path, model: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"{model}/fold_*.parquet")):
        frame = pd.read_parquet(path)
        frame["fold"] = int(path.stem.split("_")[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def check_model(root: Path, model: str, grid: list[float], top: int) -> dict:
    frame = load_model(root, model)
    if frame.empty:
        return {"model": model, "status": "no forecasts on disk"}

    skip = expected_missing(root / model, grid)
    columns = [quantile_column(q) for q in grid]
    checked = [c for c in columns if c not in skip and c in frame.columns]
    values = frame[checked].to_numpy(dtype=float)

    finite = np.isfinite(values)
    bad_rows = ~finite.all(axis=1)
    # Monotone only where the whole row is finite: a NaN row is already
    # reported, and comparing against it would double-count the same origin.
    ordered = np.ones(len(frame), dtype=bool)
    if values.shape[1] > 1:
        clean = finite.all(axis=1)
        ordered[clean] = (np.diff(values[clean], axis=1) >= -1e-12).all(axis=1)
    crossings = int((np.diff(values[finite.all(axis=1)], axis=1) < -1e-12).sum())
    pairs = int(finite.all(axis=1).sum() * max(0, values.shape[1] - 1))

    result = {
        "model": model,
        "rows": len(frame),
        "levels_checked": checked,
        "levels_skipped": sorted(skip),
        "nonfinite_rows": int(bad_rows.sum()),
        "nonmonotone_rows": int((~ordered).sum()),
        "crossings": crossings,
        "crossing_pairs": pairs,
    }

    if bad_rows.any():
        offenders = frame.loc[bad_rows]
        result["nonfinite_by_fold"] = (offenders.groupby("fold").size()
                                       .sort_index().to_dict())
        result["nonfinite_by_ticker"] = (offenders.groupby("unique_id").size()
                                         .sort_values(ascending=False)
                                         .head(top).to_dict())
        dates = pd.DatetimeIndex(offenders["ds"])
        result["nonfinite_dates"] = {
            "distinct": int(dates.nunique()),
            "first": str(dates.min().date()),
            "last": str(dates.max().date()),
            "worst": {str(k.date()): int(v) for k, v in
                      offenders.groupby("ds").size()
                      .sort_values(ascending=False).head(top).items()},
        }
        # Concentration, reported as figures rather than as a label. One origin
        # failing on every ticker is a different fault from one ticker failing
        # on many dates, and a binary "clustered/scattered" call near the
        # threshold misleads exactly when the answer matters.
        per_date = offenders.groupby("ds").size()
        per_ticker = offenders.groupby("unique_id").size()
        n_tickers = frame["unique_id"].nunique()
        n_sessions = frame["ds"].nunique()
        result["nonfinite_shape"] = (
            f"{len(per_date)} of {n_sessions} sessions affected; worst session "
            f"hits {int(per_date.max())} of {n_tickers} tickers. "
            f"{len(per_ticker)} of {n_tickers} tickers affected; worst ticker "
            f"fails on {int(per_ticker.max())} sessions")

    if (~ordered).any():
        offenders = frame.loc[~ordered]
        result["nonmonotone_by_ticker"] = (offenders.groupby("unique_id").size()
                                           .sort_values(ascending=False)
                                           .head(top).to_dict())
    return result


def report(result: dict) -> bool:
    """Print one model's findings. Returns True if the model is clean."""
    model = result["model"]
    if result.get("status"):
        print(f"\n{model}: {result['status']}")
        return True

    bad = result["nonfinite_rows"] or result["nonmonotone_rows"]
    mark = "FAIL" if bad else "ok  "
    share = (100.0 * result["crossings"] / result["crossing_pairs"]
             if result["crossing_pairs"] else 0.0)
    print(f"\n{mark} {model}: {result['rows']:,} rows, "
          f"{len(result['levels_checked'])} levels checked"
          + (f", {len(result['levels_skipped'])} declared absent "
             f"({', '.join(result['levels_skipped'])})"
             if result["levels_skipped"] else ""))
    print(f"       non-finite rows {result['nonfinite_rows']:,} | "
          f"non-monotone rows {result['nonmonotone_rows']:,} | "
          f"crossings {result['crossings']:,} of "
          f"{result['crossing_pairs']:,} pairs ({share:.1f}%)")

    if result["nonfinite_rows"]:
        print(f"       shape: {result['nonfinite_shape']}")
        print(f"       by fold: {result.get('nonfinite_by_fold')}")
        dates = result["nonfinite_dates"]
        print(f"       {dates['distinct']} distinct session(s), "
              f"{dates['first']} .. {dates['last']}")
        print(f"       worst dates: {dates['worst']}")
        print(f"       worst tickers: {result.get('nonfinite_by_ticker')}")
        print("       NOTE: np.quantile propagates NaN, so one bad path in a "
              "sample destroys")
        print("             that origin's whole row - the sample is not "
              "silently truncated.")
        print("       evaluate.py refuses to score a model with any non-finite "
              "row on a")
        print("             level its grid.json declares reachable, so this is "
              "a hard stop, not")
        print("             a wrong number.")

    if result["nonmonotone_rows"]:
        print(f"       non-monotone tickers: {result.get('nonmonotone_by_ticker')}")
        print("       NOTE: a quantile row must be non-decreasing across "
              "levels. A row that is")
        print("             not has had its levels mixed up - this is the "
              "DLinear defect.")
    return not bad


def main() -> None:
    args = parse_args()
    root = Path(args.forecasts)
    grid = list(load_configs().quantiles)
    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())

    print(f"validating {len(models)} model(s) against the frozen grid {grid}")
    clean = True
    for model in models:
        clean &= report(check_model(root, model, grid, args.top))

    print("\n" + "=" * 78)
    if clean:
        print("all checked models are finite and monotone on the levels they "
              "declare they emit")
    else:
        print("at least one model has non-finite or non-monotone rows - see "
              "above. Scoring")
        print("these would put an unusable number in a table beside usable "
              "ones.")
    raise SystemExit(0 if clean else 1)


if __name__ == "__main__":
    main()
