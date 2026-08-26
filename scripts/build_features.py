#!/usr/bin/env python
"""Build the causal feature matrix, the target, and the regime panel.

Step two of the pipeline. It reads what ``build_dataset.py`` wrote, applies the
feature construction and regime labelling already implemented (and already
tested for causality) in ``src/sp500bench/``, standardises on training
statistics alone, and writes the result.

This script adds no features of its own. The feature set is
:data:`sp500bench.features.build.FEATURE_COLUMNS`, frozen there, plus the macro
block whose publication lags were applied upstream. Anything new would have to
be added to that module and covered by ``tests/test_leakage.py`` first.

The two assertions at the end
-----------------------------
Both are adversarial, in the style of ``tests/test_leakage.py``: they construct
the situation in which a leak *would* be visible and require that it is not.

1. **No feature may correlate above 0.99 with the forward target.** Features are
   backward-looking and the target is forward-looking, so on daily equity
   returns every correlation should sit near zero. A value close to one means a
   feature is carrying its own label - the classic way a forecasting study
   reports a spectacular result that is really a bug.

2. **Scaler statistics must not move when the future is available, or corrupted.**
   Fitting on the training slice of the full panel must be bit-identical to
   fitting on a panel truncated at the training boundary, and identical again
   when every post-training row is corrupted. If any of those differ, test-period
   information reached the scaler.

What is *not* scaled
--------------------
The target. Metrics (RMSE, MASE, CRPS) and the backtest are all defined on raw
log returns, so scaling the target would require inverse-transforming every
forecast before evaluation - one more step at which a silent error changes a
headline number. It is left in its natural units.

Outputs
-------
``data/features.parquet``  long panel: unique_id, ds, target, features, macro
``data/regimes.parquet``   volatility and drawdown regime labels per session
``data/scalers.parquet``   per-ticker mean/std, so evaluation can prove the
                           statistics were fitted once, on training data only

Usage
-----
    python scripts/build_features.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.features.build import (  # noqa: E402
    FEATURE_COLUMNS,
    build_features,
    make_target,
)
from sp500bench.features.regimes import build_regime_panel  # noqa: E402

BARS = ["open", "high", "low", "close", "volume"]

# A backward-looking feature cannot track a forward-looking label this closely
# without carrying it. Well below 1.0 so a near-miss still trips the wire.
MAX_TARGET_CORRELATION = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal features, target and regime labels."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "data.yaml"),
        help="frozen data specification (default: configs/data.yaml)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    scaling = cfg["scaling"]
    if scaling["method"] != "standard":
        raise SystemExit(f"this script standardises; config asks for {scaling['method']!r}")
    if scaling["scope"] != "per_ticker":
        raise SystemExit(f"expected per_ticker scaling, config says {scaling['scope']!r}")
    if scaling["fit_on"] != "train_only":
        raise SystemExit(
            f"config sets scaling.fit_on to {scaling['fit_on']!r}. Refusing to fit "
            "scaler statistics on anything but the training split."
        )
    return cfg


def resolve_input(data_dir: Path, name: str) -> Path:
    """Find an upstream artefact, wherever build_dataset.py put it."""
    for candidate in (data_dir / "processed" / name, data_dir / name):
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"missing input: {name}\n"
        f"  looked in {data_dir / 'processed'} and {data_dir}\n"
        "  run: python scripts/build_dataset.py --config configs/data.yaml"
    )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_ticker_blocks(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Run the frozen feature builder over every ticker and stack the result.

    ``build_features`` and ``make_target`` are called per ticker and never
    across the panel, so no rolling window can straddle two names.
    """
    blocks = []
    for ticker, group in panel.groupby("unique_id", sort=True):
        bars = group.set_index("ds")[BARS].sort_index()
        block = build_features(bars)
        block["target"] = make_target(bars["close"], horizon=horizon)
        block.insert(0, "unique_id", ticker)
        blocks.append(block.reset_index())
    out = pd.concat(blocks, ignore_index=True)
    return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def attach_macro(features: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Left-join the macro block, which is already lagged and session-aligned.

    The lags were applied in ``build_dataset.py`` via ``apply_release_lag``
    before any forward fill. Nothing here re-times them; this is a join.
    """
    block = macro.copy()
    block.index.name = "ds"
    macro_columns = list(block.columns)
    merged = features.merge(block.reset_index(), on="ds", how="left", validate="many_to_one")
    return merged, macro_columns


# ---------------------------------------------------------------------------
# Scaling, fitted on training data alone
# ---------------------------------------------------------------------------

def fit_scalers(
    frame: pd.DataFrame,
    columns: list[str],
    train_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker mean and standard deviation, from training rows only.

    Population standard deviation (``ddof=0``), matching the convention of
    scikit-learn's ``StandardScaler`` so the numbers are comparable to anything
    produced downstream by a library.
    """
    train = frame.loc[frame["ds"] <= pd.Timestamp(train_end)]
    if train.empty:
        raise SystemExit(f"no rows on or before train_end {train_end}")
    grouped = train.groupby("unique_id", sort=True)[columns]
    return grouped.mean(), grouped.std(ddof=0)


def assert_scalers_ignore_the_future(
    frame: pd.DataFrame,
    columns: list[str],
    train_end: str,
    means: pd.DataFrame,
    stds: pd.DataFrame,
) -> None:
    """Two adversarial refits that must reproduce the statistics bit for bit."""
    truncated = frame.loc[frame["ds"] <= pd.Timestamp(train_end)].copy()
    t_means, t_stds = fit_scalers(truncated, columns, train_end)
    pd.testing.assert_frame_equal(means, t_means, check_exact=True,
                                  obj="means, full panel vs truncated panel")
    pd.testing.assert_frame_equal(stds, t_stds, check_exact=True,
                                  obj="stds, full panel vs truncated panel")

    corrupted = frame.copy()
    future = corrupted["ds"] > pd.Timestamp(train_end)
    corrupted.loc[future, columns] = corrupted.loc[future, columns] * 1_000.0 + 7.0
    c_means, c_stds = fit_scalers(corrupted, columns, train_end)
    pd.testing.assert_frame_equal(means, c_means, check_exact=True,
                                  obj="means after corrupting every post-train row")
    pd.testing.assert_frame_equal(stds, c_stds, check_exact=True,
                                  obj="stds after corrupting every post-train row")

    print(f"  refit on truncated panel      : bit-identical")
    print(f"  refit with corrupted future   : bit-identical")


def apply_scalers(
    frame: pd.DataFrame,
    columns: list[str],
    means: pd.DataFrame,
    stds: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Standardise in place, leaving degenerate columns shifted but unscaled."""
    ids = frame["unique_id"]
    mean_rows = means.reindex(ids)[columns].to_numpy()
    std_rows = stds.reindex(ids)[columns].to_numpy()

    degenerate = sorted({
        columns[j]
        for j in range(len(columns))
        if np.any(np.nan_to_num(std_rows[:, j], nan=1.0) == 0.0)
    })
    safe = np.where((std_rows == 0.0) | ~np.isfinite(std_rows), 1.0, std_rows)

    out = frame.copy()
    out[columns] = (frame[columns].to_numpy(dtype=float) - mean_rows) / safe
    return out, degenerate


def scaler_table(means: pd.DataFrame, stds: pd.DataFrame, train_end: str,
                 n_train: pd.Series) -> pd.DataFrame:
    """Long table of fitted statistics, for the evaluation layer to check."""
    rows = (
        means.stack().rename("mean").to_frame()
        .join(stds.stack().rename("std"))
        .reset_index()
    )
    rows.columns = ["unique_id", "column", "mean", "std"]
    rows["n_train_obs"] = rows["unique_id"].map(n_train).astype("int64")
    rows["train_end"] = pd.Timestamp(train_end)
    rows["ddof"] = 0
    return rows.sort_values(["unique_id", "column"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leak canary
# ---------------------------------------------------------------------------

def assert_no_feature_tracks_the_target(
    frame: pd.DataFrame, columns: list[str]
) -> pd.Series:
    """Every feature must be near-orthogonal to the forward return."""
    usable = frame.dropna(subset=["target"])
    corr = usable[columns].corrwith(usable["target"]).abs().sort_values(ascending=False)

    offenders = corr[corr > MAX_TARGET_CORRELATION]
    if not offenders.empty:
        lines = "\n".join(f"    {c}: |corr| = {v:.6f}" for c, v in offenders.items())
        raise SystemExit(
            f"\nfeature(s) correlate above {MAX_TARGET_CORRELATION} with the "
            f"forward target:\n{lines}\n\n"
            "  A backward-looking feature cannot track a forward-looking label\n"
            "  this closely without carrying it. Treat this as a leak in\n"
            "  src/sp500bench/features/build.py, not as a finding."
        )
    return corr


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_splits(frame: pd.DataFrame, splits: dict, feature_columns: list[str]) -> None:
    print(f"\n{'split':<18}{'rows':>10}{'sessions':>10}{'tickers':>9}"
          f"{'complete':>10}{'with target':>13}")
    print("-" * 70)
    for name, window in splits.items():
        seg = frame[(frame.ds >= window["start"]) & (frame.ds <= window["end"])]
        complete = seg.dropna(subset=feature_columns)
        print(f"{name:<18}{len(seg):>10,}{seg['ds'].nunique():>10,}"
              f"{seg['unique_id'].nunique():>9}{len(complete):>10,}"
              f"{int(seg['target'].notna().sum()):>13,}")
    print("-" * 70)
    print(f"{'all':<18}{len(frame):>10,}{frame['ds'].nunique():>10,}"
          f"{frame['unique_id'].nunique():>9}"
          f"{len(frame.dropna(subset=feature_columns)):>10,}"
          f"{int(frame['target'].notna().sum()):>13,}")


def report_nans(frame: pd.DataFrame, columns: list[str]) -> None:
    n = len(frame)
    print(f"\n{'column':<18}{'NaN':>10}{'% of rows':>12}   reason")
    print("-" * 70)
    for col in columns:
        missing = int(frame[col].isna().sum())
        note = ""
        if missing:
            first_valid = frame.groupby("unique_id")[col].apply(
                lambda s: s.isna().sum()
            )
            note = f"warm-up, up to {int(first_valid.max())} rows/ticker"
        print(f"{col:<18}{missing:>10,}{missing / n * 100:>11.2f}%   {note}")


def report_regimes(regimes: pd.DataFrame, splits: dict) -> None:
    print(f"\n{'split':<18}{'low_vol':>9}{'mid_vol':>9}{'high_vol':>9}"
          f"{'bull':>9}{'correction':>12}{'bear':>7}")
    print("-" * 74)
    for name, window in splits.items():
        seg = regimes.loc[window["start"] : window["end"]]
        v = seg["vol_regime"].value_counts()
        d = seg["direction_regime"].value_counts()
        print(f"{name:<18}{v.get('low_vol', 0):>9}{v.get('mid_vol', 0):>9}"
              f"{v.get('high_vol', 0):>9}{d.get('bull', 0):>9}"
              f"{d.get('correction', 0):>12}{d.get('bear', 0):>7}")


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    horizon = cfg["target"]["horizon"]
    splits = cfg["splits"]
    train_end = splits["train"]["end"]

    data_dir = ROOT / "data"
    panel = pd.read_parquet(resolve_input(data_dir, "panel.parquet"))
    macro = pd.read_parquet(resolve_input(data_dir, "macro.parquet"))
    index = pd.read_parquet(resolve_input(data_dir, "index.parquet"))

    print(f"Building features for {panel['unique_id'].nunique()} tickers, "
          f"horizon {horizon}, train ends {train_end}")

    print("\nFeatures")
    features = build_ticker_blocks(panel, horizon)
    print(f"  {len(features):,} rows x {len(FEATURE_COLUMNS)} features "
          "(FEATURE_COLUMNS, frozen in features/build.py)")

    features, macro_columns = attach_macro(features, macro)
    print(f"  joined {len(macro_columns)} macro columns, lagged upstream")

    scaled_columns = FEATURE_COLUMNS + macro_columns

    print("\nScaling (per ticker, training statistics only)")
    means, stds = fit_scalers(features, scaled_columns, train_end)
    n_train = (
        features.loc[features["ds"] <= pd.Timestamp(train_end)]
        .groupby("unique_id")
        .size()
    )
    print(f"  fitted {means.shape[0]} tickers x {means.shape[1]} columns "
          f"on {int(n_train.iloc[0]):,} training rows each")
    assert_scalers_ignore_the_future(features, scaled_columns, train_end, means, stds)

    features, degenerate = apply_scalers(features, scaled_columns, means, stds)
    if degenerate:
        print(f"  zero-variance in train, centred but not scaled: {degenerate}")
    print("  target left unscaled - metrics and the backtest use raw log returns")

    print("\nRegimes (VIX terciles fitted on train only)")
    regimes = build_regime_panel(macro["vix"], index["close"], train_end=train_end)
    thresholds = regimes.attrs["vol_thresholds"]
    regimes["vol_threshold_low"] = thresholds["low"]
    regimes["vol_threshold_high"] = thresholds["high"]
    print(f"  tercile cuts: low <= {thresholds['low']:.2f} < mid <= "
          f"{thresholds['high']:.2f} < high")

    print("\nLeak canary")
    corr = assert_no_feature_tracks_the_target(features, scaled_columns)
    print(f"  max |corr| with forward target: {corr.iloc[0]:.6f} ({corr.index[0]}) "
          f"- limit {MAX_TARGET_CORRELATION}")
    print("  top 5: " + ", ".join(f"{c}={v:.4f}" for c, v in corr.head(5).items()))

    print("\nWriting")
    scalers = scaler_table(means, stds, train_end, n_train)
    ordered = ["unique_id", "ds", "target", *FEATURE_COLUMNS, *macro_columns]
    features[ordered].to_parquet(data_dir / "features.parquet", index=False)
    regimes.to_parquet(data_dir / "regimes.parquet")
    scalers.to_parquet(data_dir / "scalers.parquet", index=False)
    for name, obj in (("features", features), ("regimes", regimes), ("scalers", scalers)):
        print(f"  data/{name}.parquet  ({len(obj):,} rows)")

    report_splits(features, splits, FEATURE_COLUMNS)
    report_nans(features, FEATURE_COLUMNS + macro_columns)
    report_regimes(regimes, splits)
    print("\nDone.")


if __name__ == "__main__":
    main()
