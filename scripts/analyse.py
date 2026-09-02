#!/usr/bin/env python
"""Statistical inference over the scored forecasts.

Step five. Reads ``results/losses.parquet`` - the per-observation losses written
by ``evaluate.py`` - and runs the three procedures the design commits to:

* **Diebold-Mariano** with the Harvey-Leybourne-Newbold small-sample correction,
  every model against the H1 reference on squared error;
* **Benjamini-Hochberg** FDR control across each family of comparisons;
* the **Model Confidence Set**, per arm and per regime cell.

Everything is computed from one table, so the numbers here cannot drift from the
ones in ``metrics.parquet``.

The aggregation that makes any of this valid
---------------------------------------------
Every test below runs on the **daily cross-sectional mean squared error**: one
row per trading day, the average across the 50 tickers. Not the stacked
ticker-day panel.

The reason is the same one the whole design is built around. Fifty S&P 500
constituents share a market factor, so on any given day their loss differentials
move together. Feeding 83,450 ticker-days to a test that assumes independent
draws treats 50 correlated observations as 50 independent ones, and the standard
error comes out roughly ``sqrt(50)`` too small. That is not a subtle bias: it
manufactures significance. The directional-accuracy bootstrap already showed the
size of it on this panel - a variance inflation of about 2x even after
collapsing to days.

Collapsing to a daily series first removes the cross-sectional dependence by
construction; the moving-block bootstrap inside the MCS then handles what serial
dependence remains. ``mcs.model_confidence_set`` says the same in its own
docstring, and DM has exactly the same exposure, so it gets the same treatment.

For transparency the *stacked* DM p-value is reported alongside as
``dm_p_stacked`` - never used for a decision, only to show how much the naive
version overstates the evidence.

Families for FDR
----------------
Benjamini-Hochberg is applied within a family, and the family has to be named or
the correction means nothing. Three are used, recorded in ``bh_family``:
``h1_pooled`` (each model against the reference over the whole test period),
``h1_per_fold``, and ``h3_volatility_regime``. They are corrected separately
because they answer different questions.

Usage
-----
    python scripts/analyse.py
    python scripts/analyse.py --alpha-mcs 0.10 --n-boot 5000
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

from sp500bench.config import ConfigError, load_configs  # noqa: E402
from sp500bench.evaluation.mcs import model_confidence_set  # noqa: E402
from sp500bench.evaluation.stat_tests import (benjamini_hochberg,  # noqa: E402
                                              diebold_mariano)

H1_REFERENCE = "ZeroReturn"
VOLATILITY_REFERENCE = "Log-HAR"

# The MCS needs enough independent-ish periods for the block bootstrap to mean
# anything; mcs.model_confidence_set refuses below 30 itself.
MIN_DAYS_FOR_MCS = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DM tests, BH-FDR and the MCS.")
    p.add_argument("--losses", default=str(ROOT / "results" / "losses.parquet"))
    p.add_argument("--regimes", default=str(ROOT / "data" / "regimes.parquet"))
    p.add_argument("--out", default=str(ROOT / "results" / "inference.parquet"))
    p.add_argument("--alpha-mcs", type=float, default=0.10,
                   help="MCS elimination level; the literature uses 0.10")
    p.add_argument("--alpha-dm", type=float, default=0.05,
                   help="BH-FDR level for the DM family")
    p.add_argument("--n-boot", type=int, default=5_000)
    return p.parse_args()


def daily_mse(losses: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional mean squared error: rows are days, columns models.

    This is the collapse that makes the inference valid - see the module
    docstring. One row per trading day per arm.
    """
    squared = losses[losses["loss_type"] == "squared_error"]
    wide = (squared.groupby(["arm", "ds", "model"])["value"].mean()
            .unstack("model").sort_index())
    return wide


def dm_against_reference(wide: pd.DataFrame, reference: str, horizon: int,
                         stacked: pd.DataFrame | None = None) -> list[dict]:
    """Diebold-Mariano for every model against the reference, on one cell."""
    if reference not in wide.columns:
        return []
    rows = []
    reference_loss = wide[reference].to_numpy()
    for model in wide.columns:
        if model == reference:
            continue
        both = wide[[model, reference]].dropna()
        if len(both) < MIN_DAYS_FOR_MCS:
            continue
        stat, p = diebold_mariano(both[model].to_numpy(), both[reference].to_numpy(),
                                  horizon=horizon, harvey_correction=True)
        row = {
            "model": model, "reference": reference,
            "n_days": len(both),
            "dm_stat": stat, "dm_p": p,
            "mean_loss": float(both[model].mean()),
            "reference_loss": float(both[reference].mean()),
            "loss_ratio": float(both[model].mean() / both[reference].mean()),
        }
        if stacked is not None:
            pair = stacked[[model, reference]].dropna()
            _, p_stacked = diebold_mariano(pair[model].to_numpy(),
                                           pair[reference].to_numpy(),
                                           horizon=horizon, harvey_correction=True)
            row["dm_p_stacked"] = p_stacked
            row["n_stacked"] = len(pair)
        rows.append(row)
    _ = reference_loss
    return rows


def run_mcs(wide: pd.DataFrame, alpha: float, n_boot: int) -> dict | None:
    """MCS on a daily loss frame, or None if the cell is too thin."""
    clean = wide.dropna(axis=0, how="any")
    if len(clean) < MIN_DAYS_FOR_MCS or clean.shape[1] < 2:
        return None
    result = model_confidence_set(clean, alpha=alpha, n_boot=n_boot)
    return {
        "included": result.included,
        "excluded": result.excluded,
        "p_values": result.p_values,
        "n_obs": result.n_obs,
        "block_size": result.block_size,
    }


def apply_bh(rows: list[dict], family: str, alpha: float) -> list[dict]:
    """Benjamini-Hochberg within one named family."""
    if not rows:
        return rows
    p = np.array([r["dm_p"] for r in rows], dtype=float)
    finite = np.isfinite(p)
    reject = np.zeros(len(p), dtype=bool)
    adjusted = np.full(len(p), np.nan)
    if finite.any():
        r, q = benjamini_hochberg(p[finite], alpha=alpha)
        reject[finite] = r
        adjusted[finite] = q
    for row, rj, q in zip(rows, reject, adjusted):
        row["bh_family"] = family
        row["dm_p_bh"] = float(q)
        row["dm_reject_bh"] = bool(rj)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_dm(rows: list[dict], title: str, reference: str) -> None:
    if not rows:
        return
    print(f"\n{title}")
    head = (f"  {'model':<22}{'days':>6}{'loss ratio':>12}{'DM stat':>10}"
            f"{'p':>9}{'p (BH)':>9}{'p stacked':>11}  verdict")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in sorted(rows, key=lambda x: x["loss_ratio"]):
        better = r["dm_stat"] < 0
        if r["dm_reject_bh"]:
            verdict = "BEATS ref" if better else "worse than ref"
        else:
            verdict = "indistinguishable"
        print(f"  {r['model']:<22}{r['n_days']:>6,}{r['loss_ratio']:>12.4f}"
              f"{r['dm_stat']:>10.3f}{r['dm_p']:>9.4f}{r['dm_p_bh']:>9.4f}"
              f"{r.get('dm_p_stacked', float('nan')):>11.4f}  {verdict}")
    print(f"  reference: {reference}. Loss ratio < 1 means lower squared error.")
    print("  'p stacked' ignores cross-sectional dependence and is shown only to")
    print("  size the inflation; no decision uses it.")


def print_mcs(result: dict | None, title: str, reference: str,
              reference_label: str = "reference") -> None:
    print(f"\n{title}")
    if result is None:
        print("  too few sessions for the block bootstrap; not run")
        return
    included, excluded = result["included"], result["excluded"]
    print(f"  {result['n_obs']} sessions, block size {result['block_size']}")
    print(f"  {'model':<22}{'MCS p':>9}   in set")
    print("  " + "-" * 40)
    for model, p in sorted(result["p_values"].items(), key=lambda kv: -kv[1]):
        mark = "  yes" if model in included else "  no"
        star = f"  <- {reference_label}" if model == reference else ""
        print(f"  {model:<22}{p:>9.4f}{mark}{star}")
    if reference in excluded:
        print(f"  ** {reference} is EXCLUDED - the set is a strict subset without it")
    elif reference in included:
        print(f"  {reference} survives: the data do not separate it from the best")


def main() -> None:
    args = parse_args()
    try:
        cfg = load_configs()
    except ConfigError as exc:
        raise SystemExit(str(exc))

    losses_path = Path(args.losses)
    if not losses_path.exists():
        raise SystemExit(f"missing {losses_path}\n  run: python scripts/evaluate.py")
    losses = pd.read_parquet(losses_path)
    regimes = pd.read_parquet(args.regimes)[["vol_regime", "direction_regime"]]

    squared = losses[losses["loss_type"] == "squared_error"].copy()
    wide_all = daily_mse(losses)
    stacked_all = {
        arm: (group.pivot_table(index=["unique_id", "ds"], columns="model",
                                values="value"))
        for arm, group in squared.groupby("arm", observed=True)
    }

    print(f"quantile grid: {list(cfg.quantiles)}")
    print(f"MCS alpha {args.alpha_mcs}, DM/BH alpha {args.alpha_dm}, "
          f"{args.n_boot:,} bootstrap replications")
    print("all tests on DAILY cross-sectional mean squared error "
          "(one row per session)")

    records: list[dict] = []
    fold_by_day = squared[["arm", "ds", "fold"]].drop_duplicates()

    for arm in sorted(wide_all.index.get_level_values("arm").unique()):
        # The unstack spans every model in the file, so the other arm's columns
        # come back as all-NaN for these rows. Drop them: the MCS and the DM
        # pairing both dropna(how="any"), and an all-NaN column would empty the
        # frame and silently report "too few sessions".
        wide = wide_all.loc[arm].dropna(axis=1, how="all")
        reference = H1_REFERENCE if arm == "return" else VOLATILITY_REFERENCE
        reference_label = "H1 reference" if arm == "return" else "arm reference"
        horizon = cfg.horizon_for(arm)
        stacked = stacked_all.get(arm)

        print("\n" + "=" * 92)
        print(f"{arm.upper()} ARM - reference {reference}, purge horizon {horizon}")
        print("=" * 92)

        # ---- DM pooled ----
        pooled = apply_bh(dm_against_reference(wide, reference, horizon, stacked),
                          f"{arm}_pooled", args.alpha_dm)
        print_dm(pooled, "Diebold-Mariano, pooled test period", reference)
        for r in pooled:
            records.append({"test": "dm", "arm": arm, "scope": "pooled",
                            "fold": pd.NA, "regime_kind": "all", "regime": "all", **r})

        # ---- DM per fold ----
        per_fold: list[dict] = []
        folds = sorted(fold_by_day[fold_by_day.arm == arm]["fold"].unique())
        for fold in folds:
            days = fold_by_day[(fold_by_day.arm == arm)
                               & (fold_by_day.fold == fold)]["ds"]
            cell = wide.loc[wide.index.isin(days)]
            for r in dm_against_reference(cell, reference, horizon):
                per_fold.append({**r, "fold": int(fold)})
        apply_bh(per_fold, f"{arm}_per_fold", args.alpha_dm)
        for fold in folds:
            rows = [r for r in per_fold if r["fold"] == fold]
            print_dm(rows, f"Diebold-Mariano, fold {fold}", reference)
        for r in per_fold:
            records.append({"test": "dm", "arm": arm, "scope": f"fold_{r['fold']}",
                            "regime_kind": "all", "regime": "all", **r})

        # ---- MCS pooled ----
        result = run_mcs(wide, args.alpha_mcs, args.n_boot)
        print_mcs(result, "Model Confidence Set, pooled test period", reference,
                  reference_label)
        if result:
            for model, p in result["p_values"].items():
                records.append({"test": "mcs", "arm": arm, "scope": "pooled",
                                "fold": pd.NA, "regime_kind": "all", "regime": "all",
                                "model": model, "reference": reference,
                                "mcs_p": p, "in_mcs": model in result["included"],
                                "n_days": result["n_obs"]})

        # ---- MCS per volatility regime: where H3's inferential claim sits ----
        labelled = wide.join(regimes, how="left")
        for regime in ("high_vol", "mid_vol", "low_vol"):
            cell = labelled[labelled["vol_regime"] == regime][wide.columns]
            result = run_mcs(cell, args.alpha_mcs, args.n_boot)
            note = "" if regime != "low_vol" else "  (reported, not inferential)"
            print_mcs(result, f"Model Confidence Set, {regime}{note}", reference,
                      reference_label)
            if result:
                for model, p in result["p_values"].items():
                    records.append({"test": "mcs", "arm": arm, "scope": "regime",
                                    "fold": pd.NA, "regime_kind": "volatility",
                                    "regime": regime, "model": model,
                                    "reference": reference, "mcs_p": p,
                                    "in_mcs": model in result["included"],
                                    "n_days": result["n_obs"]})

        # ---- Drawdown axis: descriptive only, per the H3 scope note ----
        for regime in ("bear",):
            cell = labelled[labelled["direction_regime"] == regime][wide.columns]
            days = len(cell.dropna(how="any"))
            print(f"\nDrawdown axis, {regime}: {days} sessions, two episodes.")
            print("  No test run. H3's directional axis is descriptive only - see the")
            print("  scope note in the README. Mean daily MSE, for reference:")
            for model in sorted(cell.columns):
                print(f"    {model:<22}{cell[model].mean():.3e}")
                records.append({"test": "descriptive", "arm": arm, "scope": "regime",
                                "fold": pd.NA, "regime_kind": "direction",
                                "regime": regime, "model": model,
                                "reference": reference, "mean_loss": float(cell[model].mean()),
                                "n_days": days})

    table = pd.DataFrame(records)
    for column in ("fold", "n_days", "n_stacked"):
        if column in table:
            table[column] = table[column].astype("Int64")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"\n{len(table)} inference rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
