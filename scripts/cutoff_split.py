"""Score the foundation models either side of a pre-training cutoff.

A zero-shot claim on financial data is a claim about data the model has not
seen, and that is checkable only where the checkpoint documents when its
training data ends. Of the five foundation specs in this roster, exactly one
does: Kronos-small states June 2024 in its paper's Appendix D. Chronos-2,
TimesFM-2.5, TTM and Moirai-2.0 publish no cutoff at all, which the registry
already records as a finding rather than an inconvenience - without it, no
zero-shot comparison on recent data can be certified contamination-free.

So this script does two different things that must not be confused:

* For **Kronos-small** it computes a genuine clean-slice score. The split date
  is the model's own documented cutoff, so the post-window is data the model
  provably did not train on.

* For every other model it computes a **date sensitivity** on the same date.
  That date is not their cutoff and the post-window is not certified clean for
  them. It answers a narrower question: whether a model's advantage is stable
  across the sample, or concentrated in the earlier part of it.

WHY THE RATIO AND NOT THE LEVEL
-------------------------------
The post-cutoff window is not a random sample of the test period. It is 539
sessions of a single regime - overwhelmingly bull, low-to-mid volatility - so
absolute error falls for *every* model in it, including a constant zero
forecast. Comparing a model's pre- and post-window CRPS therefore measures the
regime far more strongly than it measures contamination.

What survives the regime is the ratio to the baseline. ZeroReturn is scored in
the same window on the same origins, so a model that is genuinely better than
the baseline by the same margin in both windows shows a flat ratio, whatever
the regime did to the levels. A ratio that is below 1 pre-cutoff and at 1
post-cutoff is the signature the contamination story predicts.

This does not prove contamination either way. A ratio that moves has a regime
explanation available; a ratio that holds could still hide a model that
memorised the earlier window and generalises well. It is evidence, reported as
evidence.

    python scripts/cutoff_split.py
    python scripts/cutoff_split.py --cutoff 2024-06-30 --model Chronos-2
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

from sp500bench.config import load_configs  # noqa: E402
from sp500bench.evaluation.stat_tests import diebold_mariano  # noqa: E402
from sp500bench.models.broken import BROKEN_MODELS  # noqa: E402
from sp500bench.models.registry import (BY_NAME, CLEAN_SLICE_VALID_FOR,  # noqa: E402
                                        runnable)

# Kronos states June 2024 in the paper's Appendix D. Used as the split for every
# model, but only *meaningful* as a cutoff for Kronos - see the module docstring.
DEFAULT_CUTOFF = "2024-06-30"
BASELINE = "ZeroReturn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--losses", default=str(ROOT / "results" / "losses.parquet"))
    p.add_argument("--metrics", default=str(ROOT / "results" / "metrics.parquet"))
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    p.add_argument("--model", action="append", dest="models", default=None)
    p.add_argument("--within-fold", type=int, default=None,
                   help="restrict to one fold, removing the fold confound: the "
                        "post window lies entirely inside fold 3, so a naive "
                        "pre/post split compares fold 3 against folds 0-2")
    p.add_argument("--out", default=str(ROOT / "results" / "cutoff_split.parquet"))
    return p.parse_args()


def verify_crps_reconstruction(losses: pd.DataFrame, metrics: pd.DataFrame,
                               cfg) -> None:
    """The CRPS rebuilt here must equal the one evaluate.py wrote.

    Metrics are computed in exactly one place in this repository, and this
    script computes one of them a second time in order to window it. That is
    the situation in which a table quietly stops agreeing with the test that
    supports it, so the reconstruction is checked against the artefact before
    any windowed number is produced, and the run stops if it disagrees.
    """
    pooled = metrics[(metrics.regime_kind == "all") & (metrics.fold.isna())]
    checked = 0
    for _, row in pooled.iterrows():
        if not np.isfinite(row["crps"]):
            continue
        block = losses[(losses.model == row["model"])
                       & (losses.loss_type.astype(str).str.startswith("pinball_"))]
        if block.empty:
            continue
        rebuilt = 2.0 * float(block["value"].mean())
        if not np.isclose(rebuilt, row["crps"], rtol=1e-9, atol=1e-12):
            raise SystemExit(
                f"CRPS reconstruction disagrees with metrics.parquet for "
                f"{row['model']}: rebuilt {rebuilt:.10f} vs written "
                f"{row['crps']:.10f}. This script would report windowed numbers "
                "that do not reconcile with Table 4.1.")
        checked += 1
    print(f"  CRPS reconstruction verified against metrics.parquet on "
          f"{checked} models (2 x mean pinball over the scored grid)\n")


def window_scores(losses: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """RMSE and CRPS per model, plus the daily cross-sectional MSE series."""
    rows = []
    for model in models:
        block = losses[losses.model == model]
        se = block[block.loss_type == "squared_error"]
        pin = block[block.loss_type.astype(str).str.startswith("pinball_")]
        if se.empty:
            continue
        rows.append({
            "model": model,
            "n_obs": len(se),
            "n_days": int(se["ds"].nunique()),
            "rmse": float(np.sqrt(se["value"].mean())),
            "crps": 2.0 * float(pin["value"].mean()) if not pin.empty else np.nan,
        })
    return pd.DataFrame(rows).set_index("model")


def daily_mse(losses: pd.DataFrame, model: str) -> pd.Series:
    """One row per session. Never the stacked panel - see CLAUDE.md."""
    block = losses[(losses.model == model) & (losses.loss_type == "squared_error")]
    return block.groupby("ds")["value"].mean().sort_index()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    cutoff = pd.Timestamp(args.cutoff)

    losses = pd.read_parquet(args.losses)
    metrics = pd.read_parquet(args.metrics)
    losses["loss_type"] = losses["loss_type"].astype(str)
    if args.within_fold is not None:
        losses = losses[losses.fold == args.within_fold]
        print(f"restricted to fold {args.within_fold}: same models, same "
              "training data, so a pre/post difference here is not a fold "
              "difference")

    print(f"cutoff: {cutoff.date()}  baseline: {BASELINE}")
    if args.within_fold is None:
        verify_crps_reconstruction(losses, metrics, cfg)
    else:
        # The check compares a rebuilt CRPS against the pooled cell in
        # metrics.parquet. Restricted to one fold the two are different
        # quantities and the check would refuse a run it has no view of, so it
        # is skipped rather than weakened - and the unrestricted run, which does
        # reconcile, is the one that certifies the reconstruction.
        print("  CRPS reconstruction check skipped: a single fold does not "
              "reconcile with the pooled cell.\n  Run without --within-fold "
              "for the reconciled figures.\n")

    return_models = [s.name for s in runnable()
                     if BY_NAME[s.name].params.get("target") != "volatility"
                     and s.name not in BROKEN_MODELS]
    present = [m for m in return_models if m in set(losses.model)]
    wanted = args.models or present

    pre = losses[losses.ds <= cutoff]
    post = losses[losses.ds > cutoff]
    print(f"pre-cutoff  {pre.ds.min().date()}..{pre.ds.max().date()}  "
          f"{pre.ds.nunique():,} sessions")
    print(f"post-cutoff {post.ds.min().date()}..{post.ds.max().date()}  "
          f"{post.ds.nunique():,} sessions")

    pre_s = window_scores(pre, present)
    post_s = window_scores(post, present)
    full_s = window_scores(losses, present)

    rows = []
    for model in wanted:
        if model not in pre_s.index:
            continue
        record = {"model": model,
                  "cutoff_documented": model in CLEAN_SLICE_VALID_FOR,
                  "declared_cutoff": BY_NAME[model].params.get("pretrain_cutoff")}
        for label, table in (("full", full_s), ("pre", pre_s), ("post", post_s)):
            base = table.loc[BASELINE]
            row = table.loc[model]
            record[f"{label}_n_obs"] = int(row["n_obs"])
            record[f"{label}_rmse"] = row["rmse"]
            record[f"{label}_crps"] = row["crps"]
            record[f"{label}_rmse_ratio"] = row["rmse"] / base["rmse"]
            record[f"{label}_crps_ratio"] = (row["crps"] / base["crps"]
                                             if np.isfinite(row["crps"]) else np.nan)
        # DM against the baseline, daily cross-sectional MSE, in each window.
        for label, frame in (("pre", pre), ("post", post)):
            a = daily_mse(frame, model)
            b = daily_mse(frame, BASELINE)
            common = a.index.intersection(b.index)
            if model == BASELINE or len(common) < 30:
                record[f"{label}_dm_stat"] = np.nan
                record[f"{label}_dm_p"] = np.nan
                continue
            stat, p = diebold_mariano(a.loc[common].to_numpy(),
                                      b.loc[common].to_numpy(),
                                      horizon=cfg.horizon_for("return"))
            record[f"{label}_dm_stat"] = stat
            record[f"{label}_dm_p"] = p
        rows.append(record)

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    def block(rows, title, rubric):
        if rows.empty:
            return
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        for line in rubric:
            print(line)
        head = (f"\n{'model':<22}{'CRPS/base':>11}{'pre':>9}{'post':>9}"
                f"{'shift':>9}{'RMSE/base':>11}{'pre':>9}{'post':>9}")
        print(head)
        print("-" * (len(head) - 1))
        for _, r in rows.iterrows():
            shift = r["post_crps_ratio"] - r["pre_crps_ratio"]
            crps = (f"{r['full_crps_ratio']:>11.4f}{r['pre_crps_ratio']:>9.4f}"
                    f"{r['post_crps_ratio']:>9.4f}{shift:>+9.4f}"
                    if np.isfinite(r["full_crps_ratio"]) else
                    f"{'n/a':>11}{'n/a':>9}{'n/a':>9}{'n/a':>9}")
            print(f"{r['model']:<22}{crps}"
                  f"{r['full_rmse_ratio']:>11.4f}{r['pre_rmse_ratio']:>9.4f}"
                  f"{r['post_rmse_ratio']:>9.4f}")

    # Two tables, not one with a flag column. A reader scanning a single table
    # will read every row the same way, and the difference between "provably
    # unseen" and "no cutoff is published" is the entire content of a zero-shot
    # claim. Separating them makes the two unmistakable at a glance.
    documented = table[table.cutoff_documented]
    undocumented = table[~table.cutoff_documented]

    block(documented,
          "TABLE A - CLEAN-SLICE TEST (the split date is the model's OWN documented cutoff)",
          ["The post window is data the checkpoint provably did not train on.",
           "This is a zero-shot test in the sense the term is normally used."])
    block(undocumented,
          "TABLE B - DATE SENSITIVITY ONLY (these checkpoints publish NO cutoff)",
          ["The post window is NOT certified contamination-free for any model here.",
           "The split date is Kronos's cutoff, borrowed as a common date. It answers",
           "only whether a model's advantage is stable across the sample, not whether",
           "the later window is unseen. Do not read these rows as a clean slice."])

    print("\nRatios are to ZeroReturn scored on the same origins. The post window is a")
    print("single regime - overwhelmingly bull, low-to-mid volatility - so absolute error")
    print("falls for every model in it including a constant zero forecast, and only the")
    print("ratio to a baseline scored on the same origins survives that. 'shift' is post")
    print("minus pre; positive means the advantage over the baseline shrinks after the")
    print("split, the direction a contamination story predicts.")
    print("\nA stable ratio does not prove the absence of contamination, and a moving one")
    print("has a regime explanation available. This is evidence, reported as evidence.")
    print("\nCONFOUND: the post window lies entirely inside fold 3, while the pre window")
    print("spans all four folds. A pre/post difference is therefore a fold difference")
    print("unless it also holds within fold 3 - run with --within-fold 3 to check.")

    print("\n" + "=" * 104)
    print("DIEBOLD-MARIANO vs ZeroReturn, DAILY CROSS-SECTIONAL MSE, PER WINDOW")
    print("=" * 104)
    head = (f"{'model':<22}{'pre DM':>10}{'pre p':>9}{'post DM':>10}"
            f"{'post p':>9}   verdict")
    print(head)
    print("-" * (len(head) + 20))
    for _, r in table.iterrows():
        if not np.isfinite(r["pre_dm_p"]):
            continue
        def verdict(p, stat):
            if not np.isfinite(p) or p >= 0.05:
                return "indistinguishable"
            return "better" if stat < 0 else "worse"
        print(f"{r['model']:<22}{r['pre_dm_stat']:>10.3f}{r['pre_dm_p']:>9.4f}"
              f"{r['post_dm_stat']:>10.3f}{r['post_dm_p']:>9.4f}   "
              f"{verdict(r['pre_dm_p'], r['pre_dm_stat'])} -> "
              f"{verdict(r['post_dm_p'], r['post_dm_stat'])}")
    print("\nNegative statistic favours the model over ZeroReturn. These are "
          "unadjusted per-window")
    print("p-values, reported to compare the two windows against each other; "
          "they are not a")
    print("hypothesis test of the study and carry no BH correction.")
    print(f"\n{len(table)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
