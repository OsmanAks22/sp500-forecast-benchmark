"""Step five: turn forecasts into a traded strategy and price the edge.

H1's second falsification criterion is economic, not statistical: a model earns
its keep only if the cost at which its edge disappears is above 5 bps round
trip. H2 and H4 rest on the same simulation. This is the script that produces
it; the simulator itself lives in ``sp500bench.backtest.strategy`` and is
tested there.

    python scripts/backtest.py
    python scripts/backtest.py --model Chronos-2 --model ZeroReturn

Writes ``results/backtest.parquet``: one row per model x ticker x cost level,
plus one row per model x cost level holding the cross-sectional aggregate, and
the buy-and-hold benchmark.


WHAT IS SIMULATED, AND WHAT IS NOT
----------------------------------
The signal is the median forecast, which is the point forecast this study
reports; ``long_flat`` from ``configs/eval.yaml`` takes a position when it is
positive and stands aside otherwise. Long-flat rather than long-short because
there is no borrow model here and assuming free shorting would flatter every
result.

The simulation runs **per ticker** and the summary is aggregated across the
fifty afterwards. That is a deliberate limitation rather than an oversight. The
engine simulates one series, and building a portfolio-level equity curve would
mean re-implementing its cost convention in this file - a second copy of the
arithmetic that the H1 criterion is written in, which is exactly the kind of
duplication that lets a table disagree with the test that supports it. So the
per-ticker figures are the ones computed, and the aggregate is stated as what
it is: the cross-section of fifty independent single-name strategies, not a
diversified portfolio. A portfolio Sharpe would be higher than the mean of
these; the break-even cost, which is what H1 turns on, is a per-name ratio and
aggregates honestly.

``break_even_round_trip_bps`` is the number the criterion reads. It is the
round-trip cost at which the net edge reaches zero, in the same unit as
``cost_bps`` in ``configs/eval.yaml`` and as the criterion in the README.

The deflated Sharpe ratio is given the number of models tried, so its hurdle
accounts for having selected the best of a roster rather than tested one
strategy in isolation.
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

from sp500bench.backtest.strategy import (positions_from_forecast,  # noqa: E402
                                          run_backtest)
from sp500bench.config import ConfigError, load_configs  # noqa: E402
from sp500bench.models.broken import BROKEN_MODELS  # noqa: E402
from sp500bench.models.registry import BY_NAME, runnable  # noqa: E402

PENDING = "[PENDING]"
BROKEN = "[BROKEN]"

# Fields eval.yaml asks the backtest to report, in the order the tables use.
REPORTED = ["turnover_ann", "break_even_round_trip_bps", "sharpe_net",
            "deflated_sharpe_prob", "max_drawdown"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--forecasts", default=str(ROOT / "forecasts"))
    p.add_argument("--features", default=str(ROOT / "data" / "features.parquet"))
    p.add_argument("--out", default=str(ROOT / "results" / "backtest.parquet"))
    p.add_argument("--model", action="append", dest="models", default=None)
    return p.parse_args()


def arm_of(model: str) -> str:
    spec = BY_NAME.get(model)
    if spec is not None and spec.params.get("target") == "volatility":
        return "volatility"
    return "return"


def load_forecasts(root: Path, model: str) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(root.glob(f"{model}/fold_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def backtest_model(scored: pd.DataFrame, model: str, median_column: str,
                   mode: str, cost_levels: list[float], n_trials: int) -> list[dict]:
    """Every ticker, every cost level, through the tested simulator."""
    rows = []
    for cost in cost_levels:
        per_ticker = []
        for ticker, group in scored.groupby("unique_id", sort=True):
            group = group.sort_values("ds")
            signal = group[median_column].to_numpy(dtype=float)
            realised = pd.Series(group["y"].to_numpy(dtype=float),
                                 index=pd.DatetimeIndex(group["ds"]))
            positions = positions_from_forecast(
                pd.Series(signal, index=realised.index), mode=mode)
            summary = run_backtest(positions, realised, cost_bps=cost,
                                   n_trials=n_trials)
            summary.update({"model": model, "unique_id": ticker,
                            "cost_bps": float(cost), "scope": "ticker"})
            rows.append(summary)
            per_ticker.append(summary)

        frame = pd.DataFrame(per_ticker)
        aggregate = {"model": model, "unique_id": "ALL", "cost_bps": float(cost),
                     "scope": "cross_section", "n_tickers": len(frame),
                     "cost_convention": "round_trip"}
        for field in REPORTED + ["net_return_ann", "gross_return_ann",
                                 "volatility_ann", "hit_rate"]:
            values = frame[field].replace([np.inf, -np.inf], np.nan)
            aggregate[field] = float(values.mean())
            aggregate[f"{field}_median"] = float(values.median())
        # The criterion is a threshold, so the share of names that clear it is
        # more informative than the average of a ratio with a heavy tail.
        breakeven = frame["break_even_round_trip_bps"].replace(
            [np.inf, -np.inf], np.nan)
        aggregate["share_break_even_above_5bps"] = float((breakeven > 5.0).mean())
        aggregate["share_break_even_above_0bps"] = float((breakeven > 0.0).mean())
        rows.append(aggregate)
    return rows


def buy_and_hold(scored: pd.DataFrame, cost_levels: list[float],
                 n_trials: int) -> list[dict]:
    """The benchmark named in configs/eval.yaml: always long, never trades."""
    rows = []
    for cost in cost_levels:
        per_ticker = []
        for ticker, group in scored.groupby("unique_id", sort=True):
            group = group.sort_values("ds")
            realised = pd.Series(group["y"].to_numpy(dtype=float),
                                 index=pd.DatetimeIndex(group["ds"]))
            positions = pd.Series(1.0, index=realised.index)
            summary = run_backtest(positions, realised, cost_bps=cost,
                                   n_trials=n_trials)
            summary.update({"model": "BuyAndHold", "unique_id": ticker,
                            "cost_bps": float(cost), "scope": "ticker"})
            rows.append(summary)
            per_ticker.append(summary)
        frame = pd.DataFrame(per_ticker)
        aggregate = {"model": "BuyAndHold", "unique_id": "ALL",
                     "cost_bps": float(cost), "scope": "cross_section",
                     "n_tickers": len(frame), "cost_convention": "round_trip"}
        for field in REPORTED + ["net_return_ann"]:
            values = frame[field].replace([np.inf, -np.inf], np.nan)
            aggregate[field] = float(values.mean())
            aggregate[f"{field}_median"] = float(values.median())
        rows.append(aggregate)
    return rows


def print_table(table: pd.DataFrame, cost_levels: list[float],
                present: list[str], pending: list[str]) -> None:
    cross = table[table.scope == "cross_section"]
    print("\n" + "=" * 100)
    print("BACKTEST - long-flat on the median forecast, per ticker, "
          "aggregated across the cross-section")
    print("=" * 100)
    head = (f"{'model':<22}{'cost':>6}{'turnover':>10}{'break-even':>12}"
            f"{'>5bps':>8}{'Sharpe':>9}{'DSR p':>8}{'maxDD':>9}")
    print(head)
    print("-" * len(head))
    for model in sorted(cross.model.unique()):
        block = cross[cross.model == model].sort_values("cost_bps")
        for _, r in block.iterrows():
            print(f"{model if r.cost_bps == cost_levels[0] else '':<22}"
                  f"{r.cost_bps:>6.0f}{r.turnover_ann:>10.2f}"
                  f"{r.break_even_round_trip_bps:>12.2f}"
                  f"{r.share_break_even_above_5bps * 100:>7.0f}%"
                  f"{r.sharpe_net:>9.3f}{r.deflated_sharpe_prob:>8.3f}"
                  f"{r.max_drawdown * 100:>8.1f}%")
        print()

    print("break-even is the round-trip cost in bps at which the net edge "
          "reaches zero, averaged")
    print("across tickers; '>5bps' is the share of the fifty names clearing "
          "H1's threshold.")
    print("Sharpe and DSR are per-name and then averaged - a diversified "
          "portfolio would score")
    print("higher, so these understate a portfolio strategy and are not the "
          "number to quote for")
    print("one. Break-even is a per-name ratio and aggregates honestly; it is "
          "what H1(b) reads.")

    if pending:
        print()
        for model in sorted(pending):
            print(f"{model:<22}{PENDING:>6}{PENDING:>10}{PENDING:>12}"
                  f"{PENDING:>8}{PENDING:>9}{PENDING:>8}{PENDING:>9}")
    total = len(present) + len(pending)
    print(f"\nROSTER: computed on {len(present)} of {total} runnable models.")
    print(f"  present ({len(present)}): {', '.join(present)}")
    if pending:
        print(f"  {PENDING} ({len(pending)}): {', '.join(pending)}")
        print("  No simulation was run for these. Any claim about which model "
              "clears the")
        print("  5 bps threshold is provisional until they land.")


def main() -> None:
    args = parse_args()
    try:
        cfg = load_configs()
    except ConfigError as exc:
        raise SystemExit(str(exc))

    backtest_cfg = cfg.evaluation["backtest"]
    cost_levels = [float(c) for c in backtest_cfg["cost_bps"]]
    mode = backtest_cfg["mode"]
    if backtest_cfg.get("cost_convention") != "round_trip":
        raise SystemExit("configs/eval.yaml no longer says round_trip; the "
                         "simulator and the H1 criterion both assume it.")

    panel = (pd.read_parquet(args.features, columns=["unique_id", "ds", "target"])
             .rename(columns={"target": "y"}))
    root = Path(args.forecasts)

    # A broken model is excluded before the roster is split, not filtered out of
    # the table afterwards. n_trials below is the number of strategies actually
    # tried, and it sets the deflated Sharpe hurdle every other model must
    # clear - so leaving a model whose output is not a forecast in the count
    # would penalise all of them for a strategy that was never tested.
    expected = [s.name for s in runnable()
                if arm_of(s.name) == "return" and s.name not in BROKEN_MODELS]
    broken = [s.name for s in runnable()
              if arm_of(s.name) == "return" and s.name in BROKEN_MODELS]
    wanted = [m for m in (args.models or expected) if m not in BROKEN_MODELS]
    present, pending = [], []
    for name in expected:
        (present if not load_forecasts(root, name).empty else pending).append(name)

    print(f"cost levels (round trip, bps): {cost_levels}   mode: {mode}")
    print(f"deflated Sharpe hurdle uses n_trials = {len(present)} "
          "(the models actually tried)\n")

    rows = []
    for model in [m for m in wanted if m in present]:
        forecasts = load_forecasts(root, model)
        scored = forecasts.merge(panel, on=["unique_id", "ds"], how="left",
                                 validate="one_to_one")
        missing = int(scored["y"].isna().sum())
        if missing:
            raise SystemExit(f"{model}: {missing} forecasts have no outcome")
        rows.extend(backtest_model(scored, model, cfg.median_column, mode,
                                   cost_levels, len(present)))
        print(f"  {model:<22} {len(scored):>7,} forecasts x "
              f"{len(cost_levels)} cost levels")

    if not rows:
        raise SystemExit(f"no forecasts found under {root}")

    benchmark = load_forecasts(root, present[0]).merge(
        panel, on=["unique_id", "ds"], how="left", validate="one_to_one")
    rows.extend(buy_and_hold(benchmark, cost_levels, len(present)))
    print(f"  {'BuyAndHold':<22} benchmark")

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    print_table(table, cost_levels, present, pending)
    if broken:
        print(f"\n{BROKEN} ({len(broken)}): {', '.join(sorted(broken))}")
        for model in sorted(broken):
            print(f"  {model}: {BROKEN_MODELS[model]}")
        print("  No simulation was run. Excluded from the roster above and from")
        print(f"  n_trials, so the deflated Sharpe hurdle is set by the "
              f"{len(present)} strategies actually tried.")
    print(f"\n{len(table):,} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
