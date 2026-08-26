#!/usr/bin/env python
"""Download the raw dataset: OHLCV panel, index proxy, and lagged macro series.

This is step one of the pipeline. It fetches, caches, validates and writes; it
does not compute features and it does not model. Everything it emits is
regenerable, so ``data/`` stays out of git.

What ``y`` is, and what it is not
---------------------------------
``y`` in the emitted panel is the *realised* daily log return,
``log(close[t] / close[t-1])`` - the return observed at the close of ``t``. It
is not the forward return. The forward-looking label is built separately by
:func:`sp500bench.features.build.make_target`, which is the only place in this
project permitted to look ahead, and which is named so the call site says so
out loud.

Keeping the dataset backward-looking matters. In the long panel convention that
statsforecast and neuralforecast expect, forecasting ``y[t+1]`` from history
through ``t`` is exactly the one-step-ahead problem this study poses. If ``y``
were already the forward return, every model would receive its own label as an
input and the entire benchmark would report a leak.

Adjustment
----------
``auto_adjust=True``, so OHLC are adjusted for splits and dividends. Raw close
prices would put an artificial jump into the return series at every corporate
action.

Macro lags
----------
FRED series pass through :func:`sp500bench.data.macro.apply_release_lag` before
they are aligned to the trading calendar, and only then are they forward
filled. That ordering is the whole point: January CPI is published in mid
February, so a model reading the January value on a January date is reading the
future. Filling before lagging would smear a not-yet-published figure backwards
across the month and leak.

Outputs
-------
``data/raw/<ticker>.parquet``     cached raw download, one file per ticker
``data/raw/fred_<id>.parquet``    cached raw FRED series, one file per series
``data/processed/panel.parquet``  long panel: unique_id, ds, y, OHLCV
``data/processed/macro.parquet``  lagged macro, aligned to the session calendar
``data/processed/index.parquet``  index proxy, for regime labelling only
``data/quality_report.json``      per-ticker data quality audit

Usage
-----
    python scripts/build_dataset.py --config configs/data.yaml
    python scripts/build_dataset.py --config configs/data.yaml --refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Make the package importable when this runs as a plain script, so the pipeline
# works whether or not `pip install -e .` has been done.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sp500bench.data.macro import (  # noqa: E402
    MACRO_SERIES,
    align_to_sessions,
    apply_release_lag,
)
from sp500bench.data.universe import TICKERS  # noqa: E402

OHLCV = ["open", "high", "low", "close", "volume"]

# The universe is specified as continuously listed from 2005-01-01. A ticker
# whose history starts later would enter the panel mid-sample and quietly break
# that guarantee, so it is a hard failure rather than a warning. One month of
# slack absorbs ordinary calendar noise at the very start of the sample.
MIN_FIRST_SESSION = pd.Timestamp("2005-01-31")

# Daily equity moves beyond this are rare but real (COVID, earnings gaps). They
# are counted and reported, never dropped - silently winsorising the tails would
# remove exactly the episodes a forecasting model most needs to handle.
EXTREME_RETURN = 0.25


# ---------------------------------------------------------------------------
# Configuration and credentials
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and validate the S&P 500 benchmark dataset."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "data.yaml"),
        help="frozen data specification (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the raw cache and re-download everything",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Read the frozen data spec and check it still matches the code."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    size = cfg["universe"]["size"]
    if len(TICKERS) != size:
        raise SystemExit(
            f"config declares {size} tickers but universe.py holds {len(TICKERS)}"
        )

    wanted = set(cfg["macro"]["series"])
    have = {s.fred_id for s in MACRO_SERIES}
    if wanted != have:
        raise SystemExit(
            "configs/data.yaml and data/macro.py disagree on the FRED series.\n"
            f"  only in config: {sorted(wanted - have) or 'none'}\n"
            f"  only in code  : {sorted(have - wanted) or 'none'}"
        )

    kind = cfg["target"]["kind"]
    if kind != "log_return":
        raise SystemExit(f"this script emits log returns; config asks for {kind!r}")

    return cfg


def resolve_fred_key() -> str:
    """Find the FRED API key, preferring the project .env over the environment.

    ``load_dotenv`` populates ``os.environ`` from ``.env`` without overwriting
    variables already exported, so a shell export still wins if both are set.
    """
    import os

    env_path = ROOT / ".env"
    load_dotenv(env_path)

    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "FRED_API_KEY is not set.\n"
            f"  Looked in: {env_path} (project .env)\n"
            "             the FRED_API_KEY environment variable\n"
            "\n"
            "  Get a free key at https://fredaccount.stlouisfed.org/apikeys "
            "then either:\n"
            f"    echo 'FRED_API_KEY=<your key>' > {env_path}\n"
            "  or:\n"
            "    export FRED_API_KEY=<your key>"
        )

    # A FRED key is 32 lowercase alphanumeric characters. Catching a leftover
    # placeholder here gives a clear message instead of an opaque HTTP 400 from
    # the API a minute into the run.
    if len(key) != 32 or not key.isalnum() or not key.islower():
        raise SystemExit(
            f"FRED_API_KEY does not look like a FRED key: {key[:4]}... "
            f"({len(key)} chars).\n"
            "  Expected 32 lowercase alphanumeric characters. If this is still "
            "a placeholder,\n"
            f"  replace it in {env_path} with the real key."
        )
    return key


# ---------------------------------------------------------------------------
# Equity download
# ---------------------------------------------------------------------------

def _cache_path(raw_dir: Path, name: str) -> Path:
    """Cache filename for a ticker. '^GSPC' is not a legal filename stem."""
    return raw_dir / f"{name.replace('^', '_')}.parquet"


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the OHLCV columns and strip any timezone from the index.

    yfinance returns capitalised columns and, depending on version and market,
    a timezone-aware index. The rest of this project assumes lower-case names
    (``build_features`` requires them) and naive dates, so that equity data and
    FRED data can be joined on the same axis.
    """
    frame = frame.copy()
    frame.columns = [str(c).lower() for c in frame.columns]
    missing = [c for c in OHLCV if c not in frame.columns]
    if missing:
        raise ValueError(f"download is missing columns: {missing}")

    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    frame.index.name = "ds"

    frame = frame[OHLCV].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.dropna(subset=["close"])


def _split_batch(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker out of a yfinance batch response."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        frame = data[ticker]
    else:
        frame = data
    frame = frame.dropna(how="all")
    return None if frame.empty else frame


def download_equities(
    names: list[str],
    start: str,
    end: str,
    raw_dir: Path,
    refresh: bool,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for every name, using the raw cache unless ``--refresh``.

    yfinance treats ``end`` as exclusive, so the configured last session is
    included by asking for one day beyond it.
    """
    import yfinance as yf

    raw_dir.mkdir(parents=True, exist_ok=True)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frames: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []
    for name in names:
        path = _cache_path(raw_dir, name)
        if path.exists() and not refresh:
            frames[name] = _normalise(pd.read_parquet(path))
        else:
            to_fetch.append(name)

    if frames:
        print(f"  cache hit for {len(frames)} of {len(names)} names")
    if not to_fetch:
        return frames

    print(f"  downloading {len(to_fetch)} names from yfinance ...")
    failed: list[str] = []
    # Chunked so one bad symbol cannot take down the whole request, and to stay
    # clear of rate limiting.
    for i in range(0, len(to_fetch), 25):
        chunk = to_fetch[i : i + 25]
        data = yf.download(
            tickers=chunk,
            start=start,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if data is None or data.empty:
            failed.extend(chunk)
            continue
        for name in chunk:
            raw = _split_batch(data, name)
            if raw is None:
                failed.append(name)
                continue
            frame = _normalise(raw)
            frame.to_parquet(_cache_path(raw_dir, name))
            frames[name] = frame

    if failed:
        raise SystemExit(
            f"yfinance returned no data for {len(failed)} name(s): "
            f"{', '.join(sorted(failed))}\n"
            "  Check the symbols, or retry - yfinance rate-limits intermittently."
        )
    return frames


# ---------------------------------------------------------------------------
# Macro download
# ---------------------------------------------------------------------------

def download_macro(
    start: str,
    end: str,
    api_key: str,
    raw_dir: Path,
    refresh: bool,
) -> dict[str, pd.DataFrame]:
    """Fetch each FRED series, unlagged. Lagging happens downstream, once."""
    from fredapi import Fred

    raw_dir.mkdir(parents=True, exist_ok=True)
    fred = Fred(api_key=api_key)

    raw: dict[str, pd.DataFrame] = {}
    for spec in MACRO_SERIES:
        path = raw_dir / f"fred_{spec.fred_id}.parquet"
        if path.exists() and not refresh:
            raw[spec.fred_id] = pd.read_parquet(path)
            continue
        print(f"  fetching {spec.fred_id:<14} ({spec.name})")
        series = fred.get_series(
            spec.fred_id, observation_start=start, observation_end=end
        )
        frame = pd.Series(series, name=spec.name).to_frame()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        frame.index.name = "ds"
        frame = frame.sort_index().dropna()
        frame.to_parquet(path)
        raw[spec.fred_id] = frame
    return raw


def lag_and_align(
    raw: dict[str, pd.DataFrame],
    sessions: pd.DatetimeIndex,
    enforce_lag: bool,
) -> pd.DataFrame:
    """Apply publication lags, then align to the trading calendar.

    The order is load-bearing and is asserted by ``tests/test_leakage.py``:
    lag first, forward fill second. Reversing them leaks an unpublished figure
    backwards in time.
    """
    if not enforce_lag:
        raise SystemExit(
            "configs/data.yaml sets macro.enforce_release_lag: false.\n"
            "  Refusing to build a dataset that reads macro data before its "
            "publication date."
        )

    lagged = []
    for spec in MACRO_SERIES:
        frame = raw[spec.fred_id]
        lagged.append(apply_release_lag(frame, spec))

    combined = pd.concat(lagged, axis=1, sort=False).sort_index()
    return align_to_sessions(combined, sessions)


# ---------------------------------------------------------------------------
# Panel assembly and quality audit
# ---------------------------------------------------------------------------

def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-ticker frames into the long panel the model layer expects.

    The first session of each ticker has no previous close and therefore no
    return; those rows are dropped rather than filled with a zero, which would
    be a fabricated observation.
    """
    rows = []
    for ticker in sorted(frames):
        frame = frames[ticker]
        block = frame.reset_index()
        block.insert(0, "unique_id", ticker)
        block["y"] = np.log(frame["close"]).diff().to_numpy()
        rows.append(block)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["y"])
    panel = panel[["unique_id", "ds", "y", *OHLCV]]
    return panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def audit(
    frames: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> dict:
    """Per-ticker data quality audit.

    Missing sessions are counted against the reference calendar restricted to
    each ticker's own first-to-last span, so a genuine gap - a halt, a vendor
    error - is not confused with a listing history that simply starts later.
    """
    returns = {t: g.set_index("ds")["y"] for t, g in panel.groupby("unique_id")}

    per_ticker = {}
    for ticker in sorted(frames):
        frame = frames[ticker]
        first, last = frame.index.min(), frame.index.max()
        expected = calendar[(calendar >= first) & (calendar <= last)]
        missing = expected.difference(frame.index)
        ret = returns.get(ticker, pd.Series(dtype=float))
        extreme = ret[ret.abs() > EXTREME_RETURN]

        per_ticker[ticker] = {
            "n_rows": int(len(frame)),
            "first_date": first.strftime("%Y-%m-%d"),
            "last_date": last.strftime("%Y-%m-%d"),
            "missing_sessions": int(len(missing)),
            "missing_session_dates": [d.strftime("%Y-%m-%d") for d in missing[:10]],
            "zero_volume_days": int((frame["volume"].fillna(0) == 0).sum()),
            "extreme_return_days": int(len(extreme)),
            "max_abs_return": float(ret.abs().max()) if len(ret) else float("nan"),
        }

    return {
        "generated_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tickers": len(frames),
        "n_sessions": int(len(calendar)),
        "calendar_first": calendar.min().strftime("%Y-%m-%d"),
        "calendar_last": calendar.max().strftime("%Y-%m-%d"),
        "panel_rows": int(len(panel)),
        "extreme_return_threshold": EXTREME_RETURN,
        "min_first_session_required": MIN_FIRST_SESSION.strftime("%Y-%m-%d"),
        "per_ticker": per_ticker,
    }


def audit_macro(macro: pd.DataFrame, splits: dict) -> dict:
    """Coverage audit for the macro block, split window by split window.

    A macro series that is empty across training and populated only late in the
    test period is worse than a missing series. It enters the feature matrix as
    a column that is NaN for every training row, then switches on partway
    through evaluation - so its apparent contribution is confounded with the
    date at which the vendor's history happens to begin, and it lands hardest
    on the clean slice, which is precisely the comparison H4 rests on.

    FRED serves some licensed series - the ICE BofA family in particular - on a
    rolling window of a few years rather than full history, so this is a live
    risk on every rebuild, not a one-off.
    """
    out = {}
    for col in macro.columns:
        series = macro[col]
        present = series.dropna()
        by_split = {}
        for name, window in splits.items():
            segment = series.loc[window["start"] : window["end"]]
            by_split[name] = (
                round(float(segment.notna().mean()), 4) if len(segment) else None
            )
        out[col] = {
            "first_date": present.index.min().strftime("%Y-%m-%d") if len(present) else None,
            "last_date": present.index.max().strftime("%Y-%m-%d") if len(present) else None,
            "non_null": int(series.notna().sum()),
            "coverage": round(float(series.notna().mean()), 4),
            "coverage_by_split": by_split,
        }
    return out


def warn_on_macro_gaps(macro_report: dict) -> None:
    """Flag macro series that cannot support the frozen split design.

    A warning rather than a hard failure: the series list is frozen in
    configs/data.yaml, so dropping one is a specification decision that belongs
    to the researcher, not to this script. It is printed last, and loudly, so it
    cannot be scrolled past.
    """
    broken = {
        name: row
        for name, row in macro_report.items()
        if (row["coverage_by_split"].get("train") or 0.0) == 0.0
    }
    if not broken:
        return

    print()
    print("!" * 79)
    print("WARNING - macro series with no coverage in the training window")
    print("!" * 79)
    for name, row in sorted(broken.items()):
        by = row["coverage_by_split"]
        print(f"  {name}: history begins {row['first_date']}, "
              f"{row['coverage']:.0%} of sessions overall")
        print("    coverage by split: " + "  ".join(
            f"{k}={'n/a' if v is None else format(v, '.0%')}" for k, v in by.items()
        ))
    print()
    print("  These columns are NaN for every training row and populate only")
    print("  partway through the test period. Used as features they confound")
    print("  model skill with vendor history, and they favour the clean slice,")
    print("  which is the comparison H4 depends on.")
    print()
    print("  FRED serves some licensed series (the ICE BofA family among them)")
    print("  on a rolling multi-year window rather than full history.")
    print()
    print("  Decide deliberately: drop the series from configs/data.yaml, find a")
    print("  substitute with full history, or restrict it to the arms that can")
    print("  support it. Recorded under 'macro' in data/quality_report.json.")
    print("!" * 79)


def print_report(report: dict) -> None:
    print()
    print(f"{'ticker':<8}{'rows':>7}{'first':>13}{'last':>13}"
          f"{'missing':>9}{'zero vol':>10}{'|r|>25%':>9}{'max |r|':>10}")
    print("-" * 79)
    for ticker, row in report["per_ticker"].items():
        print(
            f"{ticker:<8}{row['n_rows']:>7}{row['first_date']:>13}"
            f"{row['last_date']:>13}{row['missing_sessions']:>9}"
            f"{row['zero_volume_days']:>10}{row['extreme_return_days']:>9}"
            f"{row['max_abs_return']:>10.3f}"
        )
    print("-" * 79)
    totals = report["per_ticker"].values()
    print(
        f"{'total':<8}{report['panel_rows']:>7}"
        f"{report['calendar_first']:>13}{report['calendar_last']:>13}"
        f"{sum(r['missing_sessions'] for r in totals):>9}"
        f"{sum(r['zero_volume_days'] for r in totals):>10}"
        f"{sum(r['extreme_return_days'] for r in totals):>9}"
    )


def enforce_continuous_history(report: dict) -> None:
    """Hard-fail on any ticker that does not reach back to the start of 2005.

    Called after the quality report is written, so the diagnostic file survives
    the failure and can be inspected.
    """
    late = {
        t: r["first_date"]
        for t, r in report["per_ticker"].items()
        if pd.Timestamp(r["first_date"]) > MIN_FIRST_SESSION
    }
    if late:
        lines = "\n".join(f"    {t}: first session {d}" for t, d in sorted(late.items()))
        raise SystemExit(
            f"\n{len(late)} ticker(s) lack history before "
            f"{MIN_FIRST_SESSION.date()}:\n{lines}\n\n"
            "  The universe in src/sp500bench/data/universe.py is specified as "
            "continuously\n"
            "  listed from 2005-01-01. A name entering mid-sample breaks that "
            "guarantee.\n"
            "  Either replace the name in the universe or move the calendar "
            "start in\n"
            "  configs/data.yaml - both are frozen decisions, so change them "
            "deliberately.\n"
            "  The full audit was still written to data/quality_report.json."
        )


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    start = cfg["calendar"]["start"]
    end = cfg["calendar"]["end"]
    proxy = cfg["universe"]["index_proxy"]

    data_dir = ROOT / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset for {len(TICKERS)} tickers, {start} to {end}")
    if args.refresh:
        print("  --refresh: ignoring the raw cache")

    # Credentials first: fail before spending minutes on downloads.
    fred_key = resolve_fred_key()

    print("\nEquities")
    frames = download_equities(TICKERS, start, end, raw_dir, args.refresh)

    print(f"\nIndex proxy ({proxy}, regime labelling only - never a feature)")
    index_frames = download_equities([proxy], start, end, raw_dir, args.refresh)
    index = index_frames[proxy]

    # The index proxy defines the reference session calendar: it is the market,
    # so a session it trades on is a session every constituent should have.
    calendar = pd.DatetimeIndex(index.index)

    print("\nPanel")
    panel = build_panel(frames)
    print(f"  {len(panel):,} rows x {panel['unique_id'].nunique()} tickers")

    print("\nMacro (FRED)")
    raw_macro = download_macro(start, end, fred_key, raw_dir, args.refresh)
    macro = lag_and_align(raw_macro, calendar, cfg["macro"]["enforce_release_lag"])
    print(f"  {macro.shape[1]} series aligned to {len(macro):,} sessions, "
          "publication lags applied before forward fill")

    print("\nWriting")
    panel.to_parquet(processed_dir / "panel.parquet", index=False)
    macro.to_parquet(processed_dir / "macro.parquet")
    index.to_parquet(processed_dir / "index.parquet")
    for name in ("panel", "macro", "index"):
        print(f"  data/processed/{name}.parquet")

    report = audit(frames, panel, calendar)
    report["macro"] = audit_macro(macro, cfg["splits"])
    report_path = data_dir / "quality_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  data/quality_report.json")

    print_report(report)
    enforce_continuous_history(report)
    warn_on_macro_gaps(report["macro"])
    print("\nDone.")


if __name__ == "__main__":
    main()
