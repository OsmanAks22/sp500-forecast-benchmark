"""The forecast validator must catch both defects that have reached a parquet.

Neither announced itself. DLinear's quantile columns arrived scrambled and were
scored into a CRPS six times everyone else's; Kronos can put a NaN into a
sampled close, and because ``np.quantile`` propagates rather than ignoring it,
one bad path in a hundred destroys that origin's whole row.

A detector whose detection path is unproven is worth nothing, so both are
reproduced here synthetically and the validator is pointed at them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500bench.config import load_configs, quantile_column

ROOT = Path(__file__).resolve().parents[1]
GRID = list(load_configs().quantiles)
COLUMNS = [quantile_column(q) for q in GRID]


def _load():
    spec = importlib.util.spec_from_file_location(
        "validate_forecasts_under_test", ROOT / "scripts" / "validate_forecasts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load()


def _write(root: Path, model: str, frame: pd.DataFrame, grid_note: dict | None = None):
    directory = root / model
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "fold_0.parquet", index=False)
    if grid_note is not None:
        import json
        (directory / "grid.json").write_text(json.dumps(grid_note))


def _clean_frame(n_tickers=10, n_dates=20, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-02", periods=n_dates)
    frames = []
    for i in range(n_tickers):
        values = np.sort(rng.normal(0, 0.02, (n_dates, len(GRID))), axis=1)
        frame = pd.DataFrame(values, columns=COLUMNS)
        frame.insert(0, "ds", dates)
        frame.insert(0, "unique_id", f"T{i:02d}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_a_clean_model_passes(tmp_path):
    _write(tmp_path, "Clean", _clean_frame())
    result = validator.check_model(tmp_path, "Clean", GRID, 5)
    assert result["nonfinite_rows"] == 0
    assert result["nonmonotone_rows"] == 0
    assert result["crossings"] == 0
    assert validator.report(result) is True


def test_a_nan_row_is_counted_and_located(tmp_path):
    """The Kronos shape: np.quantile propagates, so the whole row goes."""
    frame = _clean_frame()
    hit = (frame["ds"] == frame["ds"].iloc[3]) & (frame["unique_id"] < "T05")
    frame.loc[hit, COLUMNS] = np.nan
    _write(tmp_path, "KronosLike", frame)

    result = validator.check_model(tmp_path, "KronosLike", GRID, 5)
    assert result["nonfinite_rows"] == int(hit.sum())
    assert result["nonfinite_dates"]["distinct"] == 1
    assert set(result["nonfinite_by_ticker"]) <= {f"T{i:02d}" for i in range(5)}
    assert validator.report(result) is False


def test_a_partial_nan_row_counts_once_not_per_level(tmp_path):
    """One bad path destroys the row; it is one defective origin, not nine."""
    frame = _clean_frame()
    frame.loc[0, COLUMNS[0]] = np.nan
    _write(tmp_path, "Partial", frame)
    assert validator.check_model(tmp_path, "Partial", GRID, 5)["nonfinite_rows"] == 1


def test_scrambled_levels_are_caught(tmp_path):
    """The DLinear shape: a mapping fault, not a model pathology."""
    rng = np.random.default_rng(7)
    frame = _clean_frame()
    values = frame[COLUMNS].to_numpy()
    for row in values:
        rng.shuffle(row)
    frame[COLUMNS] = values
    _write(tmp_path, "ScrambledLike", frame)

    result = validator.check_model(tmp_path, "ScrambledLike", GRID, 5)
    assert result["nonfinite_rows"] == 0
    assert result["nonmonotone_rows"] > 0.8 * len(frame)
    assert result["crossings"] > 0.3 * result["crossing_pairs"]
    assert validator.report(result) is False


def test_a_single_inverted_pair_is_caught(tmp_path):
    """TimesNet's real defect is 87 rows in 83,450 - the threshold is one."""
    frame = _clean_frame()
    frame.loc[0, [COLUMNS[3], COLUMNS[4]]] = frame.loc[0, [COLUMNS[4], COLUMNS[3]]].to_numpy()
    _write(tmp_path, "OneCrossing", frame)
    result = validator.check_model(tmp_path, "OneCrossing", GRID, 5)
    assert result["nonmonotone_rows"] == 1
    assert result["crossings"] == 1


def test_declared_absent_levels_are_not_reported_as_defects(tmp_path):
    """TTM has no quantile head; eight empty columns are its stated property."""
    frame = _clean_frame()
    absent = [c for c in COLUMNS if c != quantile_column(0.5)]
    frame[absent] = np.nan
    _write(tmp_path, "TTMLike", frame, grid_note={
        "model": "TTMLike", "scored_on_frozen_grid": False, "native_levels": []})

    result = validator.check_model(tmp_path, "TTMLike", GRID, 5)
    assert result["levels_checked"] == [quantile_column(0.5)]
    assert len(result["levels_skipped"]) == 8
    assert result["nonfinite_rows"] == 0
    assert validator.report(result) is True


def test_a_partial_grid_model_skips_only_its_declared_tails(tmp_path):
    """TimesFM's four tail levels are declared unavailable, the other five are not."""
    frame = _clean_frame()
    tails = [quantile_column(q) for q in (0.01, 0.05, 0.95, 0.99)]
    frame[tails] = np.nan
    _write(tmp_path, "TimesFMLike", frame, grid_note={
        "model": "TimesFMLike", "scored_on_frozen_grid": False,
        "native_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "frozen_levels_unavailable": [0.01, 0.05, 0.95, 0.99]})

    result = validator.check_model(tmp_path, "TimesFMLike", GRID, 5)
    assert sorted(result["levels_skipped"]) == sorted(tails)
    assert result["nonfinite_rows"] == 0


def test_a_model_with_no_forecasts_is_reported_not_crashed(tmp_path):
    result = validator.check_model(tmp_path, "Missing", GRID, 5)
    assert result["status"] == "no forecasts on disk"
    assert validator.report(result) is True


def test_the_concentration_report_gives_figures_not_a_label(tmp_path):
    """A binary clustered/scattered call misleads near its own threshold."""
    frame = _clean_frame(n_tickers=50, n_dates=40)
    hit = frame["ds"].isin(frame["ds"].unique()[:2]) & (frame["unique_id"] < "T23")
    frame.loc[hit, COLUMNS] = np.nan
    _write(tmp_path, "Clustered", frame)

    shape = validator.check_model(tmp_path, "Clustered", GRID, 5)["nonfinite_shape"]
    assert "2 of 40 sessions" in shape
    assert "23 of 50 tickers" in shape
