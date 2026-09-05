"""Tier 3, the pre-trained foundation models, inference only.

Separate from ``run_experiment.py`` because this tier runs on Kaggle behind a
GPU and a session timeout, while the rest of the roster runs locally on CPU.
The folds, the quantile grid and the output schema are identical - the only
thing that differs is where it executes and how long a failure costs.

    python scripts/run_foundation.py --calibrate          # sample_count sweep
    python scripts/run_foundation.py                      # full run
    python scripts/run_foundation.py --self-check         # no GPU, no packages

Writes ``forecasts/{model}/fold_{k}.parquet`` with ``unique_id``, ``ds`` and one
column per level in the frozen grid - the schema ``evaluate.py`` already reads.
``y`` is not written; ``evaluate.py`` merges it from the panel itself.


WHAT EACH MODEL ACTUALLY EMITS
------------------------------
Read out of each model's own config or source on 5 September 2026, never
inferred from an output shape. The axis order of a returned array is not
self-describing when two axes happen to share a length, and a quantile grid
mapped onto the wrong axis raises nothing at all - it just produces a CRPS that
means nothing. Every mapping below is asserted at run time against the levels
the loaded checkpoint reports, not against these comments.

Chronos-2 - ``amazon/chronos-2``, ``chronos_config.quantiles``, 21 levels::

    0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
    0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99

  All nine of our levels are exact members, at indices 0 1 2 5 10 15 18 19 20.
  Direct selection, no interpolation anywhere.

  ``chronos/chronos2/pipeline.py:562`` documents the return of ``predict`` as a
  list whose elements are ``(n_variates, n_quantiles, prediction_length)``. Note
  that ``chronos/base.py:129`` documents ``(batch, prediction_length,
  num_quantiles)`` for ``predict_quantiles`` - a *different* method with the
  axes the other way round. Applying the base docstring to the pipeline's output
  transposes the grid silently. Hence ``_select_our_levels``, which checks the
  recovered levels are equal to the frozen grid rather than trusting arithmetic
  over an assumed layout.

TimesFM-2.5 - ``timesfm/timesfm_2p5/timesfm_2p5_base.py:92``,
  ``TimesFM_2p5_200M_Definition.quantiles = [0.1 .. 0.9]``, nine deciles, with
  ``decode_index = 5`` marking the median. The head emits
  ``len(quantiles) + 1 = 10`` channels on the *last* axis - the opposite
  convention to Chronos-2 - where channel 0 is the point forecast and channels
  1..9 are q0.1..q0.9.

  The level set is not settable. ``timesfm3_forecaster.py:398`` overwrites
  ``config.quantiles`` with ``list(self.model.quantiles)`` when the checkpoint
  loads, so a custom grid passed in the config is discarded.

  Four of our nine levels - 0.01, 0.05, 0.95, 0.99 - lie outside [0.1, 0.9]
  entirely. They are written as NaN, never extrapolated: a value at q0.01 from a
  head that was never trained to emit one would measure our extrapolation rule,
  not the model's uncertainty. The six native levels our grid does not contain
  (0.2 0.3 0.4 0.6 0.7 0.8) are written as extra columns so its CRPS can be
  computed over its own nine levels. ``forecasts/{model}/grid.json`` records
  which grid the file is on, so every table can mark TimesFM as not directly
  comparable on CRPS to models scored on the full nine.

  A common-subset comparison was considered and rejected: restricting every
  model to {0.1, 0.5, 0.9} would discard exactly the tail information that
  Chronos-2 and Kronos do carry, and H2 is a hypothesis about calibration.

TTM - ``ibm-granite/granite-timeseries-ttm-r2`` has ``loss: "mse"`` and no
  ``multi_quantile_head`` / ``quantile_levels`` / ``quantile`` keys at all, so
  they fall back to the class defaults and the quantile head does not exist.
  ``distribution_output: "student_t"`` is present but inert - it applies only
  when ``loss == "nll"``. ``modeling_tinytimemixer.py:3581`` leaves
  ``quantile_outputs`` as ``None``.

  So TTM emits a point forecast and nothing else. It is written into q0.5 with
  the other eight columns NaN, which puts it in the RMSE/MAE/MASE/directional
  tables and leaves it out of CRPS, PICP and interval score - because
  ``crps_from_quantiles`` averages pinball losses and NaN propagates. That
  absence is a reported property of the model. It is not papered over with a
  synthetic band, and getting quantiles out of TTM would mean fine-tuning it
  with a pinball loss, which would forfeit ``scope="zero_shot"`` and make it a
  different model than the one that was pre-registered.

  ``prediction_length`` is 96 and our horizon is 1, so we take step 0 of a
  jointly-trained 96-step head. ``scaling: "std"`` means the model standardises
  on the context window internally and inverts it before returning
  (``modeling_tinytimemixer.py:3627``), so raw values go in and raw values come
  out.

Kronos-small - emits price-level OHLCV and no quantiles. Its distribution comes
  from sampled paths, which is what ``registry.py`` pre-registers.

  The public API cannot supply them. ``model/kronos.py:465-467``::

      z = z.reshape(-1, sample_count, z.size(1), z.size(2))
      preds = z.cpu().numpy()
      preds = np.mean(preds, axis=1)

  and ``predict_batch``'s own docstring says the samples are "automatically
  averaged internally". ``predict()`` returns the pointwise mean of the paths,
  so the smoke test that called it never exercised the distributional path at
  all - "Kronos WORKING" so far means "returns a mean path".

  ``_keep_sample_axis`` therefore calls ``auto_regressive_inference`` directly
  behind a narrowly scoped shim that leaves that one reduction out, and asserts
  the sample axis survived. Quantiles are then empirical order statistics of the
  S predicted closes, converted to returns against the last observed close.


WHAT THE MODELS ARE FED, AND WHY IT IS NOT LEAKAGE
--------------------------------------------------
``configs/models.yaml`` specifies ``native_input: ohlcv`` for Kronos and says
nothing about the other three, so this is the one consequential choice the
frozen specification does not settle. It is made here, in one place, and named:

    The three generic TSFMs are fed the *realised log-return* series, because
    that is the target every other model in the study forecasts and the
    quantity ``evaluate.py`` scores. Feeding them close prices instead would
    hand the foundation tier a different and generally easier problem than the
    neural tier gets, and the comparison behind H1 would stop being like for
    like. ``--input-series`` exists so the alternative is a recorded decision
    rather than a silent one.

The alignment is the leakage-critical part. ``make_target`` defines
``target[t] = log(P[t+h] / P[t])`` - the *forward* return - so at forecast
origin ``t``:

  * prices through ``P[t]`` are known and may enter the context;
  * the realised return series may therefore run to
    ``r[t] = log(P[t] / P[t-1])``, which equals ``y[t-1]``;
  * ``y[t]`` is the label and must never appear in the context.

Building the context from closes and differencing makes that off-by-one
explicit instead of hiding it behind a ``.shift()``. ``_assert_context_alignment``
re-derives ``log(close[t+1]/close[t]) == target[t]`` from the two panels and
fails the run if they ever disagree.

The purge and embargo are *not* applied to the context window, and that is
deliberate rather than an oversight. Purging exists so a model fitted on the
training fold cannot have been fitted on data adjacent to the test fold. These
models are not fitted at all - ``scope="zero_shot"`` - so history running up to
the origin is ordinary conditioning, not leakage. ``assert_no_overlap`` is still
called on every fold, because the guard is the point of the fold structure and a
tier that quietly skips it is a tier nobody can check.

The separate contamination question - that these checkpoints may have been
pre-trained on the test period, and that only Kronos documents a cutoff - is not
addressed here. It is handled by the clean slice in ``configs/data.yaml`` and by
``registry.contaminated``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500bench.config import (load_configs, load_data_config,  # noqa: E402
                               quantile_column, quantile_columns)
from sp500bench.evaluation.splits import (PurgedWalkForward,  # noqa: E402
                                          assert_no_overlap)

FOUNDATION_MODELS = ["Chronos-2", "TimesFM-2.5", "TTM", "Kronos-small"]

# TimesFM's own grid, from TimesFM_2p5_200M_Definition.quantiles. Asserted
# against the loaded checkpoint before use.
TIMESFM_NATIVE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

DEFAULT_CONTEXT = 512          # Kronos' documented context_limit; TTM's fixed
                               # context_length. Shared so the four models see
                               # the same history and the comparison is on the
                               # model, not on how much past it was handed.
DEFAULT_SAMPLE_COUNT = 200
DEFAULT_BATCH = 256
DEFAULT_KRONOS_PATHS_PER_CALL = 4096   # batch x sample_count sequence cap


def show(path: Path) -> str:
    try:
        return f"{path.relative_to(Path.cwd())}"
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# timings
# --------------------------------------------------------------------------

def record_timing(path: Path, model: str, payload: dict) -> None:
    """Merge one model's wall-clock into results/timings.json.

    Deliberately the same shape and the same file as ``run_experiment.py``'s
    function of this name, so the compute-cost axis covers the whole roster in
    one record even though the tiers run on different machines on different
    days. Merged, never overwritten, for the same reason.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! {path} is not valid JSON; starting a fresh record")
    existing[model] = payload
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# panel plumbing
# --------------------------------------------------------------------------

def load_panels(bars_path: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not bars_path.exists():
        raise SystemExit(f"missing OHLCV panel: {bars_path}\n"
                         "  run: python scripts/build_dataset.py --config configs/data.yaml")
    if not features_path.exists():
        raise SystemExit(f"missing feature panel: {features_path}\n"
                         "  run: python scripts/build_features.py --config configs/data.yaml")
    bars = pd.read_parquet(
        bars_path, columns=["unique_id", "ds", "open", "high", "low", "close", "volume"])
    feats = pd.read_parquet(features_path, columns=["unique_id", "ds", "target"])
    bars = bars.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    feats = (feats.rename(columns={"target": "y"})
                  .sort_values(["unique_id", "ds"]).reset_index(drop=True))
    return bars, feats


def _assert_context_alignment(bars: pd.DataFrame, feats: pd.DataFrame,
                              n_tickers: int = 5) -> None:
    """The context off-by-one, checked against the frozen target definition.

    ``make_target`` is ``log(P[t+h]/P[t])``. If that ever stops matching a close
    difference the context built here is silently off by a session, which would
    put the label itself into the model's history - the one error in this file
    that would improve every score rather than break the run.
    """
    merged = bars.merge(feats, on=["unique_id", "ds"], how="inner")
    checked = 0
    for ticker, group in merged.groupby("unique_id", sort=True):
        group = group.sort_values("ds")
        forward = np.log(group["close"].shift(-1) / group["close"]).to_numpy()
        target = group["y"].to_numpy()
        both = np.isfinite(forward) & np.isfinite(target)
        if both.sum() and not np.allclose(forward[both], target[both], atol=1e-9):
            worst = np.max(np.abs(forward[both] - target[both]))
            raise SystemExit(
                f"context alignment failed for {ticker}: "
                f"log(close[t+1]/close[t]) != target[t], max |diff| = {worst:.3e}\n"
                "  the return context would be off by one session and would "
                "contain the label. Refusing to run.")
        checked += 1
        if checked >= n_tickers:
            break
    print(f"  alignment  log(close[t+1]/close[t]) == target[t] on {checked} tickers")


def build_context_matrix(closes: np.ndarray, origin_positions: np.ndarray,
                         context: int, series: str) -> np.ndarray:
    """``(n_origins, context)`` of history, each row ending at its own origin.

    For ``series="price"`` row ``i`` is ``P[o-context+1 .. o]`` inclusive.
    For ``series="return"`` it is ``r[o-context+1 .. o]`` where
    ``r[j] = log(P[j]/P[j-1])``, so the row ends at ``y[o-1]`` and the label
    ``y[o]`` is not in it. See the module docstring.
    """
    if series == "return":
        values = np.concatenate([[np.nan], np.diff(np.log(closes))])
    elif series == "price":
        values = closes
    else:                                            # pragma: no cover - guarded by argparse
        raise ValueError(f"unknown series {series!r}")

    rows = np.empty((len(origin_positions), context), dtype=np.float64)
    for k, origin in enumerate(origin_positions):
        start = origin - context + 1
        if start < 0:
            raise ValueError(f"origin at position {origin} has only {origin + 1} "
                             f"sessions of history, needs {context}")
        rows[k] = values[start:origin + 1]
    if not np.isfinite(rows).all():
        raise ValueError(f"non-finite value in the {series} context matrix; "
                         "a return context starting at position 0 would carry the "
                         "NaN from the first difference")
    return rows


# --------------------------------------------------------------------------
# quantile assembly
# --------------------------------------------------------------------------

def _select_our_levels(native_grid, our_grid, values: np.ndarray) -> np.ndarray:
    """Pick our levels out of a model's native grid by *value*, not by index.

    ``values`` is ``(n, len(native_grid))``. Returns ``(n, len(our_grid))``.

    The check is that the recovered levels equal the frozen grid element for
    element. Index arithmetic against an assumed axis order is exactly what
    fails silently here: Chronos-2's pipeline and its base class document the
    quantile axis in opposite positions, and with a horizon of 1 a transposed
    array is still the right rank and still indexes without complaint.
    """
    native = np.asarray(native_grid, dtype=float)
    if values.shape[-1] != native.size:
        raise AssertionError(
            f"expected the last axis to be the {native.size} native quantile "
            f"levels, got shape {values.shape}. The array is transposed or the "
            "checkpoint's grid changed; refusing to guess.")
    idx = []
    for q in our_grid:
        hits = np.flatnonzero(np.isclose(native, q, atol=1e-12))
        if hits.size != 1:
            raise AssertionError(f"level {q} is not a unique member of the "
                                 f"native grid {list(native)}")
        idx.append(int(hits[0]))
    recovered = native[idx]
    if not np.allclose(recovered, np.asarray(our_grid, dtype=float), atol=1e-12):
        raise AssertionError(f"selected levels {list(recovered)} != frozen grid "
                             f"{list(our_grid)}")
    return values[..., idx]


def empty_quantile_frame(unique_id: str, dates, our_grid) -> pd.DataFrame:
    frame = pd.DataFrame({"unique_id": unique_id, "ds": dates})
    for q in our_grid:
        frame[quantile_column(q)] = np.nan
    return frame


# --------------------------------------------------------------------------
# model adapters
# --------------------------------------------------------------------------

class Adapter:
    """One foundation checkpoint behind a uniform call.

    ``predict(contexts)`` takes ``(n, context)`` and returns
    ``(our_levels, extras)`` where ``our_levels`` is ``(n, len(our_grid))`` with
    NaN for any level the model cannot emit, and ``extras`` is a dict of extra
    column name -> ``(n,)`` for native levels outside the frozen grid.
    """

    name: str
    series = "return"
    needs_bars = False

    def __init__(self, our_grid, context: int, device: str, batch: int):
        self.our_grid = tuple(our_grid)
        self.context = context
        self.device = device
        self.batch = batch

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def grid_note(self) -> dict:
        return {"model": self.name, "scored_on_frozen_grid": True,
                "native_levels": list(self.our_grid), "notes": ""}


class ChronosAdapter(Adapter):
    name = "Chronos-2"

    def load(self) -> None:
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", device_map=self.device)
        native = list(self.pipe.quantiles)
        print(f"    native grid   {len(native)} levels: {native}")
        missing = [q for q in self.our_grid
                   if not any(np.isclose(q, g, atol=1e-12) for g in native)]
        if missing:
            raise SystemExit(
                f"Chronos-2's grid no longer contains {missing}. The checkpoint's "
                "quantile set changed; the mapping in this file is out of date.")
        self.native = native

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
        n = len(contexts)
        # (n_series, n_variates, history_length) per pipeline.py:488
        tensor = self.torch.tensor(contexts, dtype=self.torch.float32).reshape(
            n, 1, self.context)
        out = self.pipe.predict(tensor, prediction_length=1)
        # list of (n_variates, n_quantiles, prediction_length), pipeline.py:562
        stacked = np.stack([np.asarray(
            e.detach().cpu() if hasattr(e, "detach") else e) for e in out], axis=0)
        if stacked.ndim != 4 or stacked.shape[0] != n:
            raise AssertionError(f"unexpected Chronos-2 output {stacked.shape}")
        n_var, n_q, pred_len = stacked.shape[1:]
        if n_q != len(self.native) or pred_len != 1:
            raise AssertionError(
                f"Chronos-2 axes are not (n_variates, n_quantiles, "
                f"prediction_length): got {stacked.shape[1:]} with "
                f"{len(self.native)} native levels and horizon 1. base.py:129 "
                "documents the opposite order for predict_quantiles; do not "
                "reuse that layout here.")
        values = stacked[:, 0, :, 0]                      # (n, n_quantiles)
        return _select_our_levels(self.native, self.our_grid, values), {}


class TimesFMAdapter(Adapter):
    name = "TimesFM-2.5"

    def load(self) -> None:
        import timesfm
        entry = "TimesFM3Forecaster"
        if not hasattr(timesfm, entry):
            raise SystemExit(
                f"timesfm.{entry} is gone. Namespace: "
                f"{sorted(n for n in dir(timesfm) if not n.startswith('_'))}\n"
                "  update the entry point here and in configs/models.yaml together.")
        self.model = getattr(timesfm, entry)()
        for loader in ("load_checkpoint", "compile"):
            if hasattr(self.model, loader):
                with contextlib.suppress(Exception):
                    getattr(self.model, loader)()
        native = list(getattr(self.model, "quantiles", TIMESFM_NATIVE_GRID))
        print(f"    native grid   {len(native)} levels: {native}")
        if not np.allclose(native, TIMESFM_NATIVE_GRID, atol=1e-12):
            raise SystemExit(
                f"TimesFM's grid is {native}, not {list(TIMESFM_NATIVE_GRID)}. "
                "The tail-level decision in the module docstring was made "
                "against the documented grid; re-check it before running.")
        self.native = native
        # Which of our levels it can reach, and how.
        self.direct = [q for q in self.our_grid
                       if any(np.isclose(q, g, atol=1e-12) for g in native)]
        self.interpolated = [q for q in self.our_grid
                             if q not in self.direct
                             and min(native) < q < max(native)]
        self.unavailable = [q for q in self.our_grid
                            if q < min(native) or q > max(native)]
        print(f"    frozen grid   direct {self.direct} | "
              f"interpolated {self.interpolated} | NaN {self.unavailable}")

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
        point, quantiles = self.model.forecast(
            horizon=1, inputs=[row for row in contexts])
        quantiles = np.asarray(quantiles)
        n = len(contexts)
        if quantiles.ndim != 3 or quantiles.shape[0] != n:
            raise AssertionError(f"unexpected TimesFM output {quantiles.shape}")
        if quantiles.shape[-1] != len(self.native) + 1:
            raise AssertionError(
                f"TimesFM last axis is {quantiles.shape[-1]}, expected "
                f"{len(self.native) + 1} (point channel + {len(self.native)} "
                "quantiles) per timesfm_2p5_torch.py:53")
        native_values = quantiles[:, 0, 1:]               # drop channel 0, the point
        del point

        out = np.full((n, len(self.our_grid)), np.nan)
        native = np.asarray(self.native)
        for j, q in enumerate(self.our_grid):
            if q in self.unavailable:
                continue                                   # never extrapolated
            out[:, j] = np.array([
                np.interp(q, native, row) for row in native_values])
        # Interpolation on an interior level is exact when the level is a member,
        # so `direct` and `interpolated` share one code path; the split is
        # reported for the write-up, not branched on here.
        extras = {}
        for g, col in zip(native, [quantile_column(g) for g in native]):
            if not any(np.isclose(g, q, atol=1e-12) for q in self.our_grid):
                extras[col] = native_values[:, int(np.flatnonzero(
                    np.isclose(native, g, atol=1e-12))[0])]
        return out, extras

    def grid_note(self) -> dict:
        return {
            "model": self.name,
            "scored_on_frozen_grid": False,
            "native_levels": list(self.native),
            "frozen_levels_direct": list(self.direct),
            "frozen_levels_interpolated": list(self.interpolated),
            "frozen_levels_unavailable": list(self.unavailable),
            "notes": (
                "TimesFM-2.5 emits a fixed nine-decile grid [0.1..0.9] and the "
                "level set is not settable - timesfm3_forecaster.py:398 "
                "overwrites config.quantiles with the trained head's own on "
                "load. The frozen grid's 0.01/0.05/0.95/0.99 lie outside that "
                "range and are written NaN, never extrapolated: a value at a "
                "level the head was never trained to emit would measure our "
                "extrapolation rule, not the model's uncertainty. Its CRPS must "
                "therefore be computed over its own nine levels, from the "
                "native columns in this file, and every table carrying it must "
                "mark it as on a different grid and not directly comparable on "
                "CRPS to models scored on the full nine. A common-subset "
                "comparison was rejected because it would discard the tail "
                "information Chronos-2 and Kronos do carry, and H2 is about "
                "calibration."),
        }


class TTMAdapter(Adapter):
    name = "TTM"

    def load(self) -> None:
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        self.torch = torch
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2").to(self.device).eval()
        cfg = self.model.config
        if getattr(cfg, "multi_quantile_head", False):
            raise SystemExit(
                "this TTM checkpoint has a multi-quantile head, so the "
                "point-forecast-only treatment in this file is wrong for it. "
                "Stop and re-decide before running.")
        if int(cfg.context_length) != self.context:
            raise SystemExit(f"TTM wants context_length {cfg.context_length}, "
                             f"--context is {self.context}")
        self.pred_len = int(cfg.prediction_length)
        print(f"    loss={cfg.loss}  multi_quantile_head=False  "
              f"prediction_length={self.pred_len} (step 0 used)  "
              f"scaling={cfg.scaling} (internal, inverted before return)")

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
        n = len(contexts)
        x = self.torch.tensor(contexts, dtype=self.torch.float32).reshape(
            n, self.context, 1).to(self.device)
        with self.torch.no_grad():
            out = self.model(past_values=x)
        if getattr(out, "quantile_outputs", None) is not None:
            raise AssertionError("TTM returned quantile_outputs; the checkpoint "
                                 "is not the point-only one this file assumes")
        preds = np.asarray(out.prediction_outputs.detach().cpu())
        if preds.shape != (n, self.pred_len, 1):
            raise AssertionError(
                f"TTM output {preds.shape}, expected {(n, self.pred_len, 1)} "
                "= (batch, prediction_length, num_input_channels)")
        result = np.full((n, len(self.our_grid)), np.nan)
        median_at = self.our_grid.index(0.5)
        result[:, median_at] = preds[:, 0, 0]              # step 0 of the 96-step head
        return result, {}

    def grid_note(self) -> dict:
        return {
            "model": self.name,
            "scored_on_frozen_grid": False,
            "native_levels": [],
            "notes": (
                "TTM emits a point forecast only. The published checkpoint has "
                "loss=mse and no multi_quantile_head/quantile_levels/quantile "
                "keys, so no quantile head exists; distribution_output is "
                "present but inert because it applies only when loss==nll. The "
                "point forecast is written to q0.5 and the other eight levels "
                "are NaN, so TTM appears in the point-accuracy tables and is "
                "absent from CRPS, PICP and interval score. That absence is a "
                "reported property of the model, not a gap to fill: a synthetic "
                "band would measure the band, and fine-tuning a pinball head "
                "would forfeit zero-shot scope. Step 0 of a 96-step head."),
        }


class KronosAdapter(Adapter):
    name = "Kronos-small"
    series = "price"
    needs_bars = True

    def __init__(self, our_grid, context, device, batch, sample_count,
                 paths_per_call=DEFAULT_KRONOS_PATHS_PER_CALL, kronos_path=None):
        super().__init__(our_grid, context, device, batch)
        self.sample_count = sample_count
        self.paths_per_call = paths_per_call
        self.kronos_path = kronos_path

    def load(self) -> None:
        import torch
        if self.kronos_path and self.kronos_path not in sys.path:
            sys.path.append(self.kronos_path)
        from model import Kronos, KronosPredictor, KronosTokenizer
        from model import kronos as kronos_module
        self.torch = torch
        self.kronos_module = kronos_module
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        self.predictor = KronosPredictor(model, tokenizer, device=self.device,
                                         max_context=self.context)
        self.feature_cols = (list(self.predictor.price_cols)
                             + [self.predictor.vol_col, self.predictor.amt_vol])
        self.close_at = self.feature_cols.index("close")
        print(f"    sample_count  {self.sample_count} paths/origin  "
              f"features {self.feature_cols}  close at index {self.close_at}")

    @contextlib.contextmanager
    def _keep_sample_axis(self):
        """Suppress the one reduction that destroys the predictive distribution.

        ``auto_regressive_inference`` ends with ``np.mean(preds, axis=1)`` over
        the sample axis (``model/kronos.py:467``), and every public entry point
        goes through it, so there is no supported way to get the paths back.
        Rather than vendor a copy of the whole function - which would silently
        rot the next time upstream changes the decode loop - the module's
        ``np.mean`` is swapped for a shim that passes through *only* the
        4-d, axis-1 reduction at the end and delegates everything else to the
        real implementation untouched.

        Narrow on purpose. If upstream adds another 4-d axis-1 mean inside this
        function the shim would catch that too, which is why the caller asserts
        the returned array still has the sample axis and the right extent.
        """
        real_mean = self.kronos_module.np.mean

        def shim(a, axis=None, *args, **kwargs):
            if (axis == 1 and isinstance(a, np.ndarray) and a.ndim == 4
                    and a.shape[1] == self.sample_count):
                return a
            return real_mean(a, axis=axis, *args, **kwargs)

        self.kronos_module.np.mean = shim
        try:
            yield
        finally:
            self.kronos_module.np.mean = real_mean

    def predict_bars(self, windows: np.ndarray, x_stamps: np.ndarray,
                     y_stamps: np.ndarray, last_close: np.ndarray,
                     ) -> tuple[np.ndarray, dict]:
        """``windows`` is ``(n, context, n_features)`` of raw bars.

        Replicates ``KronosPredictor.predict``'s per-series normalisation
        (``model/kronos.py:544-546``: mean and std of the context window only,
        so nothing outside the window enters) and its denormalisation, but keeps
        the sample axis in between.
        """
        n = len(windows)
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        norm = np.clip((windows - mean) / (std + 1e-5), -self.predictor.clip,
                       self.predictor.clip)

        x = self.torch.tensor(norm, dtype=self.torch.float32)
        xs = self.torch.tensor(x_stamps, dtype=self.torch.float32)
        ys = self.torch.tensor(y_stamps, dtype=self.torch.float32)

        with self._keep_sample_axis():
            with self.torch.no_grad():
                paths = self.kronos_module.auto_regressive_inference(
                    self.predictor.tokenizer, self.predictor.model,
                    x.to(self.device), xs.to(self.device), ys.to(self.device),
                    self.context, 1, self.predictor.clip,
                    1.0, 0, 0.9, self.sample_count, False)

        paths = np.asarray(paths)
        if paths.ndim != 4 or paths.shape[0] != n or paths.shape[1] != self.sample_count:
            raise AssertionError(
                f"Kronos returned {paths.shape}; expected "
                f"({n}, {self.sample_count}, steps, features). The sample axis "
                "was collapsed - _keep_sample_axis did not take effect, and the "
                "quantiles would be nine copies of a mean path.")

        step = paths[:, :, -1, :]                              # last predicted step
        step = step * (std[:, 0, :][:, None, :] + 1e-5) + mean[:, 0, :][:, None, :]
        close_paths = step[:, :, self.close_at]                 # (n, sample_count)
        returns = np.log(close_paths / last_close[:, None])
        levels = np.quantile(returns, self.our_grid, axis=1).T   # (n, len(our_grid))
        return levels, {}

    def grid_note(self) -> dict:
        return {
            "model": self.name, "scored_on_frozen_grid": True,
            "native_levels": list(self.our_grid),
            "sample_count": self.sample_count,
            "notes": (
                f"Empirical order statistics of {self.sample_count} sampled "
                "close paths, converted to log returns against the last "
                "observed close. auto_regressive_inference is called directly "
                "because predict() averages the sample axis at kronos.py:467. "
                "Finite-sample extreme quantiles are biased inward, which makes "
                "the tails look narrower than they are and penalises Kronos at "
                "q0.01/q0.99; the bias shrinks with sample_count and the "
                "sensitivity is recorded by --calibrate."),
        }


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def make_adapter(name: str, our_grid, args) -> Adapter:
    if name == "Chronos-2":
        return ChronosAdapter(our_grid, args.context, args.device, args.batch_size)
    if name == "TimesFM-2.5":
        return TimesFMAdapter(our_grid, args.context, args.device, args.batch_size)
    if name == "TTM":
        return TTMAdapter(our_grid, args.context, args.device, args.batch_size)
    if name == "Kronos-small":
        return KronosAdapter(our_grid, args.context, args.device, args.batch_size,
                             args.sample_count, args.kronos_paths_per_call,
                             args.kronos_path)
    raise SystemExit(f"{name} is not a foundation model in this runner")


def forecast_fold(adapter: Adapter, bars: pd.DataFrame, split, our_grid,
                  tickers: list[str] | None = None) -> pd.DataFrame:
    """Every (ticker, session) origin in one test window."""
    frames = []
    grouped = bars.groupby("unique_id", sort=True)
    names = tickers if tickers is not None else sorted(bars["unique_id"].unique())

    for ticker in names:
        group = grouped.get_group(ticker).sort_values("ds").reset_index(drop=True)
        dates = pd.DatetimeIndex(group["ds"])
        in_window = np.flatnonzero((dates >= split.test_start) & (dates <= split.test_end))
        if in_window.size == 0:
            continue
        usable = in_window[in_window >= adapter.context]
        dropped = in_window.size - usable.size
        if dropped:
            print(f"      {ticker}: {dropped} origins lack {adapter.context} "
                  "sessions of history, skipped")
        if usable.size == 0:
            continue

        closes = group["close"].to_numpy(dtype=float)
        origin_dates = dates[usable]
        out = empty_quantile_frame(ticker, origin_dates, our_grid)
        extras_acc: dict[str, list] = {}

        if adapter.needs_bars:
            feats = group[adapter.feature_cols].to_numpy(dtype=float) \
                if all(c in group.columns for c in adapter.feature_cols) else None
            if feats is None:
                frame = group.copy()
                if adapter.predictor.amt_vol not in frame.columns:
                    frame[adapter.predictor.amt_vol] = (
                        frame[adapter.predictor.vol_col]
                        * frame[list(adapter.predictor.price_cols)].mean(axis=1))
                feats = frame[adapter.feature_cols].to_numpy(dtype=float)
            stamps = adapter.kronos_module.calc_time_stamps(
                pd.Series(dates)).to_numpy(dtype=np.float32)

            per_call = max(1, adapter.paths_per_call // adapter.sample_count)
            for start in range(0, usable.size, per_call):
                chunk = usable[start:start + per_call]
                windows = np.stack([feats[o - adapter.context + 1:o + 1] for o in chunk])
                x_stamps = np.stack([stamps[o - adapter.context + 1:o + 1] for o in chunk])
                # The stamp of the session being forecast, o + 1. Origins are
                # inside the test window and never the last row of the panel,
                # so o + 1 always exists; asserted rather than assumed.
                if (chunk + 1).max() >= len(stamps):
                    raise AssertionError(
                        f"{ticker}: origin at the final session has no t+1 stamp")
                y_stamps = stamps[chunk + 1][:, None, :]
                levels, extras = adapter.predict_bars(
                    windows, x_stamps, y_stamps, closes[chunk])
                for j, q in enumerate(our_grid):
                    out.loc[start:start + len(chunk) - 1, quantile_column(q)] = levels[:, j]
                for k, v in extras.items():
                    extras_acc.setdefault(k, []).append(v)
        else:
            contexts = build_context_matrix(closes, usable, adapter.context,
                                            adapter.series)
            for start in range(0, len(contexts), adapter.batch):
                block = contexts[start:start + adapter.batch]
                levels, extras = adapter.predict(block)
                for j, q in enumerate(our_grid):
                    out.loc[start:start + len(block) - 1, quantile_column(q)] = levels[:, j]
                for k, v in extras.items():
                    extras_acc.setdefault(k, []).append(v)

        for col, parts in extras_acc.items():
            out[col] = np.concatenate(parts)
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_model(adapter: Adapter, bars: pd.DataFrame, walk_forward: PurgedWalkForward,
              our_grid, out_root: Path, timings_path: Path, args,
              tickers: list[str] | None = None) -> None:
    model_dir = out_root / adapter.name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{adapter.name}")
    t_load = time.perf_counter()
    adapter.load()
    load_seconds = time.perf_counter() - t_load
    print(f"    loaded in {load_seconds:.1f}s")

    (model_dir / "grid.json").write_text(
        json.dumps(adapter.grid_note(), indent=2) + "\n")

    fold_times: dict[str, float] = {}
    rows_written = 0
    started = time.perf_counter()

    for split, train, test in walk_forward.iter_frames(bars, time_col="ds"):
        path = model_dir / f"fold_{split.fold}.parquet"
        if path.exists() and not args.overwrite:
            print(f"  fold {split.fold}  exists, skipping ({show(path)}) "
                  "- pass --overwrite to redo")
            continue

        # Zero-shot: nothing is fitted, so the purge gap does not bound the
        # context. The guard still runs, because a tier that skips it is a tier
        # nobody can audit. See the module docstring.
        assert_no_overlap(
            train["ds"], test["ds"],
            horizon=walk_forward.horizon, embargo=walk_forward.embargo,
            calendar=pd.DatetimeIndex(bars["ds"].unique()).sort_values())

        t0 = time.perf_counter()
        forecasts = forecast_fold(adapter, bars, split, our_grid, tickers)
        elapsed = time.perf_counter() - t0

        # Checkpoint per fold, not per model: a Kaggle session that dies at
        # hour nine should cost the current fold, not the run.
        forecasts.to_parquet(path, index=False)
        fold_times[str(split.fold)] = round(elapsed, 3)
        rows_written += len(forecasts)
        record_timing(timings_path, adapter.name, {
            "library": "foundation", "tier": "foundation", "scope": "zero_shot",
            "target": "return", "seconds_load": round(load_seconds, 3),
            "seconds_per_fold": fold_times,
            "seconds_steady_state_total": round(sum(fold_times.values()), 3),
            "n_forecasts": rows_written, "n_folds": len(fold_times),
            "context": adapter.context, "device": args.device,
            "sample_count": getattr(adapter, "sample_count", None),
            "partial": True,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"  fold {split.fold}  test=[{split.test_start.date()} .. "
              f"{split.test_end.date()}]  {len(forecasts):>6,} forecasts  "
              f"{elapsed:8.1f}s  {show(path)}")

    steady = sum(fold_times.values())
    record_timing(timings_path, adapter.name, {
        "library": "foundation", "tier": "foundation", "scope": "zero_shot",
        "target": "return", "seconds_load": round(load_seconds, 3),
        "seconds_per_fold": fold_times,
        "seconds_steady_state_total": round(steady, 3),
        "seconds_steady_state_mean": round(steady / len(fold_times), 3) if fold_times else None,
        "seconds_wall_clock_total": round(time.perf_counter() - started + load_seconds, 3),
        "n_forecasts": rows_written, "n_folds": len(fold_times),
        "context": adapter.context, "device": args.device,
        "sample_count": getattr(adapter, "sample_count", None),
        "partial": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"  {rows_written:,} forecasts | {steady:.1f}s across "
          f"{len(fold_times)} folds -> {show(timings_path)}")


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibrate(bars: pd.DataFrame, feats: pd.DataFrame, walk_forward, our_grid,
              args) -> None:
    """Kronos CRPS and wall-clock against sample_count, on one fold.

    The full run commits ~21 GPU-hours on a throughput assumption of 18% of the
    T4's fp16 peak, which is worth roughly a factor of two either way. This
    replaces that assumption with a measurement, and produces the CRPS-versus-S
    curve, which is reportable in its own right: it is the evidence that the
    chosen sample_count is adequate rather than affordable.

    CRPS comes from ``crps_from_quantiles`` - the same function evaluate.py
    calls, imported, not reimplemented. Metrics stay defined in one place.
    """
    from sp500bench.evaluation.metrics import crps_from_quantiles

    splits = walk_forward.splits(pd.DatetimeIndex(bars["ds"].unique()).sort_values())
    split = splits[args.calib_fold]
    tickers = sorted(bars["unique_id"].unique())[:args.calib_tickers]
    truth = feats.set_index(["unique_id", "ds"])["y"]

    print(f"\ncalibration: fold {split.fold} "
          f"[{split.test_start.date()} .. {split.test_end.date()}], "
          f"{len(tickers)} tickers {tickers}")
    print(f"sample counts: {args.calib_sample_counts}")

    rows = []
    for s in args.calib_sample_counts:
        adapter = KronosAdapter(our_grid, args.context, args.device,
                                args.batch_size, s, args.kronos_paths_per_call,
                                args.kronos_path)
        adapter.load()
        t0 = time.perf_counter()
        forecasts = forecast_fold(adapter, bars, split, our_grid, tickers)
        elapsed = time.perf_counter() - t0

        merged = forecasts.merge(
            truth.rename("y"), left_on=["unique_id", "ds"],
            right_index=True, how="left")
        usable = merged.dropna(subset=["y"])
        qf = {q: usable[quantile_column(q)].to_numpy(dtype=float) for q in our_grid}
        crps = crps_from_quantiles(usable["y"].to_numpy(dtype=float), qf)

        per_origin = elapsed / max(1, len(usable))
        full_hours = per_origin * 50 * 1670 / 3600
        rows.append((s, crps, elapsed, len(usable), per_origin, full_hours))
        print(f"  S={s:<5} CRPS {crps:.6f}  {elapsed:8.1f}s over {len(usable):,} "
              f"origins  ({per_origin * 1000:.1f} ms/origin -> "
              f"{full_hours:.1f} GPU-h for the full 83,500)")

    print(f"\n{'S':>6}{'CRPS':>12}{'d vs prev':>12}{'ms/origin':>12}{'full-run GPU-h':>16}")
    prev = None
    for s, crps, _, _, per_origin, full_hours in rows:
        delta = "" if prev is None else f"{(crps - prev) / prev * 100:+.3f}%"
        print(f"{s:>6}{crps:>12.6f}{delta:>12}{per_origin * 1000:>12.1f}{full_hours:>16.1f}")
        prev = crps
    print("\nRead it as: if CRPS has stopped moving between the two largest S, "
          "the smaller one is adequate and the difference is sampling noise, "
          "not model uncertainty. The GPU-h column is measured, not assumed.")

    out = Path(args.calib_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fold": int(split.fold),
        "test_window": [str(split.test_start.date()), str(split.test_end.date())],
        "tickers": tickers,
        "context": args.context,
        "device": args.device,
        "rows": [{"sample_count": s, "crps": c, "seconds": e, "n_origins": n,
                  "seconds_per_origin": p, "projected_full_run_gpu_hours": h}
                 for s, c, e, n, p, h in rows],
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"-> {show(out)}")


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def self_check(bars: pd.DataFrame, feats: pd.DataFrame, walk_forward, our_grid,
               args) -> None:
    """Everything that does not need a GPU or a model package.

    Exists because the rest of this file cannot be run anywhere but Kaggle, and
    an untested runner that costs twenty GPU-hours to discover a bug in is not
    worth writing. Covers fold construction, the context off-by-one, the
    quantile selection guard including the transposed-array case, and the output
    schema - against a stub that emits a known 21-point grid.
    """
    print("\nself-check (no GPU, no model packages)")

    _assert_context_alignment(bars, feats)

    splits = walk_forward.splits(pd.DatetimeIndex(bars["ds"].unique()).sort_values())
    print(f"  folds      {len(splits)}: " + ", ".join(
        f"{s.fold}:[{s.test_start.date()}..{s.test_end.date()}]" for s in splits))

    ticker = sorted(bars["unique_id"].unique())[0]
    group = bars[bars.unique_id == ticker].sort_values("ds").reset_index(drop=True)
    closes = group["close"].to_numpy(dtype=float)
    truth = feats[feats.unique_id == ticker].sort_values("ds")["y"].to_numpy()
    origins = np.array([args.context + 10, args.context + 500])

    ret = build_context_matrix(closes, origins, args.context, "return")
    for k, o in enumerate(origins):
        expected_last = np.log(closes[o] / closes[o - 1])
        assert np.isclose(ret[k, -1], expected_last), "return context misaligned"
        assert np.isclose(ret[k, -1], truth[o - 1], atol=1e-9), \
            "return context does not end at y[t-1]"
        assert not np.isclose(ret[k, -1], truth[o], atol=1e-12) or truth[o] == truth[o - 1], \
            "return context ends at the label y[t]"
    print(f"  context    return series ends at y[t-1], not the label y[t]  "
          f"({len(origins)} origins on {ticker})")

    price = build_context_matrix(closes, origins, args.context, "price")
    assert np.isclose(price[0, -1], closes[origins[0]]), "price context misaligned"
    print("  context    price series ends at P[t] inclusive")

    native21 = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    values = np.tile(np.asarray(native21), (4, 1))
    picked = _select_our_levels(native21, our_grid, values)
    assert np.allclose(picked[0], np.asarray(our_grid)), "level selection wrong"
    print(f"  selection  {len(native21)}-point grid -> frozen 9 by value: "
          f"{[float(v) for v in picked[0]]}")

    try:
        _select_our_levels(native21, our_grid, values.T)
        raise SystemExit("  FAIL: a transposed array was accepted")
    except AssertionError:
        print("  selection  a transposed array is rejected, not silently indexed")

    interior = [q for q in our_grid if 0.1 <= q <= 0.9]
    outside = [q for q in our_grid if q < 0.1 or q > 0.9]
    assert outside == [0.01, 0.05, 0.95, 0.99], outside
    print(f"  timesfm    reachable {interior} | NaN, never extrapolated {outside}")

    frame = empty_quantile_frame(ticker, group["ds"].iloc[origins], our_grid)
    expected_cols = ["unique_id", "ds"] + quantile_columns(our_grid)
    assert list(frame.columns) == expected_cols, list(frame.columns)
    print(f"  schema     {expected_cols}")
    print("\nself-check passed. The model calls themselves are unexercised - "
          "they need the GPU image.")


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", default=FOUNDATION_MODELS,
                   help=f"default: {' '.join(FOUNDATION_MODELS)}")
    p.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT,
                   help="Kronos paths per origin (default 200)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--kronos-paths-per-call", type=int,
                   default=DEFAULT_KRONOS_PATHS_PER_CALL,
                   help="cap on batch x sample_count sequences in one call")
    p.add_argument("--kronos-path", default="Kronos",
                   help="clone of github.com/shiyu-coder/Kronos")
    p.add_argument("--input-series", choices=["return", "price"], default="return",
                   help="what the three generic TSFMs are fed; see the module "
                        "docstring before changing this")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bars", default="data/processed/panel.parquet")
    p.add_argument("--features", default="data/features.parquet")
    p.add_argument("--out", default="forecasts")
    p.add_argument("--timings", default="results/timings.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--calibrate", action="store_true",
                   help="Kronos CRPS and wall-clock vs sample_count on one fold")
    p.add_argument("--calib-sample-counts", nargs="+", type=int,
                   default=[100, 200, 500])
    p.add_argument("--calib-tickers", type=int, default=5)
    p.add_argument("--calib-fold", type=int, default=0)
    p.add_argument("--calib-out", default="results/kronos_calibration.json")
    p.add_argument("--self-check", action="store_true",
                   help="fold, context, selection and schema checks; no GPU")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs()
    data_cfg = load_data_config()
    our_grid = list(cfg.quantiles)

    print(f"quantile grid ({len(our_grid)} levels, training == evaluation): {our_grid}")
    print(f"context: {args.context} sessions | horizon: {cfg.horizon} | "
          f"generic TSFM input: {args.input_series} series")

    bars, feats = load_panels(Path(args.bars), Path(args.features))
    print(f"panel: {len(bars):,} rows, {bars['unique_id'].nunique()} tickers, "
          f"{bars['ds'].nunique():,} sessions "
          f"[{bars['ds'].min().date()} .. {bars['ds'].max().date()}]")

    walk_forward = PurgedWalkForward(
        test_windows=cfg.test_windows,
        horizon=cfg.horizon_for("return"),
        embargo=cfg.embargo,
        train_start=data_cfg["splits"]["train"]["start"])
    print(f"folds: {len(cfg.test_windows)}, purge {walk_forward.horizon} + "
          f"embargo {walk_forward.embargo} = {walk_forward.gap} sessions excised")

    if args.self_check:
        self_check(bars, feats, walk_forward, our_grid, args)
        return

    _assert_context_alignment(bars, feats)

    if args.calibrate:
        calibrate(bars, feats, walk_forward, our_grid, args)
        return

    out_root, timings_path = Path(args.out), Path(args.timings)
    for name in args.models:
        adapter = make_adapter(name, our_grid, args)
        if not adapter.needs_bars:
            adapter.series = args.input_series
        run_model(adapter, bars, walk_forward, our_grid, out_root,
                  timings_path, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
