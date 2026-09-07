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
import importlib.metadata
import inspect
import json
import os
import subprocess
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
from sp500bench.evaluation.scale import (Q90_Q10_TO_SIGMA,  # noqa: E402,F401
                                         SIGMA_BANDS, SigmaOutOfBand,
                                         check_sigma_band, implied_daily_sigma)
from sp500bench.models.registry import BY_NAME  # noqa: E402

FOUNDATION_MODELS = ["Chronos-2", "TimesFM-2.5", "TTM", "Kronos-small"]

# TimesFM's own grid, from TimesFM_2p5_200M_Definition.quantiles. Asserted
# against the loaded checkpoint before use.
TIMESFM_NATIVE_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"

# The class that matches the checkpoint. It was in the namespace from the first
# smoke test and went unused for two runs while TimesFM3Forecaster - the 3.0
# wrapper - was used instead. Its own DEFAULT_REPO_ID is this checkpoint.
#
# The two are not interchangeable. The 3.0 wrapper builds a TimesFM3Torch from
# the repo's config.json, and that config's keys are the 2.5 architecture's
# names, of which exactly one (`quantiles`) is also a TimesFM3Torch constructor
# argument. Every other architecture field was therefore dropped and the model
# was built at 3.0 defaults, so the 2.5 weights had nowhere to land.
TIMESFM_ENTRY_POINT = "TimesFM_2p5_200M_torch"

# `forecast` first because it is this class's own inference method, inherited
# from TimesFM_2p5 and requiring compile() beforehand. The others stay as
# fallbacks in case a future release moves the surface; discovery still reports
# whichever it finds rather than this file assuming one.
TIMESFM_METHOD_PREFERENCE = ("forecast", "predict_batch", "predict",
                             "generate", "__call__")

# Names worth dumping the signature of when discovery fails.
INFERENCE_NAME_HINTS = ("forecast", "predict", "call", "compile", "decode",
                        "generate", "infer", "run")

DEFAULT_CONTEXT = 512          # Kronos' documented context_limit; TTM's fixed
                               # context_length. Shared so the four models see
                               # the same history and the comparison is on the
                               # model, not on how much past it was handed.
# 100 paths, on the full test period, rather than more paths on a narrower one.
# Read from the roster so the runner and the frozen spec cannot drift; the
# literal is the fallback if the spec ever stops carrying it.
DEFAULT_SAMPLE_COUNT = BY_NAME["Kronos-small"].params.get("sample_count", 100)
KRONOS_SAMPLE_RATIONALE = BY_NAME["Kronos-small"].params.get(
    "sample_count_rationale", "")
DEFAULT_BATCH = 256

# Kronos' memory model, which is not the same as the other three.
#
# ``auto_regressive_inference`` expands its input to ``batch * sample_count``
# along the batch axis (``model/kronos.py:394``) before the tokenizer encodes
# anything, so the sequences actually resident are the *product*, not the origin
# batch. At sample_count=100 with an origin batch of 40 that is 4,000 sequences
# of 512 bars entering rotary attention in one pass, which is what put a T4 into
# OOM trying to allocate 1.95 GiB against 1.52 GiB free.
#
# Two ceilings, because there are two axes to cap:
#   PATHS_PER_CALL   how many sample paths go in one call. The sample dimension
#                    is chunked at this width and the chunks are concatenated.
#   SEQUENCE_BUDGET  the ceiling on origin_batch * paths_per_call, i.e. on the
#                    sequences resident in a single call. The origin batch is
#                    derived from it rather than set independently.
#
# Chunking the sample dimension changes which draws are realised - the chunks
# are separate stochastic calls - but not the distribution they are drawn from,
# so CRPS is unaffected in expectation. It does mean a run is only reproducible
# at the same --paths-per-call, which is why both settings are recorded in
# grid.json and in the timings.
DEFAULT_KRONOS_PATHS_PER_CALL = 64
DEFAULT_KRONOS_SEQUENCE_BUDGET = 512
MIN_KRONOS_SEQUENCE_BUDGET = 32


def show(path: Path) -> str:
    try:
        return f"{path.relative_to(Path.cwd())}"
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# timings
# --------------------------------------------------------------------------

def _checkpoint_config(repo_id: str | None) -> dict | None:
    """The checkpoint's own config.json, from the local cache.

    What the checkpoint declares about itself, as opposed to what the class the
    runner happens to construct defaults to. Those are different objects, and
    the difference is exactly what went unnoticed on TimesFM.
    """
    if not repo_id:
        return None
    revision, _ = _hf_cache_revision(repo_id)
    roots = [os.environ.get("HF_HUB_CACHE"),
             os.environ.get("HUGGINGFACE_HUB_CACHE"),
             os.path.join(os.environ["HF_HOME"], "hub")
             if os.environ.get("HF_HOME") else None,
             os.path.expanduser("~/.cache/huggingface/hub")]
    folder = "models--" + repo_id.replace("/", "--")
    for root in roots:
        if not root:
            continue
        base = Path(root) / folder
        candidates = []
        if revision:
            candidates.append(base / "snapshots" / revision / "config.json")
        candidates.extend(sorted(base.glob("snapshots/*/config.json")))
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return json.loads(candidate.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
    return None


# A checkpoint's state dict addresses tensors inside submodules, so its keys
# carry module paths and there are many of them. A handful of bare names -
# 'weight', 'bias' - is the entire key set of one small module loading itself,
# which is a different event and says nothing about the checkpoint.
PRINCIPAL_LOAD_MIN_KEYS = 8


class _StateDictLoadRecorder:
    """Capture what ``load_state_dict`` reported, which the loaders discard.

    ``PyTorchModelHubMixin`` and friends call ``load_state_dict`` internally and
    throw away its return value, so a non-strict load that matched almost
    nothing looks exactly like a successful one - the model is constructed, the
    call returns, and the weights are whatever the constructor initialised. That
    is how a checkpoint can fail to load and still produce forecasts.

    ``torch.nn.Module.load_state_dict`` is patched for the duration of the load
    and restored in ``finally``. Narrow and temporary, for the same reason the
    numpy proxy is: the blast radius of a patched framework method should be one
    call, not a process.

    WHICH CALL IS THE CHECKPOINT'S
    ------------------------------
    Not every call to this method carries a checkpoint. Constructing a model
    also loads small submodules that own their weights - a projection, a norm,
    a scaler - and those legitimately have nothing in the checkpoint file. The
    first version of this recorder asserted on all of them and refused two
    checkpoints that were demonstrably fine, both reporting the same two missing
    keys, ``bias`` and ``weight``, from libraries that share no code. Bare names
    with no module path are the signature: a checkpoint's state dict addresses
    tensors *inside* submodules, so its keys carry paths and there are many.

    So a call is treated as principal only when its incoming state dict both
    carries module paths and holds more than a handful of keys. Everything else
    is recorded and printed, never asserted on.

    AND SOMETIMES THERE IS NO SUCH CALL
    -----------------------------------
    Several loaders never reach this method at all. transformers 5.x populates
    parameters through ``set_module_tensor_to_device`` and has no
    ``load_state_dict`` call in ``modeling_utils`` whatsoever, which covers both
    the Chronos-2 pipeline and TTM. When no principal load is observed this
    reports that the load could not be verified *here* rather than passing
    silently - the absence of evidence is itself worth printing, because it says
    which other check is carrying the weight for that model.
    """

    def __init__(self, torch_module):
        self.torch = torch_module
        self.calls: list[dict] = []

    def __enter__(self):
        real = self.torch.nn.Module.load_state_dict
        self._real = real
        calls = self.calls

        def recording(module_self, state_dict, *args, **kwargs):
            out = real(module_self, state_dict, *args, **kwargs)
            try:
                keys = list(state_dict.keys())
            except AttributeError:                       # not a mapping
                keys = []
            calls.append({
                "module": type(module_self).__name__,
                "n_incoming": len(keys),
                "dotted": any("." in str(k) for k in keys),
                "missing": list(getattr(out, "missing_keys", []) or []),
                "unexpected": list(getattr(out, "unexpected_keys", []) or []),
            })
            return out

        self.torch.nn.Module.load_state_dict = recording
        return self

    def __exit__(self, *exc):
        self.torch.nn.Module.load_state_dict = self._real
        return False

    def principal_calls(self) -> list[dict]:
        """The calls whose incoming state dict looks like a whole checkpoint."""
        return [c for c in self.calls
                if c["dotted"] and c["n_incoming"] >= PRINCIPAL_LOAD_MIN_KEYS]

    def summary(self) -> str:
        """One line, grouped by module type.

        Chronos-2 makes 170 of these calls and TTM 134; across nineteen models
        a per-call listing is thousands of lines of noise that hides anything
        real. The grouping is the readable form and the per-call detail is kept
        for :meth:`detail`, which is printed only when something is missing.
        """
        if not self.calls:
            return "no load_state_dict calls"
        by_module: dict[str, int] = {}
        for call in self.calls:
            by_module[call["module"]] = by_module.get(call["module"], 0) + 1
        parts = [f"{name} x{n}" if n > 1 else name
                 for name, n in sorted(by_module.items(),
                                       key=lambda kv: (-kv[1], kv[0]))]
        return f"{len(self.calls)} calls: " + ", ".join(parts)

    def detail(self, limit: int = 20) -> list[str]:
        """Per-call lines, for a failure that needs the individual calls."""
        lines = []
        for call in self.calls[:limit]:
            shape = "paths" if call["dotted"] else "bare"
            lines.append(f"      {call['module']}({call['n_incoming']} {shape} "
                         f"keys, -{len(call['missing'])}/"
                         f"+{len(call['unexpected'])})")
        if len(self.calls) > limit:
            lines.append(f"      ... and {len(self.calls) - limit} more")
        return lines

    def assert_clean(self, model_name: str, checkpoint: str | None) -> None:
        principal = self.principal_calls()
        if not principal:
            print(f"    weights       none checkpoint-shaped ({self.summary()}); "
                  "this loader places")
            print("                  parameters without load_state_dict, so the "
                  "weights are not")
            print("                  verifiable here - the channel probe against "
                  "the checkpoint's")
            print("                  own config is what covers this model.")
            return

        missing = sorted({k for c in principal for k in c["missing"]})
        unexpected = sorted({k for c in principal for k in c["unexpected"]})
        if not (missing or unexpected):
            print(f"    weights       {len(principal)} checkpoint-shaped load(s) "
                  f"clean ({self.summary()})")
            return
        # Only here is the per-call listing worth its space.
        raise SystemExit(
            f"{model_name}: the checkpoint did not load cleanly from "
            f"{checkpoint!r}.\n"
            f"  missing keys    ({len(missing)}): {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected[:8]}"
            f"{' ...' if len(unexpected) > 8 else ''}\n"
            f"  {self.summary()}\n" + "\n".join(self.detail()) + "\n"
            "  A non-strict load leaves the unmatched parameters at their\n"
            "  constructor initialisation, so the model runs and produces\n"
            "  forecasts that are not this checkpoint's. Refusing to continue.")


def _distribution_version(dist: str | None) -> str | None:
    if not dist:
        return None
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hf_cache_revision(repo_id: str | None) -> tuple[str | None, str]:
    """The commit the checkpoint was actually resolved to, from the local cache.

    A repo id alone does not identify a checkpoint: ``main`` moves. Read from the
    cache rather than the Hub so it works with the network off and so it reports
    the commit this machine really has, not the one the Hub serves now.

    Returns ``(sha, source)``; the source is recorded even when the sha is not
    found, so a null revision is distinguishable from one nobody looked for.
    """
    if not repo_id:
        return None, "no checkpoint declared"
    roots = [os.environ.get("HF_HUB_CACHE"),
             os.environ.get("HUGGINGFACE_HUB_CACHE"),
             os.path.join(os.environ["HF_HOME"], "hub")
             if os.environ.get("HF_HOME") else None,
             os.path.expanduser("~/.cache/huggingface/hub")]
    folder = "models--" + repo_id.replace("/", "--")
    for root in roots:
        if not root:
            continue
        refs = Path(root) / folder / "refs" / "main"
        if refs.is_file():
            try:
                return refs.read_text().strip(), f"hf cache {refs}"
            except OSError:
                continue
    return None, "not in the local hf cache"


def _git_head(path: str | None) -> str | None:
    """HEAD of a checked-out repository - Kronos ships as a clone, not a wheel."""
    if not path or not Path(path).is_dir():
        return None
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _torch_version() -> str | None:
    try:
        import torch
        return str(torch.__version__)
    except Exception:                                  # pragma: no cover - no torch locally
        return None


# Fields that identify the run a timing belongs to. Per-fold times measured
# under different settings are not the same measurement and must not be merged.
RUN_IDENTITY_FIELDS = ("context", "device", "sample_count", "precision",
                       "paths_per_call", "checkpoint", "checkpoint_revision",
                       # What the generic TSFMs were fed - returns or prices -
                       # is a documented decision of this runner and changes
                       # what the model was asked to do, not merely how fast it
                       # did it. Folds measured under different input series
                       # are different measurements.
                       "input_series")


def _merge_timing(prior: dict | None, payload: dict) -> dict:
    """Never let a call that measured less overwrite one that measured more.

    A run whose folds already exist on disk skips every one of them, so it
    reaches the end of ``run_model`` with an empty ``seconds_per_fold`` and
    writes ``n_folds=0, n_forecasts=0`` over a complete record. That is how the
    TTM entry came to claim nothing had run while 83,450 forecasts sat in
    ``forecasts/TTM/``: the loss is silent, and the number it leaves behind
    looks like a measurement rather than an absence.

    Three cases:

    * nothing measured now, something measured before - keep the prior record
      whole and note that a later invocation ran no folds;
    * something measured now under the same settings - merge per fold, so
      re-running one fold updates that fold and leaves the others standing;
    * something measured now under different settings - replace, because the
      old per-fold times describe a different configuration, and say so.
    """
    if prior is None:
        return payload

    measured_now = dict(payload.get("seconds_per_fold") or {})
    measured_before = dict(prior.get("seconds_per_fold") or {})

    if not measured_now and measured_before:
        kept = dict(prior)
        kept["last_invocation_ran_no_folds_at"] = payload.get("recorded_at")
        kept["last_invocation_note"] = (
            "a later run loaded the model but executed no folds - every fold "
            "already existed on disk and --overwrite was not passed. The timing "
            "below is from the run that produced the forecasts.")
        return kept

    identity_now = {k: payload.get(k) for k in RUN_IDENTITY_FIELDS}
    identity_before = {k: prior.get(k) for k in RUN_IDENTITY_FIELDS}
    if measured_before and identity_now != identity_before:
        changed = [k for k in RUN_IDENTITY_FIELDS
                   if identity_now.get(k) != identity_before.get(k)]
        merged = dict(payload)
        merged["replaced_prior_record"] = True
        merged["replaced_because"] = (
            f"run identity changed ({', '.join(changed)}); the previous "
            "per-fold times measured a different configuration and were not "
            "merged into these")
        return merged

    merged_folds = {**measured_before, **measured_now}
    prior_rows = prior.get("forecasts_per_fold") or {}
    payload_rows = payload.get("forecasts_per_fold") or {}

    out = dict(payload)
    out["seconds_per_fold"] = merged_folds
    out["n_folds"] = len(merged_folds)

    if prior_rows or not measured_before:
        # Both sides attribute rows per fold, so the total is recomputable.
        merged_rows = {**prior_rows, **payload_rows}
        if set(merged_rows) != set(merged_folds):
            raise ValueError(
                f"timing merge is inconsistent: folds {sorted(merged_folds)} "
                f"have times but {sorted(merged_rows)} have row counts. "
                "Writing a total over a fold set that does not match the "
                "timed set produces a record that contradicts itself, which "
                "is worse than failing here.")
        out["forecasts_per_fold"] = merged_rows
        out["n_forecasts"] = sum(merged_rows.values())
    else:
        # The prior predates forecasts_per_fold, so its rows cannot be split by
        # fold and summing only the new fold would write that fold's count over
        # the whole run's - 12,650 over 83,450, with nothing saying so.
        # Re-measuring a fold does not change how many rows it produced, so the
        # prior total stands; only genuinely new folds add to it.
        new_folds = set(payload_rows) - set(measured_before)
        out["forecasts_per_fold"] = payload_rows
        out["n_forecasts"] = (int(prior.get("n_forecasts", 0))
                              + sum(payload_rows[f] for f in new_folds))
        out["n_forecasts_from_legacy_prior"] = True
        out["n_forecasts_note"] = (
            "the prior record predates forecasts_per_fold, so rows could not "
            "be attributed per fold; the prior total was carried forward and "
            "only folds it did not contain were added to it")

    total = sum(merged_folds.values())
    out["seconds_steady_state_total"] = round(total, 3)
    out["seconds_steady_state_mean"] = (round(total / len(merged_folds), 3)
                                        if merged_folds else None)
    if set(measured_now) != set(merged_folds):
        out["merged_with_prior_folds"] = sorted(set(merged_folds) - set(measured_now))
    return out


def write_grid_note(path: Path, note: dict) -> None:
    """Write the grid note, never replacing a resolved value with a null.

    The note is written once before the first fold so it exists if the run
    dies, and again after each fold once values that can only be known by
    running - the channel layout offset, resolved on the first batch - have
    been settled. Without the merge the second write would be fine but a later
    load-only invocation would put the nulls back, which is the same failure
    the timing record already had: a file that claims a thing was verified and
    carries no value for it.
    """
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    merged = dict(existing)
    for key, value in note.items():
        if value is None and existing.get(key) is not None:
            continue
        merged[key] = value
    path.write_text(json.dumps(merged, indent=2) + "\n")


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
    merged = _merge_timing(existing.get(model), payload)
    if merged.get("last_invocation_note"):
        print(f"  timings: kept the existing {model} record "
              f"({merged.get('n_folds')} folds, {merged.get('n_forecasts', 0):,} "
              "forecasts) - this invocation ran no folds")
    existing[model] = merged
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
    # Which checkpoint, and which distribution provides the code that runs it.
    # Recorded per run: a repo id does not identify a checkpoint on its own,
    # because the branch it resolves to moves, and a forecast whose provenance
    # is only "the model name" cannot be reproduced or audited later.
    checkpoint: str | None = None
    distribution: str | None = None

    def __init__(self, our_grid, context: int, device: str, batch: int):
        self.our_grid = tuple(our_grid)
        self.context = context
        self.device = device
        self.batch = batch

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    arm = "return"                 # every foundation spec is on the return arm
    _channels_checked = False
    _scale_checked = False

    def expected_channels(self, config: dict) -> int | None:
        """How many output channels this checkpoint's own config implies.

        ``None`` means the config does not say, and the check is skipped rather
        than guessed. Overridden per adapter; the point is that the number comes
        from the checkpoint, not from the class the runner constructed - those
        are different objects and their disagreement is silent.
        """
        return None

    def declared_config(self) -> dict | None:
        """What the checkpoint says about itself, not what the class defaults to."""
        return _checkpoint_config(self.checkpoint)

    def probe_output_channels(self) -> int | None:
        """One forward pass at load time to fix the channel count, or None.

        ``None`` means this model cannot be probed cheaply - Kronos would have
        to sample - and the check falls back to the first real batch.
        """
        return None

    def assert_channels_match_checkpoint(self, observed: int) -> None:
        config = self.declared_config()
        if config is None:
            print(f"    channels      {observed} observed; checkpoint config not "
                  "in the local cache, so nothing to check it against")
            return
        expected = self.expected_channels(config)
        if expected is None:
            print(f"    channels      {observed} observed; the checkpoint config "
                  "does not imply a channel count")
            return
        if observed != expected:
            raise SystemExit(
                f"{self.name}: the model returned {observed} output channels "
                f"but {self.checkpoint!r} declares an architecture implying "
                f"{expected}.\n"
                "  The constructed class and the checkpoint disagree, which "
                "means the weights\n"
                "  were loaded into the wrong architecture - the output is "
                "that class's\n"
                "  initialisation, not this checkpoint's forecast.")
        print(f"    channels      {observed}, matching what {self.checkpoint} "
              "declares")

    def assert_sigma_in_band(self, levels: np.ndarray) -> None:
        """First-batch scale check, shared with the neural tier.

        One definition of the bands, in sp500bench.evaluation.scale, so the two
        runners cannot drift apart on what counts as a plausible width.
        """
        grid = list(self.our_grid)
        if 0.1 not in grid or 0.9 not in grid:
            return
        try:
            report = check_sigma_band(levels[:, grid.index(0.1)],
                                      levels[:, grid.index(0.9)],
                                      arm=self.arm, label=self.name)
        except SigmaOutOfBand as exc:
            raise SystemExit(str(exc)) from exc
        print(f"    scale         {report}")

    def provenance(self) -> dict:
        """Exactly which weights and which code produced this run."""
        revision, source = _hf_cache_revision(self.checkpoint)
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_revision": revision,
            "checkpoint_revision_source": source,
            "library": self.distribution,
            "library_version": _distribution_version(self.distribution),
            "torch_version": _torch_version(),
        }

    def grid_note(self) -> dict:
        """The model's own grid, and how the frozen one is recovered from it.

        ``native_levels`` is what the *model* emits, which is not always the
        frozen grid even when the model can be scored on it. Chronos-2 carries
        twenty-one levels of which our nine happen to be exact members; writing
        our nine into that field lost the finding and made the artefact claim
        the model emits a grid it does not have. A model whose native grid and
        frozen grid coincide records the same list twice, which is correct and
        says so.
        """
        native = list(getattr(self, "native", None) or self.our_grid)
        frozen = list(self.our_grid)
        same = (len(native) == len(frozen)
                and np.allclose(native, frozen, atol=1e-12))

        note = {
            "model": self.name,
            "scored_on_frozen_grid": True,
            "native_levels": native,
            "frozen_levels": frozen,
            "native_matches_frozen": bool(same),
            "notes": "",
        }
        if same:
            return note

        exact = [q for q in frozen
                 if any(np.isclose(q, g, atol=1e-12) for g in native)]
        interior = [q for q in frozen
                    if q not in exact and min(native) < q < max(native)]
        outside = [q for q in frozen if q < min(native) or q > max(native)]
        note["frozen_levels_direct"] = exact
        note["frozen_levels_interpolated"] = interior
        note["frozen_levels_unavailable"] = outside
        if len(exact) == len(frozen):
            note["notes"] = (
                f"{self.name} emits {len(native)} levels of its own, "
                f"{native}. All {len(frozen)} frozen levels are exact members "
                "of that set, so they are selected by value with no "
                "interpolation anywhere and the model is directly comparable on "
                "CRPS to models scored on the frozen grid.")
        else:
            note["notes"] = (
                f"{self.name} emits {len(native)} levels of its own, "
                f"{native}. Of the frozen {len(frozen)}: {exact} are exact "
                f"members, {interior} are interpolated inside its support, and "
                f"{outside} lie outside it.")
        return note


class ChronosAdapter(Adapter):
    name = "Chronos-2"
    checkpoint = "amazon/chronos-2"
    distribution = "chronos-forecasting"

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

    def expected_channels(self, config: dict) -> int | None:
        levels = (config.get("chronos_config") or {}).get("quantiles")
        return len(levels) if levels else None

    def grid_note(self) -> dict:
        note = super().grid_note()
        note["notes"] += (
            " The quantile axis is axis 1 of a (n_variates, n_quantiles, "
            "prediction_length) return, per the pipeline's own documentation. "
            "The base class documents the opposite order for a different "
            "method, and at horizon 1 a transposed array is still the right "
            "rank and still indexes without complaint, so the levels are "
            "selected by value and the recovered set is compared against the "
            "frozen grid element for element rather than trusted to index "
            "arithmetic.")
        return note

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
        if not self._channels_checked:
            self.assert_channels_match_checkpoint(int(n_q))
            self._channels_checked = True
        values = stacked[:, 0, :, 0]                      # (n, n_quantiles)
        return _select_our_levels(self.native, self.our_grid, values), {}


class TimesFMAdapter(Adapter):
    name = "TimesFM-2.5"
    checkpoint = TIMESFM_CHECKPOINT
    distribution = "timesfm"

    def load(self) -> None:
        import timesfm
        from timesfm import ForecastConfig
        entry = TIMESFM_ENTRY_POINT
        if not hasattr(timesfm, entry):
            raise SystemExit(
                f"timesfm.{entry} is gone. Namespace: "
                f"{sorted(n for n in dir(timesfm) if not n.startswith('_'))}\n"
                "  update the entry point here and in configs/models.yaml together.")
        cls = getattr(timesfm, entry)

        # Named, never defaulted: see TIMESFM_CHECKPOINT.
        self.model = cls.from_pretrained(TIMESFM_CHECKPOINT)
        print(f"    checkpoint    {TIMESFM_CHECKPOINT} (passed explicitly)")
        print(f"    class         timesfm.{entry}, whose DEFAULT_REPO_ID is "
              f"{getattr(cls, 'DEFAULT_REPO_ID', 'unset')!r}")
        self._assert_checkpoint_is_the_pinned_one()

        # Mandatory. forecast() raises "Model is not compiled" without it.
        # max_horizon is rounded up to the output patch size internally; we ask
        # for 1 and slice, which is what the horizon in configs/eval.yaml is.
        self.model.compile(ForecastConfig(
            max_context=self.context, max_horizon=1,
            normalize_inputs=True, per_core_batch_size=self.batch))
        print(f"    compiled      max_context={self.context} max_horizon=1 "
              f"normalize_inputs=True per_core_batch_size={self.batch}")

        self.method_name, self.method = _discover_inference_method(
            self.model, TIMESFM_METHOD_PREFERENCE, entry)

        native = list(self._native_grid() or TIMESFM_NATIVE_GRID)
        # The layout is NOT taken from the measurement that identified it. Two
        # layouts exist in this package and they disagree about where the median
        # sits; _resolve_layout settles it on the first batch against the
        # library's own returned point forecast, on this checkpoint as it would
        # on any other. A measured fact handed to the author is still a fact the
        # next person has to be able to reproduce.
        self.median_index_hint = getattr(
            getattr(self.model, "model", None), "aridx", None)
        self.channel_layout = None
        print(f"    median hint   decode_index={self.median_index_hint} "
              "(not trusted; verified against the returned point forecast)")
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

    def expected_channels(self, config: dict) -> int | None:
        """What the 2.5 checkpoint declares, not what TimesFM3Torch defaults to.

        A checkpoint whose declared architecture is the 2.5 family emits one
        channel per level plus a point channel (timesfm_2p5_torch.py:53,
        ``len(quantiles) + 1``). The 3.0 class emits one per level and no point
        channel. Loading the former into the latter yields a model with the
        right number of levels and the wrong everything else, which is what
        happened here: nine channels observed where the checkpoint implies ten.
        """
        levels = config.get("quantiles")
        if not levels:
            return None
        architectures = " ".join(config.get("architectures") or [])
        if "TimesFm" in architectures or "2p5" in architectures:
            return len(levels) + 1
        return len(levels)

    def _native_grid(self) -> list | None:
        """The nine levels the trained head emits, from the model itself."""
        for holder in (getattr(self.model, "model", None), self.model):
            config = getattr(holder, "config", None)
            levels = getattr(config, "quantiles", None)
            if levels:
                return list(levels)
        declared = self.declared_config() or {}
        return list(declared.get("quantiles") or []) or None

    def declared_config(self) -> dict | None:
        """What the checkpoint says about itself.

        ``from_pretrained`` stores the repo's config.json on the model as
        ``_hub_mixin_config``, which is more direct than reading the cache and
        works wherever the cache lives. Falls back to the cache.
        """
        # getattr on self too: this is reachable before load(), and a config
        # accessor that raises is worse than one that falls back.
        model = getattr(self, "model", None)
        config = getattr(model, "_hub_mixin_config", None)
        if isinstance(config, dict) and config:
            return config
        return _checkpoint_config(self.checkpoint)

    def _assert_checkpoint_is_the_pinned_one(self) -> None:
        """The class and the checkpoint must belong to the same generation.

        The previous failure was not a wrong repository id - that was passed
        correctly and this check passed. It was the wrong class for a correct
        id, which no comparison of path strings can see. So compare the
        architecture the checkpoint declares against the one this class is.
        """
        declared = self.declared_config() or {}
        architectures = " ".join(declared.get("architectures") or [])
        if not architectures:
            print("    architecture  checkpoint declares none; cannot verify "
                  "the class against it")
            return
        if not ("TimesFm" in architectures or "2p5" in architectures):
            raise SystemExit(
                f"{self.checkpoint!r} declares architecture "
                f"{architectures!r}, which is not the 2.5 family this class "
                f"({TIMESFM_ENTRY_POINT}) implements. Loading a checkpoint into "
                "a class of another generation leaves the weights unplaced and "
                "the model still returns forecasts.")
        print(f"    architecture  checkpoint declares {architectures!r}, "
              "matching this class's generation")

    def probe_output_channels(self) -> int | None:
        """One synthetic origin, before any fold, to fix the channel count.

        Checked here rather than on the first real batch so that a class and
        checkpoint that disagree cost one forward pass instead of a fold.
        """
        rng = np.random.default_rng(20260906)
        context = rng.normal(0.0, 0.01, self.context)      # return-scale noise
        quantiles, _ = self._as_arrays(self._call(np.stack([context])), 1)
        return int(quantiles.shape[-1])

    def _call(self, contexts: np.ndarray):
        """Invoke whichever method discovery found, with its own argument shape."""
        rows = [np.asarray(row, dtype=np.float32) for row in contexts]
        if self.method_name == "predict_batch":
            return list(self.method(contexts=rows, horizon=1,
                                    return_quantiles=True))
        if self.method_name == "predict":
            return [self.method(context=row, horizon=1, return_quantiles=True)
                    for row in rows]
        if self.method_name == "forecast":
            return self.method(horizon=1, inputs=rows)
        return self.method(rows, 1)

    def _as_arrays(self, outputs, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Normalise to ``(quantiles (n, C), point (n,))`` across return shapes.

        ``predict``/``predict_batch`` yield ``ForecastOutput`` objects carrying
        ``.quantiles`` and ``.forecast``; the older ``forecast`` method returns a
        ``(point, quantiles)`` pair. Both are reduced to the same two arrays so
        the level mapping below has one code path.
        """
        if isinstance(outputs, tuple) and len(outputs) == 2:
            point, quantiles = outputs
            quantiles = np.asarray(quantiles)
            if quantiles.ndim == 3:                     # (n, horizon, C)
                quantiles = quantiles[:, 0, :]
            return quantiles, np.asarray(point).reshape(n, -1)[:, 0]

        if not isinstance(outputs, (list, tuple)) or len(outputs) != n:
            raise AssertionError(
                f"{self.method_name} returned {type(outputs).__name__} of "
                f"length {len(outputs) if hasattr(outputs, '__len__') else '?'}, "
                f"expected {n} per-series results")

        quantiles, point = [], []
        for item in outputs:
            q = getattr(item, "quantiles", None)
            if q is None:
                raise AssertionError(
                    f"{self.method_name} returned no quantiles. It must be "
                    "called with return_quantiles=True; without it the field "
                    "is None and only the point forecast comes back.")
            q = np.asarray(q)
            quantiles.append(q[0] if q.ndim == 2 else q)     # (horizon, C) -> (C,)
            f = np.asarray(getattr(item, "forecast"))
            point.append(float(f.reshape(-1)[0]))
        return np.stack(quantiles, axis=0), np.asarray(point)

    def _resolve_layout(self, quantiles: np.ndarray, point: np.ndarray) -> int:
        """Which channels are the quantiles, settled against the library itself.

        Two layouts are documented in the same package: a 10-channel head whose
        channel 0 is the point forecast and whose channels 1..9 are the deciles
        (``timesfm_2p5_torch.py:53``, ``decode_index 5``), and a 9-channel one
        indexed directly by level (``_ModelConfig.median_quantile_index 4``).
        Guessing between them shifts every level by one and changes no shape.

        So it is decided by evidence: the library's own ``forecast`` field is by
        construction the median channel, so the offset is whichever one puts the
        median where ``forecast`` already is.
        """
        n_native = len(self.native)
        n_channels = quantiles.shape[-1]
        median_at = int(np.flatnonzero(
            np.isclose(np.asarray(self.native), 0.5, atol=1e-12))[0])

        candidates = []
        if n_channels == n_native:
            candidates.append((0, "channels are the quantiles, one per level"))
        if n_channels == n_native + 1:
            candidates.append((1, "channel 0 is the point forecast, 1.. the levels"))
        if not candidates:
            raise AssertionError(
                f"TimesFM returned {n_channels} channels, which is neither "
                f"{n_native} (one per level) nor {n_native + 1} (point channel "
                f"plus levels). Native grid {self.native}.")

        for offset, description in candidates:
            median_channel = quantiles[:, offset + median_at]
            if np.allclose(median_channel, point, rtol=1e-4, atol=1e-8):
                print(f"    layout        {n_channels} channels, offset "
                      f"{offset} - {description}; median channel matches the "
                      "returned point forecast")
                return offset

        raise AssertionError(
            f"could not identify the quantile channels: with {n_channels} "
            f"channels none of the candidate offsets {[c[0] for c in candidates]} "
            "puts the median where the library's own point forecast is. "
            "Mapping levels on an unverified offset would shift every quantile "
            "by one and raise nothing.")

    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, dict]:
        n = len(contexts)
        raw = self._call(contexts)
        quantiles, point = self._as_arrays(raw, n)
        if quantiles.shape[0] != n:
            raise AssertionError(
                f"TimesFM returned {quantiles.shape[0]} rows for {n} contexts")

        if self.channel_layout is None:
            self.assert_channels_match_checkpoint(int(quantiles.shape[-1]))
            self._channels_checked = True
            self.channel_layout = self._resolve_layout(quantiles, point)
        offset = self.channel_layout
        native_values = quantiles[:, offset:offset + len(self.native)]

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
                "TimesFM-2.5 emits nine deciles [0.1..0.9] on a head trained "
                "for exactly those levels, plus a separate point channel. The "
                "level set is not settable, and the reason is the head itself, "
                "not any particular loader: ForecastConfig carries no quantile "
                "field at all, and TimesFM_2p5_200M_Definition is a frozen "
                "dataclass whose quantiles are the ones the weights were "
                "trained on. Passing a different list would relabel the "
                "channels rather than retarget them - you would ask for q0.01 "
                "and be handed the q0.1 channel under a new name. "
                "This wording was rewritten after the checkpoint was run "
                "through the class that matches it; the previous version "
                "attributed the fixed grid to a line in the 3.0 forecaster, "
                "which was a property of the wrong loader and reached the right "
                "conclusion for the wrong reason. "
                "So the frozen grid's 0.01/0.05/0.95/0.99 lie outside what the "
                "head can express and are written NaN, never extrapolated: a "
                "value at a level the head was never trained to emit would "
                "measure our extrapolation rule, not the model's uncertainty. "
                "Its CRPS must therefore be computed over its own nine levels, "
                "from the native columns in this file, and every table carrying "
                "it must mark it as on a different grid and not directly "
                "comparable on CRPS to models scored on the full nine. A "
                "common-subset comparison was rejected because it would discard "
                "the tail information Chronos-2 and Kronos do carry, and H2 is "
                "about calibration."),
            "entry_point": f"timesfm.{TIMESFM_ENTRY_POINT}",
            "inference_method": getattr(self, "method_name", None),
            "channel_layout_offset": self.channel_layout,
            "channel_layout_verified_against": (
                "the library's own returned point forecast, on the first batch "
                "of this run - not hardcoded from a prior measurement"),
        }


class TTMAdapter(Adapter):
    name = "TTM"
    checkpoint = "ibm-granite/granite-timeseries-ttm-r2"
    distribution = "granite-tsfm"

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


class _NumpyProxy:
    """Stands in for ``numpy`` inside the kronos module, and nowhere else.

    ``kronos.py`` resolves ``np.mean`` through its own module globals, so
    rebinding the *name* ``np`` in that namespace reaches its call sites and
    leaves every other module's binding pointing at the real numpy. Mutating
    ``kronos_module.np.mean`` instead would set the attribute on the numpy module
    object itself, which is shared process-wide - an unrelated ``np.mean`` in any
    other module would hit the shim for the duration. Measured, not assumed: the
    module-mutation form makes an unrelated 4-d ``np.mean(..., axis=1)`` return
    the shimmed value.

    Everything except ``mean`` delegates. ``mean`` passes through only the exact
    reduction that destroys the sample axis and delegates every other call,
    including a 4-d axis-1 mean whose sample extent does not match.
    """

    def __init__(self, real_numpy, sample_count: int):
        self._real = real_numpy
        self._sample_count = sample_count

    def __getattr__(self, name):
        return getattr(self._real, name)

    def mean(self, a, axis=None, *args, **kwargs):
        if (axis == 1 and isinstance(a, self._real.ndarray) and a.ndim == 4
                and a.shape[1] == self._sample_count):
            return a
        return self._real.mean(a, axis=axis, *args, **kwargs)


# The parameters this runner passes by name. Checked against the installed
# function at load time: a keyword that no longer exists fails loudly at the
# call, but a *positional* argument that has shifted passes temperature where
# top_k belongs and returns plausible samples that are quietly wrong. This
# adapter's output cannot be checked by reading it, so the signature is checked
# instead.
KRONOS_INFERENCE_PARAMS = (
    "tokenizer", "model", "x", "x_stamp", "y_stamp", "max_context", "pred_len",
    "clip", "T", "top_k", "top_p", "sample_count", "verbose")


def _wrap_text(text: str, width: int) -> list[str]:
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


def _inference_surface(obj) -> list[str]:
    """Public names plus the signature of anything that looks like inference.

    Printed on discovery failure, in the same shape as the entry-point dump that
    settled ``TimesFM3Forecaster`` the first time. The point is that the next
    person reads the real namespace off the run instead of guessing a second
    time - guessing is what produced ``.forecast``, which is the TimesFM_2p5
    base-class method and not on the forecaster at all.
    """
    public = sorted(m for m in dir(obj) if not m.startswith("_"))
    lines = [f"    public attributes: {public}"]
    interesting = [m for m in public + ["__call__"]
                   if any(hint in m.lower() for hint in INFERENCE_NAME_HINTS)]
    for name in interesting:
        attr = getattr(obj, name, None)
        if not callable(attr):
            continue
        try:
            sig = str(inspect.signature(attr))
        except (TypeError, ValueError):                 # builtins, C callables
            sig = "(signature unavailable)"
        lines.append(f"      {name}{sig}")
    return lines


def _discover_inference_method(model, preference, entry_name: str):
    """The first method in ``preference`` the object actually has.

    Dispatch is by preference order rather than by a single hardcoded name, and
    the chosen name is printed so it lands in the run log and can be recorded in
    the frozen spec afterwards.
    """
    for name in preference:
        attr = getattr(model, name, None)
        if callable(attr):
            try:
                sig = str(inspect.signature(attr))
            except (TypeError, ValueError):
                sig = "(signature unavailable)"
            print(f"    inference     {entry_name}.{name}{sig}")
            print(f"    RECORD THIS   TimesFM inference method = {name!r} "
                  "- add it to configs/models.yaml and registry.py together")
            return name, attr

    surface = "\n".join(_inference_surface(model))
    raise SystemExit(
        f"{entry_name} exposes none of {list(preference)}.\n{surface}\n"
        "  Pick the right method from the signatures above, add it to "
        "TIMESFM_METHOD_PREFERENCE, and record it in configs/models.yaml and "
        "registry.py together.")


def _is_oom(exc: BaseException) -> bool:
    """CUDA OOM across torch versions.

    ``torch.OutOfMemoryError`` only exists from 2.4 and is a ``RuntimeError``
    subclass in every version that has it, so the base class is caught and the
    identity checked here. Naming the attribute in an ``except`` clause would
    itself raise ``AttributeError`` on an older image - a failure in the error
    handler, at the moment the error handler is needed.
    """
    return (type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in str(exc).lower())


def _assert_kronos_signature(kronos_module) -> None:
    """Every parameter this runner names must exist, and accept a keyword.

    Checked once at load rather than trusted at the call site. A renamed or
    removed parameter then fails here with the installed signature printed,
    instead of at hour nine of a run - or, if the call were positional, not
    failing at all.
    """
    fn = getattr(kronos_module, "auto_regressive_inference", None)
    if fn is None:
        raise SystemExit(
            "model/kronos.py no longer exposes auto_regressive_inference. "
            "predict() cannot be used instead - it averages the sample axis at "
            "kronos.py:467 and would give nine copies of a mean path.")

    params = inspect.signature(fn).parameters
    missing = [p for p in KRONOS_INFERENCE_PARAMS if p not in params]
    if missing:
        raise SystemExit(
            f"auto_regressive_inference is missing parameter(s) {missing}.\n"
            f"  installed signature: ({', '.join(params)})\n"
            "  Upstream changed the API. Re-read the function before running: "
            "the sampling arguments are not interchangeable and a wrong mapping "
            "produces plausible samples rather than an error.")

    positional_only = [p for p in KRONOS_INFERENCE_PARAMS
                       if params[p].kind is inspect.Parameter.POSITIONAL_ONLY]
    if positional_only:
        raise SystemExit(
            f"auto_regressive_inference now takes {positional_only} "
            "positional-only, so this runner cannot pass them by name.")
    print(f"    signature     auto_regressive_inference"
          f"({', '.join(KRONOS_INFERENCE_PARAMS)}) - all passed by keyword")


class KronosAdapter(Adapter):
    name = "Kronos-small"
    checkpoint = "NeoQuasar/Kronos-small"
    tokenizer_checkpoint = "NeoQuasar/Kronos-Tokenizer-base"
    distribution = None          # a git clone, not a wheel; see provenance()
    series = "price"
    needs_bars = True

    def __init__(self, our_grid, context, device, batch, sample_count,
                 paths_per_call=DEFAULT_KRONOS_PATHS_PER_CALL, kronos_path=None,
                 sequence_budget=DEFAULT_KRONOS_SEQUENCE_BUDGET,
                 precision="fp16", verification=None):
        super().__init__(our_grid, context, device, batch)
        self.precision = precision
        self.verification = verification
        self.sample_count = sample_count
        # Clamped by both: more paths than exist is meaningless, and a chunk
        # wider than the whole budget would make the printed origin batch a
        # fiction. The OOM handler applies the same clamp, so a budget set on the
        # command line and one arrived at by halving behave identically.
        self.paths_per_call = max(1, min(paths_per_call, sample_count,
                                         sequence_budget))
        self.sequence_budget = sequence_budget
        self.kronos_path = kronos_path
        self.budget_was_reduced = False
        self._reported_working_size = False

    @property
    def origin_batch(self) -> int:
        """Origins per call, derived so ``origin_batch * paths_per_call`` fits.

        Read fresh on every batch rather than computed once, so that a budget cut
        after an OOM takes effect for the rest of the run instead of the same
        oversized batch being retried repeatedly.
        """
        return max(1, self.sequence_budget // self.paths_per_call)

    def sample_chunks(self) -> list[int]:
        """The sample dimension split into calls of at most ``paths_per_call``."""
        full, remainder = divmod(self.sample_count, self.paths_per_call)
        chunks = [self.paths_per_call] * full
        if remainder:
            chunks.append(remainder)
        return chunks

    def _free(self) -> None:
        if getattr(self, "torch", None) is not None and self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def reset_peak_memory(self) -> None:
        if getattr(self, "torch", None) is not None and self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_gib(self) -> float:
        if getattr(self, "torch", None) is not None and self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 1024 ** 3
        return float("nan")

    def load(self) -> None:
        import torch
        if self.kronos_path and self.kronos_path not in sys.path:
            sys.path.append(self.kronos_path)
        from model import Kronos, KronosPredictor, KronosTokenizer
        from model import kronos as kronos_module
        self.torch = torch
        self.kronos_module = kronos_module
        _assert_kronos_signature(kronos_module)
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        self.predictor = KronosPredictor(model, tokenizer, device=self.device,
                                         max_context=self.context)
        self.feature_cols = (list(self.predictor.price_cols)
                             + [self.predictor.vol_col, self.predictor.amt_vol])
        self.close_at = self.feature_cols.index("close")
        chunks = self.sample_chunks()
        print(f"    sample_count  {self.sample_count} paths/origin  "
              f"features {self.feature_cols}  close at index {self.close_at}")
        # In the run log, not only in the parquet: the reason a sample count was
        # chosen and the direction of the bias it carries have to be legible to
        # whoever reads the run, not reconstructed from a commit later.
        print(f"    provenance    sample_count={self.sample_count} chosen for "
              "compute, on the full test period rather than a narrower window")
        for line in _wrap_text(KRONOS_SAMPLE_RATIONALE, 74):
            print(f"                  {line}")
        print(f"    precision     {self.precision} on the inference call only; "
              "normalisation, denormalisation and quantiles stay fp64")
        if self.verification:
            for line in _wrap_text(self.verification, 74):
                print(f"                  {line}")
        elif self.precision == "fp16":
            print("                  NOT VERIFIED against fp32 for this run - "
                  "Kronos is a sampler, so fp16")
            print("                  changes which paths are drawn. Run "
                  "--calibrate with --calib-precisions")
            print("                  fp32 fp16 first, or pass --precision fp32.")
        # Printed before the first call, because the product is the number that
        # decides whether this fits in the card and it is not either input.
        print(f"    memory        paths/call {self.paths_per_call} "
              f"x origin batch {self.origin_batch} = "
              f"{self.origin_batch * self.paths_per_call} sequences of "
              f"{self.context} bars per call "
              f"(budget {self.sequence_budget})")
        print(f"    sample chunks {len(chunks)} per origin batch: {chunks}")

    def _autocast(self):
        """Half precision for the forward pass only, and only on CUDA.

        Deliberately the narrowest possible scope. Kronos is a sampler, so fp16
        does not merely run the same arithmetic faster - it changes which paths
        are drawn, and a quantile built from different paths is a different
        estimate. Everything whose precision the reported numbers are actually
        made of stays in fp32 or wider: the per-window normalisation before the
        call, the denormalisation after it, and the quantile computation. Those
        are cheap, so there is nothing to win by degrading them and a
        systematically shifted tail to lose.
        """
        if self.precision == "fp16" and self.device.startswith("cuda"):
            return self.torch.autocast("cuda", dtype=self.torch.float16)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _keep_sample_axis(self, sample_extent: int):
        """Suppress the one reduction that destroys the predictive distribution.

        ``sample_extent`` is the sample count of *this call*, which is the chunk
        width and not ``self.sample_count`` once the sample dimension is chunked.
        Matching on the wrong extent would leave the reduction in place, the
        returned array would come back with the sample axis gone, and the shape
        assertion in ``predict_bars`` would stop the run - loudly, but for a
        reason that reads as an upstream change rather than as this arithmetic.

        ``auto_regressive_inference`` ends with ``np.mean(preds, axis=1)`` over
        the sample axis (``model/kronos.py:467``), and every public entry point
        goes through it, so there is no supported way to get the paths back.
        Rather than vendor a copy of the whole function - which would silently
        rot the next time upstream changes the decode loop - the name ``np`` in
        that module is rebound to a proxy for the duration.

        Rebinding the name, not mutating the module: see ``_NumpyProxy``. The
        blast radius is this one module between ``yield`` and ``finally``.

        Narrow on purpose. If upstream adds another 4-d axis-1 mean over an array
        whose second axis happens to be ``sample_count``, the proxy would pass
        that through too, which is why the caller asserts the returned array
        still carries the sample axis at the expected extent.
        """
        real_numpy = self.kronos_module.np
        self.kronos_module.np = _NumpyProxy(real_numpy, sample_extent)
        try:
            yield
        finally:
            self.kronos_module.np = real_numpy

    def predict_bars(self, windows: np.ndarray, x_stamps: np.ndarray,
                     y_stamps: np.ndarray, last_close: np.ndarray,
                     ) -> tuple[np.ndarray, dict]:
        """``windows`` is ``(n, context, n_features)`` of raw bars.

        Replicates ``KronosPredictor.predict``'s per-series normalisation
        (``model/kronos.py:544-546``: mean and std of the context window only,
        so nothing outside the window enters) and its denormalisation, but keeps
        the sample axis in between.
        """
        # Back off repeatedly, not once. Halving a single time and giving up is
        # only correct if the first guess was within a factor of two; it was not
        # on the T4, where 4000 and 2000 sequences both failed. The loop walks
        # the budget down to the floor, reports each step, and only fails when
        # the floor itself cannot run - so the working size is measured rather
        # than assumed, and a run that could have completed does not abort.
        while True:
            try:
                levels, extras = self._run_in_batches(
                    windows, x_stamps, y_stamps, last_close)
                if self.budget_was_reduced and not self._reported_working_size:
                    print(f"      settled at {self.origin_batch} origins x "
                          f"{self.paths_per_call} paths = "
                          f"{self.origin_batch * self.paths_per_call} "
                          "sequences/call")
                    self._reported_working_size = True
                return levels, extras
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                self._free()
                before = self.sequence_budget
                if before <= MIN_KRONOS_SEQUENCE_BUDGET:
                    raise SystemExit(
                        f"Kronos ran out of memory at the floor budget of "
                        f"{MIN_KRONOS_SEQUENCE_BUDGET} sequences/call "
                        f"({self.origin_batch} origins x "
                        f"{self.paths_per_call} paths), which is the smallest "
                        "this runner will attempt. The context window or the "
                        "card is the binding constraint, not the batching.\n"
                        f"  {exc}") from exc
                self.sequence_budget = max(MIN_KRONOS_SEQUENCE_BUDGET,
                                           before // 2)
                self.paths_per_call = max(1, min(self.paths_per_call,
                                                 self.sequence_budget))
                self.budget_was_reduced = True
                self._reported_working_size = False
                print(f"      OOM at {before} sequences/call; backing off to "
                      f"{self.sequence_budget} -> {self.origin_batch} origins "
                      f"x {self.paths_per_call} paths = "
                      f"{self.origin_batch * self.paths_per_call} sequences")

    def _run_in_batches(self, windows: np.ndarray, x_stamps: np.ndarray,
                        y_stamps: np.ndarray, last_close: np.ndarray,
                        ) -> tuple[np.ndarray, dict]:
        """The caller's block, re-split at the current origin batch.

        After a backoff the incoming block may be wider than the budget now
        allows, so it is re-split here rather than handed back to the caller.
        """
        step = max(1, self.origin_batch)
        if step >= len(windows):
            return self._predict_bars(windows, x_stamps, y_stamps, last_close)
        parts = []
        for start in range(0, len(windows), step):
            stop = start + step
            part, _ = self._predict_bars(
                windows[start:stop], x_stamps[start:stop],
                y_stamps[start:stop], last_close[start:stop])
            parts.append(part)
            self._free()
        return np.concatenate(parts, axis=0), {}

    def _predict_bars(self, windows: np.ndarray, x_stamps: np.ndarray,
                      y_stamps: np.ndarray, last_close: np.ndarray,
                      ) -> tuple[np.ndarray, dict]:
        """One origin batch, with the sample dimension chunked.

        The sample axis is the one that multiplies memory, so it is walked in
        ``paths_per_call``-wide chunks and the resulting close paths are
        concatenated. Each chunk is an independent stochastic call - nothing is
        reseeded between them - so the union is a valid draw of ``sample_count``
        paths from the same predictive distribution as a single call would have
        given, just not the same realisation.
        """
        n = len(windows)
        mean = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True)
        norm = np.clip((windows - mean) / (std + 1e-5), -self.predictor.clip,
                       self.predictor.clip)

        x = self.torch.tensor(norm, dtype=self.torch.float32).to(self.device)
        xs = self.torch.tensor(x_stamps, dtype=self.torch.float32).to(self.device)
        ys = self.torch.tensor(y_stamps, dtype=self.torch.float32).to(self.device)

        collected = []
        try:
            for chunk in self.sample_chunks():
                # Every argument by name. Positionally, an upstream insertion or
                # reorder would pass temperature where top_k belongs and return
                # samples that look entirely reasonable; by keyword it raises.
                # The proxy matches on `chunk`, not self.sample_count.
                with self._keep_sample_axis(chunk):
                    with self.torch.no_grad(), self._autocast():
                        paths = self.kronos_module.auto_regressive_inference(
                            tokenizer=self.predictor.tokenizer,
                            model=self.predictor.model,
                            x=x,
                            x_stamp=xs,
                            y_stamp=ys,
                            max_context=self.context,
                            pred_len=1,
                            clip=self.predictor.clip,
                            T=1.0,
                            top_k=0,
                            top_p=0.9,
                            sample_count=chunk,
                            verbose=False,
                        )

                # Back to fp64 immediately. Under autocast the decode returns
                # fp16, and everything from here on - denormalisation onto price
                # scale, the log return, the quantile - is what the reported
                # numbers are made of.
                paths = np.asarray(paths, dtype=np.float64)
                if paths.ndim != 4 or paths.shape[0] != n or paths.shape[1] != chunk:
                    raise AssertionError(
                        f"Kronos returned {paths.shape}; expected "
                        f"({n}, {chunk}, steps, features). The sample axis was "
                        "collapsed - _keep_sample_axis did not take effect, and "
                        "the quantiles would be copies of a mean path.")

                # fp16 has a maximum around 65,504, and these are price-scale
                # tensors. An overflow yields inf, np.quantile of an array
                # containing inf returns inf or a finite value drawn from the
                # wrong order statistics, and either way it reaches the parquet
                # looking like a number. Checked here rather than trusted.
                if not np.isfinite(paths).all():
                    bad = int((~np.isfinite(paths)).sum())
                    raise SystemExit(
                        f"Kronos returned {bad:,} non-finite values of "
                        f"{paths.size:,} at precision {self.precision}. In "
                        "fp16 this is overflow on a price-scale tensor; the "
                        "quantiles built from it would be silently wrong. "
                        "Re-run this model with --precision fp32.")

                # Reduce to the one column that is needed before anything else is
                # kept, so the full (n, chunk, steps, features) block can go.
                step = paths[:, :, -1, :]
                step = step * (std[:, 0, :][:, None, :] + 1e-5) \
                    + mean[:, 0, :][:, None, :]
                collected.append(step[:, :, self.close_at])
                del paths, step
        finally:
            del x, xs, ys

        close_paths = np.concatenate(collected, axis=1)          # (n, sample_count)
        if close_paths.shape != (n, self.sample_count):
            raise AssertionError(
                f"assembled {close_paths.shape} close paths, expected "
                f"({n}, {self.sample_count}) - the sample chunks did not sum to "
                "sample_count")
        returns = np.log(close_paths / last_close[:, None])
        levels = np.quantile(returns, self.our_grid, axis=1).T   # (n, len(our_grid))
        return levels, {}

    def provenance(self) -> dict:
        """Two checkpoints and a clone, so three things identify this run."""
        base = super().provenance()
        tok_rev, tok_source = _hf_cache_revision(self.tokenizer_checkpoint)
        base.update({
            "tokenizer_checkpoint": self.tokenizer_checkpoint,
            "tokenizer_revision": tok_rev,
            "tokenizer_revision_source": tok_source,
            # Kronos is cloned, not installed, so the commit of the working
            # copy is its version - there is no distribution to ask.
            "library": "kronos (git clone)",
            "library_version": _git_head(self.kronos_path),
            "library_source": self.kronos_path,
        })
        return base

    def grid_note(self) -> dict:
        return {
            "model": self.name, "scored_on_frozen_grid": True,
            "native_levels": list(self.our_grid),
            "sample_count": self.sample_count,
            # Recorded because the sample dimension is chunked at this width and
            # each chunk is a separate stochastic call: the distribution is the
            # same at any setting, the realised draws are not, so a bit-for-bit
            # rerun needs the same value.
            "paths_per_call": self.paths_per_call,
            "sequence_budget": self.sequence_budget,
            "budget_reduced_after_oom": self.budget_was_reduced,
            # Why this sample count, and which way its error points. Carried in
            # the artefact so any table built from these forecasts can state it
            # without the reader having to reconstruct it from the history.
            "sample_count_chosen_for": "compute",
            "evaluation_window": "full test period (not the clean slice)",
            "precision": self.precision,
            "precision_scope": (
                "autocast applies to the inference call only; the per-window "
                "normalisation, the denormalisation and the quantile "
                "computation are fp64"),
            "precision_verification": self.verification or (
                "NOT VERIFIED - no fp32/fp16 comparison was supplied for this "
                "run. Kronos is a sampler, so fp16 changes which paths are "
                "drawn and not merely how fast; a CRPS from fp16 paths is only "
                "defensible alongside a measured comparison against fp32."),
            "tail_bias": "inward",
            "tail_bias_direction": (
                "penalises this model: finite-sample extreme quantiles are "
                "estimated narrower than the true predictive tails, so q0.01 "
                "and q0.99 are too tight and CRPS is inflated"),
            "sample_count_rationale": KRONOS_SAMPLE_RATIONALE,
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

def _load_precision_verification(args) -> str | None:
    """The fp32/fp16 verdict from a prior calibration, if one is there.

    Copied rather than retyped so the sentence in grid.json is the one the
    measurement actually produced. Absent, mismatched or unverified all mean the
    forecasts record that no verification backs them - which is the honest state
    of an fp16 run nobody checked.
    """
    if args.precision != "fp16":
        return f"precision {args.precision}; no autocast, so nothing to verify"
    path = Path(args.precision_verification)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  ! {path} is not valid JSON; treating fp16 as unverified")
        return None
    sentence = payload.get("precision_verification")
    if not sentence:
        return None
    if payload.get("precision_within_noise") is False:
        print(f"  ! {show(path)} says the fp16 shift EXCEEDED the noise floor")
    return f"{sentence} [from {show(path)}]"


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
                             args.kronos_path, args.kronos_sequence_budget,
                             precision=args.precision,
                             verification=_load_precision_verification(args))
    raise SystemExit(f"{name} is not a foundation model in this runner")


def forecast_fold(adapter: Adapter, bars: pd.DataFrame, split, our_grid,
                  tickers: list[str] | None = None) -> pd.DataFrame:
    """Every (ticker, session) origin in one test window."""
    frames = []
    grouped = bars.groupby("unique_id", sort=True)
    names = tickers if tickers is not None else sorted(bars["unique_id"].unique())
    skipped = {"short_history": 0, "short_history_tickers": set(),
               "no_forward_return": 0, "no_forward_return_tickers": set()}

    for ticker in names:
        group = grouped.get_group(ticker).sort_values("ds").reset_index(drop=True)
        dates = pd.DatetimeIndex(group["ds"])
        in_window = np.flatnonzero((dates >= split.test_start) & (dates <= split.test_end))
        if in_window.size == 0:
            continue
        usable = in_window[in_window >= adapter.context]
        skipped["short_history"] += in_window.size - usable.size
        if in_window.size - usable.size:
            skipped["short_history_tickers"].add(ticker)
        # An origin at the final session of the panel has no t+1, so
        # make_target leaves its label NaN and there is nothing to score the
        # forecast against - evaluate.py refuses a forecast with no outcome.
        # This is what makes the count 1,669 sessions per ticker rather than the
        # 1,670 in the windows, and it matches what the existing tiers write.
        #
        # Counted, not printed per ticker. On the last fold every one of the
        # fifty hits this, and one identical line each is fifty lines per model,
        # 950 across the roster - enough noise to bury something that matters.
        labelled = usable[usable < len(group) - 1]
        if labelled.size != usable.size:
            skipped["no_forward_return"] += usable.size - labelled.size
            skipped["no_forward_return_tickers"].add(ticker)
        usable = labelled
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

            # adapter.origin_batch is read fresh each iteration: if a call OOMs
            # and halves the budget, the remaining batches use the size that fit.
            start = 0
            while start < usable.size:
                chunk = usable[start:start + adapter.origin_batch]
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
                if not adapter._scale_checked:
                    adapter.assert_sigma_in_band(levels)
                    adapter._scale_checked = True
                for j, q in enumerate(our_grid):
                    out.loc[start:start + len(chunk) - 1, quantile_column(q)] = levels[:, j]
                for k, v in extras.items():
                    extras_acc.setdefault(k, []).append(v)
                start += len(chunk)
                # Kronos holds the largest transient of the four; releasing the
                # caching allocator between batches keeps a later, slightly
                # larger allocation from failing against a fragmented pool.
                adapter._free()
        else:
            contexts = build_context_matrix(closes, usable, adapter.context,
                                            adapter.series)
            for start in range(0, len(contexts), adapter.batch):
                block = contexts[start:start + adapter.batch]
                levels, extras = adapter.predict(block)
                if not adapter._scale_checked:
                    adapter.assert_sigma_in_band(levels)
                    adapter._scale_checked = True
                for j, q in enumerate(our_grid):
                    out.loc[start:start + len(block) - 1, quantile_column(q)] = levels[:, j]
                for k, v in extras.items():
                    extras_acc.setdefault(k, []).append(v)

        for col, parts in extras_acc.items():
            out[col] = np.concatenate(parts)
        frames.append(out)

    # One line per reason for the whole fold, naming the count of tickers rather
    # than each of them.
    if skipped["short_history"]:
        print(f"      skipped {skipped['short_history']:,} origins across "
              f"{len(skipped['short_history_tickers'])} tickers: fewer than "
              f"{adapter.context} sessions of history")
    if skipped["no_forward_return"]:
        print(f"      skipped {skipped['no_forward_return']:,} origins across "
              f"{len(skipped['no_forward_return_tickers'])} tickers: final "
              "session has no forward return, so no label to score against")

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
    # Every adapter's load goes through this: the loaders discard what
    # load_state_dict reports, so a checkpoint that matched almost nothing looks
    # exactly like one that loaded.
    recorder = None
    try:
        import torch
        recorder = _StateDictLoadRecorder(torch)
    except ImportError:
        pass
    if recorder is None:
        adapter.load()
    else:
        with recorder:
            adapter.load()
        if args.allow_inexact_load:
            principal = recorder.principal_calls()
            missing = sorted({k for c in principal for k in c["missing"]})
            unexpected = sorted({k for c in principal for k in c["unexpected"]})
            if missing or unexpected:
                print(f"  ! --allow-inexact-load: {len(missing)} missing and "
                      f"{len(unexpected)} unexpected keys accepted; the "
                      "forecasts are not fully this checkpoint's")
        else:
            recorder.assert_clean(adapter.name, adapter.checkpoint)
    load_seconds = time.perf_counter() - t_load

    # Before any fold: a class and a checkpoint that disagree about the output
    # shape should cost one forward pass, not a fold of forecasts that look
    # plausible until someone checks their magnitude.
    observed_channels = adapter.probe_output_channels()
    if observed_channels is not None:
        adapter.assert_channels_match_checkpoint(observed_channels)
        adapter._channels_checked = True
    print(f"    loaded in {load_seconds:.1f}s")

    grid_path = model_dir / "grid.json"
    write_grid_note(grid_path, adapter.grid_note())

    fold_times: dict[str, float] = {}
    fold_rows: dict[str, int] = {}
    skipped_existing: list[int] = []
    rows_written = 0
    started = time.perf_counter()

    for split, train, test in walk_forward.iter_frames(bars, time_col="ds"):
        path = model_dir / f"fold_{split.fold}.parquet"
        if path.exists() and not args.overwrite:
            print(f"  fold {split.fold}  exists, skipping ({show(path)}) "
                  "- pass --overwrite to redo")
            skipped_existing.append(int(split.fold))
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
        fold_rows[str(split.fold)] = len(forecasts)
        rows_written += len(forecasts)
        record_timing(timings_path, adapter.name, {
            "library": "foundation", "tier": "foundation", "scope": "zero_shot",
            "target": "return", "seconds_load": round(load_seconds, 3),
            "seconds_per_fold": fold_times,
            "seconds_steady_state_total": round(sum(fold_times.values()), 3),
            "n_forecasts": rows_written, "n_folds": len(fold_times),
            "context": adapter.context, "device": args.device,
            "sample_count": getattr(adapter, "sample_count", None),
            "paths_per_call": getattr(adapter, "paths_per_call", None),
            "sequence_budget": getattr(adapter, "sequence_budget", None),
            "budget_reduced_after_oom": getattr(adapter, "budget_was_reduced", None),
            "precision": getattr(adapter, "precision", None),
            "precision_verification": getattr(adapter, "verification", None),
            "peak_gpu_gib": (adapter.peak_memory_gib()
                             if hasattr(adapter, "peak_memory_gib") else None),
            **adapter.provenance(),
            "forecasts_per_fold": fold_rows,
            "folds_skipped_because_present": skipped_existing,
            "input_series": getattr(adapter, "series", None),
            "partial": True,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        # The layout offset is only known after a batch has run, so the note
        # written before the first fold carries a null for it.
        write_grid_note(grid_path, adapter.grid_note())
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
        "paths_per_call": getattr(adapter, "paths_per_call", None),
        "sequence_budget": getattr(adapter, "sequence_budget", None),
        "budget_reduced_after_oom": getattr(adapter, "budget_was_reduced", None),
        "precision": getattr(adapter, "precision", None),
        "precision_verification": getattr(adapter, "verification", None),
        "peak_gpu_gib": (adapter.peak_memory_gib()
                         if hasattr(adapter, "peak_memory_gib") else None),
        **adapter.provenance(),
        "forecasts_per_fold": fold_rows,
        "folds_skipped_because_present": skipped_existing,
        "input_series": getattr(adapter, "series", None),
        "partial": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    print(f"  {rows_written:,} forecasts | {steady:.1f}s across "
          f"{len(fold_times)} folds -> {show(timings_path)}")


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def _precision_verdicts(rows: list[dict], repeats: int) -> dict:
    """Is the fp32 -> fp16 CRPS shift bigger than fp32's own run-to-run spread?

    Kronos draws different paths on every call, so two fp32 runs on the same
    origins already disagree. That disagreement is the sampling noise floor, and
    it is the only honest reference for the precision difference: a shift
    smaller than the noise is not evidence that fp16 changed the estimate, and a
    shift larger than it is.

    Hence ``--calib-repeats`` defaulting to 2. With one run per precision there
    is no noise estimate at all, and the comparison would be a bare difference
    with nothing to read it against - which is exactly the kind of number that
    looks like evidence and is not.
    """
    report: list[str] = []
    verdict = {"report": report, "sentence": None, "within_noise": None}

    by_sample: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_count"], {}).setdefault(
            row["precision"], []).append(row["crps"])

    for sample_count, by_precision in sorted(by_sample.items()):
        if set(by_precision) < {"fp32", "fp16"}:
            continue
        fp32, fp16 = by_precision["fp32"], by_precision["fp16"]
        mean32 = float(np.mean(fp32))
        mean16 = float(np.mean(fp16))
        shift = abs(mean16 - mean32)
        # Noise floor: the widest within-precision spread seen. Conservative -
        # it is easier for a real shift to clear a small floor than a large one,
        # so taking the larger spread makes the "within noise" verdict harder to
        # earn, not easier.
        spreads = [max(v) - min(v) for v in (fp32, fp16) if len(v) > 1]
        noise = max(spreads) if spreads else float("nan")

        report.append("")
        report.append("=" * 78)
        report.append(f"PRECISION COMPARISON at S={sample_count}, "
                      f"{repeats} run(s) per precision, identical origins")
        report.append("=" * 78)
        report.append(f"  fp32 CRPS   {mean32:.6f}   runs {[f'{v:.6f}' for v in fp32]}")
        report.append(f"  fp16 CRPS   {mean16:.6f}   runs {[f'{v:.6f}' for v in fp16]}")
        report.append(f"  shift       {shift:.6f}  ({shift / mean32 * 100:+.3f}% "
                      "of fp32)")
        if not np.isfinite(noise):
            report.append("  noise       not measured - only one run per "
                          "precision, so there is nothing to read the shift")
            report.append("              against. Re-run with --calib-repeats 2.")
            continue
        report.append(f"  noise floor {noise:.6f}  (widest within-precision "
                      "spread, same origins)")

        within = shift <= noise
        verdict["within_noise"] = bool(within)
        if within:
            sentence = (
                f"fp16 verified against fp32 on fold 0 at S={sample_count}: "
                f"CRPS {mean16:.6f} vs {mean32:.6f}, a shift of {shift:.6f} "
                f"({shift / mean32 * 100:+.3f}%), within the {noise:.6f} "
                "run-to-run spread of fp32 against itself on the same origins. "
                "The precision change is not distinguishable from resampling "
                "noise at this sample count.")
            report.append("")
            report.append("  VERDICT: within sampling noise. fp16 is defensible "
                          "here, and the sentence")
            report.append("  below is written into grid.json and the run log.")
        else:
            sentence = (
                f"fp16 NOT verified on fold 0 at S={sample_count}: CRPS "
                f"{mean16:.6f} vs fp32 {mean32:.6f}, a shift of {shift:.6f} "
                f"({shift / mean32 * 100:+.3f}%), LARGER than the {noise:.6f} "
                "run-to-run spread of fp32 on the same origins. The speed is "
                "not free: fp16 moves the estimate by more than resampling "
                "does.")
            report.append("")
            report.append("  VERDICT: the shift EXCEEDS the noise floor. fp16 "
                          "changes the estimate by")
            report.append("  more than resampling does - do not spend the quota "
                          "on it without deciding")
            report.append("  that the shift is acceptable. --precision fp32 "
                          "remains available.")
        verdict["sentence"] = sentence
        report.append("")
        for line in _wrap_text(sentence, 74):
            report.append(f"    {line}")
    return verdict


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
      for precision in args.calib_precisions:
       for repeat in range(args.calib_repeats):
        adapter = KronosAdapter(our_grid, args.context, args.device,
                                args.batch_size, s, args.kronos_paths_per_call,
                                args.kronos_path, args.kronos_sequence_budget,
                                precision=precision)
        adapter.load()
        adapter.reset_peak_memory()
        t0 = time.perf_counter()
        forecasts = forecast_fold(adapter, bars, split, our_grid, tickers)
        elapsed = time.perf_counter() - t0
        peak_gib = adapter.peak_memory_gib()

        merged = forecasts.merge(
            truth.rename("y"), left_on=["unique_id", "ds"],
            right_index=True, how="left")
        usable = merged.dropna(subset=["y"])
        qf = {q: usable[quantile_column(q)].to_numpy(dtype=float) for q in our_grid}
        crps = crps_from_quantiles(usable["y"].to_numpy(dtype=float), qf)

        per_origin = elapsed / max(1, len(usable))
        full_hours = per_origin * 50 * 1669 / 3600
        seqs = adapter.origin_batch * adapter.paths_per_call
        rows.append({"sample_count": s, "precision": precision, "repeat": repeat,
                     "crps": crps, "seconds": elapsed, "n_origins": len(usable),
                     "seconds_per_origin": per_origin,
                     "projected_full_run_gpu_hours": full_hours,
                     "peak_gpu_gib": peak_gib, "sequences_per_call": seqs,
                     "budget_reduced_after_oom": adapter.budget_was_reduced})
        print(f"  S={s:<5} {precision:<5} run {repeat + 1}  CRPS {crps:.6f}  "
              f"{elapsed:8.1f}s over {len(usable):,} origins  "
              f"({per_origin * 1000:.1f} ms/origin -> {full_hours:.1f} GPU-h "
              f"for the full 83,450)  peak {peak_gib:.2f} GiB @ {seqs} seq/call"
              f"{'  [budget reduced after OOM]' if adapter.budget_was_reduced else ''}")
        del adapter

    print(f"\n{'S':>6}{'prec':>7}{'run':>5}{'CRPS':>12}{'ms/origin':>12}"
          f"{'peak GiB':>10}{'seq/call':>10}{'full-run GPU-h':>16}")
    for r in rows:
        flag = " *" if r["budget_reduced_after_oom"] else ""
        print(f"{r['sample_count']:>6}{r['precision']:>7}{r['repeat'] + 1:>5}"
              f"{r['crps']:>12.6f}{r['seconds_per_origin'] * 1000:>12.1f}"
              f"{r['peak_gpu_gib']:>10.2f}{r['sequences_per_call']:>10}"
              f"{r['projected_full_run_gpu_hours']:>16.1f}{flag}")

    verdicts = _precision_verdicts(rows, args.calib_repeats)
    for line in verdicts["report"]:
        print(line)

    print("\npeak GiB is torch.cuda.max_memory_allocated over the sweep step, so "
          "it is the ceiling\nthe full run has to live under, at the seq/call the "
          "row actually used.")
    if any(r["budget_reduced_after_oom"] for r in rows):
        print("* this row hit OOM and finished at a reduced budget; seq/call is "
              "the size that worked.")

    out = Path(args.calib_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fold": int(split.fold),
        "test_window": [str(split.test_start.date()), str(split.test_end.date())],
        "tickers": tickers,
        "context": args.context,
        "device": args.device,
        "paths_per_call": args.kronos_paths_per_call,
        "sequence_budget": args.kronos_sequence_budget,
        "precisions": list(args.calib_precisions),
        "repeats_per_precision": args.calib_repeats,
        # The sentence the real run copies into grid.json and prints, so the
        # verification travels with the forecasts instead of being transcribed.
        "precision_verification": verdicts["sentence"],
        "precision_within_noise": verdicts["within_noise"],
        "rows": rows,
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
                   help="sample paths per call; the sample dimension is chunked "
                        f"at this width (default {DEFAULT_KRONOS_PATHS_PER_CALL})")
    p.add_argument("--kronos-sequence-budget", type=int,
                   default=DEFAULT_KRONOS_SEQUENCE_BUDGET,
                   help="ceiling on origin_batch x paths_per_call, i.e. the "
                        "sequences resident in one call; the origin batch is "
                        f"derived from it (default {DEFAULT_KRONOS_SEQUENCE_BUDGET})")
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
    p.add_argument("--allow-inexact-load", action="store_true",
                   help="proceed when load_state_dict reports missing or "
                        "unexpected keys. Off by default: an inexact load "
                        "leaves parameters at their constructor initialisation "
                        "and the forecasts are then not the checkpoint's. Use "
                        "only for a mismatch you have inspected and decided is "
                        "benign - it is recorded in the run log either way")
    p.add_argument("--calibrate", action="store_true",
                   help="Kronos CRPS and wall-clock vs sample_count on one fold")
    p.add_argument("--calib-sample-counts", nargs="+", type=int,
                   default=[100, 200, 500])
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                   help="Kronos inference precision. fp16 uses autocast on the "
                        "inference call only; normalisation, denormalisation "
                        "and quantiles stay fp64. Kronos is a sampler, so this "
                        "changes which paths are drawn - verify with "
                        "--calibrate --calib-precisions fp32 fp16 (default fp16)")
    p.add_argument("--precision-verification",
                   default="results/kronos_calibration.json",
                   help="calibration file whose fp32/fp16 verdict is copied "
                        "into grid.json and the run log")
    p.add_argument("--calib-precisions", nargs="+", default=["fp32", "fp16"],
                   choices=["fp32", "fp16"])
    p.add_argument("--calib-repeats", type=int, default=2,
                   help="runs per precision. 2 or more gives a run-to-run "
                        "spread to read the precision shift against; 1 gives a "
                        "bare difference with no noise floor (default 2)")
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
