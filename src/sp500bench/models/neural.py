"""Tier 2 - the supervised deep-learning generation, via neuralforecast.

Seven models: DLinear and NLinear (Zeng et al.), LSTM and GRU, PatchTST, TFT and
TimesNet. They are the generation the thesis question is about; without them the
comparison is between two generations, not three.

All seven are **global**: one model is fitted across all fifty series per fold,
not one per ticker. That asymmetry against ARIMA and ETS is intrinsic to the
architectures and is stated in the write-up - a global model sees fifty times
the data - but it is also what makes seven models affordable here, because a
fold costs one fit rather than fifty.


WHAT THIS TIER COSTS A KAGGLE SESSION, LEARNED THE HARD WAY
-----------------------------------------------------------
Both of these cost a session already. They are recorded here rather than in the
notebook so that whoever reads this file does not have to find the notebook.

**Lightning cannot fork workers once CUDA is initialised.** Running this tier in
a kernel that has already touched CUDA - the foundation smoke test does, twice -
fails with::

    RuntimeError: Lightning can't create new processes if CUDA is already
    initialized

So the neural tier belongs in its own Kaggle session, on a fresh kernel. Not a
kernel restart: a session that has already imported and run a foundation model
has an initialised CUDA context, and restarting the kernel does not clear what
pip and the driver have already done. Start a new session and run this tier
alone.

**Dataloader settings go in ``dataloader_kwargs``, trainer settings go loose.**
The first attempt passed ``num_workers_loader=0`` and got::

    TypeError: Trainer.__init__() got an unexpected keyword argument
    'num_workers_loader'

because that parameter does not exist in neuralforecast. Anything the model
constructor does not name falls into ``**trainer_kwargs`` and is handed straight
to ``pl.Trainer``, which rejects it. The chain that works is::

    DLinear(dataloader_kwargs={"num_workers": n})
      -> BaseModel.dataloader_kwargs
      -> TimeSeriesDataModule(**dataloader_kwargs)
      -> the DataLoader

while ``accelerator`` and ``devices`` legitimately travel through
``**trainer_kwargs``. The smoke test used zero workers, which serialises data
loading; that was a workaround for cell ordering and is wrong for seven models
over fifty tickers and four folds. ``--num-workers`` exists so it can be lowered
again if the image objects.


HOW A FOLD IS FORECAST
----------------------
Fit once on the purged training frame, then roll.

The models are fitted on ``train`` alone, which the walk-forward has already
purged and embargoed, so nothing inside the gap or the test window reaches the
optimiser. Forecasting is then a sequence of one-step-ahead predictions: for
each test session ``t`` the history through ``t`` is handed to ``predict`` and
the model returns ``t + 1`` for all fifty tickers at once.

Rolling this way rather than through ``cross_validation`` is deliberate.
``cross_validation`` decides internally where training stops, and this study's
purge and embargo are not its default; passing the whole frame and trusting it
to split correctly would put the gap sessions into the fit. Here the fit sees
exactly ``train`` and nothing else, which is checkable by reading this file.

Using sessions inside the gap as *context* for a prediction is not leakage - it
is ordinary conditioning on the past, the same treatment the foundation tier's
zero-shot models get. The constraint that matters is that the label at ``t`` is
the forward return to ``t + 1``, so the context may run to ``t`` and no further.


THE QUANTILE GRID
-----------------
Every model is trained with ``MQLoss`` on the frozen nine levels. MQLoss emits
exactly the levels it was trained on, so a model trained on a different set
cannot supply the missing columns and would surface as NaN in ``evaluate.py``
long after the GPU hours were spent. :func:`assert_training_grid` gates that at
construction instead, and the grid actually passed to the loss is printed with
the run.

The output is *named* by the symmetric interval each level bounds rather than by
the level - ``lo-98.0`` for 0.01, ``hi-98.0`` for 0.99, ``median`` for 0.5 - so
reading it back assumes the quantile set is symmetric about 0.5. The frozen grid
is symmetric and this is sound for it. A future grid that is not could not be
read back from these names at all: a level and its reflection share an interval
width and differ only by the lo/hi prefix, so an unpaired level would silently
resolve to a column that belongs to nothing. :func:`interval_column_names`
derives the name and the caller asserts it is present, so that case fails at the
mapping instead of producing a plausible column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from sp500bench.config import quantile_column

__all__ = ["NEURAL_MODELS", "NeuralRuntime", "RUNTIME", "configure",
           "assert_training_grid", "build_model", "neural_forecast_fold",
           "make_executor", "executors"]

# The seven specs in registry.py whose library is neuralforecast, mapped to the
# class that implements each. Names must match the roster exactly.
NEURAL_MODELS = {
    "DLinear": "DLinear",
    "NLinear": "NLinear",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "PatchTST": "PatchTST",
    "TFT": "TFT",
    "TimesNet": "TimesNet",
}


@dataclass
class NeuralRuntime:
    """Settings that come from the command line rather than the frozen spec.

    Held here rather than threaded through the executor signature because every
    other executor in the roster shares that signature and none of them has a
    device. ``run_experiment`` sets this once, before any model runs, and prints
    it - so it is configuration, not hidden state.
    """

    device: str = "auto"
    num_workers: int = 4
    max_steps: int = 500
    input_size: int = 60
    seed: int = 20260826
    batch_size: int = 32
    val_size: int = 0

    @property
    def accelerator(self) -> str:
        if self.device in ("cuda", "gpu"):
            return "gpu"
        if self.device == "cpu":
            return "cpu"
        return "auto"

    def trainer_kwargs(self) -> dict:
        kwargs = {"accelerator": self.accelerator,
                  "enable_progress_bar": False,
                  "logger": False,
                  "enable_checkpointing": False}
        if self.accelerator == "gpu":
            kwargs["devices"] = 1
        return kwargs

    def describe(self) -> str:
        return (f"device={self.device} (accelerator={self.accelerator}) "
                f"num_workers={self.num_workers} max_steps={self.max_steps} "
                f"input_size={self.input_size} seed={self.seed}")


RUNTIME = NeuralRuntime()


def configure(**kwargs) -> NeuralRuntime:
    """Set the runtime once, from parsed arguments."""
    global RUNTIME
    RUNTIME = NeuralRuntime(**{k: v for k, v in kwargs.items() if v is not None})
    return RUNTIME


def assert_training_grid(quantiles: Sequence[float], frozen: Sequence[float]) -> list[float]:
    """The loss must be given the frozen grid, element for element, in order.

    ``config.load_configs`` already gates the two halves of the specification
    against each other. This extends the same gate to the executor, because the
    configs agreeing with each other does not prove that what reached MQLoss was
    either of them. A model trained on a subset emits only that subset, and the
    missing columns arrive as NaN in ``evaluate.py`` - after the GPU hours.
    """
    levels = [float(q) for q in quantiles]
    expected = [float(q) for q in frozen]
    if levels != expected:
        missing = sorted(set(expected) - set(levels))
        extra = sorted(set(levels) - set(expected))
        detail = ""
        if missing:
            detail += f"\n  levels scored but not trained: {missing}"
        if extra:
            detail += f"\n  levels trained but not scored: {extra}"
        if not missing and not extra:
            detail = "\n  same levels, different order - MQLoss emits them in the order given"
        raise ValueError(
            "the training grid handed to MQLoss is not the frozen grid.\n"
            f"  passed to the loss : {levels}\n"
            f"  configs/eval.yaml  : {expected}{detail}\n"
            "  MQLoss emits exactly the levels it was trained on, so this would\n"
            "  surface as NaN columns in evaluate.py rather than as an error.")
    return levels


def build_model(name: str, quantiles: Sequence[float], horizon: int,
                runtime: NeuralRuntime | None = None):
    """One neuralforecast model, with MQLoss on the frozen grid."""
    from neuralforecast import models as nf_models
    from neuralforecast.losses.pytorch import MQLoss

    runtime = runtime or RUNTIME
    if name not in NEURAL_MODELS:
        raise KeyError(f"{name!r} is not a Tier 2 model; known: {sorted(NEURAL_MODELS)}")
    cls = getattr(nf_models, NEURAL_MODELS[name], None)
    if cls is None:
        raise SystemExit(
            f"neuralforecast has no model named {NEURAL_MODELS[name]!r}. "
            f"Available: {sorted(n for n in dir(nf_models) if not n.startswith('_'))}")

    levels = list(quantiles)
    kwargs = dict(
        h=horizon,
        input_size=runtime.input_size,
        loss=MQLoss(quantiles=levels),
        max_steps=runtime.max_steps,
        random_seed=runtime.seed,
        batch_size=runtime.batch_size,
        # Dataloader settings belong in this dict; trainer settings go loose.
        # See the module docstring - the other way round costs a session.
        dataloader_kwargs={"num_workers": runtime.num_workers},
        **runtime.trainer_kwargs(),
    )
    return cls(**kwargs)


def _histories(panel: pd.DataFrame, id_col: str, time_col: str,
               target_col: str) -> dict[str, tuple[pd.DataFrame, pd.DatetimeIndex]]:
    """Per-ticker frames plus their date index, sorted once.

    The index is kept alongside so the rolling loop can find an origin by
    ``searchsorted`` and slice by position. Masking the whole series per origin
    instead is 6.3x slower - measured at 15.4 against 2.5 ms per origin across
    fifty tickers - which is three minutes against half a minute over the tier.
    Small either way, but the loop runs 1,669 times per model and there is no
    reason to scan five thousand rows to find the last sixty.
    """
    out, trimmed = {}, 0
    for ticker, group in panel.groupby(id_col, sort=True):
        frame = (group[[id_col, time_col, target_col]]
                 .sort_values(time_col).reset_index(drop=True))
        finite = np.isfinite(frame[target_col].to_numpy(dtype=float))
        if not finite.all():
            if not finite.any():
                raise ValueError(f"{ticker}: no finite target anywhere in the panel")
            last_good = int(np.max(np.flatnonzero(finite)))
            # Only the trailing run is dropped. The last session of the panel
            # has no target by construction - make_target is log(P[t+1]/P[t])
            # and there is no t+1 - and neuralforecast refuses a frame holding
            # NaN in y. A NaN anywhere before that is a data defect, not a
            # boundary, and dropping it globally would hide it.
            if not finite[:last_good + 1].all():
                gaps = frame.loc[~np.isfinite(
                    frame[target_col].to_numpy(dtype=float))][time_col]
                interior = [d for d in gaps if d <= frame[time_col].iloc[last_good]]
                raise ValueError(
                    f"{ticker}: target is NaN at {len(interior)} session(s) "
                    f"inside the series, first at {pd.Timestamp(interior[0]).date()}. "
                    "That is a data defect rather than the end-of-panel "
                    "boundary, and it is not dropped silently.")
            trimmed += len(frame) - (last_good + 1)
            frame = frame.iloc[:last_good + 1].reset_index(drop=True)
        out[ticker] = (frame, pd.DatetimeIndex(frame[time_col]))
    return out, trimmed


def neural_forecast_fold(
    model_name: str,
    panel: pd.DataFrame,
    train: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    quantiles: Sequence[float],
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    horizon: int = 1,
    frozen_grid: Sequence[float] | None = None,
    runtime: NeuralRuntime | None = None,
    **_,
) -> pd.DataFrame:
    """Fit one global model on the purged training frame, then roll the window.

    The grid gate and the leakage guard run before neuralforecast is imported,
    deliberately. Both are statements about the protocol rather than about the
    library, and a protocol violation should be reported as itself - not as a
    missing package on a machine where the library happens to be absent, and not
    after a model has already been constructed.
    """
    runtime = runtime or RUNTIME
    levels = assert_training_grid(quantiles, frozen_grid or quantiles)
    columns = [quantile_column(q) for q in levels]

    fit_frame = train[[id_col, time_col, target_col]].rename(
        columns={id_col: "unique_id", time_col: "ds", target_col: "y"})
    train_end = pd.Timestamp(fit_frame["ds"].max())
    if train_end >= pd.Timestamp(test_start):
        raise AssertionError(
            f"{model_name}: the training frame reaches {train_end.date()}, on or "
            f"after the test window opens at {pd.Timestamp(test_start).date()}. "
            "The fit would see the test period.")

    from neuralforecast import NeuralForecast

    model = build_model(model_name, levels, horizon, runtime)
    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(df=fit_frame, val_size=runtime.val_size)

    histories, trimmed = _histories(panel, id_col, time_col, target_col)
    calendar = pd.DatetimeIndex(sorted(panel[time_col].unique()))
    in_window = calendar[(calendar >= pd.Timestamp(test_start))
                         & (calendar <= pd.Timestamp(test_end))]

    # An origin whose own label is NaN has no outcome to be scored against, and
    # evaluate.py refuses a forecast with none. The foundation runner drops the
    # same origin for the same reason; both tiers must forecast the same set or
    # they are not comparable.
    labelled = pd.DatetimeIndex(sorted(
        pd.unique(panel.loc[np.isfinite(panel[target_col].to_numpy(dtype=float)),
                            time_col])))
    origins = in_window.intersection(labelled)
    dropped = len(in_window) - len(origins)
    if dropped or trimmed:
        print(f"      {model_name}: {dropped} origin(s) with no forward return "
              f"skipped, {trimmed} trailing NaN row(s) trimmed from the history")

    rows = []
    width = runtime.input_size + horizon
    for origin in origins:
        blocks = []
        for ticker, (frame, index) in histories.items():
            # The context ends at the session BEFORE the origin.
            #
            # y[t] is the target, log(P[t+1]/P[t]), which is not known at t: the
            # last target known at the origin is y[t-1]. Including y[t] would put
            # the value being predicted into the model's own input, and the
            # one-step-ahead forecast would then be of y[t+1] while carrying
            # ds = t - scored against y[t], which the model had just been shown.
            # side="left" excludes the origin row; side="right" would include it.
            stop = int(index.searchsorted(origin, side="left"))
            if stop < width:
                continue
            block = frame.iloc[stop - width:stop]
            blocks.append(block.rename(columns={id_col: "unique_id",
                                               time_col: "ds",
                                               target_col: "y"}))
        if not blocks:
            continue
        history = pd.concat(blocks, ignore_index=True)
        prediction = nf.predict(df=history)
        rows.append(_to_grid(prediction, model_name, origin, levels, columns))

    if not rows:
        return pd.DataFrame(columns=["unique_id", "ds"] + columns)
    out = pd.concat(rows, ignore_index=True)
    _assert_origins_match(out, origins, model_name)
    return out


def _to_grid(prediction: pd.DataFrame, model_name: str, origin: pd.Timestamp,
             levels: list[float], columns: list[str]) -> pd.DataFrame:
    """Map neuralforecast's wide quantile output onto the frozen column names.

    The output columns are named by the model and the level, and the exact
    spelling has changed between releases, so they are located by matching the
    level rather than by assuming a format. A level that cannot be found is an
    error and not a NaN column: the whole point of the grid gate is that a
    missing level should never reach the parquet quietly.
    """
    frame = prediction.reset_index() if prediction.index.name else prediction.copy()
    out = pd.DataFrame({"unique_id": frame["unique_id"].to_numpy(),
                        "ds": origin})
    for level, column in zip(levels, columns):
        source = _find_level_column(frame.columns, model_name, level, levels)
        out[column] = frame[source].to_numpy(dtype=float)
    return out


def interval_column_names(model_name: str, level: float) -> list[str]:
    """The names neuralforecast gives one quantile level.

    MQLoss output is named by the *symmetric interval* the level bounds, not by
    the level itself. For a level below the median the column is the low edge of
    the ``100 * (1 - 2q)`` percent interval, above it the high edge of the
    ``100 * (2q - 1)`` percent one, and the median has its own name::

        0.01 -> lo-98.0    0.99 -> hi-98.0
        0.05 -> lo-90.0    0.95 -> hi-90.0
        0.10 -> lo-80.0    0.90 -> hi-80.0
        0.25 -> lo-50.0    0.75 -> hi-50.0
        0.50 -> median

    Derived from the level rather than tabulated, so a change to the frozen grid
    is followed automatically instead of silently missing a row of a hardcoded
    table. Two spellings of the width are returned because the decimal form is
    what the installed release emits and the bare integer is a plausible
    variant.

    THIS NAMING CANNOT EXPRESS AN ASYMMETRIC GRID. A level and its reflection
    about 0.5 map to the *same* interval width and are told apart only by the
    lo/hi prefix, so a grid holding 0.01 without 0.99 would still produce a
    ``lo-98.0`` column and nothing would mark the absence. The frozen grid is
    symmetric, so this is sound here; a future grid that is not would break at
    this function, which is the correct place for it to break.
    """
    if abs(level - 0.5) < 1e-12:
        return [f"{model_name}-median"]
    prefix = "lo" if level < 0.5 else "hi"
    width = 100.0 * (1.0 - 2.0 * level) if level < 0.5 else 100.0 * (2.0 * level - 1.0)
    return [f"{model_name}-{prefix}-{width:.1f}", f"{model_name}-{prefix}-{width:g}"]


def _is_symmetric(levels: Sequence[float] | None) -> bool:
    if not levels:
        return True
    values = sorted(float(q) for q in levels)
    return all(any(abs((1.0 - q) - other) < 1e-12 for other in values) for q in values)


def _find_level_column(columns, model_name: str, level: float,
                       levels: Sequence[float] | None = None) -> str:
    """Locate the column carrying one quantile level.

    The interval name is tried first because it is what neuralforecast actually
    emits; the older spellings are kept as fallbacks in case a release changes
    it back. A level that resolves to no column at all is an error rather than a
    NaN column - the point of the grid gate is that a gap must never reach the
    parquet quietly.
    """
    present = [c for c in columns if isinstance(c, str)]
    for name in interval_column_names(model_name, level):
        if name in present:
            return name

    candidates = [c for c in present if model_name in c]
    target = f"{level:g}"
    exact = [c for c in candidates
             if c.endswith(f"-ql{target}") or c.endswith(f"-q-{target}")
             or c.endswith(f"-{target}") or c.endswith(f"-q{target}")]
    if len(exact) == 1:
        return exact[0]

    detail = ""
    if not _is_symmetric(levels):
        detail = (
            "\n  The grid is NOT symmetric about 0.5. neuralforecast names its "
            "output by the\n  symmetric interval a level bounds, so an "
            "asymmetric grid cannot be read back\n  from the column names - a "
            "level and its reflection share an interval width and\n  are told "
            "apart only by the lo/hi prefix. This is the correct place to fail.")
    raise KeyError(
        f"{model_name}: no output column for quantile level {level}. "
        f"Tried {interval_column_names(model_name, level)}. "
        f"Columns present: {sorted(str(c) for c in columns)}. MQLoss emits only "
        "the levels it was trained on, so a missing level means the loss did "
        f"not receive the frozen grid.{detail}")


def _assert_origins_match(out: pd.DataFrame, origins, model_name: str) -> None:
    produced = pd.DatetimeIndex(sorted(pd.unique(out["ds"])))
    expected = pd.DatetimeIndex(sorted(origins))
    if not produced.equals(expected):
        missing = expected.difference(produced)
        extra = produced.difference(expected)
        raise AssertionError(
            f"{model_name}: forecast sessions do not match the test window. "
            f"{len(missing)} missing, {len(extra)} unexpected. "
            "A misaligned rolling window would score each forecast against the "
            "wrong outcome and nothing downstream would notice.")


def make_executor(model_name: str):
    """Bind one model name into the executor signature the runner calls."""

    def executor(panel, train, test_start, test_end, quantiles, **kwargs):
        return neural_forecast_fold(
            model_name, panel=panel, train=train, test_start=test_start,
            test_end=test_end, quantiles=quantiles, **kwargs)

    executor.__name__ = f"{model_name.lower()}_forecast_fold"
    executor.__doc__ = f"Tier 2 executor for {model_name}; see sp500bench.models.neural."
    return executor


def executors() -> dict:
    """The seven Tier 2 executors, keyed by roster name."""
    return {name: make_executor(name) for name in NEURAL_MODELS}
