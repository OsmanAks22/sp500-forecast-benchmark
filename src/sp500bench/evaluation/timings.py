"""One timing record per model, merged rather than overwritten.

Both runners write ``results/timings.json``: the CPU and neural tiers through
``run_experiment``, the foundation tier through ``run_foundation``. Compute cost
is one of the axes this study reports and it cannot be reconstructed after the
fact, so the record has to survive being written to twice.

It did not. A run whose fold files already exist skips every fold and still
reaches the end of the model loop, where the timing was written unconditionally
- so it recorded zero folds and zero forecasts over a complete record. That is
how the TTM entry came to claim nothing had run while four folds and 83,450
forecasts sat on disk beside it. Nothing failed, and the number left behind
reads as a measurement rather than an absence.

The merge is here, in one place, because the same shape of pass exists in both
runners and a guard in only one of them is not a guard.
"""

from __future__ import annotations

import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["ROSTER_KEY", "RUN_IDENTITY_FIELDS", "SESSION_ID", "comparable",
           "hardware_fingerprint", "merge_timing", "record_timing"]


# Fields that identify the run a timing belongs to. Per-fold times measured
# under different settings are not the same measurement and must not be merged.
RUN_IDENTITY_FIELDS = ("context", "device", "sample_count", "precision",
                       "paths_per_call", "checkpoint", "checkpoint_revision",
                       # What the generic TSFMs were fed - returns or prices -
                       # is a documented decision of this runner and changes
                       # what the model was asked to do, not merely how fast it
                       # did it. Folds measured under different input series
                       # are different measurements.
                       "input_series",
                       # Tier 2: a fold trained for a different number of steps
                       # or on a different lookback is not the same measurement
                       # either. `device` is already in the list above.
                       "max_steps", "input_size", "num_workers",
                       # A fold re-timed on another machine, or under a
                       # different numeric stack, is not the same measurement;
                       # merging the two would produce a per-fold average across
                       # two environments.
                       "platform", "accelerator", "torch_version",
                       "pandas_version")


def merge_timing(prior: dict | None, payload: dict) -> dict:
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




ROSTER_KEY = "_roster"

# One id per process. Two timings share a session only if they were measured by
# the same invocation on the same machine.
SESSION_ID = f"{uuid.uuid4().hex[:12]}"


def hardware_fingerprint() -> dict:
    """What measured this timing, so a ratio between two rows can be checked.

    A wall-clock number is meaningless without the machine that produced it, and
    a *ratio* between two numbers measured on different machines is worse than
    meaningless - it looks like a comparison. This study lost one timing file to
    an overwrite and rebuilt half of it on a different machine from the other
    half, which is exactly the situation in which a cross-model ratio silently
    stops being a measurement of the models.

    So every record carries its machine and its session, and the rule that
    follows is mechanical rather than remembered: a ratio between two models is
    legitimate only where both rows agree on ``hardware`` and ``session_id``.
    """
    info = {
        "session_id": SESSION_ID,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "accelerator": "cpu",
    }
    try:
        import pandas
        info["pandas_version"] = str(pandas.__version__)
    except Exception:
        pass
    try:
        import torch
        # Recorded whenever torch is importable, not only under CUDA: the
        # numeric stack is part of what a wall-clock ratio depends on, and the
        # foundation runner's records name this field, so a CPU row that omits
        # it is not comparable with anything and silently drops out.
        info["torch_version"] = str(torch.__version__)
        if torch.cuda.is_available():                  # pragma: no cover - no GPU locally
            info["accelerator"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


# What must match for a ratio between two timings to be a measurement of the
# models rather than of two computers.
#
# Session id is deliberately NOT in this list, and was at first. A hosted
# notebook runs each model as a separate `!python scripts/...` invocation, so a
# per-process id differs between every row even when eleven models ran on the
# same accelerator, in the same container, minutes apart. Gating on it made the
# entire within-generation comparison unquotable, which is a stricter answer
# than the question deserves: process identity is not what a wall-clock ratio
# depends on.
#
# What it does depend on is the machine and the software stack. `pandas_version`
# is here because it has already mattered - three foundation models ran after a
# pip install pulled pandas to 3.0.5 while the rest of that session ran on 2.x,
# and a session id alone would have implied a shared environment they did not
# have.
# Always recordable, so absence means the row was written before stamping and
# nothing can be said about it. Must be present on both sides and equal.
REQUIRED_COMPARABLE = ("platform", "accelerator", "pandas_version")

# Recordable only where the library is installed. The CPU tier has no torch, so
# demanding it would make every statistical model non-comparable with every
# other one - including with itself. Compared when both sides have it, ignored
# when neither does, and treated as a mismatch when only one does, because a row
# that ran under torch and a row that did not are not the same environment.
OPTIONAL_COMPARABLE = ("torch_version",)

COMPARABLE_FIELDS = REQUIRED_COMPARABLE + OPTIONAL_COMPARABLE


def comparable(a: dict, b: dict) -> bool:
    """Whether a ratio between two timing records is a measurement at all.

    Session id is recorded on every row and reported beside the figures, but it
    is not a gate. Two rows from different processes on the same machine and the
    same stack are comparable; two rows from one process on different stacks are
    not.

    A record missing any of these fields is not comparable with anything,
    including another record missing the same fields. Absence is not a match:
    two unstamped rows are not known to share a machine, they are known to say
    nothing about it, and treating None == None as agreement would manufacture
    exactly the false comparison this function exists to refuse.
    """
    if not all(a.get(k) is not None and a.get(k) == b.get(k)
               for k in REQUIRED_COMPARABLE):
        return False
    return all(a.get(k) == b.get(k) for k in OPTIONAL_COMPARABLE)


def _roster_block(models: list[str]) -> dict:
    """What this file claims to describe, so a reader can check the claim.

    The merge below protects the record against a pass that measured nothing
    *inside* a session. It cannot protect against the file being replaced from
    outside one - a copy back from a hosted notebook overwrote fifteen models'
    timings with two, and nothing in the file said it had once held more. A
    guard that cannot fire is not worth adding, so the answer is not another
    check but a self-describing artefact: the file records which models it
    covers and when, and a reader comparing that list against the forecasts on
    disk sees the gap immediately.
    """
    return {
        "models": sorted(models),
        "n_models": len(models),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("The models this file holds timings for. Compare it against "
                 "the directories under forecasts/: a model with forecasts and "
                 "no entry here means this file was replaced rather than "
                 "merged, and the missing timings are gone rather than "
                 "pending."),
    }


def record_timing(path: Path, model: str, payload: dict) -> None:
    """Merge one model's wall-clock into the shared record.

    Merged rather than overwritten so tiers run on different days, on different
    machines, accumulate into one file - and so a pass that measured nothing
    cannot erase one that measured everything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! {path} is not valid JSON; starting a fresh record")
    payload = {**hardware_fingerprint(), **payload}
    merged = merge_timing(existing.get(model), payload)
    if merged.get("last_invocation_note"):
        print(f"  timings: kept the existing {model} record "
              f"({merged.get('n_folds')} folds, {merged.get('n_forecasts', 0):,} "
              "forecasts) - this invocation ran no folds")
    existing[model] = merged
    existing[ROSTER_KEY] = _roster_block(
        [k for k in existing if k != ROSTER_KEY])
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
