"""Purged, embargoed, expanding-window cross-validation for financial time series.

Why this module exists
----------------------
Plain walk-forward CV still leaks when the prediction target spans multiple
periods. A sample observed at time ``t`` carries a label built from data up to
``t + h``. If ``t`` sits in the training fold and ``t + h`` reaches into the test
fold, the model has seen the future.

Two corrections, following Lopez de Prado (2018, *Advances in Financial Machine
Learning*, ch. 7):

* **Purge** - drop training samples whose label window overlaps the test window.
* **Embargo** - drop a further block of training samples immediately before the
  test window, to absorb serial correlation that survives purging.

Both are applied on the *time axis*, never on the row index, so the logic is
identical whether the panel is balanced or ragged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

__all__ = ["Split", "PurgedWalkForward", "assert_no_overlap"]


@dataclass(frozen=True)
class Split:
    """One fold of the walk-forward scheme.

    Attributes
    ----------
    fold:
        Zero-based fold index.
    train_end:
        Last timestamp a training label is permitted to *depend on*, after
        purging and embargo have been subtracted. Inclusive.
    test_start, test_end:
        Inclusive bounds of the test window.
    """

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Split(fold={self.fold}, "
            f"train<=|{self.train_end.date()}| "
            f"test=[{self.test_start.date()} .. {self.test_end.date()}])"
        )


class PurgedWalkForward:
    """Expanding-window CV with purging and an embargo.

    Parameters
    ----------
    test_windows:
        Sequence of ``(start, end)`` pairs delimiting each test fold. Supplied
        explicitly rather than derived, so the folds are auditable and frozen in
        config rather than recomputed at runtime.
    horizon:
        Forecast horizon ``h`` in trading days. Determines the purge width: a
        label stamped at ``t`` consumes data through ``t + h``.
    embargo:
        Extra trading days removed from the tail of the training set, on top of
        the purge.
    train_start:
        Earliest timestamp available for training. Expanding-window means every
        fold trains from here.
    """

    def __init__(
        self,
        test_windows: Sequence[tuple[str, str]],
        horizon: int,
        embargo: int,
        train_start: str,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.test_windows = [
            (pd.Timestamp(a), pd.Timestamp(b)) for a, b in test_windows
        ]
        self.horizon = int(horizon)
        self.embargo = int(embargo)
        self.train_start = pd.Timestamp(train_start)

        for i, (start, end) in enumerate(self.test_windows):
            if start > end:
                raise ValueError(f"test window {i} is inverted: {start} > {end}")
        for i in range(1, len(self.test_windows)):
            if self.test_windows[i][0] <= self.test_windows[i - 1][1]:
                raise ValueError(
                    f"test windows {i - 1} and {i} overlap; folds must be disjoint"
                )

    @property
    def gap(self) -> int:
        """Total trading days excised between train and test."""
        return self.horizon + self.embargo

    def splits(self, calendar: pd.DatetimeIndex) -> list[Split]:
        """Materialise the folds against a trading calendar.

        ``calendar`` must be the sorted set of session dates. The gap is counted
        in *sessions*, not calendar days, so weekends and holidays do not
        silently shrink the embargo.
        """
        calendar = pd.DatetimeIndex(calendar).sort_values().unique()
        out: list[Split] = []
        for fold, (test_start, test_end) in enumerate(self.test_windows):
            pos = calendar.searchsorted(test_start, side="left")
            cut = pos - self.gap
            if cut <= 0:
                raise ValueError(
                    f"fold {fold}: calendar too short for a gap of {self.gap} "
                    "sessions before the test window"
                )
            train_end = calendar[cut - 1]
            if train_end < self.train_start:
                raise ValueError(f"fold {fold}: no training data survives the gap")
            out.append(
                Split(
                    fold=fold,
                    train_end=train_end,
                    test_start=pd.Timestamp(test_start),
                    test_end=pd.Timestamp(test_end),
                )
            )
        return out

    def iter_frames(
        self,
        panel: pd.DataFrame,
        time_col: str = "ds",
    ) -> Iterator[tuple[Split, pd.DataFrame, pd.DataFrame]]:
        """Yield ``(split, train_df, test_df)`` for each fold.

        The training frame is truncated at ``split.train_end`` *inclusive*, which
        already accounts for purge and embargo.
        """
        calendar = pd.DatetimeIndex(panel[time_col].unique())
        for split in self.splits(calendar):
            train_mask = (panel[time_col] >= self.train_start) & (
                panel[time_col] <= split.train_end
            )
            test_mask = (panel[time_col] >= split.test_start) & (
                panel[time_col] <= split.test_end
            )
            yield split, panel.loc[train_mask].copy(), panel.loc[test_mask].copy()


def assert_no_overlap(
    train_time: pd.Series,
    test_time: pd.Series,
    horizon: int,
    embargo: int,
) -> None:
    """Raise if any training label window reaches into the test window.

    Called from the test suite and, defensively, from the experiment runner.
    Cheap enough to leave switched on in production runs.
    """
    if train_time.empty or test_time.empty:
        return
    last_train_label_end = train_time.max() + pd.Timedelta(days=horizon)
    first_test = test_time.min()
    if last_train_label_end >= first_test:
        raise AssertionError(
            f"leakage: last training label extends to {last_train_label_end.date()} "
            f"but test begins {first_test.date()}"
        )
    sessions_between = (first_test - train_time.max()).days
    if sessions_between < embargo:
        raise AssertionError(
            f"embargo violated: only {sessions_between} days between folds, "
            f"{embargo} required"
        )
