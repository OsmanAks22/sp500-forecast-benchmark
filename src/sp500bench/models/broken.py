"""Models whose written output is not the model's, and the evidence for saying so.

Separate from ``registry.py`` because this is not a roster fact. The roster is
twenty specs and stays twenty; ``configs/models.yaml`` and the registry still
agree on every field they are checked against. What is recorded here is that one
of those specs ran and produced a file that does not contain forecasts.

It lives in the package rather than in a script because more than one consumer
needs it and they must not disagree. ``evaluate.py`` excludes these models from
scoring; ``backtest.py`` must exclude them too, and not only so the row is
absent - the deflated Sharpe hurdle is parameterised by the number of strategies
tried, so a broken model left in the count raises the bar every other model has
to clear. A guard in one script and not the other is not a guard.

The reason string is part of the artefact, not a comment. A model dropped
without a stated reason is indistinguishable from a model dropped for scoring
badly, and in a pre-registered study that difference is the whole question.
Every entry therefore cites what was measured, not what was concluded.
"""

from __future__ import annotations

__all__ = ["BROKEN_MODELS", "is_broken", "reason"]


BROKEN_MODELS: dict[str, str] = {
    "NLinear": (
        "median carries a constant shift of about -6.2% per day, invariant "
        "across all four folds (-0.06231, -0.06173, -0.06292, -0.06211) and "
        "across all fifty tickers (-0.06310 to -0.06045). A model responding to "
        "data does not hold a location constant while the regime changes, so "
        "the point axes measure the shift rather than the model. Its quantile "
        "columns are additionally non-monotone on 100% of rows, with 54.7% of "
        "adjacent pairs inverted."),
}


def is_broken(model: str) -> bool:
    return model in BROKEN_MODELS


def reason(model: str) -> str:
    return BROKEN_MODELS[model]
