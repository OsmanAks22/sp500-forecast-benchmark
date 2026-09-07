"""Scale guardrails on a predictive distribution, shared by both runners.

Every silent failure in this project so far produced plausible numbers. The
TimesFM run loaded a checkpoint into the wrong class, completed without error,
passed every structural assertion in its adapter, and wrote 83,450 forecasts
whose only wrong feature was magnitude: a median of +2.45% per day against a
realised +0.03%. Nothing was checking magnitude.

A model trained on the wrong quantile grid fails the same way. The columns are
present, the ordering is monotone, the numbers are finite, and the band is the
wrong width. So the check lives here rather than in one runner, and both the
foundation tier and the neural tier call it on their first fold.

The bounds are NOT principled. They are empirical guardrails calibrated to this
panel - fifty large-cap US equities, 2005-2026 - and would be wrong for another
asset class, another universe or another horizon. They are sized to catch an
error of a factor, not one of a few percent.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Q90_Q10_TO_SIGMA", "SIGMA_BANDS", "implied_daily_sigma",
           "check_sigma_band", "SigmaOutOfBand"]

# q0.9 - q0.1 spans 2 * Phi^-1(0.9) standard deviations of a normal.
Q90_Q10_TO_SIGMA = 2.0 * 1.2815515655446004          # 2.5631

# Return arm: the realised 1-day log return has sd 0.0183 on this panel, 0.0109
# to 0.0305 per ticker. The band runs from about a ninth of that to four and a
# half times it, so a fold as violent as 2020 sits comfortably inside while a
# fivefold error does not.
#
# Volatility arm: a different quantity on a different scale - the 21-session
# forward Yang-Zhang estimate, whose values run 0.0065 to 0.0572 between the 1st
# and 99th percentiles. A single global band would pass anything the return arm
# allows and would therefore never fire here.
SIGMA_BANDS = {
    "return": (0.002, 0.08),
    "volatility": (0.0002, 0.03),
}


class SigmaOutOfBand(RuntimeError):
    """The predictive distribution is the wrong width by a factor."""


def implied_daily_sigma(q10, q90) -> float:
    """Sigma implied by the 80% interval, ignoring non-finite rows.

    Non-finite rows are skipped rather than propagated because a model that
    legitimately emits no interval - a point-forecast-only model - should be
    absent from this check, not failed by it.
    """
    width = np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float)
    width = width[np.isfinite(width)]
    if not width.size:
        return float("nan")
    return float(np.mean(width) / Q90_Q10_TO_SIGMA)


def check_sigma_band(q10, q90, arm: str = "return", label: str = "") -> str:
    """Return a one-line report, or raise :class:`SigmaOutOfBand`.

    Returns rather than prints so the caller decides where the line goes, and
    raises rather than warning because a distribution off by a factor is a
    loading or normalisation fault and not a forecast worth spending a run on.
    """
    if arm not in SIGMA_BANDS:
        raise KeyError(f"no sigma band for arm {arm!r}; known: {sorted(SIGMA_BANDS)}")
    sigma = implied_daily_sigma(q10, q90)
    low, high = SIGMA_BANDS[arm]
    if not np.isfinite(sigma):
        return f"no finite 80% interval{' for ' + label if label else ''}; band check skipped"
    annualised = sigma * np.sqrt(252) * 100
    if not (low <= sigma <= high):
        raise SigmaOutOfBand(
            f"{label or 'model'}: implied daily sigma {sigma:.5f} "
            f"({annualised:.1f}% annualised) is outside the {arm}-arm band "
            f"[{low}, {high}].\n"
            "  This is an empirical guardrail calibrated to this panel, not a "
            "principled bound;\n"
            "  it fires on errors of a factor, not of a few percent. A "
            "predictive distribution\n"
            "  this far from the realised scale is almost always a loading, "
            "normalisation or\n"
            "  training-grid fault rather than a forecast.")
    return (f"implied daily sigma {sigma:.5f} ({annualised:.1f}% ann), "
            f"inside the {arm} band [{low}, {high}]")
