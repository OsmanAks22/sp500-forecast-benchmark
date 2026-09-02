"""Uniform interface over four generations of forecasters.

Every model, from a random walk to a 100M-parameter pre-trained transformer,
exposes the same two calls::

    model.fit(train_panel)
    model.predict(test_panel) -> DataFrame[unique_id, ds, q0.01 ... q0.99]

That uniformity is what makes the comparison fair: the evaluation layer cannot
accidentally give one family an easier path than another, because it cannot tell
them apart.

Local vs global
---------------
ARIMA and ETS fit one model per ticker; the neural models fit a single model
across all fifty. This asymmetry is intrinsic to the architectures, not a
design flaw - but it must be stated in the write-up, because a global model sees
fifty times the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["baseline", "statistical", "linear", "recurrent", "transformer", "foundation"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tier: Tier
    library: str
    scope: Literal["local", "global", "zero_shot"]
    probabilistic: bool
    in_proposal: bool
    notes: str = ""
    params: dict = field(default_factory=dict)


REGISTRY: list[ModelSpec] = [
    # Tier 0 - the benchmark that actually matters on daily equity returns.
    #
    # ZeroReturn is the H1 reference, and the distinction from LastReturn is not
    # cosmetic. A random walk in *prices* forecasts zero return; predicting
    # yesterday's return instead is a different model with sqrt(2(1-rho)) times
    # the RMSE - about 1.45x on this panel - so scoring against it flatters
    # everything else by roughly 30%. That error was live in this repository
    # until 2 September 2026; see the note under H1 in the README.
    ModelSpec("ZeroReturn", "baseline", "sp500bench", "local", True, True,
              "The random walk on prices: the forecast is always zero. This is the "
              "H1 reference. Its RMSE is the sample standard deviation of test "
              "returns by construction, which is the number every other model has "
              "to beat."),
    ModelSpec("LastReturn", "baseline", "statsforecast", "local", True, False,
              "ADDED. statsforecast's Naive: tomorrow's return equals today's. NOT "
              "the random-walk benchmark - it differences an already-differenced "
              "series, so its RMSE is sqrt(2(1-rho)) times ZeroReturn's. Kept as an "
              "informative baseline for how much a spurious lag costs, never as the "
              "H1 reference."),
    ModelSpec("RandomWalkWithDrift", "baseline", "statsforecast", "local", True, True,
              "LastReturn plus the mean per-session change; inherits the same "
              "differencing penalty and is likewise not the H1 reference."),

    # Tier 1 - classical statistical.
    ModelSpec("AutoARIMA", "statistical", "statsforecast", "local", True, True,
              "Refit once per fold, not per session - 50 tickers x daily is wasteful."),
    ModelSpec("AutoETS", "statistical", "statsforecast", "local", True, True),

    # Tier 2a - linear deep learning. Zeng et al. (2023).
    ModelSpec("DLinear", "linear", "neuralforecast", "global", True, False,
              "ADDED. The proposal cites Zeng et al.; omitting their model is indefensible."),
    ModelSpec("NLinear", "linear", "neuralforecast", "global", True, False, "ADDED."),

    # Tier 2b - recurrent.
    ModelSpec("LSTM", "recurrent", "neuralforecast", "global", True, True),
    ModelSpec("GRU", "recurrent", "neuralforecast", "global", True, True),

    # Tier 2c - supervised transformers.
    ModelSpec("PatchTST", "transformer", "neuralforecast", "global", True, True,
              "Patches the series into tokens; the only model given a tuning budget."),
    ModelSpec("TFT", "transformer", "neuralforecast", "global", True, True,
              "Variable-selection network gives per-feature attribution - use it."),
    ModelSpec("TimesNet", "transformer", "neuralforecast", "global", True, True,
              "Heaviest model in the study. Smoke-test on 5 tickers before committing."),

    # Tier 3 - pre-trained foundation models, inference only.
    # Second-generation checkpoints; the 2024-era models they replace are one
    # generation behind as of August 2026.
    ModelSpec("Chronos-2", "foundation", "chronos", "zero_shot", True, False,
              "ADDED. Current SOTA generic TSFM; native quantile + covariate support.",
              {"checkpoint": "amazon/chronos-2", "pretrain_cutoff": "undocumented"}),
    ModelSpec("TimesFM-2.5", "foundation", "timesfm", "zero_shot", True, False,
              "ADDED. Decoder-only, patch-based, continuous quantile head. The "
              "entry point is timesfm.TimesFM3Forecaster - NOT timesfm.TimesFm, "
              "which does not exist in the installed release and cost one Kaggle "
              "smoke-test run to discover. Recorded here so the real run does not "
              "rediscover it.",
              {"checkpoint": "google/timesfm-2.5-200m-pytorch",
               "entry_point": "timesfm.TimesFM3Forecaster",
               "pretrain_cutoff": "undocumented"}),
    ModelSpec("Moirai-2.0", "foundation", "uni2ts", "zero_shot", True, False,
              "ADDED. Decoder-only with multi-token prediction; top GIFT-Eval ranking.",
              {"checkpoint": "Salesforce/moirai-2.0-R-small",
               "pretrain_cutoff": "undocumented"}),
    ModelSpec("Kronos-small", "foundation", "kronos", "zero_shot", True, False,
              "ADDED. Finance-specific TSFM (AAAI 2026), native OHLCV tokenisation. "
              "Autoregressive sampling - derive returns from predicted close; "
              "CRPS must come from sampled paths, not a closed-form quantile head.",
              {"checkpoint": "NeoQuasar/Kronos-small",
               "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
               "pretrain_cutoff": "2024-06",
               "context_limit": 512,
               "us_equities_in_distribution": True}),
    ModelSpec("TTM", "foundation", "granite-tsfm", "zero_shot", True, False,
              "ADDED. TinyTimeMixers (IBM Granite), a sub-1M-parameter MLP-Mixer TSFM. "
              "Direct replication probe: Brini (2026) reports TTM as the only TSFM to "
              "beat Log-HAR at every horizon, by 1.3-1.8%. If that margin does not "
              "reproduce here, the discrepancy is the finding.",
              {"checkpoint": "ibm-granite/granite-timeseries-ttm-r2",
               "pretrain_cutoff": "undocumented"}),

    # Volatility arm only - the mandatory econometric benchmark.
    ModelSpec("HAR-RV", "statistical", "sp500bench", "local", True, False,
              "ADDED. Corsi (2009) heterogeneous autoregressive cascade.",
              {"target": "volatility"}),
    ModelSpec("Log-HAR", "statistical", "sp500bench", "local", True, False,
              "ADDED. The benchmark most TSFMs fail to beat (Brini, 2026). "
              "Smearing correction applied on back-transform.",
              {"target": "volatility"}),
    ModelSpec("GJR-GARCH", "statistical", "arch", "local", True, False,
              "ADDED. GJR-GARCH(1,1) with skew-t innovations. Captures the leverage "
              "effect - a negative return raises next-session volatility more than a "
              "positive return of the same size - which the HAR cascade cannot "
              "represent, since HAR regresses on unsigned realised variance over "
              "daily/weekly/monthly windows and is blind to the sign of the shock. "
              "Skew-t innovations also give the conditional density a fat left tail, "
              "so the arm is not conceding the new 0.01/0.05 eval levels to Gaussian "
              "HAR quantiles by construction.",
              {"target": "volatility"}),
]

# Pre-training cutoffs are documented only for Kronos (June 2024, stated in the
# paper's Appendix D). For the other four the cutoff is undocumented, which is
# itself a reportable finding: without it, no zero-shot comparison on recent data
# can be certified contamination-free.
CLEAN_SLICE_VALID_FOR = {"Kronos-small"}

BY_NAME = {m.name: m for m in REGISTRY}


def contaminated(spec: ModelSpec) -> bool:
    """Whether a model's pre-training corpus may overlap the test period.

    Foundation models were pre-trained on public market data through roughly
    2025. Their headline test-period numbers are therefore not comparable to
    models trained only on the 2005-2016 window. Report them, then repeat the
    comparison on the clean slice defined in ``configs/data.yaml``.
    """
    return spec.tier == "foundation"


def summary() -> str:
    rows = [f"{'model':<22}{'tier':<14}{'scope':<11}{'source'}"]
    rows.append("-" * 62)
    for m in REGISTRY:
        origin = "proposal" if m.in_proposal else "ADDED"
        rows.append(f"{m.name:<22}{m.tier:<14}{m.scope:<11}{origin}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(summary())
    print(f"\n{len(REGISTRY)} models | "
          f"{sum(m.in_proposal for m in REGISTRY)} from proposal | "
          f"{sum(not m.in_proposal for m in REGISTRY)} added")
