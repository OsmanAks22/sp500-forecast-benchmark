# Three Generations of Forecasting Models on S&P 500 Daily Returns

A leakage-controlled, regime-stratified benchmark of statistical, supervised
deep-learning and pre-trained foundation models, evaluated on 50 S&P 500
constituents under purged walk-forward cross-validation.

MSc Data Science and Analytics, University of Westminster (7COSC012W)
Supervisor: Dr Panagiotis Chountas

---

## What this study asks

The 2023 question — *do transformers beat classical models on time series?* —
has largely been answered. This study asks the 2026 version:

> Across three architectural generations — classical statistical models,
> supervised deep learning, and pre-trained time-series foundation models — does
> any family produce a statistically significant, economically meaningful
> improvement in forecasting daily S&P 500 equity returns, once leakage is
> purged, multiple testing is corrected for, and transaction costs are charged?

Three secondary questions carry most of the contribution:

1. **Does the advantage live in the distribution rather than the point?** Point
   accuracy on daily returns is close to a coin flip for every model. Calibrated
   uncertainty may not be.
2. **Does the ranking survive regime stratification?** Aggregate RMSE hides
   models that are competent in calm markets and useless in crises.
3. **How much of a foundation model's apparent edge is pre-training
   contamination?** Their corpora include public market data through roughly
   2025 — overlapping this study's test period.

## The expected finding, stated in advance

Daily equity returns are close to a random walk. Directional accuracy of
50–53% is the realistic band; anything above 55% is a leak until proven
otherwise. **A null result is the expected outcome and is reported as a
finding, not a failure.** The design is pre-registered here so that the
conclusion cannot be reverse-engineered from the results.

---

## Pre-registered hypotheses

**Frozen 26 August 2026, before any model was run.** Committed ahead of every
result file so the ordering cannot be revised after the fact. Reordering these
once results are known would be HARKing, and an examiner will check the commit
history.

**H1 (primary).** No model family produces a statistically significant
improvement over the random-walk benchmark in *point* accuracy on 1-day-ahead
S&P 500 log returns, once leakage is purged, cross-sectional dependence is
respected, and multiple testing is corrected. *Expected outcome: fail to reject
the null.*

**H2 (primary).** Modern probabilistic models produce better-calibrated
predictive distributions than ARIMA/ETS, whose Gaussian intervals cannot
represent the fat tails and volatility clustering of daily equity returns.
*Expected outcome: reject the null - a genuine difference.*

**H3 (primary).** Model rankings are not stable across volatility and drawdown
regimes; aggregate metrics conceal regime-specific failure.

*Scope note, recorded 26 August 2026, before any model was run.* The test
period supports volatility stratification comfortably — 194 low-vol, 761
mid-vol and 715 high-vol sessions — but the drawdown axis is thin at the tail:
`bear` is 73 sessions, 4.4% of the test period. Confidence intervals in that
cell will be wide, and a model that looks best there may simply be the one that
happened to fit 73 days. Any `bear` result is reported with its interval and
treated as indicative, never as a ranking.

The decision **not** to merge `correction` and `bear` into one drawdown bucket
was taken on the same date, before any results existed. Merging would buy
statistical power at the cost of the distinction the hypothesis is about: a
10% pullback and a 20% bear market are different regimes for a forecasting
model, and collapsing them to widen a cell would answer an easier question than
the one registered. The thresholds in `features/regimes.py` are unchanged.

**H4 (primary).** The apparent advantage of pre-trained foundation models
narrows on the clean slice that post-dates their pre-training cutoff.
*Only Kronos has a documented cutoff (June 2024); for the others this is a
limitation, not a controlled comparison.*

*Scope narrowed 26 August 2026, before any model was run.* **H4 is tested on
aggregate performance only.** The clean slice is 247 sessions: 100% bull, 78%
mid-vol, and just 4 low-vol days. It cannot support regime stratification —
several cells are empty and the rest are too small to carry an interval — so
the contamination comparison is made on aggregate metrics across the slice.

The consequence is worth stating plainly: **H3 and H4 cannot be answered on the
same data.** H3 needs regime variation, which lives in the full test period;
H4 needs the post-cutoff window, which has almost none. Results for the two are
reported separately and are not combined into a claim about regime-specific
contamination effects, which this design cannot support.

Sessions by regime, from `data/regimes.parquet`:

| split | window | sessions | low | mid | high | bull | corr | bear |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2005-01-01 – 2016-12-31 | 3,021 | 1,007 | 1,006 | 1,007 | 2,199 | 314 | 257 |
| val | 2017-01-01 – 2019-12-31 | 754 | 442 | 231 | 81 | 725 | 29 | 0 |
| test | 2020-01-01 – 2026-08-25 | 1,670 | 194 | 761 | 715 | 1,337 | 260 | 73 |
| clean slice | 2025-09-01 – 2026-08-25 | 247 | 4 | 193 | 50 | 247 | 0 | 0 |

The clean slice is a subset of test, not a fifth disjoint period, so the rows do
not sum to the full sample. Train splits 1,007/1,006/1,007 by construction — the
terciles are fitted there — which is the check that they were never refitted
later. Train carries 251 sessions with no drawdown label, the warm-up for the
trailing 252-session peak.

**H5 (secondary, pre-registered).** On 21-day realised volatility, models
separate clearly and at least one beats Log-HAR - but the margin is small.
Brini (2026) finds only TTM beats Log-HAR at every horizon, by roughly 1.3-1.8%.

**Prediction registered in advance:** directional accuracy falls in the 50-53%
band. Anything above ~53% is treated as evidence of a defect in this pipeline
until proven otherwise, not as a finding.

### Falsification criteria

The study reports a positive result for H1 only if a model both (a) enters the
Model Confidence Set as a strict subset excluding the random walk, and (b) shows
a break-even transaction cost above 5 bps round trip. Either alone is
insufficient.

---

## What the design does *not* claim

- **Not a trading strategy.** The backtest exists to price the forecast in
  economic terms, not to propose a deployable system.
- **Not a contamination-free comparison for three of four foundation models.**
  Only Kronos publishes a cutoff. This is disclosed, not papered over.
- **Not free of survivorship bias.** See Limitations.
- **Not a claim of novelty in modelling.** The contribution is the protocol: the
  combination of purged CV, a post-cutoff clean slice, MCS *and* FDR control,
  cost-charged evaluation with a Deflated Sharpe, and regime stratification, on
  one panel, at zero cost.

---

## Design decisions, frozen

| Decision | Value | Why |
|---|---|---|
| Target | 1-day forward log return | Forecasting price *levels* yields R² ≈ 0.99 by copying yesterday's price. It measures nothing. |
| Lookback | 60 sessions | Standard for patch-based transformers; ARIMA selects its own order. |
| Train | 2005-01-01 → 2016-12-31 | |
| Validation | 2017-01-01 → 2019-12-31 | Model selection only. |
| Test | 2020-01-01 → 2026-08-25 | Contains a crash, a low-vol bull, a rate-driven bear and a recovery. |
| Clean slice | 2025-09-01 → 2026-08-25 | Post-dates every foundation-model checkpoint. |
| Purge | horizon = 1 session | A label at *t* consumes data through *t+h*. |
| Embargo | 5 sessions | Absorbs residual serial correlation. |
| Folds | 4 expanding windows | Annual refit. 78 daily refits would cost 20× the compute for no extra insight. |
| Scaling | per ticker, train statistics only | |
| Loss | `MQLoss`, q ∈ {0.1, 0.25, 0.5, 0.75, 0.9} | Median is the point forecast — one training run serves both. |

## The seventeen models

| Tier | Models | Scope | Source |
|---|---|---|---|
| 0 · Baseline | Naive, RandomWalkWithDrift | local | proposal |
| 1 · Statistical | AutoARIMA, AutoETS | local | proposal |
| 2a · Linear DL | **DLinear, NLinear** | global | added |
| 2b · Recurrent | LSTM, GRU | global | proposal |
| 2c · Transformer | PatchTST, TFT, TimesNet | global | proposal |
| 3 · Foundation | **Chronos-2, TimesFM-2.5, Moirai-2.0, Kronos-small** | zero-shot | added |
| V · Volatility arm | **HAR-RV, Log-HAR** | local | added |

**Why second-generation foundation models.** Chronos-Bolt and the 2024-era
checkpoints are one generation behind as of August 2026. Chronos-2, TimesFM-2.5
and Moirai-2.0 are the models the current literature ranks highest.

**Why HAR-RV is not optional.** It is the field-standard volatility benchmark.
Any volatility claim made without it is indefensible.

**Compute sacrifice, designated in advance.** If the Kaggle quota binds, the
annual refits of TimesNet are cut first — the trained deep models dominate GPU
cost, while every Tier 3 model is inference-only.

**Why DLinear was added.** The proposal cites Zeng et al. (2023), whose entire
argument is that a linear model outperforms transformers. Citing the paper while
omitting its model leaves an indefensible gap.

**Why foundation models were added.** They are the current frontier, and
zero-shot inference costs almost no GPU time. Kronos is pre-trained natively on
OHLCV candlesticks from 45 exchanges (AAAI 2026), making it the domain-specific
entry in the comparison.

**Local vs global asymmetry.** ARIMA and ETS fit one model per ticker; neural
models fit one across all fifty and therefore see fifty times the data. This is
intrinsic to the architectures and is discussed rather than hidden.

---

## Contamination protocol

Foundation models were pre-trained on public market data. Benchmarks show that
where evaluation data overlaps a pre-training corpus, the apparent advantage
inflates dramatically compared with genuinely unseen data. Kronos has almost
certainly seen these tickers over 2020–2025.

The protocol:

1. Report full-test-period results for all seventeen models, with foundation
   models flagged as potentially contaminated.
2. Repeat the entire comparison on the clean slice.
3. Report the gap between the two as a finding in its own right.

Handled this way, contamination becomes a contribution instead of a flaw.

---

## Evaluation

**Point** — RMSE, MAE, MASE, directional accuracy.
Never average raw RMSE across tickers at different price levels; MASE is
unit-free and aggregates honestly.

**Probabilistic** — CRPS (pinball loss averaged over the quantile grid), PICP,
Winkler interval score. PICP alone can be gamed by widening the band; the
interval score cannot.

**Significance** — two layers, because one is not enough here.

*Primary verdict:* the **Model Confidence Set** (Hansen, Lunde & Nason, 2011).
The 50 tickers share a market factor, so per-ticker loss differentials are
cross-sectionally dependent and BH-FDR across them is anti-conservative. The MCS
handles many models and data snooping jointly, and returns the set that cannot
be separated from the best. *A large surviving set is a finding, not a failure.*

*Supplementary:* Diebold–Mariano with the Harvey–Leybourne–Newbold correction,
then Benjamini–Hochberg FDR, for pairwise comparisons against the benchmark.

*Directional accuracy:* reported with a **date-clustered moving-block bootstrap**.
The naive binomial standard error treats 50 × 1,650 ticker-days as 82,500
independent trials and implies a detectable effect of ~0.2pp. The design effect
is roughly √(1 + 49ρ): at a plausible ρ ≈ 0.2 the true standard error is over
three times larger. Both figures are reported side by side, along with the
minimum detectable effect.

**Economic** — long/flat strategy at 0, 2, 5 and 10 bps round-trip. Reports
annualised turnover, **break-even cost in round-trip basis points**, and the
**Deflated Sharpe Ratio** (Bailey & López de Prado, 2014), which discounts the
observed Sharpe by the number of variants tried.

*"Profitable below 4 bps round-trip"* is a professional claim.
*"Returns 18% annually"* without costs is not.

---

## Leakage controls

Enforced in code and verified by `tests/test_leakage.py` — 28 adversarial tests
that corrupt future data and assert nothing upstream moves.

- Purged, embargoed, expanding-window CV on the session calendar — both the
  purge and the embargo are counted in *sessions*, by position in the calendar,
  never in calendar days, so a holiday cannot silently shorten the gap
- Trailing rolling windows only; no centred windows, no backward interpolation
- Macro series shifted by explicit publication lags (CPI = 45 days) *before*
  any forward fill
- Volatility regime thresholds fitted on training data, applied unchanged
- Scalers fitted on training data only
- Chronological splits throughout; shuffling is never invoked
- Volatility estimators use trailing windows only (Yang–Zhang, gap-aware)

**Known limitation — survivorship bias.** The universe is drawn from current
index membership, so firms that failed or were delisted are absent. Correcting
this needs point-in-time constituent data (CRSP/Compustat), which is licensed
and outside the project's zero-cost constraint. Stated in Limitations; expect it
in the viva.

**Known limitation — validation contains no drawdown regime.** The validation
period, 2017-2019, holds zero `bear` sessions and 29 `correction` sessions:
96.2% of it is `bull`. Every hyperparameter choice, and the single tuning budget
spent on PatchTST, is therefore made on a stretch of market history containing
no crisis. Model selection in this study is structurally blind to the drawdown
regime that H3 concerns.

This is not a fixable flaw so much as an honest one. It mirrors real practice —
hyperparameters cannot be chosen on a crisis that has not happened yet, and a
practitioner in December 2019 faced exactly this validation set. But it must not
be presented as regime-robust selection. Where a model performs well in the test
period's `correction` or `bear` cells, that is out-of-sample in a stronger sense
than the aggregate figures suggest; where it performs badly, selection never had
the chance to catch it.

**Known limitation — credit spread substitution.** The proposal specified
BAMLH0A0HYM2, the ICE BofA high-yield option-adjusted spread. FRED serves the
licensed ICE BofA family on a rolling window of roughly three years, so on the
build date the series began 2023-08-28 and its coverage across the frozen splits
was: train 0%, validation 0%, test 45%, clean slice 100%.

That gradient is not a gap, it is a confound. A column empty for every training
row that switches on partway through the test period and is complete precisely
on the clean slice would let vendor history masquerade as model skill in the one
comparison H4 exists to make. It was substituted with **BAA10Y** — Moody's Baa
corporate bond yield minus the 10-year Treasury — which is daily back to 1986 and
covers 100% of every split.

BAMLH0A0HYM2 is the better instrument on the merits: an option-adjusted
high-yield spread is a cleaner read on credit risk than a Baa-minus-Treasury
difference, which carries some duration exposure. The substitution trades
instrument quality for sample coverage, and it is coverage that the design
cannot do without. Recorded here because the swap is a deviation from the
approved proposal.

The defect was found on day one, before any model was run, by the per-split
macro coverage audit in `scripts/build_dataset.py`; it is reported on every
build and written to `data/quality_report.json`. `configs/data.yaml` is at
version 2 for this change. No split date, hypothesis or metric moved.

---

## Repository layout

```
configs/          frozen experiment specification (data, eval, models)
src/sp500bench/
  data/           universe, yfinance download, FRED with release lags
  features/       causal feature construction, regime labelling
  evaluation/     purged CV, metrics, DM + BH-FDR + DSR
  models/         uniform registry over all four tiers
  backtest/       cost-aware simulation
tests/            leakage (28) + numerical (13) + MCS/HAR/robust-SE (21)
                  + backtest cost accounting (18)
scripts/          pipeline entry points — not yet written
```

## Reproducing

The library and its test suite are complete and run today:

```bash
make setup        # installs the package and the [full] extra
make test         # 80 tests, ~5 s
```

**The pipeline scripts are not yet written.** `scripts/` is empty, so
`make data`, `make features`, `make baselines`, `make neural`,
`make foundation`, `make evaluate`, `make backtest` and `make all` will fail
until the five entry points they invoke exist. What is frozen and working so
far is the specification in `configs/`, the evaluation and leakage machinery in
`src/sp500bench/`, and the tests that hold them to it. The pre-registration in
this file is timestamped independently of any of that, by commit date.

Once the scripts land:

```bash
make all          # data → features → models → evaluation → backtest
```

Seeds fixed at 20260826; `torch.use_deterministic_algorithms(True)`.
Every run logs its config hash.

## Cost

Zero. yfinance and FRED are free; compute is Kaggle's free GPU tier
(~30 h/week). Budgeted usage: **≈17 GPU hours**, including reruns.

---

## References

Brini, A. (2026) 'Forecasting realized volatility with time series foundation
models: a comparison with econometric benchmarks', arXiv:2607.05291.
Corsi, F. (2009) 'A simple approximate long-memory model of realized
volatility', *Journal of Financial Econometrics*, 7(2).
Hansen, P.R., Lunde, A. and Nason, J.M. (2011) 'The model confidence set',
*Econometrica*, 79(2).
Kapoor, S. et al. (2024) 'REFORMS: consensus-based recommendations for
machine-learning-based science', *Science Advances*, 10(18).
Meyer, A. et al. (2026) 'Rethinking evaluation in the era of time series
foundation models', arXiv:2510.13654.
Yang, D. and Zhang, Q. (2000) 'Drift-independent volatility estimation based on
high, low, open, and close prices', *Journal of Business*, 73(3).
Bailey, D.H. and López de Prado, M. (2014) 'The deflated Sharpe ratio',
*Journal of Portfolio Management*, 40(5).
Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy',
*Journal of Business & Economic Statistics*, 13(3).
Harvey, D., Leybourne, S. and Newbold, P. (1997) 'Testing the equality of
prediction mean squared errors', *International Journal of Forecasting*, 13(2).
Hochreiter, S. and Schmidhuber, J. (1997) 'Long short-term memory',
*Neural Computation*, 9(8).
López de Prado, M. (2018) *Advances in Financial Machine Learning*. Wiley.
Shi, Y. et al. (2025) 'Kronos: a foundation model for the language of financial
markets', arXiv:2508.02739 (AAAI 2026).
Zeng, A. et al. (2023) 'Are transformers effective for time series
forecasting?', *AAAI*.
