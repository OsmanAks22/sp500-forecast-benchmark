# Three Generations of Forecasting Models on S&P 500 Daily Returns

A leakage-controlled, regime-stratified benchmark of statistical, supervised
deep-learning and pre-trained foundation models, evaluated on 50 S&P 500
constituents under purged walk-forward cross-validation.

MSc Data Science and Analytics, University of Westminster (7COSC012W)
Supervisor: Dr Panagiotis Chountas

---

## Final results (dissertation submitted September 2026)

- **Scope:** 18 models scored (15 return models, 3 volatility models) on 50 S&P 500 stocks, 2005–2026 (272,200 stock-day observations), under purged walk-forward cross-validation with an embargo.
- **Point forecasts:** no model significantly beats the zero-return forecast; the Model Confidence Set retains ZeroReturn (p = 0.114).
- **Probabilistic forecasts:** zero-shot Chronos-2 has the lowest CRPS, 4.2% below the zero-return baseline; six other models also score below the baseline. In high-volatility periods its 80% intervals cover 77% of outcomes, against 68% for the baseline.
- **Reproducibility:** 271 automated tests; every reported metric re-computes from the saved forecasts.

The sections below are the pre-registration as frozen on 26 August 2026 (commit `d35c812`). *Results to date* is the interim position recorded on 2 September, before the GPU tiers were run; the dissertation reports the final results above.

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

*Benchmark identified 2 September 2026, after Tiers 0 and 1 were run and before
any hypothesis test.* **The random-walk benchmark H1 names is `ZeroReturn`.** A
random walk in *prices* implies an expected return of zero, so the forecast is
zero at every session.

This had been implemented as `statsforecast`'s `Naive`, which on a return series
predicts `y[t+1] = y[t]` — a different model. The target is already a first
difference of log prices, so predicting the previous return differences it a
second time and inflates the error by `sqrt(2(1-rho))`, where `rho` is the lag-1
autocorrelation of returns. On this panel `rho = -0.059`, so the factor is 1.455
— and the measured ratio is 1.455. The wrong benchmark made AutoARIMA look 31%
better than "the random walk"; against the correct one it is 0.5% *worse*.

The old model is kept as `LastReturn`, named for what it does, and reported as a
diagnostic of what a spurious lag costs. It is never the H1 reference.

`ZeroReturn`'s directional accuracy is reported as n/a rather than 0%. Its
forecast is identically zero, `sign(0)` matches nothing, and the convention that
drops flat *outcomes* does not drop flat *forecasts* — so the mechanical 0% is
an artefact of the statistic, not a directional failure. A random walk makes no
directional call; that is the point of it.

*Reading the scaled errors.* On a return target, MASE's no-skill line is **not
1.0**. Its denominator is the in-sample `mean |y_t - y_{t-1}|`, which differences
an already-differenced series, so a zero forecast scores about
`1/sqrt(2(1-rho))` — **0.707 at `rho = 0`**, 0.687 at the `rho` measured here —
before any train-versus-test volatility difference is taken into account. A
model at 0.77 is therefore *not* beating the benchmark by 23%. A second scaled
error is reported alongside it, dividing by the in-sample `mean |y_t|` instead;
its no-skill line is 1.0 and it can be read directly. Both appear in
`results/metrics.parquet` as `mase` and `scaled_error_mad`. In either case the
operative comparison is the `ZeroReturn` row itself, which absorbs the
volatility difference between the training and test windows.

**H2 (primary).** Modern probabilistic models produce better-calibrated
predictive distributions than ARIMA/ETS, whose Gaussian intervals cannot
represent the fat tails and volatility clustering of daily equity returns.
*Expected outcome: reject the null - a genuine difference.*

*Benchmark recorded 1 September 2026, after the Tier 0 baselines were run and
before any competing model.* H2 is a claim about calibration, so the number it
is measured against is fixed here rather than chosen later. Both baselines carry
an empirical-residual predictive distribution fitted on each fold's training
window, and both undercover the nominal 80% band:

| window | PICP80 | CRPS |
|---|---:|---:|
| fold 0 · 2020 | 62.4% | 0.01665 |
| fold 1 · 2021 | 81.7% | 0.00770 |
| fold 2 · 2022 | 69.0% | 0.01032 |
| fold 3 · 2023–26 | 80.0% | 0.00836 |
| **pooled test period** | **75.9%** | **0.00981** |

Naive and RandomWalkWithDrift agree to three decimals on every cell. The pooled
shortfall — 75.9% against 80% nominal — is the benchmark; the 2020 figure of
62.4% is the harder one, and the more informative, because it is where a band
fitted on calmer training data fails. A model claiming better calibration under
H2 has to beat both, and beating the pooled figure while missing 2020 is a
result about quiet markets, not about calibration.

**H3 (primary).** Model rankings are not stable across volatility and drawdown
regimes; aggregate metrics conceal regime-specific failure.

*Scope note, recorded 26 August 2026, before any model was run.* The test
period supports volatility stratification comfortably — 194 low-vol, 761
mid-vol and 715 high-vol sessions — but the drawdown axis is thin at the tail:
`bear` is 73 sessions, 4.4% of the test period. Confidence intervals in that
cell will be wide, and a model that looks best there may simply be the one that
happened to fit 73 days. Any `bear` result is reported with its interval and
treated as indicative, never as a ranking.

*Scope narrowed 1 September 2026, after the Tier 0 baselines were run and before
any competing model. The per-fold counts below were checked against the regime
panel before they entered this document.* **The directional axis of H3 is
reported descriptively only. No hypothesis test is run on the `bear` cell.**

The 73 bear sessions are two episodes, not a sample of bear regimes:

| fold | window | bear sessions | episode |
|---|---|---:|---|
| 0 | 2020 | 18 | COVID crash, 12 Mar – 7 Apr 2020 |
| 1 | 2021 | 0 | — |
| 2 | 2022 | 54 | rate-driven bear, 13 Jun – 28 Dec 2022 |
| 3 | 2023–26 | 1 | 3 Jan 2023, the tail of the 2022 episode |

The two episodes are not two draws from one population. March 2020 was a fast
liquidity shock — a near-vertical drawdown over four weeks, with volatility
arriving faster than any conditioning variable could track it. June to December
2022 was a slow rate repricing: a six-month grind at moderate daily volatility,
driven by discount-rate expectations rather than by a scramble for liquidity.
A forecasting model can plausibly do well in one and badly in the other for
reasons that have nothing to do with a common factor called `bear`.

The consequence for inference is the sample size. Within an episode, consecutive
sessions are highly dependent — that is what makes it an episode — so the
effective sample is closer to **2** than to 73. The date-clustered bootstrap
does not repair this: block resampling over dates assumes many exchangeable
blocks, and two contiguous runs supply about two. An interval computed there
would be a number, not evidence.

So the narrowing rests on this, and not on a claim that replication is
impossible: it is available, in one comparison. **Any inferential claim about
the `bear` cell would generalise from n = 2 episodes**, and two episodes of
structurally different kinds. Results for the cell are reported with their
counts and read descriptively.

**H3's inferential claim therefore rests on the volatility axis** — and within
it, on the `high_vol` versus `mid_vol` contrast, which is the only pairing
genuinely spread across the design:

| fold | low_vol | mid_vol | high_vol |
|---|---:|---:|---:|
| 0 · 2020 | 18 | 18 | 217 |
| 1 · 2021 | 0 | 157 | 95 |
| 2 · 2022 | 0 | 21 | 230 |
| 3 · 2023–26 | 176 | 565 | 173 |

`high_vol` appears in all four folds (217/95/230/173), and `mid_vol` in all
four, so a ranking that reverses between them can be checked for replication
fold by fold. `low_vol` cannot carry that weight: 194 sessions sounds
comfortable, but it is **absent from folds 1 and 2 entirely and 91%
concentrated in fold 3**, which makes it closer to a single late-period episode
than to a regime observed throughout. Low-vol results are reported with their
intervals and carry the same caveat as the drawdown axis, in weaker form.

The decision **not** to merge `correction` and `bear` into one drawdown bucket
was taken on the same date, before any results existed. Merging would buy
statistical power at the cost of the distinction the hypothesis is about: a
10% pullback and a 20% bear market are different regimes for a forecasting
model, and collapsing them to widen a cell would answer an easier question than
the one registered. The thresholds in `features/regimes.py` are unchanged.

**H4 (primary).** The apparent advantage of pre-trained foundation models
narrows on the clean slice that post-dates their pre-training cutoff.
*Only Kronos has a documented cutoff (June 2024); for the other four this is a
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

*Target defined 2 September 2026, after the volatility arm was first run and
before any hypothesis test.* **"21-day realised volatility" means the Yang-Zhang
estimate over sessions `t+1` to `t+21`, forecast from information through `t`.**
The forward window, not a trailing one.

The distinction decides the hypothesis. A trailing 21-session window overlaps its
own predecessor in 20 of 21 sessions, so `rv_d = y(t-1)` shares 20/21 of its
sessions with `y(t)` and the HAR cascade regresses on lags of the target itself.
That is smoothing, not forecasting. Measured that way the arm reported GJR-GARCH
at 4.2x Log-HAR's RMSE — not because it forecasts volatility badly (its
predictions correlate 0.86 with the target and sit within 3% of its mean level)
but because it does not use the target's own lag, and under a trailing target
nothing else matters. H5 asks whether models separate on volatility; a trailing
target answers with the target's construction instead. On the forward target the
same three models sit at 1.00, 1.01 and 1.03.

The purge follows the target. A forward label consumes data 21 sessions ahead,
so this arm is purged at `horizon: 21` — recorded in `configs/eval.yaml` v3 as
`volatility_arm.horizon` — against the return arm's `horizon: 1`. Its training
windows therefore end 26 sessions before each test window rather than 6.
`assert_no_overlap` receives the arm's own horizon; given the return arm's, it
would wave through 15 sessions of leakage and report a clean fold. The folds,
the embargo and the quantile grid are shared; only the purge width differs.

**Prediction registered in advance:** directional accuracy falls in the 50-53%
band. Anything above ~53% is treated as evidence of a defect in this pipeline
until proven otherwise, not as a finding.

### Falsification criteria

The study reports a positive result for H1 only if a model both (a) enters the
Model Confidence Set as a strict subset excluding the random walk, and (b) shows
a break-even transaction cost above 5 bps round trip. Either alone is
insufficient.

---

## Results to date

**Recorded 2 September 2026, on eight models: Tier 0, Tier 1 and the volatility
arm. No GPU tier has been run.** Twelve models remain — the linear, recurrent and
transformer tiers, and the five foundation models — and every number below is
subject to revision when they land. It is recorded now so that the position
before those runs is timestamped rather than reconstructed afterwards.

Reproduce with `make evaluate` then `python scripts/analyse.py`; all figures come
from `results/inference.parquet`.

### H1 — the falsification criterion is not met

The pooled return-arm Model Confidence Set is **{ZeroReturn, AutoETS}** at
α = 0.10, over 1,669 sessions.

| model | MCS p | in set |
|---|---:|---|
| AutoETS | 0.691 | yes |
| **ZeroReturn** | 0.691 | **yes** |
| AutoARIMA | 0.0004 | no |
| LastReturn | 0.0000 | no |
| RandomWalkWithDrift | 0.0000 | no |

The random walk survives, so no model enters the set as a strict subset
excluding it. **Criterion (a) fails, and criterion (b) — break-even above 5 bps
round trip — therefore does not come into play.** The same holds in every
volatility regime cell: `high_vol`, `mid_vol` and `low_vol` all return
{ZeroReturn, AutoETS}.

- **AutoETS is statistically indistinguishable from a constant zero forecast.**
  Diebold-Mariano on daily cross-sectional MSE: p = 0.718, loss ratio 0.9997.
- **AutoARIMA is eliminated for being worse**, at a loss ratio of 1.009. Its
  pooled DM does not survive BH on its own (p = 0.065, BH-adjusted 0.087), but
  the MCS eliminates it at p = 0.0004, and it is significantly worse in folds 2
  and 3 after BH (adjusted p = 0.0040 and 0.0029). The two procedures agree on
  the direction and the MCS is the one the criterion is written against.
- LastReturn and RandomWalkWithDrift sit at a loss ratio of 2.12 — the
  double-differencing penalty described under H1, not a modelling result.

This is the outcome H1 registered in advance.

### H3 — first inferential result, on the volatility axis

The volatility arm separates by regime, and it separates in the direction that
is easy to get backwards:

| regime | sessions | MCS |
|---|---:|---|
| `high_vol` | 714 | **all three survive** — GJR-GARCH, HAR-RV, Log-HAR, p = 0.706 |
| `mid_vol` | 741 | **Log-HAR only** — GJR-GARCH and HAR-RV eliminated at p = 0.0002 |

**The separation appears in calm markets, not turbulent ones.** In high
volatility the data cannot distinguish the three; in mid volatility Log-HAR is
alone in the set. `low_vol` reproduces the `mid_vol` pattern (Log-HAR only,
others at p = 0.0006 and 0.0016) but is reported rather than treated as
inferential, per the H3 scope note. The `bear` cell is descriptive only, as
registered.

### A methodological finding: the cost of ignoring cross-sectional dependence

Every test above runs on the **daily cross-sectional mean squared error** — one
row per session, averaged across the 50 tickers. Run instead on the stacked
83,450-row ticker-day panel, the same comparisons read:

| comparison | p, daily (used) | p, stacked (diagnostic) |
|---|---:|---:|
| AutoARIMA vs ZeroReturn, pooled | 0.065 | 1.5 × 10⁻⁹ |
| HAR-RV vs Log-HAR, pooled | 0.352 | 4.5 × 10⁻⁶ |
| GJR-GARCH vs Log-HAR, pooled | 0.234 | 1.3 × 10⁻⁴ |

Both of the first two would have been reported as decisive by a study that
stacks. Neither is. Fifty constituents share a market factor, so stacking treats
50 correlated observations as 50 independent ones and shrinks the standard error
by roughly `sqrt(50)`. This is the same inflation the directional-accuracy
bootstrap measures at about 2× *after* collapsing to days.

**The stacked column is a diagnostic and is never a result.** It exists to size
the error, and no decision anywhere in this study uses it. It is carried in
`results/inference.parquet` as `dm_p_stacked`, named so it cannot be mistaken
for the figure that was tested.

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
| Loss | `MQLoss`, q ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99} | Median is the point forecast — one training run serves both. Identical to the evaluation grid, and checked at load time. |

## The twenty models

| Tier | Models | Scope | Source |
|---|---|---|---|
| 0 · Baseline | **ZeroReturn** (H1 reference), **LastReturn**, RandomWalkWithDrift | local | proposal + added |
| 1 · Statistical | AutoARIMA, AutoETS | local | proposal |
| 2a · Linear DL | **DLinear, NLinear** | global | added |
| 2b · Recurrent | LSTM, GRU | global | proposal |
| 2c · Transformer | PatchTST, TFT, TimesNet | global | proposal |
| 3 · Foundation | **Chronos-2, TimesFM-2.5, Moirai-2.0, Kronos-small, TTM** | zero-shot | added |
| V · Volatility arm | **HAR-RV, Log-HAR, GJR-GARCH** | local | added |

**Why second-generation foundation models.** Chronos-Bolt and the 2024-era
checkpoints are one generation behind as of August 2026. Chronos-2, TimesFM-2.5
and Moirai-2.0 are the models the current literature ranks highest.

**Why HAR-RV is not optional.** It is the field-standard volatility benchmark.
Any volatility claim made without it is indefensible.

**Why GJR-GARCH was added.** The HAR cascade regresses on unsigned realised
variance over daily, weekly and monthly windows, so it is structurally blind to
the sign of the shock and cannot represent the leverage effect. GJR-GARCH(1,1)
with skew-t innovations can. Without it the volatility arm would test HAR
against models that share HAR's blind spot.

**Why TTM was added.** Brini (2026) reports TinyTimeMixers as the only TSFM to
beat Log-HAR at every horizon, by 1.3-1.8%. That is the single strongest claim
in the literature this study is positioned against, and it is cited in H5. TTM
is included as a direct replication probe: if the margin does not reproduce on
this panel, the discrepancy is a finding in its own right.

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

1. Report full-test-period results for all twenty models, with foundation
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
  config.py       loads and cross-checks the frozen specification
  models/         uniform registry over all four tiers; Tier 0 baselines and
                  Tier 1 statistical share one statsforecast driver
  backtest/       cost-aware simulation
tests/            leakage (28) + numerical (17) + MCS/HAR/robust-SE (21)
                  + backtest cost accounting (18) + config gate (10)
                  + H1 benchmark identity (5)
scripts/          pipeline entry points — build_dataset, build_features,
                  run_experiment and evaluate are written; run_experiment
                  executes Tiers 0 and 1 and reports every other model as
                  pending
results/          metrics.parquet (model x fold x regime cells), losses.parquet
                  (per-observation losses for DM and the MCS), timings.json
```

## Reproducing

```bash
make setup        # installs the package and the [full] extra
make test         # 271 tests
```

The full pipeline, end to end:

```bash
make data         # download OHLCV + FRED, apply release lags
make features     # causal features and regime labels
make baselines    # statistical tiers (CPU)
make neural       # linear, recurrent and transformer tiers (GPU)
make foundation   # foundation models, inference only (GPU)
make evaluate     # metrics, DM tests, BH-FDR, regime stratification
make backtest     # cost-aware strategy simulation
python scripts/analyse.py
```

`data/`, `forecasts/` and `results/` are not tracked in git. The GPU tiers were run on Kaggle's free tier.

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

