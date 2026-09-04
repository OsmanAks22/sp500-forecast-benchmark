# Standing rules

This is a **pre-registered study**, not an ordinary codebase. The value of the
result depends on the protocol having been fixed before the results were known,
and an examiner will check the commit history to confirm it. Most of the rules
below exist because breaking them would not cause a test to fail — it would
quietly turn a defensible finding into an indefensible one.

## Frozen — never edit

- **Hypotheses H1–H5 in `README.md`** and **split dates in `configs/data.yaml`**
  are frozen. Never edit them. If a change seems necessary, **stop and ask.**
- `Edit`/`Write` on `README.md` and `configs/` are denied in
  `.claude/settings.json`. That guard is deliberate. Do not work around it by
  shelling out to `sed`, `python`, `cat >`, or a heredoc.
- Scope *notes* may be **added** under a hypothesis when something material is
  discovered (see the existing notes under H3, H4 and H5). Additions are how
  this repository records a narrowing. Rewording an existing hypothesis is not.

## Roster

- The roster is **fixed at 20 specs**. Do not add models.
- If a model seems worth adding, **say so and stop.** Do not add it and explain
  afterwards.
- `src/sp500bench/models/registry.py` is the executable source of truth;
  `configs/models.yaml` must agree with it exactly. Both must be updated
  together, and the agreement is checkable — name, tier, library, scope,
  checkpoint, entry point, pre-training cutoff, target.

## Tests

- **Never modify `tests/` to make a failing test pass.** Determine whether the
  *code* or the *test* is wrong, say which and why, then fix that one.
- Adding a new test for new behaviour is fine and expected. Weakening an
  existing assertion to get green is not.
- If it is genuinely ambiguous which side is wrong, **stop and ask.**

## Figures

- **Verify every figure against the data before it enters `README.md`.**
  Recompute it from `data/*.parquet`, `results/*.parquet` or the code — do not
  quote from memory, from an earlier message, or from a summary.
- Both the user and the assistant have stated wrong numbers from memory more
  than once in this project. The bear-cell distribution and the ratio behind the
  H1 benchmark were both wrong on first statement and caught only by checking.
- When a stated figure and the data disagree, **report the discrepancy** and
  record the computed value. Do not quietly adopt either.

## Commits

- Every commit message states **what changed** and **that H1–H5 and the split
  dates are unchanged** — or, if a scope note was added, exactly what was added
  and that no hypothesis wording moved.
- Explain *why*, not just *what*. The commit history is part of the evidence
  that the protocol was followed.

## Layout

| path | role |
|---|---|
| `configs/` | the frozen specification — data, eval, models |
| `src/sp500bench/config.py` | loads and cross-checks the two config halves |
| `src/sp500bench/models/registry.py` | executable roster, source of truth |
| `scripts/run_experiment.py` | runs models, writes `forecasts/` |
| `scripts/evaluate.py` | scores, writes `metrics.parquet` + `losses.parquet` |
| `scripts/analyse.py` | DM tests, BH-FDR, Model Confidence Set |
| `tests/` | leakage, numerical, MCS/HAR, backtest, config gate, benchmark |

Metrics are computed in exactly one place (`evaluate.py`). Everything
downstream reads the parquet rather than recomputing, so a table in the write-up
cannot disagree with the test that supports it.

## Environment

- conda env **`sp500`**, Python 3.11. Not a virtualenv.
- **GPU work goes to Kaggle** — see `notebooks/kaggle_smoke_test.ipynb`. There is
  no local GPU.
- Detachable sessions use **`screen`**, not tmux. Homebrew is not installed on
  this machine, so `brew install` is not available.
- `statsforecast` import costs ~25s on a cold bytecode cache. Warm-up is timed
  separately from fold cost; do not fold it back in.

## Inference conventions

- All DM tests and the MCS run on **daily cross-sectional mean squared error** —
  one row per session — never the stacked ticker-day panel. Fifty constituents
  share a market factor; stacking shrinks the standard error by roughly
  `sqrt(50)` and manufactures significance.
- Stacked p-values are reported as `dm_p_stacked`, **diagnostic only, never a
  result.**
- The H1 reference is **`ZeroReturn`**, the random walk on prices. Not
  `LastReturn`. See the note under H1.
- On a return target MASE's no-skill line is `1/sqrt(2(1-rho))` ≈ 0.707, not 1.0.
  Report `scaled_error_mad` alongside it, whose no-skill line is 1.0.

## STOP AND ASK

Do not decide these alone. Say what you would do and why, then wait:

- **Roster changes** — adding, removing or substituting any model.
- **Hypothesis scope** — narrowing, widening or reinterpreting H1–H5.
- **Regime definitions** — volatility terciles, drawdown thresholds, the
  `correction`/`bear` split.
- **Data source substitutions** — a different ticker, index proxy, or FRED
  series.
- **Anything that changes what the thesis claims.**
- **Any test failure where the correct fix is ambiguous.**

A wrong answer here is recoverable. A silent decision here is not, because it
will not look like a decision when the write-up is read.
