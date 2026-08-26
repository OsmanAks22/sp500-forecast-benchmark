# Setup

## 1. Dependencies

```bash
pip install pytest pandas numpy scipy
```

Later, on Kaggle, add: `statsforecast neuralforecast yfinance fredapi pyarrow`

## 2. Verify

```bash
PYTHONPATH=src pytest tests/ -q
```

Expected: `59 passed`

## 3. Freeze the pre-registration

```bash
git init
git add .
git commit -m "Initial: frozen spec, evaluation framework, pre-registered hypotheses"
```

This commit timestamps the pre-registered hypotheses in README.md. It must
precede any results file — the commit history is the evidence that the
hypothesis ordering was not chosen after seeing the outcome.

## 4. Optional: install as a package

```bash
pip install -e .
```

Removes the need for the `PYTHONPATH=src` prefix.

## Directory map

```
configs/          frozen experiment specification
src/sp500bench/   the package — do not flatten this tree, imports depend on it
tests/            59 tests: leakage (25), numerical (13), MCS/HAR/robust-SE (21)
scripts/          pipeline entry points (to be written)
notebooks/        exploration only
```
