.PHONY: help setup data features baselines neural foundation evaluate backtest report test all clean clean-all

help:
	@echo "make setup       install dependencies"
	@echo "make data        download OHLCV + FRED, apply release lags"
	@echo "make features    build causal features and regime labels"
	@echo "make baselines   Naive, RWD, AutoARIMA, AutoETS  (CPU, overnight)"
	@echo "make neural      DLinear, NLinear, LSTM, GRU, PatchTST, TFT, TimesNet"
	@echo "make foundation  Chronos-2, TimesFM-2.5, Moirai-2.0, Kronos-small, TTM (inference only)"
	@echo "make evaluate    metrics, DM tests, BH-FDR, regime stratification"
	@echo "make backtest    cost-aware strategy simulation + deflated Sharpe"
	@echo "make test        leakage and numerical test suites"
	@echo "make all         full pipeline, end to end"
	@echo "make clean       remove derived data; keeps the raw download cache"
	@echo "make clean-all   also remove data/raw - forces a full re-download"

setup:
	pip install -e ".[full]"

data:
	python scripts/build_dataset.py --config configs/data.yaml

features:
	python scripts/build_features.py --config configs/data.yaml

baselines:
	python scripts/run_experiment.py --tier baseline --tier statistical

neural:
	python scripts/run_experiment.py --tier linear --tier recurrent --tier transformer

foundation:
	python scripts/run_experiment.py --tier foundation

evaluate:
	python scripts/evaluate.py

backtest:
	python scripts/backtest.py

test:
	pytest tests/ -v

all: data features baselines neural foundation evaluate backtest

# Everything derived. data/raw is deliberately spared so a rebuild does not
# re-hit yfinance and FRED; use clean-all for that.
clean:
	rm -rf data/interim data/processed forecasts/ .pytest_cache
	rm -f data/features.parquet data/regimes.parquet data/scalers.parquet
	rm -f data/quality_report.json

# Additionally drops the raw download cache. The next `make data` therefore
# re-downloads all 50 tickers, the index proxy and every FRED series from
# scratch - minutes, not seconds, and subject to yfinance rate limiting.
clean-all: clean
	rm -rf data/raw
