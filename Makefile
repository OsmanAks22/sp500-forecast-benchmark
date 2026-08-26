.PHONY: help setup data features baselines neural foundation evaluate backtest report test all clean

help:
	@echo "make setup       install dependencies"
	@echo "make data        download OHLCV + FRED, apply release lags"
	@echo "make features    build causal features and regime labels"
	@echo "make baselines   Naive, RWD, AutoARIMA, AutoETS  (CPU, overnight)"
	@echo "make neural      DLinear, NLinear, LSTM, GRU, PatchTST, TFT, TimesNet"
	@echo "make foundation  Chronos-Bolt, Kronos  (inference only)"
	@echo "make evaluate    metrics, DM tests, BH-FDR, regime stratification"
	@echo "make backtest    cost-aware strategy simulation + deflated Sharpe"
	@echo "make test        leakage and numerical test suites"
	@echo "make all         full pipeline, end to end"

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
	python scripts/evaluate.py --config configs/eval.yaml

backtest:
	python scripts/backtest.py --config configs/eval.yaml

test:
	pytest tests/ -v

all: data features baselines neural foundation evaluate backtest

clean:
	rm -rf data/interim data/processed forecasts/ .pytest_cache
