.PHONY: conda_env install test run benchmark lint analyze_correlations analyze_correlations_latest

# Stream logs live from recipes; avoid per-target buffering
MAKEFLAGS += --output-sync=none

# Absolute Python in conda env (use tilde to avoid hardcoding username)
PY_ABS := ~/miniconda3/envs/metaculus-bot/bin/python

# for reference, won't actually persist to the shell
# conda_env:
# 	conda activate metaculus-bot

install:
	conda run -n metaculus-bot poetry install

lock:
	conda run -n metaculus-bot poetry lock

lint:
	conda run -n metaculus-bot poetry run black --check .
	conda run -n metaculus-bot poetry run isort --check .

format:
	conda run -n metaculus-bot poetry run black .
	conda run -n metaculus-bot poetry run isort .

test:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u -m pytest" /dev/null

run:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u main.py" /dev/null

# Warning: this will fire off requests to OpenRouter and cost (a very small amount) of money.
benchmark_run_smoke_test_binary:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode run --num-questions 1" /dev/null

benchmark_run_smoke_test:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode custom --num-questions 4 --mixed" /dev/null

benchmark_run_binary_only:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode run --num-questions 30" /dev/null

benchmark_run_small:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode custom --num-questions 12 --mixed" /dev/null

benchmark_run_medium:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode custom --num-questions 32 --mixed" /dev/null

benchmark_run_large:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode custom --num-questions 100 --mixed" /dev/null

benchmark_display:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u community_benchmark.py --mode display" /dev/null

analyze_correlations:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u analyze_correlations.py $(if $(FILE),$(FILE),benchmarks/)" /dev/null

analyze_correlations_latest:
	PYTHONUNBUFFERED=1 PYTHONPATH=. stdbuf -oL -eL script -q -c "$(PY_ABS) -u analyze_correlations.py $$(ls -t benchmarks/benchmarks_*.jsonl | head -1)" /dev/null
