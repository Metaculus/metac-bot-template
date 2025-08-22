.PHONY: conda_env install test run benchmark lint analyze_correlations analyze_correlations_latest

# Optionally load secrets from ~/.keys into env for commands that hit APIs.
# These are best-effort; if files are missing, values stay empty.
ENV_FROM_KEYS=export METACULUS_TOKEN=$$(cat $$HOME/.keys/metaculus_token 2>/dev/null); \
              export OPENAI_API_KEY=$$(cat $$HOME/.keys/openai_api_key 2>/dev/null); \
              export OPENROUTER_API_KEY=$$(cat $$HOME/.keys/openrouter_key 2>/dev/null); \
              export OAI_ANTH_OPENROUTER_KEY=$$(cat $$HOME/.keys/oai_anth_openrouter_key 2>/dev/null); \
              export ASKNEWS_CLIENT_ID=$$(cat $$HOME/.keys/asknews_client_id 2>/dev/null); \
              export ASKNEWS_SECRET=$$(cat $$HOME/.keys/asknews_secret 2>/dev/null); \
              export ASKNEWS_MAX_CONCURRENCY=$$(cat $$HOME/.keys/asknews_max_concurrency 2>/dev/null); \
              export ASKNEWS_MAX_RPS=$$(cat $$HOME/.keys/asknews_max_rps 2>/dev/null);

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
	conda run -n metaculus-bot PYTHONPATH=. poetry run pytest

run:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python main.py

# Warning: this will fire off requests to OpenRouter and cost (a very small amount) of money.
benchmark_run_smoke_test:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python community_benchmark.py --mode run --num-questions 1

benchmark_run_smoke_test_all_q_types:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python community_benchmark.py --mode custom --num-questions 4 --mixed

benchmark_run_binary_only:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python community_benchmark.py --mode run --num-questions 30

benchmark_run_small:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python community_benchmark.py --mode custom --num-questions 12 --mixed

benchmark_run_medium:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python community_benchmark.py --mode custom --num-questions 32 --mixed

benchmark_run_large:
	$(ENV_FROM_KEYS) conda run -n metaculus-bot poetry run python community_benchmark.py --mode custom --num-questions 100 --mixed

benchmark_display:
	conda run -n metaculus-bot poetry run python community_benchmark.py --mode display

analyze_correlations:
	conda run -n metaculus-bot poetry run python analyze_correlations.py $(if $(FILE),$(FILE),benchmarks/)

analyze_correlations_latest:
	conda run -n metaculus-bot poetry run python analyze_correlations.py $$(ls -t benchmarks/benchmarks_*.jsonl | head -1)
