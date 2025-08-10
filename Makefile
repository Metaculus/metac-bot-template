.PHONY: conda_env install test run benchmark lint

conda_env:
	conda activate metaculus-bot

install:
	poetry install

lock:
	poetry lock

lint:
	conda run -n metaculus-bot && poetry run black --check .
	conda run -n metaculus-bot && poetry run isort --check.

format:
	conda run -n metaculus-bot && poetry run black .
	conda run -n metaculus-bot && poetry run isort .

test:
	conda run -n metaculus-bot PYTHONPATH=. poetry run pytest

run:
	conda run -n metaculus-bot && poetry run python main.py

# Warning: this will fire off requests to OpenRouter and cost money.
benchmark_run_smoke_test:
	export METACULUS_TOKEN=$$(cat ~/.keys/metaculus_token) && export OPENAI_API_KEY=$$(cat ~/.keys/openai_api_key) && export OPENROUTER_API_KEY=$$(cat ~/.keys/openrouter_key) && conda run -n metaculus-bot && poetry run python community_benchmark.py --mode run --num-questions 2

benchmark_run_mini:
	export METACULUS_TOKEN=$$(cat ~/.keys/metaculus_token) && export OPENAI_API_KEY=$$(cat ~/.keys/openai_api_key) && export OPENROUTER_API_KEY=$$(cat ~/.keys/openrouter_key) && conda run -n metaculus-bot && poetry run python community_benchmark.py --mode run --num-questions 10

benchmark_run_small:
	export METACULUS_TOKEN=$$(cat ~/.keys/metaculus_token) && export OPENAI_API_KEY=$$(cat ~/.keys/openai_api_key) && export OPENROUTER_API_KEY=$$(cat ~/.keys/openrouter_key) && conda run -n metaculus-bot && poetry run python community_benchmark.py --mode run --num-questions 30

benchmark_display:
	conda run -n metaculus-bot && poetry run python community_benchmark.py --mode display
