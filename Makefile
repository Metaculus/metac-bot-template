.PHONY: conda_env install test run benchmark lint

conda_env:
	conda activate metaculus-bot

install:
	conda activate metaculus-bot && poetry install

test:
	conda activate metaculus-bot && PYTHONPATH=. poetry run pytest tests/test_aggregation.p

run:
	conda activate metaculus-bot && poetry run python main.py

benchmark:
	conda activate metaculus-bot && poetry run python community_benchmark.py

lint:
	conda activate metaculus-bot && poetry run black .
	conda activate metaculus-bot && poetry run isort .
