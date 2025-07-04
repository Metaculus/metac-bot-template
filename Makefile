.PHONY: conda_env install test run benchmark lint

conda_env:
	conda activate metaculus-bot

install:
	poetry install

test:
	PYTHONPATH=. poetry run pytest

run:
	poetry run python main.py

benchmark:
	poetry run python community_benchmark.py

lint:
	poetry run black .
	poetry run isort .
