.PHONY: conda_env install test run benchmark lint

conda_env:
	conda activate metaculus-bot

install:
	poetry install

test:
	conda run -n metaculus-bot PYTHONPATH=. poetry run pytest

run:
	conda activate metaculus-bot && poetry run python main.py

benchmark:
	conda activate metaculus-bot && poetry run python community_benchmark.py

lint:
	conda activate metaculus-bot && poetry run black --check .
	conda activate metaculus-bot && poetry run isort --check.

format:
	conda activate metaculus-bot && poetry run black .
	conda activate metaculus-bot && poetry run isort .

lock:
	poetry lock
