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

benchmark:
	conda run -n metaculus-bot && poetry run python community_benchmark.py --mode display

