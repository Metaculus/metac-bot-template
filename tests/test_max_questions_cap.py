from unittest.mock import MagicMock

import pytest

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_cap_applied_after_skip(monkeypatch):
    # Create 15 questions, first 5 already forecasted -> 10 unforecasted remain
    class Q:
        def __init__(self, done: bool):
            self.already_forecasted = done

    questions = [Q(True) for _ in range(5)] + [Q(False) for _ in range(10)]

    captured = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        captured.append(len(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    # Patch base class method to observe how many questions are forwarded after capping
    from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        },
        max_questions_per_run=10,
    )
    bot.skip_previously_forecasted_questions = True

    results = await bot.forecast_questions(questions)

    assert captured == [10]
    assert len(results) == 10


@pytest.mark.asyncio
async def test_cap_limits_to_10(monkeypatch):
    # 12 unforecasted questions -> expect 10 due to cap
    class Q:
        def __init__(self):
            self.already_forecasted = False

    questions = [Q() for _ in range(12)]

    captured = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        captured.append(len(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        }
    )  # default cap = 10
    bot.skip_previously_forecasted_questions = False

    results = await bot.forecast_questions(questions)

    assert captured == [10]
    assert len(results) == 10


@pytest.mark.asyncio
async def test_no_cap_when_below_limit(monkeypatch):
    # 7 unforecasted questions -> should pass through unchanged
    class Q:
        def __init__(self):
            self.already_forecasted = False

    questions = [Q() for _ in range(7)]

    captured = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        captured.append(len(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        }
    )  # default cap = 10

    results = await bot.forecast_questions(questions)

    assert captured == [7]
    assert len(results) == 7
