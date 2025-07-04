from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_binary_question_aggregation_is_mean():
    # Arrange
    bot = TemplateForecaster(llms={"default": "mock_model"})
    mock_binary_question = MagicMock(spec=BinaryQuestion)
    predictions = [0.1, 0.2, 0.3]

    # Act
    aggregated_prediction = await bot._aggregate_predictions(
        predictions, mock_binary_question
    )

    # Assert
    assert aggregated_prediction == 0.2

@pytest.mark.asyncio
async def test_binary_question_aggregation_is_mean_rounded():
    # Arrange
    bot = TemplateForecaster(llms={"default": "mock_model"})
    mock_binary_question = MagicMock(spec=BinaryQuestion)
    predictions = [0.1234, 0.2468, 0.3579]

    # Act
    aggregated_prediction = await bot._aggregate_predictions(
        predictions, mock_binary_question
    )

    # Assert
    expected_mean = (0.1234 + 0.2468 + 0.3579) / 3
    expected_rounded_mean = round(expected_mean, 3)
    assert aggregated_prediction == expected_rounded_mean
