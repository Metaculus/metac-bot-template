from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion

from main import TemplateForecaster


@pytest.mark.asyncio
async def test_bot_binary_aggregate_rounding():
    bot = TemplateForecaster(llms={"default": "mock"})
    preds = [0.3331, 0.3332]
    q = MagicMock(spec=BinaryQuestion)
    agg = await bot._aggregate_predictions(preds, q)
    assert agg == 0.333

