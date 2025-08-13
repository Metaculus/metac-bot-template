#!/usr/bin/env python3
"""
Simple test to verify research caching functionality works as expected.
This test demonstrates that:
1. Research cache is used when enabled in benchmarking mode
2. Research cache is ignored when not in benchmarking mode
3. Cache sharing works between multiple bot instances
"""

import asyncio

# Mock dependencies that might not be available
import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules["forecasting_tools"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["metaculus_bot.constants"] = MagicMock()
sys.modules["metaculus_bot.numeric_utils"] = MagicMock()
sys.modules["metaculus_bot.research_providers"] = MagicMock()
sys.modules["metaculus_bot.utils.logging_utils"] = MagicMock()

# Mock the imports before importing main
from main import TemplateForecaster


async def test_research_cache():
    """Test that research caching works correctly"""

    # Create shared cache
    shared_cache = {}

    # Mock question object
    mock_question = MagicMock()
    mock_question.id_of_question = 12345
    mock_question.question_text = "Will it rain tomorrow?"
    mock_question.page_url = "https://example.com/q/12345"

    # Mock LLMs config
    mock_llms = {"default": MagicMock(), "parser": MagicMock(), "researcher": MagicMock(), "summarizer": MagicMock()}

    # Create first bot instance with caching enabled (benchmarking mode)
    bot1 = TemplateForecaster(is_benchmarking=True, research_cache=shared_cache, llms=mock_llms)

    # Mock the research provider to return a test result
    bot1._custom_research_provider = AsyncMock(return_value="Test research result")

    # First call should hit the provider and cache the result
    result1 = await bot1.run_research(mock_question)
    assert result1 == "Test research result"
    assert shared_cache[12345] == "Test research result"
    print("✓ First call cached result successfully")

    # Create second bot instance sharing the same cache
    bot2 = TemplateForecaster(is_benchmarking=True, research_cache=shared_cache, llms=mock_llms)

    # Mock the provider for the second bot to return something different
    bot2._custom_research_provider = AsyncMock(return_value="Different result")

    # Second call should use cache and NOT hit the provider
    result2 = await bot2.run_research(mock_question)
    assert result2 == "Test research result"  # Should get cached result
    assert not bot2._custom_research_provider.called  # Provider should not be called
    print("✓ Second bot used cached result successfully")

    # Test non-benchmarking mode ignores cache
    bot3 = TemplateForecaster(is_benchmarking=False, research_cache=shared_cache, llms=mock_llms)  # Not benchmarking
    bot3._custom_research_provider = AsyncMock(return_value="Non-cached result")

    result3 = await bot3.run_research(mock_question)
    assert result3 == "Non-cached result"  # Should get fresh result
    assert bot3._custom_research_provider.called  # Provider should be called
    print("✓ Non-benchmarking mode bypassed cache correctly")

    print("\n🎉 All tests passed! Research caching implementation is working correctly.")


if __name__ == "__main__":
    asyncio.run(test_research_cache())
