"""Test correlation analysis with the new ensemble naming convention."""

from unittest.mock import Mock

import pytest

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.correlation_analysis import CorrelationAnalyzer


def test_extract_model_name_with_new_ensemble_naming():
    """Test that _extract_model_name works with new ensemble bot names."""
    analyzer = CorrelationAnalyzer()

    # Test single model bot
    single_benchmark = Mock()
    single_benchmark.name = "qwen3-235b"
    single_benchmark.forecast_bot_config = {"llms": {"forecasters": []}}

    result = analyzer._extract_model_name(single_benchmark)
    assert result == "qwen3-235b"

    # Test ensemble bot with mean aggregation
    ensemble_mean_benchmark = Mock()
    ensemble_mean_benchmark.name = "qwen3_glm_mean"
    ensemble_mean_benchmark.forecast_bot_config = {"llms": {"forecasters": []}}

    result = analyzer._extract_model_name(ensemble_mean_benchmark)
    assert result == "qwen3_glm_mean"

    # Test ensemble bot with median aggregation
    ensemble_median_benchmark = Mock()
    ensemble_median_benchmark.name = "qwen3_glm_median"
    ensemble_median_benchmark.forecast_bot_config = {"llms": {"forecasters": []}}

    result = analyzer._extract_model_name(ensemble_median_benchmark)
    assert result == "qwen3_glm_median"


def test_extract_model_name_legacy_fallback():
    """Test that _extract_model_name falls back to legacy behavior for unknown patterns."""
    analyzer = CorrelationAnalyzer()

    # Test legacy benchmark that doesn't match new patterns
    legacy_benchmark = Mock()
    legacy_benchmark.name = "Legacy Bot | Config | some-unknown-model"
    legacy_benchmark.forecast_bot_config = {"llms": {"forecasters": [{"model": "openrouter/unknown/model-xyz"}]}}

    result = analyzer._extract_model_name(legacy_benchmark)
    # Should extract from the single forecaster model (legacy behavior)
    assert result == "model-xyz"


def test_extract_model_name_ensemble_from_forecasters():
    """Test ensemble name generation from forecaster list when bot name is not available."""
    analyzer = CorrelationAnalyzer()

    # Test ensemble without explicit bot name but with multiple forecasters
    ensemble_benchmark = Mock()
    ensemble_benchmark.name = "Unknown Ensemble"
    ensemble_benchmark.forecast_bot_config = {
        "llms": {
            "forecasters": [
                {"model": "openrouter/qwen/qwen3-235b-a22b-thinking-2507"},
                {"model": "openrouter/z-ai/glm-4.5"},
            ]
        },
        "aggregation_strategy": AggregationStrategy.MEAN,
    }

    result = analyzer._extract_model_name(ensemble_benchmark)
    # Should generate ensemble name from components
    assert result == "glm_qwen3_mean"  # sorted alphabetically


if __name__ == "__main__":
    pytest.main([__file__])
