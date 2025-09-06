"""Minimal tests for correlation analysis functionality."""

from datetime import datetime

from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion


def create_mock_benchmark(model_name: str, total_cost: float, num_questions: int = 3) -> BenchmarkForBot:
    """Create a mock benchmark with specified model and cost."""
    # Create mock binary questions and reports
    questions = [
        BinaryQuestion(
            id_of_question=i,
            id_of_post=i,
            page_url=f"https://example.com/{i}",
            question_text=f"Test question {i}?",
            background_info="",
            resolution_criteria="",
            fine_print="",
            published_time=None,
            close_time=None,
        )
        for i in range(num_questions)
    ]

    # Create mock reports with different predictions
    reports = []
    for i, question in enumerate(questions):
        # Create realistic reasoning length for premium models (real usage ~12,000 chars)
        if model_name in ["gpt-5", "o3"]:
            reasoning_text = "Mock detailed reasoning analysis. " * 300  # ~12,000 chars
        else:
            reasoning_text = "Mock reasoning text. " * 10  # ~200 chars

        report = BinaryReport(
            question=question,
            prediction=0.3 + (i * 0.2),  # Vary predictions: 0.3, 0.5, 0.7
            explanation=f"# Mock reasoning for question {i}\n{reasoning_text}",
            price_estimate=total_cost / num_questions,
            minutes_taken=1.0,
            errors=[],
        )
        reports.append(report)

    # Create benchmark with LLM config that includes model name
    benchmark = BenchmarkForBot(
        forecast_bot_class_name="TemplateForecaster",
        num_input_questions=num_questions,
        timestamp=datetime.now(),
        time_taken_in_minutes=5.0,
        total_cost=total_cost,
        forecast_bot_config={
            "llms": {
                "forecasters": [{"model": f"openrouter/{model_name}"}],
                "default": {"model": f"openrouter/{model_name}"},
            },
            "research_reports_per_question": 1,
            "predictions_per_research_report": 1,
        },
        forecast_reports=reports,
    )

    # Mock the average_expected_baseline_score property since it's read-only and calculated from reports
    # Use a reasonable baseline score based on the model name (vary by model for diversity)
    model_score_base = {
        "gpt-5": 18.0,
        "o3": 17.0,
        "gpt-4o": 16.0,
        "claude-3-5-sonnet": 15.0,
        "gemini-pro": 14.0,
        "grok-4": 13.0,
    }.get(model_name.split("/")[-1], 12.0)

    # Monkey patch the property to return our test value
    type(benchmark).average_expected_baseline_score = property(lambda self: model_score_base)

    return benchmark


def test_import_correlation_analysis():
    """Test that the module can be imported without errors."""
    from metaculus_bot.correlation_analysis import (
        CorrelationAnalyzer,
        CorrelationMatrix,
        EnsembleCandidate,
    )

    # Should not raise any import errors
    assert CorrelationAnalyzer is not None
    assert CorrelationMatrix is not None
    assert EnsembleCandidate is not None


def test_correlation_analyzer_instantiation():
    """Test that CorrelationAnalyzer can be created and has expected attributes."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()

    assert hasattr(analyzer, "predictions")
    assert hasattr(analyzer, "benchmarks")
    assert analyzer.predictions == []
    assert analyzer.benchmarks == []


def test_add_benchmark_results():
    """Test that benchmark results can be added to the analyzer."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmarks = [
        create_mock_benchmark("gpt-4o", 0.50, 2),
        create_mock_benchmark("claude-3-5-sonnet", 0.30, 2),
    ]

    analyzer.add_benchmark_results(benchmarks)

    assert len(analyzer.benchmarks) == 2
    assert len(analyzer.predictions) == 4  # 2 models * 2 questions each


def test_calculate_correlation_matrix():
    """Test that correlation matrix calculation works with mock data."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmarks = [
        create_mock_benchmark("gpt-4o", 0.50, 3),
        create_mock_benchmark("claude-3-5-sonnet", 0.30, 3),
        create_mock_benchmark("gemini-pro", 0.40, 3),
    ]

    analyzer.add_benchmark_results(benchmarks)
    corr_matrix = analyzer.calculate_correlation_matrix()

    assert len(corr_matrix.model_names) == 3
    assert corr_matrix.num_questions == 3
    assert corr_matrix.pearson_matrix.shape == (3, 3)
    assert corr_matrix.spearman_matrix.shape == (3, 3)


def test_find_optimal_ensembles():
    """Test that ensemble optimization runs without crashing."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmarks = [
        create_mock_benchmark("gpt-4o", 0.50, 2),
        create_mock_benchmark("claude-3-5-sonnet", 0.30, 2),
        create_mock_benchmark("gemini-pro", 0.40, 2),
    ]

    analyzer.add_benchmark_results(benchmarks)
    ensembles = analyzer.find_optimal_ensembles(max_ensemble_size=3, max_cost_per_question=1.0)

    # Should return some ensemble candidates
    assert isinstance(ensembles, list)
    if ensembles:  # If any ensembles found
        ensemble = ensembles[0]
        assert hasattr(ensemble, "model_names")
        assert hasattr(ensemble, "avg_performance")
        assert hasattr(ensemble, "avg_cost")
        assert hasattr(ensemble, "ensemble_score")


def test_cost_adjustment_for_premium_models():
    """Test that premium models with low costs get adjusted."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmarks = [
        create_mock_benchmark("gpt-5", 0.01, 2),  # Unrealistically low cost
        create_mock_benchmark("o3", 0.02, 2),  # Unrealistically low cost
        create_mock_benchmark("gpt-4o", 0.50, 2),  # Realistic cost
    ]

    analyzer.add_benchmark_results(benchmarks)
    model_stats = analyzer._calculate_model_statistics()

    # Premium models should have adjusted costs based on realistic reasoning length
    # With ~12,000 chars: estimated_tokens = (12000 * 0.3) + 1000 = 4600
    # gpt-5: 2 questions * (4600 * $1.25/M) = ~$0.0115
    # o3: 2 questions * (4600 * $2.0/M) = ~$0.0184
    assert model_stats["gpt-5"]["total_cost"] > 0.01  # Should be ~0.0115
    assert model_stats["o3"]["total_cost"] > 0.015  # Should be ~0.0184

    # Non-premium model should keep original cost
    assert abs(model_stats["gpt-4o"]["total_cost"] - 0.50) < 0.01


def test_extract_model_name():
    """Test model name extraction from benchmark configs."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmark = create_mock_benchmark("openai/gpt-4o", 0.50, 1)

    model_name = analyzer._extract_model_name(benchmark)

    # Should extract the model name part
    assert "gpt-4o" in model_name or "openai/gpt-4o" in model_name


def test_extract_prediction_value():
    """Test prediction value extraction from different report types."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmark = create_mock_benchmark("gpt-4o", 0.50, 1)
    report = benchmark.forecast_reports[0]

    pred_value = analyzer._extract_prediction_value(report)

    # Should return a float value
    assert isinstance(pred_value, float)
    assert 0.0 <= pred_value <= 1.0  # For binary predictions


def test_correlation_matrix_methods():
    """Test CorrelationMatrix utility methods."""
    import pandas as pd

    from metaculus_bot.correlation_analysis import CorrelationMatrix

    # Create simple correlation data
    models = ["gpt-4o", "claude-3-5-sonnet"]
    pearson_data = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=models, columns=models)
    spearman_data = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]], index=models, columns=models)

    corr_matrix = CorrelationMatrix(
        pearson_matrix=pearson_data,
        spearman_matrix=spearman_data,
        model_names=models,
        num_questions=5,
    )

    # Test correlation lookup
    assert corr_matrix.get_correlation("gpt-4o", "claude-3-5-sonnet", "pearson") == 0.5
    assert corr_matrix.get_correlation("gpt-4o", "claude-3-5-sonnet", "spearman") == 0.6

    # Test least correlated pairs
    pairs = corr_matrix.get_least_correlated_pairs(threshold=0.8)
    assert len(pairs) == 1  # Only one pair with correlation < 0.8
    assert pairs[0][2] == 0.5  # Correlation value


def test_ensemble_candidate_scoring():
    """Test EnsembleCandidate scoring calculation."""
    from metaculus_bot.correlation_analysis import EnsembleCandidate

    candidate = EnsembleCandidate(
        model_names=["gpt-4o", "claude-3-5-sonnet"],
        avg_performance=15.0,
        avg_cost=0.40,
        avg_correlation=0.6,
        diversity_score=0.4,
        efficiency_ratio=37.5,
        aggregation_strategy="mean",
    )

    score = candidate.ensemble_score

    # Should return a reasonable score
    assert isinstance(score, float)
    assert score > 0.0


def test_generate_correlation_report():
    """Test that correlation report generation works."""
    from metaculus_bot.correlation_analysis import CorrelationAnalyzer

    analyzer = CorrelationAnalyzer()
    benchmarks = [
        create_mock_benchmark("gpt-4o", 0.50, 2),
        create_mock_benchmark("claude-3-5-sonnet", 0.30, 2),
    ]

    analyzer.add_benchmark_results(benchmarks)
    report = analyzer.generate_correlation_report()  # No file output

    # Should return a string report
    assert isinstance(report, str)
    assert "Model Correlation Analysis Report" in report
    assert "gpt-4o" in report
    assert "claude-3-5-sonnet" in report
