"""Minimal tests for analyze_correlations.py CLI utility functions."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def test_import_analyze_correlations():
    """Test that the CLI script can be imported without errors."""
    import analyze_correlations  # Should not raise any import errors

    assert hasattr(analyze_correlations, "main")
    assert hasattr(analyze_correlations, "extract_timestamp_from_filename")
    assert hasattr(analyze_correlations, "load_benchmarks_from_path")


def test_extract_timestamp_from_filename():
    """Test timestamp extraction from various filename formats."""
    from analyze_correlations import extract_timestamp_from_filename

    # Standard format
    assert extract_timestamp_from_filename("benchmarks_2025-08-10_15-04-51.jsonl") == "2025-08-10_15-04-51"

    # With path
    assert extract_timestamp_from_filename("benchmarks/benchmarks_2025-12-25_23-59-59.json") == "2025-12-25_23-59-59"

    # No timestamp
    assert extract_timestamp_from_filename("simple.json") is None

    # Different format
    assert extract_timestamp_from_filename("other_2024-01-01_00-00-00_suffix.jsonl") == "2024-01-01_00-00-00"


def create_mock_benchmark_data():
    """Create mock benchmark data for binary questions only."""
    return {
        "forecast_bot_class_name": "TemplateForecaster",
        "name": "Test Bot | Model | test-model | 2025-08-11_12-00-00",
        "num_input_questions": 1,
        "timestamp": datetime.now().isoformat(),
        "time_taken_in_minutes": 5.0,
        "total_cost": 0.50,
        "average_expected_baseline_score": 15.5,
        "forecast_bot_config": {
            "llms": {
                "forecasters": [{"model": "openrouter/openai/gpt-4o"}],
                "default": {"model": "openrouter/openai/gpt-4o"},
            },
            "research_reports_per_question": 1,
            "predictions_per_research_report": 1,
        },
        "forecast_reports": [
            {
                "question": {
                    "id_of_question": 1,
                    "id_of_post": 1,
                    "page_url": "https://example.com/1",
                    "question_text": "Will this binary event happen?",
                    "background_info": "",
                    "resolution_criteria": "",
                    "fine_print": "",
                    "published_time": None,
                    "close_time": None,
                },
                "prediction": 0.6,
                "explanation": "# Binary Analysis\nThis is mock reasoning for correlation analysis.",
                "price_estimate": 0.25,
                "minutes_taken": 2.5,
                "expected_baseline_score": 15.5,
                "errors": [],
            }
        ],
        "failed_report_errors": [],
        "git_commit_hash": "abc123",
        "code": "# mock code",
    }


def test_load_benchmarks_from_json_file():
    """Test loading benchmarks from a single JSON file."""
    from analyze_correlations import load_benchmarks_from_path

    mock_data = create_mock_benchmark_data()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([mock_data], f)  # List of benchmarks
        json_path = f.name

    try:
        benchmarks = load_benchmarks_from_path(json_path)
        assert len(benchmarks) == 1
        assert benchmarks[0].forecast_bot_class_name == "TemplateForecaster"
    finally:
        Path(json_path).unlink()  # Clean up


def test_load_benchmarks_from_jsonl_file():
    """Test loading benchmarks from a JSONL file."""
    from analyze_correlations import load_benchmarks_from_path

    mock_data1 = create_mock_benchmark_data()
    mock_data2 = create_mock_benchmark_data()
    mock_data2["total_cost"] = 0.30  # Different cost

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(mock_data1) + "\n")
        f.write(json.dumps(mock_data2) + "\n")
        jsonl_path = f.name

    try:
        benchmarks = load_benchmarks_from_path(jsonl_path)
        assert len(benchmarks) == 2
        assert benchmarks[0].total_cost == 0.50
        assert benchmarks[1].total_cost == 0.30
    finally:
        Path(jsonl_path).unlink()  # Clean up


def test_load_benchmarks_from_directory():
    """Test loading benchmarks from a directory with multiple files."""
    from analyze_correlations import load_benchmarks_from_path

    mock_data = create_mock_benchmark_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a JSON file
        json_file = Path(temp_dir) / "bench1.json"
        with open(json_file, "w") as f:
            json.dump([mock_data], f)

        # Create a JSONL file
        jsonl_file = Path(temp_dir) / "bench2.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(mock_data) + "\n")

        benchmarks = load_benchmarks_from_path(temp_dir)
        assert len(benchmarks) == 2


def test_load_nonexistent_path():
    """Test loading from a path that doesn't exist."""
    from analyze_correlations import load_benchmarks_from_path

    benchmarks = load_benchmarks_from_path("/nonexistent/path")
    assert len(benchmarks) == 0


def test_argument_parsing():
    """Test CLI argument parsing."""
    import argparse

    # Create parser similar to main function
    parser = argparse.ArgumentParser(description="Analyze model correlations from benchmark results")
    parser.add_argument("benchmark_path", help="Path to benchmark file (.json/.jsonl) or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for correlation report (default: correlation_analysis.md)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="Maximum cost per question for ensemble recommendations",
    )
    parser.add_argument("--max-size", type=int, default=5, help="Maximum ensemble size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Test default args
    args = parser.parse_args(["benchmarks/"])
    assert args.benchmark_path == "benchmarks/"
    assert args.max_cost == 1.0
    assert args.max_size == 5
    assert args.verbose is False

    # Test custom args
    args = parser.parse_args(["test.jsonl", "--max-cost", "0.5", "--max-size", "3", "--verbose"])
    assert args.benchmark_path == "test.jsonl"
    assert args.max_cost == 0.5
    assert args.max_size == 3
    assert args.verbose is True


@patch("analyze_correlations.CorrelationAnalyzer")
@patch("analyze_correlations.load_benchmarks_from_path")
def test_main_function_flow(mock_load, mock_analyzer_class):
    """Test main function flow without actual file operations."""
    import analyze_correlations

    # Mock the loading
    mock_benchmarks = [Mock(), Mock()]  # Two mock benchmarks
    mock_load.return_value = mock_benchmarks

    # Mock the analyzer
    mock_analyzer = Mock()
    mock_analyzer.generate_correlation_report.return_value = "Mock report"
    mock_analyzer.find_optimal_ensembles.return_value = []

    # Mock correlation matrix with proper get_least_correlated_pairs return value
    mock_corr_matrix = Mock()
    mock_corr_matrix.get_least_correlated_pairs.return_value = [
        ("model1", "model2", 0.1),
        ("model3", "model4", 0.2),
        ("model5", "model6", 0.3),
    ]
    mock_analyzer.calculate_correlation_matrix.return_value = mock_corr_matrix
    mock_analyzer.calculate_correlation_matrix_by_components.return_value = mock_corr_matrix
    mock_analyzer._has_mixed_question_types.return_value = False  # Use simple correlation
    mock_analyzer._get_question_type_breakdown.return_value = {"binary": 10}
    mock_analyzer_class.return_value = mock_analyzer

    # Mock sys.argv to avoid parsing real command line
    with patch("sys.argv", ["analyze_correlations.py", "test.jsonl"]):
        # Should run without errors
        try:
            analyze_correlations.main()
        except SystemExit:
            pass  # Expected from successful completion


@patch("analyze_correlations.load_benchmarks_from_path")
def test_main_with_insufficient_benchmarks(mock_load):
    """Test main function with too few benchmarks."""
    import analyze_correlations

    # Mock loading only one benchmark (need 2+ for correlation)
    mock_load.return_value = [Mock()]

    with patch("sys.argv", ["analyze_correlations.py", "test.jsonl"]):
        with pytest.raises(SystemExit):
            analyze_correlations.main()


def test_timestamped_output_filename():
    """Test that output filename includes timestamp from input file."""

    from analyze_correlations import extract_timestamp_from_filename

    input_file = "benchmarks/benchmarks_2025-08-10_15-04-51.jsonl"
    timestamp = extract_timestamp_from_filename(input_file)

    # Simulate the logic in main()
    if timestamp:
        filename = f"correlation_analysis_{timestamp}.md"
    else:
        filename = "correlation_analysis.md"

    expected_filename = "correlation_analysis_2025-08-10_15-04-51.md"
    assert filename == expected_filename
