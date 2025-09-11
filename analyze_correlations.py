#!/usr/bin/env python3
"""
Standalone script to analyze correlations from existing benchmark results.

Usage:
    python analyze_correlations.py benchmarks/benchmarks_2025-08-10_15-04-51.jsonl
    python analyze_correlations.py benchmarks/ --max-cost 0.3 --max-size 3

Or via Makefile:
    make analyze_correlations FILE=benchmarks/benchmarks_2025-08-10_15-04-51.jsonl
    make analyze_correlations_latest
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot

from metaculus_bot.correlation_analysis import CorrelationAnalyzer
from metaculus_bot.scoring_patches import apply_scoring_patches

logger = logging.getLogger(__name__)


def extract_timestamp_from_filename(filepath: str) -> Optional[str]:
    """Extract timestamp from benchmark filename like 'benchmarks_2025-08-10_15-04-51.jsonl'"""
    filename = Path(filepath).name
    # Match pattern: benchmarks_YYYY-MM-DD_HH-MM-SS
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None


def load_benchmarks_from_path(benchmark_path: str) -> List[BenchmarkForBot]:
    """Load benchmark data from a file or directory."""
    path = Path(benchmark_path)
    benchmarks = []

    if path.is_file():
        # Single file - handle both .json and .jsonl
        try:
            with open(path, "r") as f:
                if path.suffix == ".jsonl":
                    # JSON Lines format - one benchmark per line
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            benchmark = BenchmarkForBot.model_validate(data)
                            benchmarks.append(benchmark)
                else:
                    # Regular JSON
                    data = json.load(f)
                    if isinstance(data, list):
                        for bench_data in data:
                            benchmark = BenchmarkForBot.model_validate(bench_data)
                            benchmarks.append(benchmark)
                    else:
                        benchmark = BenchmarkForBot.model_validate(data)
                        benchmarks.append(benchmark)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            return []

    elif path.is_dir():
        # Directory - load all .json and .jsonl files
        for pattern in ["*.json", "*.jsonl"]:
            for json_file in path.glob(pattern):
                if json_file.name.startswith("correlation_"):
                    continue  # Skip correlation analysis files
                benchmarks.extend(load_benchmarks_from_path(str(json_file)))

    else:
        logger.error(f"Path does not exist: {benchmark_path}")
        return []

    logger.info(f"Loaded {len(benchmarks)} benchmarks from {benchmark_path}")
    return benchmarks


def main():
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
    parser.add_argument(
        "--question-types",
        nargs="*",
        choices=["binary", "numeric", "multiple_choice"],
        help="Filter analysis to specific question types",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help=(
            "Exclude models by substring match (case-insensitive). " "Example: --exclude-models grok-4 gemini-2.5-pro"
        ),
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help=(
            "Only include models matching these substrings (case-insensitive). "
            "Mutually exclusive with --exclude-models."
        ),
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load benchmarks
    try:
        benchmarks = load_benchmarks_from_path(args.benchmark_path)
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        sys.exit(1)

    if len(benchmarks) < 2:
        logger.error("Need at least 2 benchmark results for correlation analysis")
        sys.exit(1)

    # Apply scoring patches for mixed question types
    apply_scoring_patches()

    # Perform analysis
    analyzer = CorrelationAnalyzer()
    analyzer.add_benchmark_results(benchmarks)

    # Apply include/exclude filtering before analysis
    if args.include_models and args.exclude_models:
        logger.error("--include-models and --exclude-models are mutually exclusive")
        sys.exit(2)

    filter_summary = analyzer.filter_models_inplace(include=args.include_models, exclude=args.exclude_models)
    if args.include_models or args.exclude_models:
        print("Applied model filters:")
        if args.include_models:
            print(f"  include tokens: {args.include_models}")
        if args.exclude_models:
            print(f"  exclude tokens: {args.exclude_models}")
        unmatched_inc = filter_summary.get("unmatched_includes", [])
        unmatched_exc = filter_summary.get("unmatched_excludes", [])
        if unmatched_inc:
            print(f"  unmatched include tokens: {unmatched_inc}")
        if unmatched_exc:
            print(f"  unmatched exclude tokens: {unmatched_exc}")

    # Ensure at least two models remain
    remaining_models = analyzer.get_model_names()
    if len(remaining_models) < 2:
        logger.error(
            f"Analysis requires ≥2 models after filtering. Remaining: {remaining_models if remaining_models else 'none'}"
        )
        sys.exit(1)

    # Check if we have mixed question types
    has_mixed_types = analyzer._has_mixed_question_types()
    if has_mixed_types:
        logger.info("Detected mixed question types - using component-wise correlation analysis")
        type_breakdown = analyzer._get_question_type_breakdown()
        logger.info(f"Question type distribution: {type_breakdown}")
    else:
        logger.info("Using traditional correlation analysis for binary questions")

    # Generate report with timestamped filename
    if args.output:
        output_file = args.output
    else:
        # Default output location with timestamp from input file
        benchmark_path = Path(args.benchmark_path)
        timestamp = extract_timestamp_from_filename(args.benchmark_path)

        if timestamp:
            filename = f"correlation_analysis_{timestamp}.md"
        else:
            # Fallback to current timestamp if can't extract from input
            from datetime import datetime

            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"correlation_analysis_{current_timestamp}.md"

        if benchmark_path.is_file():
            output_file = benchmark_path.parent / filename
        else:
            output_file = benchmark_path / filename
    report = analyzer.generate_correlation_report(output_file)

    print("=" * 60)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 60)
    print(report)

    # Show top ensemble recommendations
    print("\n" + "=" * 60)
    print("ENSEMBLE RECOMMENDATIONS")
    print("=" * 60)

    optimal_ensembles = analyzer.find_optimal_ensembles(
        max_ensemble_size=args.max_size, max_cost_per_question=args.max_cost
    )

    if optimal_ensembles:
        print(f"\nTop 10 Ensembles (Both Aggregation Strategies, Cost ≤ ${args.max_cost}/question):")
        for i, ensemble in enumerate(optimal_ensembles[:10], 1):
            models = " + ".join(ensemble.model_names)
            print(f"{i:2}. {models} ({ensemble.aggregation_strategy.upper()})")
            print(
                f"    Score: {ensemble.avg_performance:.2f} | "
                f"Cost: ${ensemble.avg_cost:.3f} | "
                f"Diversity: {ensemble.diversity_score:.3f} | "
                f"Overall: {ensemble.ensemble_score:.3f}"
            )
    else:
        print("No ensembles found meeting the cost constraint.")

    # Show correlation matrix highlights
    # Use appropriate correlation method based on question types
    if has_mixed_types:
        corr_matrix = analyzer.calculate_correlation_matrix_by_components()
    else:
        corr_matrix = analyzer.calculate_correlation_matrix()
    print(f"\n{'-' * 40}")
    print("CORRELATION HIGHLIGHTS")
    print(f"{'-' * 40}")

    least_correlated = corr_matrix.get_least_correlated_pairs(threshold=0.8)
    print("\nMost Independent Model Pairs:")
    for model1, model2, corr in least_correlated[:8]:
        print(f"  {model1:20} ↔ {model2:20} | r = {corr:6.3f}")

    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    main()
