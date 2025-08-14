"""
Correlation analysis utilities for ensemble optimization.

Tracks inter-model correlations to optimize ensemble composition by balancing
performance with diversity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.questions import MetaculusQuestion
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ModelPrediction:
    """Single model's prediction on a question."""

    model_name: str
    question_id: int
    question_url: str
    prediction_value: float  # For binary: probability, for numeric: aggregated value
    baseline_score: float
    cost: float


@dataclass
class CorrelationMatrix:
    """Inter-model correlation analysis results."""

    pearson_matrix: pd.DataFrame
    spearman_matrix: pd.DataFrame
    model_names: List[str]
    num_questions: int

    def get_correlation(self, model1: str, model2: str, method: str = "pearson") -> float:
        """Get correlation coefficient between two models."""
        matrix = self.pearson_matrix if method == "pearson" else self.spearman_matrix
        return matrix.loc[model1, model2]

    def get_least_correlated_pairs(
        self, threshold: float = 0.7, method: str = "pearson"
    ) -> List[Tuple[str, str, float]]:
        """Find model pairs with correlation below threshold."""
        matrix = self.pearson_matrix if method == "pearson" else self.spearman_matrix
        pairs = []

        for i in range(len(self.model_names)):
            for j in range(i + 1, len(self.model_names)):
                model1, model2 = self.model_names[i], self.model_names[j]
                corr = matrix.iloc[i, j]
                if abs(corr) < threshold:
                    pairs.append((model1, model2, corr))

        return sorted(pairs, key=lambda x: abs(x[2]))  # Sort by absolute correlation


@dataclass
class EnsembleCandidate:
    """Potential ensemble configuration with performance metrics."""

    model_names: List[str]
    avg_performance: float  # Average baseline score
    avg_cost: float
    avg_correlation: float  # Average pairwise correlation
    diversity_score: float  # Lower correlation = higher diversity
    efficiency_ratio: float  # Performance per dollar

    @property
    def ensemble_score(self) -> float:
        """Combined score balancing performance, cost, and diversity."""
        normalized_perf = self.avg_performance / 20.0  # Normalize typical scores
        normalized_efficiency = min(self.efficiency_ratio / 1500.0, 1.0)  # Cap at 1000
        diversity_bonus = 1.0 - self.avg_correlation
        PERF_WT, EFFIC_WT, DIVERS_WT = 0.9, 0.025, 0.075
        perf_score, effic_score, divers_score = (
            PERF_WT * normalized_perf,
            EFFIC_WT * normalized_efficiency,
            DIVERS_WT * diversity_bonus,
        )
        logger.debug(
            f"Score components: performance/accuracy {perf_score:.4f}, efficiency {effic_score:.4f}, diversity {divers_score:.4f}"
        )

        return perf_score + effic_score + divers_score


class CorrelationAnalyzer:
    """Analyzes correlations between forecasting models for ensemble optimization."""

    def __init__(self):
        self.predictions: List[ModelPrediction] = []
        self.benchmarks: List[BenchmarkForBot] = []
        self._model_stats_cache: Optional[Dict[str, Dict[str, float]]] = None

    def add_benchmark_results(self, benchmarks: List[BenchmarkForBot]) -> None:
        """Extract predictions from benchmark results."""
        self.benchmarks = benchmarks
        self.predictions.clear()
        self._model_stats_cache = None  # Clear cache when data changes

        for benchmark in benchmarks:
            model_name = self._extract_model_name(benchmark)

            for report in benchmark.forecast_reports:
                # Convert prediction to float for correlation analysis
                pred_value = self._extract_prediction_value(report)

                prediction = ModelPrediction(
                    model_name=model_name,
                    question_id=report.question.id_of_question or 0,
                    question_url=report.question.page_url,
                    prediction_value=pred_value,
                    baseline_score=report.expected_baseline_score or 0.0,
                    cost=report.price_estimate or 0.0,
                )
                self.predictions.append(prediction)

        logger.info(f"Loaded {len(self.predictions)} predictions from {len(benchmarks)} models")

    def calculate_correlation_matrix(self) -> CorrelationMatrix:
        """Calculate Pearson and Spearman correlations between all model pairs."""
        # Create pivot table: questions × models
        df = pd.DataFrame(
            [
                {
                    "question_id": pred.question_id,
                    "model": pred.model_name,
                    "prediction": pred.prediction_value,
                }
                for pred in self.predictions
            ]
        )

        pivot_df = df.pivot(index="question_id", columns="model", values="prediction")

        # Remove questions where any model failed to predict
        pivot_df = pivot_df.dropna()

        logger.info(f"Correlation analysis using {len(pivot_df)} questions and {len(pivot_df.columns)} models")

        # Calculate correlation matrices
        pearson_corr = pivot_df.corr(method="pearson")
        spearman_corr = pivot_df.corr(method="spearman")

        return CorrelationMatrix(
            pearson_matrix=pearson_corr,
            spearman_matrix=spearman_corr,
            model_names=list(pivot_df.columns),
            num_questions=len(pivot_df),
        )

    def calculate_correlation_matrix_by_components(self) -> CorrelationMatrix:
        """Calculate correlations using component-wise analysis for mixed question types.

        For each question, extracts prediction components and calculates correlations:
        - Binary: Direct correlation on probabilities
        - Numeric: Average correlation across percentiles (10, 20, 40, 60, 80, 90)
        - Multiple Choice: Average correlation across option probabilities
        """
        # Group predictions by question and extract components
        question_data = {}

        for pred in self.predictions:
            q_id = pred.question_id
            if q_id not in question_data:
                question_data[q_id] = {}

            # Get the full report to extract components
            report = None
            for benchmark in self.benchmarks:
                for report_candidate in benchmark.forecast_reports:
                    if (report_candidate.question.id_of_question or 0) == q_id:
                        if self._extract_model_name(benchmark) == pred.model_name:
                            report = report_candidate
                            break
                if report:
                    break

            if report:
                q_type, components = self._extract_prediction_components(report)
                question_data[q_id][pred.model_name] = (q_type, components)

        # Calculate correlations for each question, then average
        model_names = list(set(pred.model_name for pred in self.predictions))
        n_models = len(model_names)

        # Initialize correlation matrices
        correlation_sums = np.zeros((n_models, n_models))
        correlation_counts = np.zeros((n_models, n_models))

        for q_id, model_data in question_data.items():
            # Only process questions where we have data for multiple models
            available_models = list(model_data.keys())
            if len(available_models) < 2:
                continue

            # Group by question type
            q_types = set(data[0] for data in model_data.values())
            if len(q_types) > 1:
                logger.warning(f"Question {q_id} has mixed types across models: {q_types}")
                continue

            q_type = list(q_types)[0]

            # Calculate correlation for this question
            model_indices = {name: i for i, name in enumerate(model_names)}

            for i, model1 in enumerate(available_models):
                for j, model2 in enumerate(available_models):
                    if i >= j:  # Skip duplicates and self-correlation
                        continue

                    # Get components for both models
                    _, components1 = model_data[model1]
                    _, components2 = model_data[model2]

                    # Calculate component-wise correlation
                    if q_type == "binary":
                        # Direct correlation for binary
                        if len(components1) == 1 and len(components2) == 1:
                            corr = 1.0 if components1[0] == components2[0] else 0.0
                        else:
                            corr = 0.0

                    elif q_type in ["numeric", "multiple_choice"]:
                        # Average correlation across components
                        if len(components1) == len(components2) and len(components1) > 1:
                            # Use scipy.stats.pearsonr for component pairs
                            try:
                                corr_val, _ = pearsonr(components1, components2)
                                corr = corr_val if not np.isnan(corr_val) else 0.0
                            except:
                                corr = 0.0
                        else:
                            corr = 0.0
                    else:
                        corr = 0.0

                    # Add to correlation matrix
                    idx1 = model_indices[model1]
                    idx2 = model_indices[model2]
                    correlation_sums[idx1, idx2] += corr
                    correlation_sums[idx2, idx1] += corr  # Symmetric
                    correlation_counts[idx1, idx2] += 1
                    correlation_counts[idx2, idx1] += 1

        # Calculate average correlations
        correlation_matrix = np.zeros((n_models, n_models))
        for i in range(n_models):
            correlation_matrix[i, i] = 1.0  # Self-correlation is 1
            for j in range(i + 1, n_models):
                if correlation_counts[i, j] > 0:
                    avg_corr = correlation_sums[i, j] / correlation_counts[i, j]
                    correlation_matrix[i, j] = avg_corr
                    correlation_matrix[j, i] = avg_corr
                else:
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0

        # Convert to DataFrame
        corr_df = pd.DataFrame(correlation_matrix, index=model_names, columns=model_names)

        logger.info(
            f"Component-wise correlation analysis using {len(question_data)} questions and {len(model_names)} models"
        )

        return CorrelationMatrix(
            pearson_matrix=corr_df,
            spearman_matrix=corr_df,  # For now, use same matrix for both
            model_names=model_names,
            num_questions=len(question_data),
        )

    def find_optimal_ensembles(
        self,
        max_ensemble_size: int = 5,
        max_cost_per_question: float = 1.0,
        min_performance: float = 10.0,
        use_component_analysis: bool = True,
    ) -> List[EnsembleCandidate]:
        """Find optimal ensemble configurations using performance + correlation data."""
        model_stats = self._calculate_model_statistics()

        # Use component-wise analysis for mixed question types if available
        if use_component_analysis and self._has_mixed_question_types():
            correlation_matrix = self.calculate_correlation_matrix_by_components()
            logger.info("Using component-wise correlation analysis for mixed question types")
        else:
            correlation_matrix = self.calculate_correlation_matrix()
            logger.info("Using traditional correlation analysis")

        candidates = []

        # Generate all possible ensemble combinations up to max_ensemble_size
        from itertools import combinations

        for size in range(2, max_ensemble_size + 1):
            for model_combo in combinations(model_stats.keys(), size):
                candidate = self._evaluate_ensemble(model_combo, model_stats, correlation_matrix)

                # Filter by constraints
                if candidate.avg_cost <= max_cost_per_question and candidate.avg_performance >= min_performance:
                    candidates.append(candidate)

        # Sort by ensemble score (descending)
        candidates.sort(key=lambda x: x.ensemble_score, reverse=True)

        logger.info(f"Generated {len(candidates)} viable ensemble candidates")
        return candidates

    def _extract_model_name(self, benchmark: BenchmarkForBot) -> str:
        """Extract clean model name from benchmark."""
        try:
            # Try to get the forecaster model name
            llms = benchmark.forecast_bot_config.get("llms", {})
            if "forecasters" in llms and llms["forecasters"]:
                first_forecaster = llms["forecasters"][0]
                if isinstance(first_forecaster, dict) and "original_model" in first_forecaster:
                    return first_forecaster["original_model"].split("/")[-1]
                elif isinstance(first_forecaster, dict) and "model" in first_forecaster:
                    return first_forecaster["model"].split("/")[-1]

            # Fallback to benchmark name parsing
            name_parts = benchmark.name.split(" | ")
            if len(name_parts) >= 3:
                return name_parts[2]  # Model name is usually third part
        except Exception as e:
            logger.warning(f"Could not extract model name from benchmark: {e}")

        return f"model_{hash(benchmark.name) % 10000}"

    def _extract_prediction_value(self, report) -> float:
        """Convert prediction to float for correlation analysis.

        This method is used for backward compatibility. For mixed question types,
        use _extract_prediction_components() instead.
        """
        prediction = report.prediction

        # Binary questions: return probability directly
        if isinstance(prediction, (int, float)):
            return float(prediction)

        # Numeric questions: use median or mean of distribution
        if hasattr(prediction, "median"):
            return float(prediction.median)
        elif hasattr(prediction, "declared_percentiles") and prediction.declared_percentiles:
            # Use the 50th percentile as central tendency
            percentiles = prediction.declared_percentiles
            median_percentile = next((p for p in percentiles if p.percentile == 50), None)
            if median_percentile:
                return float(median_percentile.value)
            # Fallback to average of available percentiles
            return float(np.mean([p.value for p in percentiles]))

        # Multiple choice: convert to single numeric score (entropy or max probability)
        if hasattr(prediction, "predicted_options"):
            # Use maximum probability as representative value
            return max(opt.probability for opt in prediction.predicted_options)

        # Last resort: hash the prediction for some numeric value
        return float(hash(str(prediction)) % 1000) / 1000.0

    def _extract_prediction_components(self, report) -> Tuple[str, List[float]]:
        """Extract prediction components for improved correlation analysis.

        Returns:
            Tuple of (question_type, component_values)
            - Binary: ("binary", [probability])
            - Numeric: ("numeric", [p10, p20, p40, p60, p80, p90])
            - Multiple Choice: ("multiple_choice", [prob_option1, prob_option2, ...])
        """
        prediction = report.prediction

        # Binary questions: return probability directly
        if isinstance(prediction, (int, float)):
            return ("binary", [float(prediction)])

        # Multiple choice: extract all option probabilities (check this first to avoid median conflicts)
        if (
            hasattr(prediction, "predicted_options")
            and prediction.predicted_options is not None
            and prediction.predicted_options
        ):
            try:
                # Sort by option name for consistency across models
                sorted_options = sorted(
                    prediction.predicted_options, key=lambda opt: getattr(opt, "option", "") or str(opt)
                )
                option_probs = [float(getattr(opt, "probability", 0)) for opt in sorted_options]
                return ("multiple_choice", option_probs)
            except (TypeError, AttributeError):
                # Handle case where predicted_options is not iterable (e.g., in tests)
                return ("multiple_choice", [0.5, 0.5])  # Default equal probability for 2 options

        # Numeric questions: extract all percentiles
        if (
            hasattr(prediction, "declared_percentiles")
            and prediction.declared_percentiles is not None
            and prediction.declared_percentiles
        ):
            # Extract the standard percentiles (10, 20, 40, 60, 80, 90)
            target_percentiles = [10, 20, 40, 60, 80, 90]
            percentile_values = []

            # Create dict for quick lookup - handle both real and mock objects
            try:
                percentile_dict = {p.percentile: p.value for p in prediction.declared_percentiles}
            except (TypeError, AttributeError):
                # Handle case where declared_percentiles is not iterable (e.g., in tests)
                percentile_dict = {}

            for target_p in target_percentiles:
                if target_p in percentile_dict:
                    percentile_values.append(float(percentile_dict[target_p]))
                else:
                    # If missing, interpolate or use median as fallback
                    if hasattr(prediction, "median"):
                        try:
                            percentile_values.append(float(prediction.median))
                        except (TypeError, ValueError):
                            # Handle case where median is a Mock or invalid
                            percentile_values.append(0.0)
                    else:
                        # Last resort: use mean of available percentiles
                        available_values = list(percentile_dict.values())
                        percentile_values.append(float(np.mean(available_values)) if available_values else 0.0)

            return ("numeric", percentile_values)
        elif hasattr(prediction, "median") and prediction.median is not None:
            # Fallback for numeric with only median
            try:
                median_val = float(prediction.median)
                return ("numeric", [median_val] * 6)  # Repeat median for all percentiles
            except (TypeError, ValueError):
                # Handle case where median is a Mock or invalid
                return ("numeric", [0.0] * 6)

        # Fallback: treat as binary with neutral prediction
        return ("binary", [0.5])

    def _has_mixed_question_types(self) -> bool:
        """Check if the benchmarks contain mixed question types."""
        question_types = set()

        for benchmark in self.benchmarks:
            for report in benchmark.forecast_reports:
                q_type, _ = self._extract_prediction_components(report)
                question_types.add(q_type)

        return len(question_types) > 1

    def _get_question_type_breakdown(self) -> Dict[str, int]:
        """Get count of each question type in the benchmarks."""
        type_counts = {}

        for benchmark in self.benchmarks:
            for report in benchmark.forecast_reports:
                q_type, _ = self._extract_prediction_components(report)
                type_counts[q_type] = type_counts.get(q_type, 0) + 1

        return type_counts

    def _calculate_model_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance and cost statistics per model."""
        if self._model_stats_cache is not None:
            return self._model_stats_cache

        model_stats = {}

        for benchmark in self.benchmarks:
            model_name = self._extract_model_name(benchmark)
            total_cost = benchmark.total_cost
            num_questions = len(benchmark.forecast_reports)

            # Hack: Fix unrealistic costs for premium models
            if model_name in ["gpt-5", "o3"] and total_cost < 0.10:
                # Estimate based on average reasoning length and known pricing
                avg_reasoning_length = self._estimate_avg_reasoning_length(benchmark)
                estimated_tokens = (avg_reasoning_length * 0.3) + 1000  # chars*0.3 + base prompt

                if model_name == "gpt-5":
                    total_cost = num_questions * (
                        estimated_tokens * 1.25 / 1_000_000
                    )  # $1.25 input + conservative output
                elif model_name == "o3":
                    total_cost = num_questions * (estimated_tokens * 2.0 / 1_000_000)  # $2 input + conservative output

                logger.info(
                    f"Adjusted {model_name} cost from ${benchmark.total_cost:.4f} to ${total_cost:.4f} "
                    f"(avg reasoning: {avg_reasoning_length} chars)"
                )

            model_stats[model_name] = {
                "avg_performance": benchmark.average_expected_baseline_score,
                "avg_cost": total_cost / max(num_questions, 1),
                "total_cost": total_cost,
                "num_questions": num_questions,
                "efficiency_ratio": benchmark.average_expected_baseline_score / max(total_cost, 0.001),
            }

        self._model_stats_cache = model_stats  # Cache the results
        return model_stats

    def _estimate_avg_reasoning_length(self, benchmark: BenchmarkForBot) -> float:
        """Estimate average reasoning text length for cost calculation."""
        total_chars = 0
        count = 0

        for report in benchmark.forecast_reports:
            if hasattr(report, "explanation") and report.explanation:
                total_chars += len(report.explanation)
                count += 1

        return total_chars / max(count, 1) if count > 0 else 2000  # Default estimate

    def _evaluate_ensemble(
        self, model_names: Tuple[str, ...], model_stats: Dict[str, Dict[str, float]], corr_matrix: CorrelationMatrix
    ) -> EnsembleCandidate:
        """Evaluate a specific ensemble configuration."""
        models = list(model_names)

        # Calculate average performance and cost
        avg_performance = np.mean([model_stats[m]["avg_performance"] for m in models])
        avg_cost = np.mean([model_stats[m]["avg_cost"] for m in models])

        # Calculate average pairwise correlation
        correlations = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                try:
                    corr = corr_matrix.get_correlation(models[i], models[j], "pearson")
                    correlations.append(abs(corr))
                except KeyError:
                    # Models might not have overlapping predictions
                    correlations.append(0.5)  # Neutral correlation

        avg_correlation = np.mean(correlations) if correlations else 0.5
        diversity_score = 1.0 - avg_correlation
        efficiency_ratio = avg_performance / max(avg_cost, 0.001)

        return EnsembleCandidate(
            model_names=models,
            avg_performance=avg_performance,
            avg_cost=avg_cost,
            avg_correlation=avg_correlation,
            diversity_score=diversity_score,
            efficiency_ratio=efficiency_ratio,
        )

    def generate_correlation_report(self, output_path: Optional[str] = None) -> str:
        """Generate human-readable correlation analysis report."""
        if not self.predictions:
            return "No prediction data available for correlation analysis."

        # Use component-wise analysis for mixed question types
        use_component_analysis = self._has_mixed_question_types()
        if use_component_analysis:
            correlation_matrix = self.calculate_correlation_matrix_by_components()
        else:
            correlation_matrix = self.calculate_correlation_matrix()

        model_stats = self._calculate_model_statistics()
        optimal_ensembles = self.find_optimal_ensembles(use_component_analysis=use_component_analysis)

        report = []
        report.append("# Model Correlation Analysis Report")
        report.append(
            f"Based on {correlation_matrix.num_questions} questions across {len(correlation_matrix.model_names)} models\n"
        )

        # Add question type breakdown if mixed
        if use_component_analysis:
            type_counts = self._get_question_type_breakdown()
            report.append("## Question Type Distribution")
            for q_type, count in sorted(type_counts.items()):
                report.append(f"- **{q_type.title()}**: {count} questions")
            report.append(f"- **Analysis Method**: Component-wise correlation (improved for mixed types)\n")

        # Model Performance Summary
        report.append("## Individual Model Performance")
        for model, stats in sorted(model_stats.items(), key=lambda x: x[1]["avg_performance"], reverse=True):
            report.append(
                f"- **{model}**: Score {stats['avg_performance']:.2f}, "
                f"Cost ${stats['avg_cost']:.3f}/question, "
                f"Efficiency {stats['efficiency_ratio']:.1f}"
            )

        # Correlation Highlights
        report.append("\n## Model Correlations (Pearson)")
        least_correlated = correlation_matrix.get_least_correlated_pairs(threshold=0.8)
        report.append("**Most Independent Model Pairs:**")
        for model1, model2, corr in least_correlated[:5]:
            report.append(f"- {model1} ↔ {model2}: r = {corr:.3f}")

        # Optimal Ensembles
        report.append("\n## Recommended Ensembles")
        for i, ensemble in enumerate(optimal_ensembles[:5], 1):
            models_str = " + ".join(ensemble.model_names)
            report.append(f"**{i}. {models_str}**")
            report.append(f"   - Performance: {ensemble.avg_performance:.2f}")
            report.append(f"   - Cost: ${ensemble.avg_cost:.3f}/question")
            report.append(f"   - Diversity: {ensemble.diversity_score:.3f}")
            report.append(f"   - Ensemble Score: {ensemble.ensemble_score:.3f}")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Correlation report saved to {output_path}")

        return report_text
