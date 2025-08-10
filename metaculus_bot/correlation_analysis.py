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
        # Weighted combination: performance (0.5) + efficiency (0.3) + diversity (0.2)
        normalized_perf = self.avg_performance / 20.0  # Normalize typical scores
        normalized_efficiency = min(self.efficiency_ratio / 1000.0, 1.0)  # Cap at 1000
        diversity_bonus = 1.0 - self.avg_correlation
        PERF_WT, EFFIC_WT, DIVERS_WT = 0.5, 0.1, 0.4

        return PERF_WT * normalized_perf + EFFIC_WT * normalized_efficiency + DIVERS_WT * diversity_bonus


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

    def find_optimal_ensembles(
        self,
        max_ensemble_size: int = 5,
        max_cost_per_question: float = 1.0,
        min_performance: float = 10.0,
    ) -> List[EnsembleCandidate]:
        """Find optimal ensemble configurations using performance + correlation data."""
        model_stats = self._calculate_model_statistics()
        correlation_matrix = self.calculate_correlation_matrix()

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
        """Convert prediction to float for correlation analysis."""
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

        correlation_matrix = self.calculate_correlation_matrix()
        model_stats = self._calculate_model_statistics()
        optimal_ensembles = self.find_optimal_ensembles()

        report = []
        report.append("# Model Correlation Analysis Report")
        report.append(
            f"Based on {correlation_matrix.num_questions} questions across {len(correlation_matrix.model_names)} models\n"
        )

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
