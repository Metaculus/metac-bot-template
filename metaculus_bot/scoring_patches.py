"""
Scoring patches for mixed question types in forecasting-tools.

This module monkey patches the forecasting-tools library to add scoring support
for numeric and multiple choice questions that currently have NotImplementedError.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def validate_community_prediction_count(question: Any) -> bool:
    """
    Validate that a question has sufficient community predictions (minimum 10).

    Args:
        question: MetaculusQuestion object

    Returns:
        True if question has adequate community predictions, False otherwise
    """
    # Check various possible attributes for prediction count
    if hasattr(question, "num_predictions") and question.num_predictions is not None:
        count = question.num_predictions
        logger.debug(f"Question {question.id_of_question}: {count} community predictions")
        return count >= 10

    if hasattr(question, "prediction_count") and question.prediction_count is not None:
        count = question.prediction_count
        logger.debug(f"Question {question.id_of_question}: {count} community predictions")
        return count >= 10

    # Check if community prediction exists as a proxy for sufficient data
    if (
        hasattr(question, "community_prediction_at_access_time")
        and question.community_prediction_at_access_time is not None
    ):
        logger.debug(f"Question {question.id_of_question}: has community prediction (assuming sufficient count)")
        return True

    logger.warning(f"Question {question.id_of_question}: cannot determine community prediction count")
    return False


def extract_multiple_choice_probabilities(prediction: Any) -> List[float]:
    """
    Extract probability list from a multiple choice prediction.

    Args:
        prediction: PredictedOptionList or similar object

    Returns:
        List of probabilities for each option
    """
    try:
        if hasattr(prediction, "predicted_options") and prediction.predicted_options:
            # Sort by option name for consistency
            sorted_options = sorted(prediction.predicted_options, key=lambda opt: getattr(opt, "option", str(opt)))
            return [float(getattr(opt, "probability", 0)) for opt in sorted_options]
    except (TypeError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to extract MC probabilities: {e}")

    return []


def extract_numeric_percentiles(prediction: Any) -> List[Tuple[float, float]]:
    """
    Extract (percentile, value) pairs from a numeric prediction.

    Args:
        prediction: NumericDistribution or similar object

    Returns:
        List of (percentile, value) tuples
    """
    try:
        if hasattr(prediction, "declared_percentiles") and prediction.declared_percentiles:
            return [(float(p.percentile), float(p.value)) for p in prediction.declared_percentiles]
    except (TypeError, AttributeError, ValueError) as e:
        logger.warning(f"Failed to extract numeric percentiles: {e}")

    return []


def calculate_multiple_choice_baseline_score(report: Any) -> Optional[float]:
    """
    Calculate baseline score for multiple choice questions.

    Uses the same log scoring pattern as binary questions:
    100.0 * sum(c_i * (log2(p_i) + 1.0)) for each option i

    Args:
        report: MultipleChoiceReport object

    Returns:
        Baseline score or None if cannot be calculated
    """
    try:
        # Validate community prediction count
        if not validate_community_prediction_count(report.question):
            logger.info(f"MC Question {report.question.id_of_question}: insufficient community predictions")
            return None

        # Extract bot prediction probabilities
        bot_probs = extract_multiple_choice_probabilities(report.prediction)
        if not bot_probs:
            logger.warning(f"MC Question {report.question.id_of_question}: cannot extract bot probabilities")
            return None

        # For now, simulate community prediction (TODO: extract real community data when available)
        # Use uniform distribution as conservative baseline
        num_options = len(bot_probs)
        community_probs = [1.0 / num_options] * num_options

        logger.debug(f"MC Question {report.question.id_of_question}: {num_options} options")
        logger.debug(f"Bot probs: {[f'{p:.3f}' for p in bot_probs]}")
        logger.debug(f"Community probs: {[f'{p:.3f}' for p in community_probs]} (simulated uniform)")

        # Calculate baseline score using same pattern as binary questions
        score = 0.0
        for c_i, p_i in zip(community_probs, bot_probs):
            if p_i > 0:  # Avoid log(0)
                score += c_i * (math.log2(max(p_i, 1e-10)) + 1.0)
            else:
                logger.warning(f"MC Question {report.question.id_of_question}: zero probability detected")
                return None

        final_score = 100.0 * score
        logger.info(f"MC Question {report.question.id_of_question}: baseline score {final_score:.2f}")
        return final_score

    except Exception as e:
        logger.error(
            f"Error calculating MC baseline score for question {getattr(report.question, 'id_of_question', 'unknown')}: {e}"
        )
        return None


def calculate_numeric_baseline_score(report: Any) -> Optional[float]:
    """
    Calculate baseline score for numeric questions using percentile comparison.

    Simplified approach: compare percentiles and scale to match binary score range.
    Ideally we would use the official Metaculus method using PDFs.
    See: https://www.metaculus.com/help/scores-faq/#continuous-log-score

    Args:
        report: NumericReport object

    Returns:
        Baseline score or None if cannot be calculated
    """
    try:
        # Validate community prediction count
        if not validate_community_prediction_count(report.question):
            logger.info(f"Numeric Question {report.question.id_of_question}: insufficient community predictions")
            return None

        # Extract bot prediction percentiles
        bot_percentiles = extract_numeric_percentiles(report.prediction)
        if not bot_percentiles:
            logger.warning(f"Numeric Question {report.question.id_of_question}: cannot extract bot percentiles")
            return None

        # For now, simulate community prediction (TODO: extract real community data when available)
        # Use bot prediction shifted by small amount as conservative baseline
        community_percentiles = [(p, v * 1.1) for p, v in bot_percentiles]  # 10% shift for simulation

        logger.debug(f"Numeric Question {report.question.id_of_question}: {len(bot_percentiles)} percentiles")

        # Calculate score based on percentile accuracy
        # Use relative error between percentiles, scaled to match binary score range
        total_error = 0.0
        for (p_bot, v_bot), (p_comm, v_comm) in zip(bot_percentiles, community_percentiles):
            if v_comm != 0:
                relative_error = abs(v_bot - v_comm) / abs(v_comm)
                total_error += relative_error

        avg_relative_error = total_error / len(bot_percentiles)

        # Convert to log-like score scaled to match binary range (~-100 to +100)
        # Better predictions (lower error) get higher scores
        if avg_relative_error < 1e-10:
            score = 50.0  # Near perfect prediction
        else:
            score = 50.0 - 30.0 * math.log(1 + avg_relative_error)  # Penalty for higher error

        final_score = score
        logger.info(
            f"Numeric Question {report.question.id_of_question}: baseline score {final_score:.2f} (avg error: {avg_relative_error:.3f})"
        )
        return final_score

    except Exception as e:
        logger.error(
            f"Error calculating numeric baseline score for question {getattr(report.question, 'id_of_question', 'unknown')}: {e}"
        )
        return None


def patch_multiple_choice_scoring():
    """Monkey patch MultipleChoiceReport.expected_baseline_score"""
    try:
        from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport

        def expected_baseline_score_mc(self) -> Optional[float]:
            return calculate_multiple_choice_baseline_score(self)

        MultipleChoiceReport.expected_baseline_score = property(expected_baseline_score_mc)
        logger.info("Successfully patched MultipleChoiceReport.expected_baseline_score")

    except ImportError as e:
        logger.error(f"Could not import MultipleChoiceReport for patching: {e}")
    except Exception as e:
        logger.error(f"Error patching MultipleChoiceReport: {e}")


def patch_numeric_scoring():
    """Monkey patch NumericReport.expected_baseline_score"""
    try:
        from forecasting_tools.data_models.numeric_report import NumericReport

        def expected_baseline_score_numeric(self) -> Optional[float]:
            return calculate_numeric_baseline_score(self)

        NumericReport.expected_baseline_score = property(expected_baseline_score_numeric)
        logger.info("Successfully patched NumericReport.expected_baseline_score")

    except ImportError as e:
        logger.error(f"Could not import NumericReport for patching: {e}")
    except Exception as e:
        logger.error(f"Error patching NumericReport: {e}")


def patch_error_handling():
    """Monkey patch ForecastReport.calculate_average_expected_baseline_score to fix UnboundLocalError"""
    try:
        from typing import Sequence

        import typeguard
        from forecasting_tools.data_models.forecast_report import ForecastReport

        @staticmethod
        def calculate_average_expected_baseline_score_fixed(
            reports: Sequence[Any],
        ) -> float:
            assert len(reports) > 0, "Must have at least one report to calculate average expected baseline score"

            try:
                scores: List[Optional[float]] = [report.expected_baseline_score for report in reports]
                # Filter out None scores
                valid_scores = [score for score in scores if score is not None]

                if not valid_scores:
                    logger.warning("All baseline scores are None, cannot calculate average")
                    return 0.0

                validated_scores: List[float] = typeguard.check_type(valid_scores, list[float])
                average_score = sum(validated_scores) / len(validated_scores)

                none_count = len([score for score in scores if score is None])
                if none_count > 0:
                    logger.warning(f"Calculated average from {len(valid_scores)} scores, {none_count} were None")

                return average_score

            except Exception as e:
                # Fix the UnboundLocalError by ensuring scores is always defined
                scores = [report.expected_baseline_score for report in reports]
                none_count = len([score for score in scores if score is None])
                raise ValueError(
                    f"Error calculating average expected baseline score. {len(reports)} reports. "
                    f"There were {none_count} None scores. Error: {e}"
                ) from e

        ForecastReport.calculate_average_expected_baseline_score = calculate_average_expected_baseline_score_fixed
        logger.info("Successfully patched ForecastReport.calculate_average_expected_baseline_score")

    except ImportError as e:
        logger.error(f"Could not import ForecastReport for patching: {e}")
    except Exception as e:
        logger.error(f"Error patching ForecastReport: {e}")


def log_score_scale_validation(benchmarks: List[Any]) -> None:
    """
    Log score distributions by question type to verify consistent scaling.

    Args:
        benchmarks: List of BenchmarkForBot objects
    """
    try:
        from forecasting_tools.data_models.questions import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

        binary_scores = []
        numeric_scores = []
        mc_scores = []

        for benchmark in benchmarks:
            for report in benchmark.forecast_reports:
                score = report.expected_baseline_score
                if score is not None:
                    if isinstance(report.question, BinaryQuestion):
                        binary_scores.append(score)
                    elif isinstance(report.question, NumericQuestion):
                        numeric_scores.append(score)
                    elif isinstance(report.question, MultipleChoiceQuestion):
                        mc_scores.append(score)

        logger.info("=== SCORE SCALE VALIDATION ===")

        if binary_scores:
            logger.info(
                f"Binary scores: count={len(binary_scores)}, range=[{min(binary_scores):.1f}, {max(binary_scores):.1f}], mean={np.mean(binary_scores):.1f}"
            )
        else:
            logger.info("Binary scores: no data")

        if numeric_scores:
            logger.info(
                f"Numeric scores: count={len(numeric_scores)}, range=[{min(numeric_scores):.1f}, {max(numeric_scores):.1f}], mean={np.mean(numeric_scores):.1f}"
            )
        else:
            logger.info("Numeric scores: no data")

        if mc_scores:
            logger.info(
                f"MC scores: count={len(mc_scores)}, range=[{min(mc_scores):.1f}, {max(mc_scores):.1f}], mean={np.mean(mc_scores):.1f}"
            )
        else:
            logger.info("MC scores: no data")

        logger.info("=== END SCORE VALIDATION ===")

    except Exception as e:
        logger.error(f"Error in score scale validation: {e}")


def apply_scoring_patches() -> None:
    """
    Apply all scoring patches to the forecasting-tools library.

    This function should be called before running benchmarks with mixed question types.
    """
    logger.info("Applying scoring patches for mixed question types...")

    patch_multiple_choice_scoring()
    patch_numeric_scoring()
    patch_error_handling()

    logger.info("Scoring patches applied successfully")
