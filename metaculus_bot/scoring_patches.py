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
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Scoring path counters for diagnostics across a run
_NUMERIC_PMF_ATTEMPTS = 0
_NUMERIC_PMF_SUCCESSES = 0
_NUMERIC_FALLBACK_ATTEMPTS = 0
_NUMERIC_FALLBACK_SUCCESSES = 0
_MC_ATTEMPTS = 0
_MC_MISSING_COMMUNITY = 0
_MC_SUCCESSES = 0
# MC diagnostics breakdown
_MC_MISSING_API_JSON = 0
_MC_MISSING_QUESTION_NODE = 0
_MC_MISSING_AGGREGATIONS = 0
_MC_MISSING_PYC = 0


def reset_scoring_path_stats() -> None:
    global _NUMERIC_PMF_ATTEMPTS, _NUMERIC_PMF_SUCCESSES
    global _NUMERIC_FALLBACK_ATTEMPTS, _NUMERIC_FALLBACK_SUCCESSES
    global _MC_ATTEMPTS, _MC_MISSING_COMMUNITY, _MC_SUCCESSES
    global _MC_MISSING_API_JSON, _MC_MISSING_QUESTION_NODE, _MC_MISSING_AGGREGATIONS, _MC_MISSING_PYC
    _NUMERIC_PMF_ATTEMPTS = 0
    _NUMERIC_PMF_SUCCESSES = 0
    _NUMERIC_FALLBACK_ATTEMPTS = 0
    _NUMERIC_FALLBACK_SUCCESSES = 0
    _MC_ATTEMPTS = 0
    _MC_MISSING_COMMUNITY = 0
    _MC_SUCCESSES = 0
    _MC_MISSING_API_JSON = 0
    _MC_MISSING_QUESTION_NODE = 0
    _MC_MISSING_AGGREGATIONS = 0
    _MC_MISSING_PYC = 0


def get_scoring_path_stats() -> dict[str, float | int]:
    total_numeric = _NUMERIC_PMF_ATTEMPTS + _NUMERIC_FALLBACK_ATTEMPTS
    total_mc = _MC_ATTEMPTS
    return {
        "numeric_pmf_attempts": _NUMERIC_PMF_ATTEMPTS,
        "numeric_pmf_successes": _NUMERIC_PMF_SUCCESSES,
        "numeric_fallback_attempts": _NUMERIC_FALLBACK_ATTEMPTS,
        "numeric_fallback_successes": _NUMERIC_FALLBACK_SUCCESSES,
        "numeric_total": total_numeric,
        "numeric_fallback_rate": ((_NUMERIC_FALLBACK_ATTEMPTS / total_numeric) if total_numeric > 0 else 0.0),
        "mc_attempts": total_mc,
        "mc_successes": _MC_SUCCESSES,
        "mc_missing_community": _MC_MISSING_COMMUNITY,
        "mc_missing_rate": ((_MC_MISSING_COMMUNITY / total_mc) if total_mc > 0 else 0.0),
        # MC breakdown
        "mc_missing_api_json": _MC_MISSING_API_JSON,
        "mc_missing_question_node": _MC_MISSING_QUESTION_NODE,
        "mc_missing_aggregations": _MC_MISSING_AGGREGATIONS,
        "mc_missing_pyc": _MC_MISSING_PYC,
    }


def log_scoring_path_stats() -> None:
    stats = get_scoring_path_stats()
    logger.info("=== SCORING PATH SUMMARY ===")
    logger.info(
        "Numeric: pmf_attempts=%d pmf_successes=%d fallback_attempts=%d fallback_successes=%d total=%d fallback_rate=%.2f",
        stats["numeric_pmf_attempts"],
        stats["numeric_pmf_successes"],
        stats["numeric_fallback_attempts"],
        stats["numeric_fallback_successes"],
        stats["numeric_total"],
        stats["numeric_fallback_rate"],
    )
    logger.info(
        "MC: attempts=%d successes=%d missing_community=%d missing_rate=%.2f",
        stats["mc_attempts"],
        stats["mc_successes"],
        stats["mc_missing_community"],
        stats["mc_missing_rate"],
    )
    logger.info(
        "MC missing breakdown: api_json=%d question_node=%d aggregations=%d pyc=%d",
        stats["mc_missing_api_json"],
        stats["mc_missing_question_node"],
        stats["mc_missing_aggregations"],
        stats["mc_missing_pyc"],
    )

    # Bright warnings when fallbacks dominate
    if stats["numeric_total"] and stats["numeric_fallback_rate"] >= 0.8:
        logger.warning(
            "⚠️  ALERT: Numeric scoring fallback used for %.0f%% of items. Check that model predictions expose CDFs.",
            100 * stats["numeric_fallback_rate"],
        )
    logger.info("=== END SCORING SUMMARY ===")


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


def extract_multiple_choice_probabilities(prediction: Any) -> list[float]:
    """
    Safely extracts probabilities from a PredictedOptionList, sorting by option name.

    Note: forecasting_tools PredictedOption uses the field `option_name`.
    """
    if not prediction or not hasattr(prediction, "predicted_options") or prediction.predicted_options is None:
        return []
    # Sort by option name to ensure consistent order
    try:
        sorted_options = sorted(prediction.predicted_options, key=lambda o: o.option_name)
    except AttributeError:
        # Fallback if mocks used a different attribute during tests
        sorted_options = sorted(prediction.predicted_options, key=lambda o: getattr(o, "option", ""))
    return [opt.probability for opt in sorted_options]


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


def _extract_mc_community_probs(question: Any) -> Optional[List[float]]:
    global _MC_MISSING_API_JSON, _MC_MISSING_QUESTION_NODE, _MC_MISSING_AGGREGATIONS, _MC_MISSING_PYC, _MC_MISSING_COMMUNITY
    """Extract community option probabilities for an MC question from api_json.

    According to the Metaculus API, community MC aggregations expose
    `probability_yes_per_category` under `aggregations.recency_weighted.latest`.
    We align the resulting vector to `question.options` order.
    """
    try:
        # Basic fingerprint
        post_id = getattr(question, "id_of_post", None)
        qid = getattr(question, "id_of_question", None)
        api_json = getattr(question, "api_json", None)
        if not isinstance(api_json, dict):
            logger.warning(
                "MC q=%s post=%s: api_json missing or not dict (type=%s)", qid, post_id, type(api_json).__name__
            )
            global _MC_MISSING_API_JSON, _MC_MISSING_COMMUNITY
            _MC_MISSING_API_JSON += 1
            _MC_MISSING_COMMUNITY += 1
            return None

        # Detect the question node
        api_has_question = isinstance(api_json.get("question"), dict)
        question_obj = api_json.get("question") if api_has_question else api_json
        if not isinstance(question_obj, dict):
            logger.warning(
                "MC q=%s post=%s: missing question object (api_has_question=%s, type=%s)",
                qid,
                post_id,
                api_has_question,
                type(question_obj).__name__,
            )
            global _MC_MISSING_QUESTION_NODE
            _MC_MISSING_QUESTION_NODE += 1
            _MC_MISSING_COMMUNITY += 1
            return None

        qtype = question_obj.get("type")
        options = getattr(question, "options", None)
        if options is None and isinstance(question_obj.get("options"), list):
            options = question_obj.get("options")

        aggregations = question_obj.get("aggregations")
        if not isinstance(aggregations, dict):
            logger.info(
                "MC q=%s: aggregations missing (question.type=%s). keys=%s",
                qid,
                qtype,
                list(question_obj.keys()),
            )
            global _MC_MISSING_AGGREGATIONS
            _MC_MISSING_AGGREGATIONS += 1
            _MC_MISSING_COMMUNITY += 1
            return None

        rw = aggregations.get("recency_weighted")
        rw_latest = rw.get("latest") if isinstance(rw, dict) else None
        rw_keys = list(rw_latest.keys()) if isinstance(rw_latest, dict) else None
        logger.debug(
            "MC q=%s: rw.latest keys=%s (agg.keys=%s)",
            qid,
            rw_keys,
            list(aggregations.keys()),
        )

        if not isinstance(rw_latest, dict):
            logger.info("MC q=%s: recency_weighted.latest missing", qid)
            global _MC_MISSING_PYC
            _MC_MISSING_PYC += 1
            _MC_MISSING_COMMUNITY += 1
            return None

        # First, prefer forecast_values aligned by index with options
        fv = rw_latest.get("forecast_values")
        if isinstance(fv, list):
            if not options or not isinstance(options, list):
                logger.info("MC q=%s: options unavailable; cannot align forecast_values", qid)
                _MC_MISSING_COMMUNITY += 1
                return None
            if len(fv) != len(options):
                logger.warning(
                    "MC q=%s: forecast_values length %d != options length %d",
                    qid,
                    len(fv),
                    len(options),
                )
                _MC_MISSING_PYC += 1
                _MC_MISSING_COMMUNITY += 1
                return None
            try:
                probs = [float(x) for x in fv]
            except Exception as e:
                logger.warning("MC q=%s: forecast_values cast error: %s", qid, e)
                _MC_MISSING_PYC += 1
                _MC_MISSING_COMMUNITY += 1
                return None
            within = all(0.0 <= p <= 1.0 for p in probs)
            total = sum(probs)
            if not within:
                logger.warning("MC q=%s: forecast_values contain out-of-range probabilities", qid)
                _MC_MISSING_PYC += 1
                _MC_MISSING_COMMUNITY += 1
                return None
            if abs(total - 1.0) > 1e-3:
                logger.warning("MC q=%s: forecast_values sum %.6f far from 1.0", qid, total)
                _MC_MISSING_PYC += 1
                _MC_MISSING_COMMUNITY += 1
                return None
            if abs(total - 1.0) > 1e-6:
                logger.info("MC q=%s: normalizing forecast_values (sum=%.6f)", qid, total)
                probs = [p / total for p in probs]
            logger.debug("MC q=%s: using rw.latest.forecast_values aligned to options", qid)
            return probs

        # If forecast_values missing, try probability_yes_per_category dict
        pyc = rw_latest.get("probability_yes_per_category")
        if isinstance(pyc, dict):
            if options and isinstance(options, list):
                keys = sorted(pyc.keys())
                missing = [opt for opt in options if opt not in pyc]
                extra = [k for k in keys if k not in options]
                if missing or extra:
                    logger.warning("MC q=%s: option mismatch vs pyc. missing=%s extra=%s", qid, missing, extra)
                probs = [float(pyc.get(opt, 0.0)) for opt in options]
                total = sum(probs)
                if abs(total - 1.0) > 1e-6 and abs(total - 1.0) <= 1e-3:
                    logger.info("MC q=%s: normalizing pyc (sum=%.6f)", qid, total)
                    probs = [p / total for p in probs]
                elif abs(total - 1.0) > 1e-3:
                    logger.warning("MC q=%s: pyc sum %.6f far from 1.0", qid, total)
                    _MC_MISSING_PYC += 1
                    _MC_MISSING_COMMUNITY += 1
                    return None
                logger.debug("MC q=%s: using rw.latest.probability_yes_per_category", qid)
                return probs
            else:
                logger.info("MC q=%s: options unavailable; cannot align pyc", qid)
                _MC_MISSING_COMMUNITY += 1
                return None

        logger.info("MC q=%s: neither forecast_values nor pyc available in rw.latest (keys=%s)", qid, rw_keys)
        _MC_MISSING_PYC += 1
        _MC_MISSING_COMMUNITY += 1
        return None

    except Exception as e:
        logger.warning(f"Failed to extract MC community probabilities: {e}")
        _MC_MISSING_COMMUNITY += 1
    return None


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
    global _MC_ATTEMPTS, _MC_MISSING_COMMUNITY, _MC_SUCCESSES
    try:
        _MC_ATTEMPTS += 1
        # Extract bot prediction probabilities
        bot_probs = extract_multiple_choice_probabilities(report.prediction)
        if not bot_probs:
            logger.warning(
                f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: cannot extract bot probabilities"
            )
            return None

        # Extract community probabilities (extractor logs causes and increments counters)
        community_probs = _extract_mc_community_probs(report.question)
        if not community_probs:
            logger.info(
                f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: missing community probabilities"
            )
            return None
        if len(community_probs) != len(bot_probs):
            logger.warning(
                f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: community vector length mismatch"
            )
            return None

        # Clamp and normalize both
        eps = 1e-9
        bot_probs = [max(min(p, 0.999), 0.001) for p in bot_probs]
        s = sum(bot_probs)
        bot_probs = [p / s for p in bot_probs] if s > 0 else [1.0 / len(bot_probs)] * len(bot_probs)

        community_probs = [max(min(float(c), 0.999), 0.001) for c in community_probs]
        s2 = sum(community_probs)
        community_probs = (
            [c / s2 for c in community_probs] if s2 > 0 else [1.0 / len(community_probs)] * len(community_probs)
        )

        # Expected baseline-style score vs community:
        # 100 * (E_c[ ln p ] / ln K + 1)
        K = max(1, len(bot_probs))
        lnK = math.log(K) if K > 1 else 1.0
        sum_ln = 0.0
        for c_i, p_i in zip(community_probs, bot_probs):
            sum_ln += c_i * math.log(max(p_i, eps))
        final_score = 100.0 * (sum_ln / lnK + 1.0)
        _MC_SUCCESSES += 1
        logger.info(
            f"MC Question {getattr(report.question, 'id_of_question', 'unknown')}: baseline score {final_score:.2f}"
        )
        return final_score

    except Exception as e:
        logger.error(
            f"Error calculating MC baseline score for question {getattr(report.question, 'id_of_question', 'unknown')}: {e}"
        )
        return None


def _extract_numeric_community_cdf(question: Any) -> Optional[List[float]]:
    """Extract community CDF (forecast_values) from api_json with structured logging; no fallback."""
    try:
        post_id = getattr(question, "id_of_post", None)
        qid = getattr(question, "id_of_question", None)
        api_json = getattr(question, "api_json", None)
        if not isinstance(api_json, dict):
            logger.warning(
                "Numeric q=%s post=%s: api_json missing or not dict (type=%s)", qid, post_id, type(api_json).__name__
            )
            return None

        api_has_question = isinstance(api_json.get("question"), dict)
        question_obj = api_json.get("question") if api_has_question else api_json
        if not isinstance(question_obj, dict):
            logger.warning(
                "Numeric q=%s post=%s: missing question object (api_has_question=%s, type=%s)",
                qid,
                post_id,
                api_has_question,
                type(question_obj).__name__,
            )
            return None

        expected_len = None
        try:
            scaling = question_obj.get("scaling", {})
            inbound = scaling.get("inbound_outcome_count")
            if inbound is not None:
                expected_len = int(inbound) + 1
        except Exception:
            expected_len = None

        aggregations = question_obj.get("aggregations")
        if not isinstance(aggregations, dict):
            logger.info(
                "Numeric q=%s: aggregations missing. keys=%s",
                qid,
                list(question_obj.keys()),
            )
            return None

        rw = aggregations.get("recency_weighted")
        rw_latest = rw.get("latest") if isinstance(rw, dict) else None
        rw_keys = list(rw_latest.keys()) if isinstance(rw_latest, dict) else None
        logger.debug(
            "Numeric q=%s: rw.latest keys=%s (agg.keys=%s)",
            qid,
            rw_keys,
            list(aggregations.keys()),
        )
        if not isinstance(rw_latest, dict):
            logger.info("Numeric q=%s: recency_weighted.latest missing", qid)
            return None

        fv = rw_latest.get("forecast_values")
        if isinstance(fv, list) and len(fv) >= 2:
            if expected_len and len(fv) != expected_len:
                logger.warning(
                    "Numeric q=%s: forecast_values length %d != expected %d",
                    qid,
                    len(fv),
                    expected_len,
                )
            logger.debug(
                "Numeric q=%s: using rw.latest.forecast_values len=%d first=%.5f last=%.5f",
                qid,
                len(fv),
                float(fv[0]),
                float(fv[-1]),
            )
            return [float(x) for x in fv]

        logger.info("Numeric q=%s: forecast_values missing in rw.latest (keys=%s)", qid, rw_keys)
    except Exception as e:
        logger.warning(f"Failed to extract numeric community CDF: {e}")
    return None


def calculate_numeric_baseline_score(report: Any) -> Optional[float]:
    """
    Calculate baseline score for numeric questions using CDF→PMF comparison.

    Preferred approach: convert both model and community CDFs to discretized PMFs and
    compute 50 * E_c[ln(p_model / p_baseline)]. This mirrors the continuous log-score
    style and aligns scale with binary baselines. Falls back to a simple PDF estimate
    from declared percentiles only when CDFs are unavailable.

    Args:
        report: NumericReport-like object with `.prediction.cdf` and question `api_json`.

    Returns:
        Baseline score or None if cannot be calculated
    """
    global _NUMERIC_PMF_ATTEMPTS, _NUMERIC_PMF_SUCCESSES
    global _NUMERIC_FALLBACK_ATTEMPTS, _NUMERIC_FALLBACK_SUCCESSES
    try:
        # Try to obtain model CDF percentiles (list of objects with .percentile in [0,1])
        model_cdf_percentiles = None
        try:
            candidate_cdf = getattr(report.prediction, "cdf", None)
            # Validate the CDF looks like a sequence of percentile-like objects
            if isinstance(candidate_cdf, (list, tuple)) and len(candidate_cdf) >= 2:
                model_cdf_percentiles = candidate_cdf
            else:
                model_cdf_percentiles = None
        except Exception as e:
            logger.warning(
                f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: cannot compute model CDF: {e}"
            )

        # Extract community CDF from API JSON
        community_cdf = _extract_numeric_community_cdf(report.question)

        # If either CDF is missing or too short, fall back to a PDF-like approximation
        if not community_cdf or len(community_cdf) < 2 or model_cdf_percentiles is None:
            logger.info(
                f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: missing community/model CDF; using legacy PDF fallback"
            )
            try:
                _NUMERIC_FALLBACK_ATTEMPTS += 1
                # Log basic context for debugging
                if community_cdf is None:
                    logger.debug("Numeric: community_cdf is None")
                else:
                    logger.debug("Numeric: community_cdf length=%d", len(community_cdf))
                # Use declared percentiles if available to build interpolation over values
                declared = getattr(report.prediction, "declared_percentiles", None)
                if not declared and model_cdf_percentiles:
                    declared = model_cdf_percentiles[::40]

                if not declared:
                    return None

                values = [float(p.value) for p in declared]
                perc = [
                    float(getattr(p, "percentile", 0)) / (100.0 if getattr(p, "percentile", 1) > 1 else 1.0)
                    for p in declared
                ]
                if len(values) < 3:
                    return None

                # Interpolate CDF(value)->percentile and estimate local PDF near median
                cdf_func = interp1d(values, perc, bounds_error=False, fill_value=(0, 1), kind="linear")
                vmin, vmax = min(values), max(values)
                dx = (vmax - vmin) / 1000.0 if vmax > vmin else 1.0
                x0 = np.median(values)
                cdf_left = float(cdf_func(x0 - dx / 2))
                cdf_right = float(cdf_func(x0 + dx / 2))
                pdf = max(1e-10, (cdf_right - cdf_left) / dx)
                baseline_pdf = 1.0 / (vmax - vmin) if vmax > vmin else 1.0
                final_score = 50.0 * (math.log(pdf / baseline_pdf))
                final_score = max(min(final_score, 100.0), -100.0)
                _NUMERIC_FALLBACK_SUCCESSES += 1
                logger.info(
                    f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: baseline score {final_score:.2f} (legacy PDF fallback)"
                )
                return final_score
            except Exception:
                return None

        # Convert CDFs to PMFs and normalize
        _NUMERIC_PMF_ATTEMPTS += 1
        model_cdf_values = np.clip(
            np.array([float(p.percentile) for p in model_cdf_percentiles], dtype=float), 0.0, 1.0
        )
        model_pmf = np.diff(model_cdf_values)
        model_pmf = np.maximum(model_pmf, 0.0)
        if model_pmf.sum() <= 0:
            logger.warning(
                f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: model PMF degenerate"
            )
            return None
        model_pmf = model_pmf / model_pmf.sum()

        community_cdf_arr = np.clip(np.array(community_cdf, dtype=float), 0.0, 1.0)
        community_pmf = np.diff(community_cdf_arr)
        community_pmf = np.maximum(community_pmf, 0.0)
        if community_pmf.sum() <= 0:
            logger.warning(
                f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: community PMF degenerate"
            )
            return None
        community_pmf = community_pmf / community_pmf.sum()

        # Align lengths (guard, though both should be 200)
        m = min(len(model_pmf), len(community_pmf))
        model_pmf = model_pmf[:m]
        community_pmf = community_pmf[:m]

        # Expected baseline for continuous:
        # 100/2 * E_c[ ln( p_k / baseline_k ) ]
        open_upper = bool(getattr(report.question, "open_upper_bound", False))
        open_lower = bool(getattr(report.question, "open_lower_bound", False))
        open_count = int(open_upper) + int(open_lower)
        baseline = np.zeros(m, dtype=float)
        if m >= 1 and open_lower:
            baseline[0] = 0.05
        if m >= 2 and open_upper:
            baseline[-1] = 0.05
        remaining_mass = max(0.0, 1.0 - 0.05 * open_count)
        inner_bins = max(m - 2, 0)
        if inner_bins > 0:
            baseline[1:-1] = remaining_mass / inner_bins
        else:
            # Degenerate tiny grid: put everything into available bin(s)
            if m == 1:
                baseline[0] = 1.0
            elif m == 2:
                distribute = remaining_mass / 2.0
                baseline[0] += distribute
                baseline[1] += distribute
        eps = 1e-12
        terms = community_pmf * (np.log(np.maximum(model_pmf, eps) / np.maximum(baseline, eps)))
        final_score = float(50.0 * terms.sum())
        _NUMERIC_PMF_SUCCESSES += 1
        logger.info(
            f"Numeric Question {getattr(report.question, 'id_of_question', 'unknown')}: baseline score {final_score:.2f} (PMF-based vs community)"
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
