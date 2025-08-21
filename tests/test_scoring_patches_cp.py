import types
from types import SimpleNamespace

import numpy as np

from metaculus_bot.scoring_patches import calculate_multiple_choice_baseline_score, calculate_numeric_baseline_score


def _make_mc_question(options, cp_probs):
    # Build api_json with forecast_values for MC
    latest = {
        "forecast_values": list(cp_probs),
    }
    api_q = {"aggregations": {"recency_weighted": {"latest": latest}}}
    question = SimpleNamespace(
        id_of_question=123,
        options=list(options),
        api_json={"question": api_q},
    )
    return question


def _make_mc_prediction(options, probs):
    # Build a minimal predicted_options list with option and probability
    opts = [SimpleNamespace(option=o, probability=float(p)) for o, p in zip(options, probs)]
    return SimpleNamespace(predicted_options=opts)


def test_mc_scoring_prefers_matching_distribution():
    options = ["A", "B", "C"]
    cp = [0.2, 0.3, 0.5]
    q = _make_mc_question(options, cp)

    # Matching prediction
    pred_match = _make_mc_prediction(options, cp)
    rep_match = SimpleNamespace(question=q, prediction=pred_match)
    score_match = calculate_multiple_choice_baseline_score(rep_match)

    # Mismatched prediction (swap mass)
    pred_mismatch = _make_mc_prediction(options, [0.5, 0.3, 0.2])
    rep_mismatch = SimpleNamespace(question=q, prediction=pred_mismatch)
    score_mismatch = calculate_multiple_choice_baseline_score(rep_mismatch)

    assert score_match is not None and score_mismatch is not None
    assert score_match > score_mismatch


def _make_numeric_question(cdf_values, range_min=0.0, range_max=1.0, zero_point=None):
    latest = {"forecast_values": list(cdf_values)}
    api_q = {
        "aggregations": {"recency_weighted": {"latest": latest}},
        "scaling": {"range_min": float(range_min), "range_max": float(range_max), "zero_point": zero_point},
    }
    return SimpleNamespace(id_of_question=456, api_json={"question": api_q})


class _Perc:
    __slots__ = ("value", "percentile")

    def __init__(self, value: float, percentile: float):
        self.value = value
        self.percentile = percentile


class _NumericPred:
    def __init__(self, x, cdf):
        self._cdf = [_Perc(v, p) for v, p in zip(x, cdf)]

    @property
    def cdf(self):
        return self._cdf


def test_numeric_scoring_prefers_matching_distribution():
    # Community CDF: linear 0..1 across 201 points
    cp_cdf = np.linspace(0.0, 1.0, 201)
    q = _make_numeric_question(cp_cdf, 0.0, 1.0, None)

    # Model pred 1: matches community
    x_axis = np.linspace(0.0, 1.0, 201)
    pred_match = _NumericPred(x_axis, cp_cdf)
    rep_match = SimpleNamespace(question=q, prediction=pred_match)
    score_match = calculate_numeric_baseline_score(rep_match)

    # Model pred 2: skewed distribution (cdf^2)
    pred_mismatch = _NumericPred(x_axis, np.square(cp_cdf))
    rep_mismatch = SimpleNamespace(question=q, prediction=pred_mismatch)
    score_mismatch = calculate_numeric_baseline_score(rep_mismatch)

    assert score_match is not None and score_mismatch is not None
    assert score_match > score_mismatch
