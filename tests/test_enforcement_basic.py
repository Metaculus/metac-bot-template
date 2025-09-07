import types

from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.cluster_processing import compute_cluster_parameters
from metaculus_bot.numeric_validation import detect_unit_mismatch


def _fake_question(lower: float, upper: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=False,
        open_upper_bound=False,
        id_of_question=123,
        page_url="https://example/q/123",
    )


def test_unit_mismatch_detector_flags_tiny_span():
    # Range is large (1e9); values are nearly identical and tiny → mismatch
    q = _fake_question(0.0, 1_000_000_000.0)
    plist = [Percentile(percentile=p, value=1.0) for p in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95]]

    mismatch, reason = detect_unit_mismatch(plist, q)
    assert mismatch is True
    assert "span_ratio" in reason or "near-duplicate" in reason or "tiny" in reason


def test_cluster_parameters_use_span_based_delta():
    # Small range (1e6) but substantial span (1e5) should yield spread >= 0.02 * span
    range_size = 1_000_000.0
    span = 100_000.0
    count_like = True

    value_eps, base_delta, spread_delta = compute_cluster_parameters(range_size, count_like, span)

    # Base delta is range * 1e-6 = 1.0; span-based is 2000.0
    assert base_delta >= 1.0
    assert spread_delta >= 2000.0
