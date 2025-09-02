"""
Unit tests for numeric validation utilities.

Tests for percentile validation and processing functions extracted from main.py.
"""

from types import SimpleNamespace

import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from metaculus_bot.numeric_validation import (
    check_discrete_question_properties,
    sort_percentiles_by_value,
    validate_percentile_count_and_values,
)


def _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0, zero_point=None, cdf_size=None):
    return SimpleNamespace(
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        upper_bound=upper,
        lower_bound=lower,
        zero_point=zero_point,
        id_of_question=999,
        cdf_size=cdf_size,
    )


class TestPercentileValidation:
    """Test percentile validation functions."""

    def test_validate_percentile_count_and_values_success(self):
        """Test successful validation of correct percentiles."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.40, value=40.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        # Should not raise any exception
        validate_percentile_count_and_values(percentiles)

    def test_validate_percentile_count_wrong_count(self):
        """Test validation fails with wrong number of percentiles."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(percentiles)

        assert "Expected 8 declared percentiles" in str(exc_info.value)

    def test_validate_percentile_wrong_values(self):
        """Test validation fails with wrong percentile values."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.15, value=15.0),  # Wrong percentile (should be 0.10)
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.40, value=40.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(percentiles)

        assert "Expected percentile set" in str(exc_info.value)

    def test_sort_percentiles_by_value(self):
        """Test sorting percentiles by percentile value."""
        # Create unsorted percentiles
        percentiles = [
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.20, value=20.0),
        ]

        sorted_percentiles = sort_percentiles_by_value(percentiles)

        # Check they are sorted by percentile value
        expected_order = [0.05, 0.20, 0.60, 0.90]
        actual_order = [p.percentile for p in sorted_percentiles]

        assert actual_order == expected_order

    def test_check_discrete_question_properties_discrete(self):
        """Test discrete question detection."""
        question = _make_question(cdf_size=100)  # Not 201, so discrete
        question.zero_point = 1.0

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is True
        assert should_force_none is True

    def test_check_discrete_question_properties_continuous(self):
        """Test continuous question detection."""
        question = _make_question(cdf_size=201)  # 201, so continuous
        question.zero_point = None

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is False
        assert should_force_none is False

    def test_check_discrete_question_properties_zero_point_equals_lower_bound(self):
        """Test zero_point equals lower_bound case."""
        question = _make_question(lower=0.0, cdf_size=201)
        question.zero_point = 0.0  # Same as lower_bound

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is False
        assert should_force_none is True  # Should force zero_point to None

    def test_check_discrete_question_properties_no_cdf_size(self):
        """Test question with no cdf_size attribute."""
        question = _make_question()
        # Don't set cdf_size attribute at all
        if hasattr(question, "cdf_size"):
            delattr(question, "cdf_size")
        question.zero_point = None

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is False
        assert should_force_none is False
