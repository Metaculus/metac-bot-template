import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.tail_widening import widen_declared_percentiles


def _make_question(lower=0.0, upper=100.0, open_lower=False, open_upper=False):
    return SimpleNamespace(
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        id_of_question=777,
        page_url="https://ex/q/777",
    )


def _eleven_percentiles(values: list[float]) -> list[Percentile]:
    ps = [0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975]
    assert len(values) == len(ps)
    return [Percentile(percentile=p, value=v) for p, v in zip(ps, values)]


class TestTailWideningUnit:
    def test_noop_when_k_is_one(self):
        q = _make_question(0.0, 100.0, False, False)
        base = _eleven_percentiles([5, 6, 8, 12, 20, 50, 80, 90, 94, 97, 98])
        out = widen_declared_percentiles(base, q, k_tail=1.0, tail_start=0.2, span_floor_gamma=0.0)
        assert [p.value for p in out] == [p.value for p in base]
        assert next(p.value for p in out if math.isclose(p.percentile, 0.5)) == 50

    def test_closed_bounds_widening_increases_tail_spans(self):
        q = _make_question(0.0, 100.0, False, False)
        base = _eleven_percentiles([5, 6, 8, 12, 20, 50, 80, 90, 94, 97, 98])
        p = widen_declared_percentiles(base, q, k_tail=1.3, tail_start=0.2, span_floor_gamma=0.0)

        def _get(ps, target):
            return next(x.value for x in ps if math.isclose(x.percentile, target))

        # Median unchanged
        assert math.isclose(_get(p, 0.5), 50.0, rel_tol=0, abs_tol=1e-9)

        # Tail spans increased
        base_p025, base_p05, base_p10 = 5.0, 6.0, 8.0
        base_p10 - base_p05
        base_low_span_outer = base_p05 - base_p025
        new_low_span_inner = _get(p, 0.10) - _get(p, 0.05)
        new_low_span_outer = _get(p, 0.05) - _get(p, 0.025)
        assert new_low_span_outer > base_low_span_outer - 1e-12
        # Inner span may adjust slightly; ensure it's still positive
        assert new_low_span_inner > 0.0

        base_p90, base_p95, base_p975 = 94.0, 97.0, 98.0
        base_p95 - base_p90
        base_up_span_outer = base_p975 - base_p95
        new_up_span_inner = _get(p, 0.95) - _get(p, 0.90)
        new_up_span_outer = _get(p, 0.975) - _get(p, 0.95)
        assert new_up_span_outer > base_up_span_outer - 1e-12
        assert new_up_span_inner > 0.0

        # Bounds respected and strictly increasing
        vals = [pp.value for pp in p]
        assert all(0.0 <= v <= 100.0 for v in vals)
        assert all(b > a for a, b in zip(vals, vals[1:])), vals

    @pytest.mark.parametrize(
        "open_lower, open_upper",
        [
            (False, True),  # lower-bounded
            (True, False),  # upper-bounded
        ],
    )
    def test_semibounded_open_cases(self, open_lower: bool, open_upper: bool):
        L, U = 0.0, 100.0
        q = _make_question(L, U, open_lower, open_upper)
        base = _eleven_percentiles([10, 11, 13, 18, 30, 50, 70, 85, 92, 96, 98])
        out = widen_declared_percentiles(base, q, k_tail=1.3, tail_start=0.2, span_floor_gamma=0.0)
        # Median unchanged; monotone; values clamped within numeric range
        assert math.isclose(next(p.value for p in out if math.isclose(p.percentile, 0.5)), 50.0, abs_tol=1e-9)
        vals = [pp.value for pp in out]
        assert all(L <= v <= U for v in vals)
        assert all(b > a for a, b in zip(vals, vals[1:])), vals

    def test_span_floor_enforced(self):
        q = _make_question(0.0, 100.0, False, False)
        # Intentionally compressed outer tails vs inner spans
        base = _eleven_percentiles([10, 10.4, 12.0, 16.0, 30.0, 50.0, 70.0, 85.0, 92.0, 95.0, 96.0])
        out = widen_declared_percentiles(base, q, k_tail=1.0, tail_start=0.2, span_floor_gamma=1.0)

        def _get(ps, target):
            return next(x.value for x in ps if math.isclose(x.percentile, target))

        assert (_get(out, 0.05) - _get(out, 0.025)) >= (_get(out, 0.10) - _get(out, 0.05)) - 1e-12
        assert (_get(out, 0.975) - _get(out, 0.95)) >= (_get(out, 0.95) - _get(out, 0.90)) - 1e-12
        vals = [pp.value for pp in out]
        assert all(b > a for a, b in zip(vals, vals[1:])), vals


class TestTailWideningIntegration:
    @pytest.mark.asyncio
    async def test_integration_discrete_enabled(self, monkeypatch):
        # Enable tail widening globally for this test
        from metaculus_bot import numeric_config as cfg

        monkeypatch.setattr(cfg, "TAIL_WIDENING_ENABLE", True, raising=False)
        monkeypatch.setattr(cfg, "TAIL_WIDEN_K_TAIL", 1.35, raising=False)
        monkeypatch.setattr(cfg, "TAIL_WIDEN_TAIL_START", 0.2, raising=False)
        monkeypatch.setattr(cfg, "TAIL_WIDEN_SPAN_FLOOR_GAMMA", 1.0, raising=False)

        from main import TemplateForecaster

        bot = TemplateForecaster(
            llms={
                "default": MagicMock(),
                "parser": MagicMock(),
                "researcher": MagicMock(),
                "summarizer": MagicMock(),
            },
            publish_reports_to_metaculus=False,
        )

        # Minimal numeric question with discrete cdf_size but we still expect 201-point PCHIP CDF
        nq = SimpleNamespace(
            question_text="num?",
            background_info="",
            resolution_criteria="",
            fine_print="",
            unit_of_measure=None,
            open_upper_bound=False,
            open_lower_bound=False,
            lower_bound=0.0,
            upper_bound=100.0,
            page_url="url",
            zero_point=None,
            id_of_question=4242,
            cdf_size=101,
        )

        # Compressed tails baseline
        baseline_vals = [10, 11, 13, 18, 30, 50, 70, 85, 92, 95, 96]
        declared = _eleven_percentiles(baseline_vals)

        # Patch structure_output to return our declared percentiles
        async def _fake_structure_output(*args, **kwargs):
            return declared

        llm = MagicMock()
        llm.model = "m"
        llm.invoke = AsyncMock(return_value="rationale")

        with patch("main.structure_output", _fake_structure_output):
            result = await bot._run_forecast_on_numeric(nq, "", llm)

        # Ensure declared percentiles changed (tails widened)
        out_decl = [p.value for p in result.prediction_value.declared_percentiles]  # type: ignore[attr-defined]
        assert out_decl[0] <= baseline_vals[0] - 1e-6 or out_decl[-1] >= baseline_vals[-1] + 1e-6

        # CDF is produced and monotone
        cdf = result.prediction_value.cdf  # type: ignore[attr-defined]
        probs = [p.percentile for p in cdf]
        assert len(cdf) == 201
        assert all(a <= b for a, b in zip(probs[:-1], probs[1:]))
