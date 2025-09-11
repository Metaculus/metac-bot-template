from typing import Any, Dict, List

from metaculus_bot.correlation_analysis import CorrelationAnalyzer


class FakeQuestion:
    def __init__(self, qid: int):
        self.id_of_question = qid
        self.page_url = f"https://metaculus.com/questions/{qid}"
        self.community_prediction_at_access_time = None


class FakeReport:
    def __init__(self, qid: int, prediction: float, score: float, cost: float):
        self.question = FakeQuestion(qid)
        self.prediction = prediction  # float for binary
        self.expected_baseline_score = score
        self.price_estimate = cost
        self.explanation = ""


class FakeBenchmark:
    def __init__(self, name: str, model_path: str):
        self.name = name
        self.total_cost = 0.01
        self.forecast_reports = [FakeReport(42, 0.6, 12.3, 0.001)]
        # Emulate the llms structure used for identifier extraction
        self.forecast_bot_config: Dict[str, Any] = {
            "aggregation_strategy": "mean",
            "llms": {
                "default": {"model": model_path},
                "forecasters": [
                    {"model": model_path},
                ],
            },
        }


def build_analyzer_with_models(names_and_paths: List[tuple[str, str]]) -> CorrelationAnalyzer:
    benches = [FakeBenchmark(n, p) for n, p in names_and_paths]
    analyzer = CorrelationAnalyzer()
    analyzer.add_benchmark_results(benches)  # type: ignore[arg-type]
    return analyzer


def test_exclude_models_by_substring_simple():
    analyzer = build_analyzer_with_models(
        [
            ("qwen3-235b", "openrouter/qwen/qwen3-235b-a22b-thinking-2507"),
            ("o3", "openrouter/openai/o3"),
            ("grok-4", "openrouter/x-ai/grok-4"),
            ("gemini-2.5-pro", "openrouter/google/gemini-2.5-pro"),
        ]
    )

    before = analyzer.get_model_names()
    assert set(before) >= {"qwen3-235b", "o3", "grok-4", "gemini-2.5-pro"}

    analyzer.filter_models_inplace(exclude=["grok-4", "gemini-2.5-pro"])  # remove two
    after = analyzer.get_model_names()
    assert set(after) == {"qwen3-235b", "o3"}

    # Predictions should also be pruned to the remaining models (one report each)
    preds_models = {p.model_name for p in analyzer.predictions}
    assert preds_models == {"qwen3-235b", "o3"}


def test_exclude_by_model_path_substring():
    analyzer = build_analyzer_with_models(
        [
            ("grok-4", "openrouter/x-ai/grok-4"),
            ("o3", "openrouter/openai/o3"),
        ]
    )

    # Exclude via a substring of the model path, not the clean name
    analyzer.filter_models_inplace(exclude=["x-ai/grok-4"])  # substring match
    remaining = analyzer.get_model_names()
    assert remaining == ["o3"]


def test_include_only_subset():
    analyzer = build_analyzer_with_models(
        [
            ("qwen3-235b", "openrouter/qwen/qwen3-235b-a22b-thinking-2507"),
            ("o3", "openrouter/openai/o3"),
            ("grok-4", "openrouter/x-ai/grok-4"),
        ]
    )

    analyzer.filter_models_inplace(include=["o3", "qwen3-235b"])  # include only two
    names = set(analyzer.get_model_names())
    assert names == {"qwen3-235b", "o3"}
