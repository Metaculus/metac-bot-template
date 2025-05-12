import random
from forecasting_tools import ForecastBot, ReasonedPrediction, BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, NumericDistribution, PredictedOptionList
from tools import get_related_markets_from_adjacent_news, get_web_search_results_from_openrouter

class DummyBot(ForecastBot):
    def __init__(self):
        super().__init__()
        self.name = "DummyBot"

    async def run_research(self, question):
        return ""  # No research

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prob = random.uniform(0, 1)
        reasoning = f"Random guess: {prob:.2f}"
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        n = len(question.options)
        probs = [random.random() for _ in range(n)]
        total = sum(probs)
        norm_probs = [p / total for p in probs]
        prediction = {option: prob for option, prob in zip(question.options, norm_probs)}
        reasoning = f"Random guess: {prediction}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        cdf = [random.uniform(lower, upper) for _ in range(201)]
        cdf.sort()
        prediction = NumericDistribution(declared_percentiles=cdf)
        reasoning = f"Random CDF between {lower} and {upper}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

class AdjacentNewsRelatedMarketsBot(ForecastBot):
    def __init__(self):
        super().__init__()
        self.name = "AdjacentNewsRelatedMarketsBot"

    async def run_research(self, question):
        # Use the Adjacent News tool to get related markets
        return get_related_markets_from_adjacent_news(question.question_text)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prob = random.uniform(0, 1)
        reasoning = f"Related markets info:\n{research}\nRandom guess: {prob:.2f}"
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        n = len(question.options)
        probs = [random.random() for _ in range(n)]
        total = sum(probs)
        norm_probs = [p / total for p in probs]
        prediction = {option: prob for option, prob in zip(question.options, norm_probs)}
        reasoning = f"Related markets info:\n{research}\nRandom guess: {prediction}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        cdf = [random.uniform(lower, upper) for _ in range(201)]
        cdf.sort()
        prediction = NumericDistribution(declared_percentiles=cdf)
        reasoning = f"Related markets info:\n{research}\nRandom CDF between {lower} and {upper}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

class OpenRouterWebSearchBot(ForecastBot):
    def __init__(self):
        super().__init__()
        self.name = "OpenRouterWebSearchBot"

    async def run_research(self, question):
        # Use the OpenRouter web search tool to get relevant news/info
        return get_web_search_results_from_openrouter(question.question_text)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prob = random.uniform(0, 1)
        reasoning = f"Web search info:\n{research}\nRandom guess: {prob:.2f}"
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        n = len(question.options)
        probs = [random.random() for _ in range(n)]
        total = sum(probs)
        norm_probs = [p / total for p in probs]
        prediction = {option: prob for option, prob in zip(question.options, norm_probs)}
        reasoning = f"Web search info:\n{research}\nRandom guess: {prediction}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        cdf = [random.uniform(lower, upper) for _ in range(201)]
        cdf.sort()
        prediction = NumericDistribution(declared_percentiles=cdf)
        reasoning = f"Web search info:\n{research}\nRandom CDF between {lower} and {upper}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

class CombinedWebAndAdjacentNewsBot(ForecastBot):
    def __init__(self):
        super().__init__()
        self.name = "CombinedWebAndAdjacentNewsBot"

    async def run_research(self, question):
        # Use both OpenRouter web search and Adjacent News related markets
        web_results = get_web_search_results_from_openrouter(question.question_text)
        related_markets = get_related_markets_from_adjacent_news(question.question_text)
        return f"{web_results}\n\n{related_markets}"

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prob = random.uniform(0, 1)
        reasoning = f"Web + Related markets info:\n{research}\nRandom guess: {prob:.2f}"
        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        n = len(question.options)
        probs = [random.random() for _ in range(n)]
        total = sum(probs)
        norm_probs = [p / total for p in probs]
        prediction = {option: prob for option, prob in zip(question.options, norm_probs)}
        reasoning = f"Web + Related markets info:\n{research}\nRandom guess: {prediction}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        cdf = [random.uniform(lower, upper) for _ in range(201)]
        cdf.sort()
        prediction = NumericDistribution(declared_percentiles=cdf)
        reasoning = f"Web + Related markets info:\n{research}\nRandom CDF between {lower} and {upper}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning) 