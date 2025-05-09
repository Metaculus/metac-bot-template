import random
from forecasting_tools import ForecastBot, ReasonedPrediction, BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, NumericDistribution, PredictedOptionList

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
        # Generate a random CDF of 201 values between lower and upper bound
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        cdf = [random.uniform(lower, upper) for _ in range(201)]
        cdf.sort()
        prediction = NumericDistribution(declared_percentiles=cdf)
        reasoning = f"Random CDF between {lower} and {upper}"
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning) 