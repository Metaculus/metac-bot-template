import random
from forecasting_tools import (
    ForecastBot, ReasonedPrediction, BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, NumericDistribution, PredictedOptionList,
    GeneralLlm, PredictionExtractor, clean_indents
)
from tools import get_related_markets_from_adjacent_news, get_web_search_results_from_openrouter, fermi_estimate_with_llm
from datetime import datetime

class AdjacentNewsRelatedMarketsBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = "AdjacentNewsRelatedMarketsBot"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        return get_related_markets_from_adjacent_news(question.question_text)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant found related markets info:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant found related markets info:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(question, 'upper_bound') else ""
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}

            Your research assistant found related markets info:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

class OpenRouterWebSearchBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = "OpenRouterWebSearchBot"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        return get_web_search_results_from_openrouter(question.question_text)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant found web search results:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant found web search results:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(question, 'upper_bound') else ""
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}

            Your research assistant found web search results:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

class CombinedWebAndAdjacentNewsBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = "CombinedWebAndAdjacentNewsBot"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        web_results = get_web_search_results_from_openrouter(question.question_text)
        related_markets = get_related_markets_from_adjacent_news(question.question_text)
        return f"Web search results:\n{web_results}\n\nRelated markets info:\n{related_markets}"

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant found web search results and related markets info:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant found web search results and related markets info:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(question, 'upper_bound') else ""
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}

            Your research assistant found web search results and related markets info:
            {research}

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

class FermiEstimationBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = "FermiEstimationBot"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    async def run_research(self, question):
        return ""  # Fermi estimation bot does not use external research

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        # Use Fermi estimation to answer the binary question
        reasoning = fermi_estimate_with_llm(question.question_text, self.llm_model, self.llm_temperature)
        if hasattr(reasoning, '__await__'):
            import asyncio
            reasoning = await reasoning
        prediction = PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        reasoning = fermi_estimate_with_llm(question.question_text, self.llm_model, self.llm_temperature)
        if hasattr(reasoning, '__await__'):
            import asyncio
            reasoning = await reasoning
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        reasoning = fermi_estimate_with_llm(question.question_text, self.llm_model, self.llm_temperature)
        if hasattr(reasoning, '__await__'):
            import asyncio
            reasoning = await reasoning
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning) 