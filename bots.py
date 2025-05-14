from forecasting_tools import (
    ForecastBot, ReasonedPrediction, BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, NumericDistribution, PredictedOptionList,
    GeneralLlm, PredictionExtractor, clean_indents
)
from tools import get_related_markets_from_adjacent_news, get_web_search_results_from_openrouter, fermi_estimate_with_llm, get_perplexity_research_from_openrouter
from datetime import datetime
import traceback


class AdjacentNewsRelatedMarketsBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = f"AdjacentNewsRelatedMarketsBot | {llm_model}"
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
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
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
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
            question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
            question, 'upper_bound') else ""
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
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class OpenRouterWebSearchBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = f"OpenRouterWebSearchBot | {llm_model}"
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
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
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
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
            question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
            question, 'upper_bound') else ""
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
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class CombinedWebAndAdjacentNewsBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2, predictions_per_research_report=1):
        super().__init__(predictions_per_research_report=predictions_per_research_report)
        self.name = f"CombinedWebAndAdjacentNewsBot | {llm_model}"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        web_results = get_web_search_results_from_openrouter(
            question.question_text)
        related_markets = get_related_markets_from_adjacent_news(
            question.question_text)
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
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
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
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
            question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
            question, 'upper_bound') else ""
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
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class FermiEstimationBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = f"FermiEstimationBot | {llm_model}"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

    async def run_research(self, question):
        return ""  # Fermi estimation bot does not use external research

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        # Use Fermi estimation to answer the binary question
        reasoning = await fermi_estimate_with_llm(question.question_text, self.llm_model, self.llm_temperature)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        reasoning = await fermi_estimate_with_llm(question.question_text, self.llm_model, self.llm_temperature)
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await fermi_estimate_with_llm(question.question_text, self.llm_model, self.llm_temperature)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class PerplexityRelatedMarketsBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2, predictions_per_research_report=1):
        super().__init__(predictions_per_research_report=predictions_per_research_report)
        self.name = f"PerplexityRelatedMarketsBot | {llm_model}"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        # Use Perplexity via OpenRouter (sonar-reasoning) for web research
        web_results = await get_perplexity_research_from_openrouter(question.question_text, model_name="openrouter/perplexity/sonar-reasoning")
        related_markets = get_related_markets_from_adjacent_news(
            question.question_text)
        return f"Web search results (Perplexity Sonar Reasoning):\n{web_results}\n\nRelated markets info:\n{related_markets}"

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
            (e) Write out the question again, and acknolwedge that if the No outcome is more likely your answers should be closer to 0 and if the Yes outcome is more likely your answers should be closer to 100.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
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
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
            question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
            question, 'upper_bound') else ""
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
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class OpenSearchPerpAdjMarkets(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = f"OpenSearchPerpAdjMarkets | {llm_model}"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        web_results = get_web_search_results_from_openrouter(
            question.question_text)
        perp_results = await get_perplexity_research_from_openrouter(
            question.question_text, model_name="openrouter/perplexity/sonar-reasoning"
        )
        related_markets = get_related_markets_from_adjacent_news(
            question.question_text)
        return (
            f"Web search results (OpenRouter):\n{web_results}\n\n"
            f"Web search results (Perplexity Sonar Reasoning):\n{perp_results}\n\n"
            f"Related markets info:\n{related_markets}"
        )

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

            Your research assistant found web search results (OpenRouter), web search results (Perplexity Sonar Reasoning), and related markets info:
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
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
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

            Your research assistant found web search results (OpenRouter), web search results (Perplexity Sonar Reasoning), and related markets info:
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
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
            question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
            question, 'upper_bound') else ""
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

            Your research assistant found web search results (OpenRouter), web search results (Perplexity Sonar Reasoning), and related markets info:
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
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class FermiResearchFirstBot(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = f"FermiResearchFirst | {llm_model}"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        web_results = get_web_search_results_from_openrouter(
            question.question_text)
        perp_results = await get_perplexity_research_from_openrouter(
            question.question_text, model_name="openrouter/perplexity/sonar-reasoning"
        )
        related_markets = get_related_markets_from_adjacent_news(
            question.question_text)
        return (
            f"Web search results (OpenRouter):\n{web_results}\n\n"
            f"Web search results (Perplexity Sonar Reasoning):\n{perp_results}\n\n"
            f"Related markets info:\n{related_markets}"
        )

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster skilled in Fermi estimation (back-of-the-envelope reasoning).
            Your task is to answer the following question using a Fermi estimation approach, making use of the research provided below.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Research provided:
            {research}

            Instructions:
            - Break the problem down into smaller, logical components.
            - For each component, make explicit, reasonable guesses or estimates, and clearly state your assumptions.
            - Document every step and calculation in detail, showing your work.
            - Do not make any unstated assumptions; explain your reasoning for each guess.
            - Proceed step by step, combining your estimates to reach a final answer.
            - At the end, summarize your Fermi estimate and show the final calculation.
            - If the No outcome is more likely, your answer should be closer to 0; if Yes, closer to 100.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster skilled in Fermi estimation (back-of-the-envelope reasoning).
            Your task is to answer the following question using a Fermi estimation approach, making use of the research provided below.

            Question:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Research provided:
            {research}

            Instructions:
            - Break the problem down into smaller, logical components.
            - For each component, make explicit, reasonable guesses or estimates, and clearly state your assumptions.
            - Document every step and calculation in detail, showing your work.
            - Do not make any unstated assumptions; explain your reasoning for each guess.
            - Proceed step by step, combining your estimates to reach a final answer.
            - At the end, summarize your Fermi estimate and show the final calculation.
            - Good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.llm.invoke(prompt)
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        lower = getattr(question, 'lower_bound', 0)
        upper = getattr(question, 'upper_bound', 100)
        lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
            question, 'lower_bound') else ""
        upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
            question, 'upper_bound') else ""
        prompt = clean_indents(
            f"""
            You are a professional forecaster skilled in Fermi estimation (back-of-the-envelope reasoning).
            Your task is to answer the following question using a Fermi estimation approach, making use of the research provided below.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}

            Research provided:
            {research}

            {lower_bound_message}
            {upper_bound_message}

            Instructions:
            - Break the problem down into smaller, logical components.
            - For each component, make explicit, reasonable guesses or estimates, and clearly state your assumptions.
            - Document every step and calculation in detail, showing your work.
            - Do not make any unstated assumptions; explain your reasoning for each guess.
            - Proceed step by step, combining your estimates to reach a final answer.
            - At the end, summarize your Fermi estimate and show the final calculation.
            - Good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

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
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)


class FermiWithSearchControl(ForecastBot):
    def __init__(self, llm_model: str = "gpt-4o-mini", llm_temperature: float = 0.2):
        super().__init__()
        self.name = f"FermiWithSearchControl | {llm_model}"
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm = GeneralLlm(model=llm_model, temperature=llm_temperature)

    async def run_research(self, question):
        print("[FermiWithSearchControl] Entering run_research")
        try:
            print("[FermiWithSearchControl] Step 1: Building Fermi prompt")
            # Step 1: Fermi estimation and search term generation
            fermi_prompt = clean_indents(
                f"""
                You are a professional forecaster skilled in Fermi estimation (back-of-the-envelope reasoning).
                Your task is to answer the following question using a Fermi estimation approach.

                {question.question_text}

                Instructions:
                - Break the problem down into smaller, logical components.
                - For each component, make explicit, reasonable guesses or estimates, and clearly state your assumptions.
                - Document every step and calculation in detail, showing your work.
                - Do not make any unstated assumptions; explain your reasoning for each guess.
                - Proceed step by step, combining your estimates to reach a final answer.
                - At the end, summarize your Fermi estimate and show the final calculation.
                - After your Fermi estimate, list between 1 and 4 web search queries (as an array of strings) that would help you reduce your uncertainty or check the weakest parts of your reasoning. If you don't need any, return an empty array [].
                - Format your search queries as a valid Python list of strings, e.g. [\"search 1\", \"search 2\"].
                """
            )
            print("[FermiWithSearchControl] Step 1: Prompt built, invoking LLM...")
            fermi_response = await self.llm.invoke(fermi_prompt)
            print(
                "[FermiWithSearchControl] Step 1: Fermi response received:", fermi_response)

            # Extract search queries (as a Python list) from the LLM output
            import ast
            import re
            search_queries = []
            print(
                "[FermiWithSearchControl] Step 2: Extracting search queries from response...")
            match = re.search(r'\[.*?\]', fermi_response, re.DOTALL)
            if match:
                try:
                    search_queries = ast.literal_eval(match.group(0))
                    if not isinstance(search_queries, list):
                        print(
                            "[FermiWithSearchControl] Step 2: Search queries not a list, resetting to []")
                        search_queries = []
                except Exception as e:
                    print(
                        "[FermiWithSearchControl] Error parsing search queries:", e)
                    traceback.print_exc()
                    search_queries = []
            print(
                "[FermiWithSearchControl] Step 2: Search queries extracted:", search_queries)

            # Step 2: Run OpenRouter web search for each query
            search_results = []
            print(
                "[FermiWithSearchControl] Step 3: Running web search for each query...")
            for query_str in search_queries:
                try:
                    print(
                        f"[FermiWithSearchControl] Step 3: Running web search for query: '{query_str}'")
                    result = get_web_search_results_from_openrouter(query_str)
                    print(
                        f"[FermiWithSearchControl] Step 3: Search result for '{query_str}':", result)
                except Exception as e:
                    print(
                        f"[FermiWithSearchControl] Error during web search for query '{query_str}':", e)
                    traceback.print_exc()
                    result = f"Error: {e}"
                search_results.append({"query": query_str, "result": result})
            print("[FermiWithSearchControl] Step 3: All search results:",
                  search_results)

            # Step 3: Aggregate all research into a markdown string (no question text)
            search_results_str = "\n\n".join([
                f"Search: {item['query']}\nResult: {item['result']}" for item in search_results
            ])
            research_md = clean_indents(f"""
                ## Fermi Estimate and Reasoning
                {fermi_response}

                ## Search Queries
                {search_queries}

                ## Web Search Results
                {search_results_str}
            """)
            print(
                "[FermiWithSearchControl] Step 4: Exiting run_research with research_md:", research_md)
            return research_md
        except Exception as e:
            print("[FermiWithSearchControl] Exception in run_research:", e)
            traceback.print_exc()
            return f"Exception in run_research: {e}"

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        print("[FermiWithSearchControl] Entering _run_forecast_on_binary")
        try:
            prompt = clean_indents(
                f"""
                You previously made a Fermi estimate for the following question:
                {question.question_text}

                Here is your original Fermi estimate and reasoning, followed by the search queries you generated for the parts you were most uncertain about, and the results of those searches:

                {research}

                Now, update your Fermi estimate and reasoning in light of this new information. Clearly state any changes or updates you are making.

                IMPORTANT: Your final answer must be an integer percentage between 0 and 100 (e.g., \"Probability: 24%\"). Do not use decimals or fractions. Always round to the nearest integer.

                At the end, write your final answer as: \"Probability: ZZ%\", 0-100
                """
            )
            reasoning = await self.llm.invoke(prompt)
            print("[FermiWithSearchControl] Reasoning:", reasoning)
            prediction = PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0)
            print(
                "[FermiWithSearchControl] Exiting _run_forecast_on_binary with prediction:", prediction)
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            print("[FermiWithSearchControl] Exception in _run_forecast_on_binary:", e)
            traceback.print_exc()
            return ReasonedPrediction(prediction_value=0.0, reasoning=f"Exception: {e}")

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        print("[FermiWithSearchControl] Entering _run_forecast_on_multiple_choice")
        try:
            prompt = clean_indents(
                f"""
                You previously made a Fermi estimate for the following question:
                {question.question_text}

                Here is your original Fermi estimate and reasoning, followed by the search queries you generated for the parts you were most uncertain about, and the results of those searches:

                {research}

                Now, update your Fermi estimate and reasoning in light of this new information. Clearly state any changes or updates you are making.

                IMPORTANT: Your final answer must be an integer percentage between 0 and 100 (e.g., \"Probability: 24%\"). Do not use decimals or fractions. Always round to the nearest integer.

                At the end, write your final probabilities for the N options in this order {question.options} as:
                Option_A: Probability_A
                Option_B: Probability_B
                ...
                Option_N: Probability_N
                """
            )
            reasoning = await self.llm.invoke(prompt)
            prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options)
            print(
                "[FermiWithSearchControl] Exiting _run_forecast_on_multiple_choice with prediction:", prediction)
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            print(
                "[FermiWithSearchControl] Exception in _run_forecast_on_multiple_choice:", e)
            traceback.print_exc()
            return ReasonedPrediction(prediction_value=[], reasoning=f"Exception: {e}")

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        print("[FermiWithSearchControl] Entering _run_forecast_on_numeric")
        try:
            lower = getattr(question, 'lower_bound', 0)
            upper = getattr(question, 'upper_bound', 100)
            lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
                question, 'lower_bound') else ""
            upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
                question, 'upper_bound') else ""
            prompt = clean_indents(
                f"""
                You previously made a Fermi estimate for the following question:
                {question.question_text}

                Here is your original Fermi estimate and reasoning, followed by the search queries you generated for the parts you were most uncertain about, and the results of those searches:

                {research}

                {lower_bound_message}
                {upper_bound_message}

                Now, update your Fermi estimate and reasoning in light of this new information. Clearly state any changes or updates you are making.

                IMPORTANT: Your final answer must be an integer percentage between 0 and 100 (e.g., \"Probability: 24%\"). Do not use decimals or fractions. Always round to the nearest integer.

                At the end, write your final answer as:
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
            prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question)
            print(
                "[FermiWithSearchControl] Exiting _run_forecast_on_numeric with prediction:", prediction)
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            print("[FermiWithSearchControl] Exception in _run_forecast_on_numeric:", e)
            traceback.print_exc()
            return ReasonedPrediction(prediction_value=None, reasoning=f"Exception: {e}")
