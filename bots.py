from forecasting_tools import (
    ForecastBot, ReasonedPrediction, BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, NumericDistribution, PredictedOptionList,
    GeneralLlm, PredictionExtractor, clean_indents
)
from tools import get_related_markets_from_adjacent_news, get_web_search_results_from_openrouter, fermi_estimate_with_llm, get_perplexity_research_from_openrouter, log_report_summary_returning_str, get_related_markets_raw, format_markets, IntegerExtractor, FactsExtractor, FollowUpQuestionsExtractor
from datetime import datetime
import traceback

# OpenRouter model names
CLAUDE_SONNET = "openrouter/anthropic/claude-3.7-sonnet"
PERPLEXITY_SONAR = "openrouter/perplexity/sonar-reasoning"

PROBABILITY_FINAL_ANSWER_LINE = (
    "Before giving your final answer, rewrite the question as a probability statement (e.g., "
    "\"What is the probability that [event] will happen?\"), making sure it matches the outcome you are forecasting. "
    "Then, the last thing you write is your final answer as: \"Probability: ZZ%\", 0-100 (no decimals, do not include a space between the number and the % sign)."
)


class AdjacentNewsRelatedMarketsBot(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm]):
        super().__init__(llms=llms)

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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class OpenRouterWebSearchBot(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm]):
        super().__init__(llms=llms)

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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class CombinedWebAndAdjacentNewsBot(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm], predictions_per_research_report=1):
        super().__init__(llms=llms, predictions_per_research_report=predictions_per_research_report)

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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class FermiEstimationBot(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm]):
        super().__init__(llms=llms)

    async def run_research(self, question):
        return ""  # Fermi estimation bot does not use external research

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        # Use Fermi estimation to answer the binary question
        reasoning = await fermi_estimate_with_llm(question.question_text, self.get_llm())
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        reasoning = await fermi_estimate_with_llm(question.question_text, self.get_llm())
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await fermi_estimate_with_llm(question.question_text, self.get_llm())
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class PerplexityRelatedMarketsBot(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm], predictions_per_research_report=1):
        super().__init__(llms=llms, predictions_per_research_report=predictions_per_research_report)

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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            (e) Write out the question again, and acknolwedge that if the No outcome is more likely your answers should be closer to 0 and if the Yes outcome is more likely your answers should be closer to 100.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class OpenSearchPerpAdjMarkets(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm]):
        super().__init__(llms=llms)

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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

            Today is {datetime.now().strftime('%Y-%m-%d')}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
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

            IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

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
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class FermiResearchFirstBot(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm]):
        super().__init__(llms=llms)

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

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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
        reasoning = await self.get_llm().invoke(prompt)
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
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class FermiWithSearchControl(ForecastBot):
    def __init__(self, llms: dict[str, GeneralLlm]):
        super().__init__(llms=llms)

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
            fermi_response = await self.get_llm().invoke(fermi_prompt)
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

                {PROBABILITY_FINAL_ANSWER_LINE}
                """
            )
            reasoning = await self.get_llm().invoke(prompt)
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
            reasoning = await self.get_llm().invoke(prompt)
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
            reasoning = await self.get_llm().invoke(prompt)
            prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question)
            print(
                "[FermiWithSearchControl] Exiting _run_forecast_on_numeric with prediction:", prediction)
            return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
        except Exception as e:
            print("[FermiWithSearchControl] Exception in _run_forecast_on_numeric:", e)
            traceback.print_exc()
            return ReasonedPrediction(prediction_value=None, reasoning=f"Exception: {e}")

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class PerplexityRelatedMarketsScenarioBot(ForecastBot):
    name = "perplexity-scenario"
    
    def __init__(self, llms: dict[str, GeneralLlm], predictions_per_research_report=1):
        super().__init__(llms=llms, predictions_per_research_report=predictions_per_research_report)

    async def run_research(self, question):
        # Use Perplexity via OpenRouter (sonar-reasoning) for web research
        web_results = await get_perplexity_research_from_openrouter(question.question_text, model_name="openrouter/perplexity/sonar-reasoning")
        related_markets = get_related_markets_from_adjacent_news(
            question.question_text)
        return f"Web search results (Perplexity Sonar Reasoning):\n{web_results}\n\nRelated markets info:\n{related_markets}"

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:

            {question.question_text}
            
            The question background:
            
            {question.background_info}
            
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            
            {question.resolution_criteria}
            {question.fine_print}
            
            Here is some research and related forecasts:
            
            {research}
            
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one where the question would resolve no and one where it would resolve yes. The scenarios must be moderately different. If they aren't, just use yes and no.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Write up to 5 prediction markets that you would like to exist that haven't been supplied in the research above.
            (g) Consider all these scenarios and then give an overall probability as [number]%
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            
            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:
            {question.question_text}
            The options are: {question.options}
            The question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Here is some research and related forecasts:
            {research}
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each option. The scenarios must be moderately different.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Consider all these scenarios and then give probabilities for each option.
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:
            {question.question_text}
            The question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}
            Here is some research and related forecasts:
            {research}
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            {lower_bound_message}
            {upper_bound_message}
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each possible outcome range. The scenarios must be moderately different.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Consider all these scenarios and then give your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            You write your rationale remembering that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class PerplexityFilteredRelatedMarketsScenarioBot(ForecastBot):
    name = "perplexity-scenario-filtered"
    
    def __init__(self, llms: dict[str, GeneralLlm], predictions_per_research_report=1):
        super().__init__(llms=llms, predictions_per_research_report=predictions_per_research_report)
        # Use Claude Sonnet through OpenRouter for lightweight tasks
        self.lightweight_llm = GeneralLlm(model=CLAUDE_SONNET, temperature=0.2)

    async def score_market_relevance(self, market: dict, question: str, current_date: str) -> dict:
        """
        Score how relevant a market is to the question using Claude Sonnet.
        Returns a dict containing the score and reasoning.
        """
        # Get market name using the same logic as in run_research
        name = market.get("name", market.get("question", "Unnamed Market"))
        
        prompt = clean_indents(
            f"""
            You are evaluating how relevant a prediction market is to a forecasting question.
            
            The forecasting question is:
            {question}

            The current date is: 
            {current_date}
            
            The prediction market is:
            Name: {name}
            Platform: {market.get('platform', 'Unknown Platform')}
            Volume: {market.get('volume', 'N/A')}
            Status: {market.get('status', 'N/A')}
            End Date: {market.get('end_date', 'N/A')}
            
            Before giving your final score, please:
            1. Analyze how similar the topics/events are between the market and question
            2. Compare the timeframes of both
            3. Evaluate how similar the outcomes being predicted are
            4. Consider any other relevant factors
            
            Then, assign a score between 1 and 5 where:
            1 = Completely irrelevant
            2 = Largely irrelevant
            3 = Some useful information
            4 = Useful information
            5 = Perfect/near-perfect match
            
            Write out your analysis and reasoning, then end with "Score: X" where X is your final score between 1 and 5.
            """
        )
        response = await self.lightweight_llm.invoke(prompt)
        try:
            score = IntegerExtractor.extract_last_integer_value(response)
            return {'score': score, 'reasoning': response}
        except ValueError as e:
            # If we can't extract a score, default to 1 (completely irrelevant)
            # and include the error in the reasoning
            error_msg = f"Error extracting score: {str(e)}. Defaulting to score 1 (completely irrelevant)."
            return {'score': 1, 'reasoning': f"{response}\n\n{error_msg}"}

    async def run_research(self, question):
        # Get raw market data
        markets = get_related_markets_raw(question.question_text)
        
        # Score each market's relevance
        scored_markets = []
        filtered_markets = []
        print("\nAnalyzing market relevance:")
        print("=" * 80)
        for market in markets:
            name = market.get("name", market.get("question", "Unnamed Market"))
            print(f"\nEvaluating market: {name}")
            print("-" * 80)
            result = await self.score_market_relevance(market, question.question_text, datetime.now().strftime('%Y-%m-%d'))
            score = result['score']
            reasoning = result['reasoning']
            print(f"Reasoning:\n{reasoning}")
            print(f"Final score: {score}")
            print("-" * 80)
            
            if score >= 4:  # Only include markets with relevance score >= 4 (useful information or better)
                scored_markets.append(market)
            else:
                filtered_markets.append(market)
        
        # Format the filtered markets
        formatted_markets = format_markets(scored_markets)
        
        # Get web results from Perplexity
        web_results = await get_perplexity_research_from_openrouter(
            question.question_text, 
            model_name=PERPLEXITY_SONAR
        )
        
        # Extract facts and follow-up questions from the web results
        facts = FactsExtractor.extract_facts(web_results)
        follow_up_questions = FollowUpQuestionsExtractor.extract_follow_up_questions(web_results)
        
        # Verify each fact with Perplexity
        verified_facts = []
        print("\nVerifying facts:")
        print("=" * 80)
        for fact in facts:
            print(f"\nVerifying fact: {fact}")
            print("-" * 80)
            verification = await get_perplexity_research_from_openrouter(
                f"Is this true? {fact}",
                model_name=PERPLEXITY_SONAR
            )
            verified_facts.append({
                'fact': fact,
                'verification': verification
            })
            print(f"Verification result:\n{verification}")
            print("-" * 80)
        
        # Process follow-up questions
        follow_up_results = []
        if follow_up_questions:
            print("\nProcessing follow-up questions:")
            print("=" * 80)
            for q in follow_up_questions:
                print(f"\nProcessing question: {q}")
                print("-" * 80)
                result = await get_perplexity_research_from_openrouter(
                    q,
                    model_name=PERPLEXITY_SONAR
                )
                follow_up_results.append({
                    'question': q,
                    'answer': result
                })
                print(f"Answer:\n{result}")
                print("-" * 80)
        
        # Add report of filtered markets
        filtered_report = "\nMarkets which were considered not relevant:\n"
        for market in filtered_markets:
            filtered_report += f"- {market.get('name', 'Unnamed Market')}\n"
        
        # Format facts with their verifications
        facts_section = "\nKey Facts (with verifications):\n"
        for vf in verified_facts:
            facts_section += f"- Fact: {vf['fact']}\n  Verification: {vf['verification']}\n\n"
        
        # Format follow-up questions with their answers
        questions_section = "\nFollow-up Questions (with answers):\n"
        for fr in follow_up_results:
            questions_section += f"- Question: {fr['question']}\n  Answer: {fr['answer']}\n\n"
        
        # Create the initial research report
        initial_research = f"Web search results (Perplexity Sonar Reasoning):\n{web_results}\n\nRelated markets info:\n{formatted_markets}\n{filtered_report}{facts_section}{questions_section}"
        
        # Run a second prompt to process all the research and follow-up results
        second_prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.

            Your forecasting question is:
            {question.question_text}

            Here is a previous conversation you had. There were a number of questions and things you wanted to check. After the conversation will be printed the responses to those checks. All searches are done via LM. There may be mistakes, but given that they are more specific, they are a bit more likely to be correct than the original search material was.

            Original Research and Thinking:
            {initial_research}

            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 2 - 5 sentence-length scenarios based on the question, including at least one where the question would resolve no and one where it would resolve yes. The scenarios must be moderately different. If they aren't, just use yes and no.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation. Start this with the headline "Key Facts"
            (e) Write 5 follow up questions which would allow you to make a better decision. Start this with the headline "Follow Up Questions"
            (f) Suggest up to 5 prediction markets that could be created to aid this question. Start this with the headline "Prediction Markets"
            (g) Consider all these scenarios and then give an overall probability as a number in the format described below.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        
        # Get the second analysis
        second_analysis = await self.get_llm().invoke(second_prompt)
        
        # Combine everything into the final report
        final_report = f"""
        Original Research and Initial Analysis:
        {initial_research}

        Updated Analysis After Follow-up Research:
        {second_analysis}
        """
        
        return final_report

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:

            {question.question_text}
            
            The question background:
            
            {question.background_info}
            
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            
            {question.resolution_criteria}
            {question.fine_print}
            
            Here is some research and related forecasts:
            
            {research}
            
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one where the question would resolve no and one where it would resolve yes. The scenarios must be moderately different. If they aren't, just use yes and no.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Suggest up to 5 prediction markets that could be created to aid this question
            (g) Consider all these scenarios and then give an overall probability as a number in the format described below.
            
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            
            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:
            {question.question_text}
            The options are: {question.options}
            The question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Here is some research and related forecasts:
            {research}
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each option. The scenarios must be moderately different.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Consider all these scenarios and then give probabilities for each option.
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:
            {question.question_text}
            The question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}
            Here is some research and related forecasts:
            {research}
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            {lower_bound_message}
            {upper_bound_message}
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each possible outcome range. The scenarios must be moderately different.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Consider all these scenarios and then give your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            You write your rationale remembering that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)


class PerplexityFilteredRelatedMarketsScenarioPerplexityBot(ForecastBot):
    name = "perplexity-scenario-filtered"
    
    def __init__(self, llms: dict[str, GeneralLlm], predictions_per_research_report=1):
        super().__init__(llms=llms, predictions_per_research_report=predictions_per_research_report)
        # Use Claude Sonnet through OpenRouter for lightweight tasks
        self.lightweight_llm = GeneralLlm(model=CLAUDE_SONNET, temperature=0.2)

    async def score_market_relevance(self, market: dict, question: str, current_date: str) -> dict:
        """
        Score how relevant a market is to the question using Claude Sonnet.
        Returns a dict containing the score and reasoning.
        """
        # Get market name using the same logic as in run_research
        name = market.get("name", market.get("question", "Unnamed Market"))
        
        prompt = clean_indents(
            f"""
            You are evaluating how relevant a prediction market is to a forecasting question.
            
            The forecasting question is:
            {question}

            The current date is: 
            {current_date}
            
            The prediction market is:
            Name: {name}
            Platform: {market.get('platform', 'Unknown Platform')}
            Volume: {market.get('volume', 'N/A')}
            Status: {market.get('status', 'N/A')}
            End Date: {market.get('end_date', 'N/A')}
            
            Before giving your final score, please:
            1. Analyze how similar the topics/events are between the market and question
            2. Compare the timeframes of both
            3. Evaluate how similar the outcomes being predicted are
            4. Consider any other relevant factors
            
            Then, assign a score between 1 and 5 where:
            1 = Completely irrelevant
            2 = Largely irrelevant
            3 = Some useful information
            4 = Useful information
            5 = Perfect/near-perfect match
            
            Write out your analysis and reasoning, then end with "Score: X" where X is your final score between 1 and 5.
            """
        )
        response = await self.lightweight_llm.invoke(prompt)
        try:
            score = IntegerExtractor.extract_last_integer_value(response)
            return {'score': score, 'reasoning': response}
        except ValueError as e:
            # If we can't extract a score, default to 1 (completely irrelevant)
            # and include the error in the reasoning
            error_msg = f"Error extracting score: {str(e)}. Defaulting to score 1 (completely irrelevant)."
            return {'score': 1, 'reasoning': f"{response}\n\n{error_msg}"}

    async def run_research(self, question):
        # Get raw market data
        markets = get_related_markets_raw(question.question_text)
        
        # Score each market's relevance
        scored_markets = []
        filtered_markets = []
        print("\nAnalyzing market relevance:")
        print("=" * 80)
        for market in markets:
            name = market.get("name", market.get("question", "Unnamed Market"))
            print(f"\nEvaluating market: {name}")
            print("-" * 80)
            result = await self.score_market_relevance(market, question.question_text, datetime.now().strftime('%Y-%m-%d'))
            score = result['score']
            reasoning = result['reasoning']
            print(f"Reasoning:\n{reasoning}")
            print(f"Final score: {score}")
            print("-" * 80)
            
            if score >= 4:  # Only include markets with relevance score >= 4 (useful information or better)
                scored_markets.append(market)
            else:
                filtered_markets.append(market)
        
        # Format the filtered markets
        formatted_markets = format_markets(scored_markets)
        
        # Get web results from Perplexity
        web_results = await get_perplexity_research_from_openrouter(
            question.question_text, 
            model_name=PERPLEXITY_SONAR
        )
        
        # Extract facts and follow-up questions from the web results
        facts = FactsExtractor.extract_facts(web_results)
        follow_up_questions = FollowUpQuestionsExtractor.extract_follow_up_questions(web_results)
        
        # Verify each fact with Perplexity
        verified_facts = []
        print("\nVerifying facts:")
        print("=" * 80)
        for fact in facts:
            print(f"\nVerifying fact: {fact}")
            print("-" * 80)
            verification = await get_perplexity_research_from_openrouter(
                f"Is this true? {fact}",
                model_name=PERPLEXITY_SONAR
            )
            verified_facts.append({
                'fact': fact,
                'verification': verification
            })
            print(f"Verification result:\n{verification}")
            print("-" * 80)
        
        # Process follow-up questions
        follow_up_results = []
        if follow_up_questions:
            print("\nProcessing follow-up questions:")
            print("=" * 80)
            for q in follow_up_questions:
                print(f"\nProcessing question: {q}")
                print("-" * 80)
                result = await get_perplexity_research_from_openrouter(
                    q,
                    model_name=PERPLEXITY_SONAR
                )
                follow_up_results.append({
                    'question': q,
                    'answer': result
                })
                print(f"Answer:\n{result}")
                print("-" * 80)
        
        # Add report of filtered markets
        filtered_report = "\nMarkets which were considered not relevant:\n"
        for market in filtered_markets:
            filtered_report += f"- {market.get('name', 'Unnamed Market')}\n"
        
        # Format facts with their verifications
        facts_section = "\nKey Facts (with verifications):\n"
        for vf in verified_facts:
            facts_section += f"- Fact: {vf['fact']}\n  Verification: {vf['verification']}\n\n"
        
        # Format follow-up questions with their answers
        questions_section = "\nFollow-up Questions (with answers):\n"
        for fr in follow_up_results:
            questions_section += f"- Question: {fr['question']}\n  Answer: {fr['answer']}\n\n"
        
        # Create the initial research report
        initial_research = f"Web search results (Perplexity Sonar Reasoning):\n{web_results}\n\nRelated markets info:\n{formatted_markets}\n{filtered_report}{facts_section}{questions_section}"
        
        # Run a second prompt to process all the research and follow-up results
        second_prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.

            Your forecasting question is:
            {question.question_text}

            Here is a previous conversation you had. There were a number of questions and things you wanted to check. After the conversation will be printed the responses to those checks. All searches are done via LM. There may be mistakes, but given that they are more specific, they are a bit more likely to be correct than the original search material was.

            Original Research and Thinking:
            {initial_research}

            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one where the question would resolve no and one where it would resolve yes. The scenarios must be moderately different. If they aren't, just use yes and no.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation. Start this with the headline "Key Facts"
            (e) Write 5 follow up questions which would allow you to make a better decision. Start this with the headline "Follow Up Questions"
            (f) Suggest up to 5 prediction markets that could be created to aid this question. Start this with the headline "Prediction Markets"
            (g) Consider all these scenarios and then give an overall probability as a number in the format described below.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        
        # Get the second analysis
        second_analysis = await self.get_llm().invoke(second_prompt)
        
        # Combine everything into the final report
        final_report = f"""
Original Research and Initial Analysis:
{initial_research}

Updated Analysis After Follow-up Research:
{second_analysis}
"""
        
        return final_report

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:

            {question.question_text}
            
            The question background:
            
            {question.background_info}
            
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            
            {question.resolution_criteria}
            {question.fine_print}
            
            Here is some research and related forecasts:
            
            {research}
            
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one where the question would resolve no and one where it would resolve yes. The scenarios must be moderately different. If they aren't, just use yes and no.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation. Start this with the headline "Key Facts"
            (e) Write 5 follow up questions which would allow you to make a better decision. Start this with the headline "Follow Up Questions"
            (f) Suggest up to 5 prediction markets that could be created to aid this question. Start this with the headline "Prediction Markets"
            (g) Consider all these scenarios and then give an overall probability as a number in the format described below.
            
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            
            {PROBABILITY_FINAL_ANSWER_LINE}
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:
            {question.question_text}
            The options are: {question.options}
            The question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Here is some research and related forecasts:
            {research}
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each option. The scenarios must be moderately different.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Consider all these scenarios and then give probabilities for each option.
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
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
            You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.
            Your forecasting question is:
            {question.question_text}
            The question background:
            {question.background_info}
            This question's outcome will be determined by the specific criteria below. These criteria have almost certainly not yet been satisfied:
            {question.resolution_criteria}
            {question.fine_print}
            Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}
            Here is some research and related forecasts:
            {research}
            IMPORTANT: The research above was gathered by LLM search. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you can ignore it when making your forecast.
            Today is {datetime.now().strftime('%Y-%m-%d')}.
            {lower_bound_message}
            {upper_bound_message}
            Before answering you write:
            (a) Write out a quick summary of the situation as we know it, from the research. Please include the amount of time left between now and the question's resolution.
            (b) Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each possible outcome range. The scenarios must be moderately different.
            (c) Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
            (d) Write 5 facts which are key parts of our current understanding of the situation.
            (e) Write 5 follow up questions which would allow you to make a better decision.
            (f) Consider all these scenarios and then give your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            You write your rationale remembering that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.
            """
        )
        reasoning = await self.get_llm().invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def log_report_summary(forecast_reports):
        return log_report_summary_returning_str(forecast_reports)