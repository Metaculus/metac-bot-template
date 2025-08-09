import argparse
import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Any, Coroutine, Literal, Sequence, cast

import numpy as np
from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
    Percentile,
    BinaryPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import (
    ForecastReport, ResearchWithPredictions)
from forecasting_tools.data_models.numeric_report import (  # type: ignore
    NumericReport, Percentile)
from forecasting_tools.data_models.questions import DateQuestion
from pydantic import ValidationError
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot
from metaculus_bot.numeric_utils import aggregate_binary_mean, aggregate_numeric, bound_messages
from metaculus_bot.research_providers import choose_provider, ResearchCallable
from forecasting_tools import ForecastBot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


<<<<<<< HEAD
class TemplateForecaster(CompactLoggingForecastBot):

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Determine provider each call unless a custom one was supplied.
            if self._custom_research_provider is not None:
                provider = self._custom_research_provider
            else:
                default_llm = self.get_llm("default", "llm") if hasattr(self, "get_llm") else None  # type: ignore[attr-defined]
                provider = choose_provider(
                    default_llm,
                    exa_callback=self._call_exa_smart_searcher,
                    perplexity_callback=self._call_perplexity,
                    openrouter_callback=lambda q: self._call_perplexity(q, use_open_router=True),
                )

            research = await provider(question.question_text)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    # Override _research_and_make_predictions to support multiple LLMs
    async def _research_and_make_predictions(
        self,
        question: MetaculusQuestion,
    ) -> ResearchWithPredictions[PredictionTypes]:
        # Call the parent class's method if no specific forecaster LLMs are provided
        if not self._forecaster_llms:
            return await super()._research_and_make_predictions(question)

        notepad = await self._get_notepad(question)
        notepad.num_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = summary_report if self.use_research_summary_to_forecast else research

        # Generate tasks for each forecaster LLM
        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [self._make_prediction(question, research_to_use, llm_instance) for llm_instance in self._forecaster_llms],
        )
        valid_predictions, errors, exception_group = await self._gather_results_and_exceptions(tasks)
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")
        if len(valid_predictions) == 0:
            assert exception_group, "Exception group should not be None"
            self._reraise_exception_with_prepended_message(
                exception_group,
                "Error while running research and predictions",
            )
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

    async def _make_prediction(
        self,
        question: MetaculusQuestion,
        research: str,
        llm_to_use: GeneralLlm | None = None,
    ) -> ReasonedPrediction[PredictionTypes]:
        notepad = await self._get_notepad(question)
        notepad.num_predictions_attempted += 1

        # Determine which LLM to use
        actual_llm = llm_to_use if llm_to_use else self.get_llm("default", "llm")

        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_binary(q, r, llm)
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_multiple_choice(q, r, llm)
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_numeric(q, r, llm)
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported yet")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        prediction = await forecast_function(question, research, actual_llm)
        # Embed model name in reasoning for reporting
        prediction.reasoning = f"Model: {actual_llm.model}\n\n{prediction.reasoning}"
        return prediction  # type: ignore

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            In addition to news, consider and research all relevant prediction markets that are relevant to the question.
            You do not produce forecasts yourself; you must provide all relevant data to the superforecaster so they can make an expert judgment.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning-pro"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give" 
            "you a question they intend to forecast on. To be a great assistant, you generate" 
            "a concise but detailed rundown of the most relevant news, including if the question" 
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}" 
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[float]:
        from metaculus_bot.prompts import binary_prompt
        prompt = binary_prompt(question, research)
        reasoning = await llm_to_use.invoke(prompt)
        self._log_raw_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        from metaculus_bot.prompts import multiple_choice_prompt
        prompt = multiple_choice_prompt(question, research)
        reasoning = await llm_to_use.invoke(prompt)
        self._log_raw_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        from metaculus_bot.prompts import numeric_prompt
        prompt = numeric_prompt(question, research, lower_bound_message, upper_bound_message)
        reasoning = await llm_to_use.invoke(prompt)

        self._log_raw_llm_output(llm_to_use, question.id_of_question, reasoning)  # type: ignore

        try:
            percentile_list: list[Percentile] = await structure_output(
                reasoning, list[Percentile], model=self.get_llm("parser", "llm")
            )
            prediction = NumericDistribution.from_question(percentile_list, question)
            # Ensure we extracted all 6 required percentiles (10,20,40,60,80,90).
            if (
                hasattr(prediction, "declared_percentiles")
                and isinstance(prediction.declared_percentiles, list)
                and len(prediction.declared_percentiles) != 6
            ):
                raise ValidationError.from_exception_data(
                    "NumericDistribution",  # title
                    [
                        {
                            "type": "value_error",
                            "loc": ("declared_percentiles",),
                            "input": prediction.declared_percentiles,
                            "ctx": {
                                "error": "Expected 6 declared percentiles (10,20,40,60,80,90).",
                            },
                        }
                    ],
                )
        except ValidationError as err:
            # Fallback: find lines with "percentile", extract the last number,
            # and assume they correspond to the required 10/20/40/60/80/90.
            # Use strict filtering for percentile lines.
            logger.warning("Attempting to repair numeric distribution from LLM output.")
            percentile_lines = []
            for line in reasoning.split("\n"):
                match = re.match(
                    r"^[Pp]ercentile\s+\d+:\s+[-]?\d+(?:,\d{3})*(?:\.\d+)?$",
                    line.strip(),
                )
                if match:
                    percentile_lines.append(line)

            if len(percentile_lines) != 6:
                logger.warning("Did not receive exactly 6 valid percentile lines after strict filtering.")
                raise err  # Re-raise original error if strict filtering fails to find 6 lines

            values = []
            for line in percentile_lines:
                # Extract the last number from the strictly filtered line
                numbers = re.findall(r"-?\d+(?:\.\d+)?", line)
                if numbers:
                    values.append(float(numbers[-1].replace(",", "")))

            if len(values) != 6:
                raise ValueError("Could not extract 6 numeric values from strictly filtered percentile lines.")

            percentiles_template = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            repaired_percentiles = [Percentile(value=v, percentile=p) for v, p in zip(values, percentiles_template)]

            prediction = NumericDistribution(
                declared_percentiles=repaired_percentiles,
                open_upper_bound=question.open_upper_bound,
                open_lower_bound=question.open_lower_bound,
                upper_bound=question.upper_bound,
                lower_bound=question.lower_bound,
                zero_point=question.zero_point,
            )

        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

<<<<<<< HEAD

# ──────────────────────────────────────────────────────────────────────────
# Compact logging now provided by CompactLoggingForecastBot; removed
# legacy monkey-patch code.
# ──────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from metaculus_bot.cli import main as cli_main

    cli_main()
