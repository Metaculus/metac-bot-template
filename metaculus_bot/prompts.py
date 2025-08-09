from __future__ import annotations

from datetime import datetime

from forecasting_tools import (BinaryQuestion, MultipleChoiceQuestion,
                               NumericQuestion)

__all__ = [
    "binary_prompt",
    "multiple_choice_prompt",
    "numeric_prompt",
]


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def binary_prompt(question: BinaryQuestion, research: str) -> str:
    """Return the forecasting prompt for binary questions.

    The body is copied verbatim from the original TemplateForecaster implementation
    to ensure behaviour is unchanged.
    """

    from forecasting_tools import clean_indents  # local import to avoid heavy deps at module import time

    return clean_indents(
        f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {_today_str()}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
    )


def multiple_choice_prompt(question: MultipleChoiceQuestion, research: str) -> str:
    from forecasting_tools import clean_indents

    return clean_indents(
        f"""
        You are a professional forecaster interviewing for a job.

        Your interview question is:
        {question.question_text}

        The options are: {question.options}


        Background:
        {question.background_info}

        {question.resolution_criteria}

        {question.fine_print}


        Your research assistant says:
        {research}

        Today is {_today_str()}.

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


def numeric_prompt(
    question: NumericQuestion,
    research: str,
    lower_bound_message: str,
    upper_bound_message: str,
) -> str:
    from forecasting_tools import clean_indents

    return clean_indents(
        f"""
        You are a professional forecaster interviewing for a job.

        Your interview question is:
        {question.question_text}

        Question background:
        {question.background_info}


        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
        {question.resolution_criteria}

        {question.fine_print}


        Your research assistant says:
        {research}

        Today is {_today_str()}.

        {lower_bound_message}
        {upper_bound_message}

        Before answering you write:
        (a) The time left until the outcome to the question is known.
        (b) The status quo outcome if nothing changed.
        (c) A description of an scenario that results in an unexpected outcome.

        You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

        The last thing you write is your final answer as a list of percentiles and values.
        """
    ) 