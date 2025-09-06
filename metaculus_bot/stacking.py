from __future__ import annotations

from typing import List, Sequence, Tuple

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    structure_output,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from .constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from .numeric_utils import clamp_and_renormalize_mc
from .prompts import stacking_binary_prompt, stacking_multiple_choice_prompt, stacking_numeric_prompt


def strip_model_tag(text: str) -> str:
    """Remove a leading "Model: ...\n\n" tag if present.

    This normalizes base-model reasoning snippets before feeding them to the stacker.
    """
    if text.startswith("Model: "):
        parts = text.split("\n", 2)
        if len(parts) >= 3 and parts[1] == "":
            return parts[2]
    return text


async def run_stacking_binary(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: BinaryQuestion,
    research: str,
    base_texts: Sequence[str],
) -> Tuple[float, str]:
    """Invoke the stacker for a binary question and parse to a decimal probability.

    Returns (prediction_in_decimal, meta_reasoning_text).
    """
    prompt = stacking_binary_prompt(question, research, list(base_texts))
    meta_reasoning = await stacker_llm.invoke(prompt)

    parse_instructions = (
        "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
        "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
        "Do not return percentages, strings, or any extra fields."
    )
    binary_prediction: BinaryPrediction = await structure_output(
        meta_reasoning,
        BinaryPrediction,
        model=parser_llm,
        additional_instructions=parse_instructions,
    )
    decimal_pred = max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, binary_prediction.prediction_in_decimal))
    return decimal_pred, meta_reasoning


async def run_stacking_mc(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: MultipleChoiceQuestion,
    research: str,
    base_texts: Sequence[str],
) -> Tuple[PredictedOptionList, str]:
    """Invoke the stacker for a multiple choice question and parse options.

    Returns (PredictedOptionList, meta_reasoning_text).
    """
    prompt = stacking_multiple_choice_prompt(question, research, list(base_texts))
    meta_reasoning = await stacker_llm.invoke(prompt)

    parsing_instructions = (
        f"Make sure that all option names are one of the following:\n{question.options}\n"
        'The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.'
    )
    predicted_option_list: PredictedOptionList = await structure_output(
        text_to_structure=meta_reasoning,
        output_type=PredictedOptionList,
        model=parser_llm,
        additional_instructions=parsing_instructions,
    )

    predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
    return predicted_option_list, meta_reasoning


async def run_stacking_numeric(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: NumericQuestion,
    research: str,
    base_texts: Sequence[str],
    lower_bound_message: str,
    upper_bound_message: str,
) -> Tuple[List[Percentile], str]:
    """Invoke the stacker for a numeric question and parse percentiles.

    Returns (declared_percentiles, meta_reasoning_text). The caller should perform
    numeric validation, jitter/clamping, and CDF construction.
    """
    prompt = stacking_numeric_prompt(question, research, list(base_texts), lower_bound_message, upper_bound_message)
    meta_reasoning = await stacker_llm.invoke(prompt)

    percentile_list: List[Percentile] = await structure_output(meta_reasoning, list[Percentile], model=parser_llm)
    return percentile_list, meta_reasoning
