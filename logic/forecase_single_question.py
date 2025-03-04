from typing import Tuple, Dict, List, Union

import numpy as np

from agents.agent_creator import (
    create_summarization_assistant,
)
from logic.call_asknews import run_research
from logic.summarization import run_summarization_phase
from logic.utils import build_and_write_json, extract_probabilities, perform_forecasting_phase, create_experts, \
    extract_question_details, identify_experts, strip_title_to_filename, get_all_experts
from utils.config import get_gpt_config
from utils.utils import normalize_and_average


async def forecast_single_question(
        question_details: dict,
        cache_seed: int = 42,
        is_multiple_choice: bool = False,
        options: List[str] = None,
        is_woc:bool=False,
        num_of_experts:str|None = None,
        news: str = None
) -> Tuple[Union[int, Dict[str, float]], str]:
    title, description, fine_print, resolution_criteria, forecast_date = extract_question_details(question_details)
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    if not is_woc:
        # Extract news
        news = await run_research(question_details)

    # Identify and create experts
    all_experts = await get_all_experts(config, question_details , is_multiple_choice, options,is_woc, num_of_experts)


    # Forecasting
    results = await perform_forecasting_phase(all_experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options)

    # Extract probabilities
    first_step_key = "final_probability"
    second_step_key = "revised_distribution" if is_multiple_choice else "revised_probability"
    first_step_probabilities, second_step_probabilities = extract_probabilities(results, first_step_key,
                                                                                second_step_key)

    # Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(results, question_details,
                                                  summarization_assistant)

    # Compute final probabilities
    final_result = int(round(np.mean(second_step_probabilities))) if not is_multiple_choice else {
        opt: val / 100.0 for opt, val in normalize_and_average(second_step_probabilities, options=options).items()
    }

    sd_first_step = np.std(first_step_probabilities, ddof=1)
    sd_second_step = np.std(second_step_probabilities, ddof=1)
    mean_first_step = np.mean(first_step_probabilities)
    mean_second_step = np.mean(second_step_probabilities)

    # Build final JSON
    final_json = {
        "question_details": question_details,
        "date": forecast_date, "news": news,
        "results": results,
        "summary": summarization,
        "statistics": {"mean_first_step": mean_first_step, "mean_second_step": mean_second_step,
                       "sd_first_step": sd_first_step, "sd_second_step": sd_second_step,
                       "final_result": final_result},
    }

    # Save JSON
    filename = strip_title_to_filename(title)
    build_and_write_json(filename, final_json, is_woc)

    return final_result, summarization
