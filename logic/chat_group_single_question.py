from typing import List, Dict, Union, Tuple

from agents.agent_creator import create_group, create_summarization_assistant
from logic.call_asknews import run_research
from logic.chat import validate_and_parse_response
from logic.summarization import run_summarization_phase
from logic.utils import extract_question_details, get_all_experts, perform_forecasting_phase, \
    format_phase1_results_to_string, \
    perform_revised_forecasting_step, strip_title_to_filename, build_and_write_json, get_probabilities, \
    enrich_probabilities, get_first_phase_probabilities
from utils.PROMPTS import GROUP_INSTRUCTIONS
from utils.config import get_gpt_config


async def chat_group_single_question(
        question_details: dict,
        cache_seed: int = 42,
        is_multiple_choice: bool = False,
        options: List[str] = None,
        is_woc: bool = False,
        use_hyde: bool = True,
        num_of_experts: str | None = None,
) -> Tuple[Union[int, Dict[str, float]], str]:
    title, description, fine_print, resolution_criteria, forecast_date = extract_question_details(question_details)
    config = get_gpt_config(cache_seed, 1, "gpt-4.1", 120)

    news = await run_research(question_details,use_hyde = use_hyde)

    # Identify and create experts
    all_experts = await get_all_experts(config, question_details, is_multiple_choice, options, is_woc, num_of_experts)
    forecasters_names = [expert.name for expert in all_experts]

    group_chat = create_group(all_experts)
    # Forecasting
    results = await perform_forecasting_phase(all_experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options)

    phase1_results_as_string = format_phase1_results_to_string(results)

    probabilities = get_first_phase_probabilities(results, is_multiple_choice, options)

    group_results = await group_chat.run(
        task=GROUP_INSTRUCTIONS.format(phase1_results_json_string=phase1_results_as_string,
                                       forecasters_list=forecasters_names))

    parsed_group_results = {group_single_answer.source: validate_and_parse_response(group_single_answer.content) for
                            group_single_answer in group_results.messages if group_single_answer.source != "user"}

    revision_results = await perform_revised_forecasting_step(all_experts, question_details, news=news,
                                                              is_multiple_choice=is_multiple_choice, options=options)

    # Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(results, question_details,
                                                  summarization_assistant)

    # Extract probabilities
    probabilities = get_probabilities(results, revision_results, parsed_group_results, is_multiple_choice, options, probabilities)

    enrich_probabilities(probabilities, question_details, news, forecast_date, summarization)

    final_answer = probabilities['revision_probability_result']

    # Save JSON
    filename = strip_title_to_filename(title)
    await build_and_write_json(filename, probabilities, is_woc)

    return final_answer, summarization
