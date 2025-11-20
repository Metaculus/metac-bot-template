from typing import List, Dict, Union, Tuple
import logging
from agents.agent_creator import create_summarization_assistant, create_group
from logic.call_asknews import run_research
from logic.chat import validate_and_parse_response
from logic.summarization import run_summarization_phase
from logic.utils import extract_question_details, get_all_experts, perform_forecasting_phase, \
    perform_revised_forecasting_step, strip_title_to_filename, build_and_write_json, get_probabilities, \
    enrich_probabilities, get_first_phase_probabilities, get_relevant_contexts_to_group_discussion
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
    title, description, fine_print, resolution_criteria, forecast_date, aggregations = extract_question_details(question_details)
    logging.info("=== Starting main pipeline for question: %s ===", title[:100] + "..." if len(title) > 100 else title)
    logging.info("Pipeline parameters: cache_seed=%s, is_multiple_choice=%s, options=%s, is_woc=%s, use_hyde=%s, num_of_experts=%s", 
                 cache_seed, is_multiple_choice, options, is_woc, use_hyde, num_of_experts)
    config = get_gpt_config(cache_seed, 1, "gpt-4.1", 120)
    logging.info("Configuration created with cache_seed=%s", cache_seed)

    logging.info("Starting research phase with use_hyde=%s", use_hyde)
    news = await run_research(question_details, use_hyde=use_hyde)
    logging.info("Research phase completed. Found %s news articles", len(news) if isinstance(news, list) else "N/A")

    # Identify and create experts
    logging.info("Creating experts with num_of_experts=%s", num_of_experts)
    all_experts = await get_all_experts(config, question_details, is_multiple_choice, options, is_woc, num_of_experts)
    forecasters_names = [expert.name for expert in all_experts]
    forecasters_display_names = [getattr(expert, "display_name", expert.name) for expert in all_experts]
    logging.info("Created %s experts: %s", len(all_experts), forecasters_display_names)

    logging.info("Creating group chat with %s experts", len(all_experts))
    group_chat = create_group(all_experts)
    
    # Forecasting
    logging.info("Starting first phase forecasting with %s experts", len(all_experts))
    results = await perform_forecasting_phase(all_experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options)

    logging.info("Finished first phase forecasting. Generated %s results", len(results) if results else 0)


    logging.info("Preparing group discussion contextualization")
    group_contextualization = get_relevant_contexts_to_group_discussion(results)

    logging.info("Extracting first phase probabilities")
    probabilities = get_first_phase_probabilities(results, is_multiple_choice, options)

    logging.info("Starting group chat discussion with forecasters: %s", forecasters_names)
    group_results = await group_chat.run(
        task=GROUP_INSTRUCTIONS.format(phase1_results_json_string=group_contextualization,
                                       forecasters_list=forecasters_names))

    logging.info("Finished group chat. Generated %s group messages", len(group_results.messages) if hasattr(group_results, 'messages') else 0)

    logging.info("Parsing group chat results")
    parsed_group_results = {group_single_answer.source: validate_and_parse_response(group_single_answer.content) for
                            group_single_answer in group_results.messages if group_single_answer.source != "user"}
    logging.info("Parsed %s group results from sources: %s", len(parsed_group_results), list(parsed_group_results.keys()))

    logging.info("Starting revised forecasting step with %s experts", len(all_experts))
    revision_results = await perform_revised_forecasting_step(all_experts, question_details, news=news,
                                                              is_multiple_choice=is_multiple_choice, options=options)
    logging.info("Finished revised forecasting step. Generated %s revision results", len(revision_results) if revision_results else 0)
    
    # Summarization
    logging.info("Creating summarization assistant")
    summarization_assistant = create_summarization_assistant(config)
    logging.info("Starting summarization phase")
    summarization = await run_summarization_phase(results, question_details,
                                                  summarization_assistant)
    logging.info("Finished summarization phase. Summary length: %s characters", len(summarization) if summarization else 0)

    # Extract probabilities
    logging.info("Extracting and calculating final probabilities")
    probabilities = get_probabilities(results, revision_results, parsed_group_results, is_multiple_choice, options,
                                      probabilities)

    logging.info("Enriching probabilities with additional metadata")
    enrich_probabilities(probabilities, question_details, news, forecast_date, summarization, forecasters_display_names)

    final_answer = probabilities['revision_probability_result']

    logging.info("Pipeline completed. Final answer: %s", final_answer)

    # Save JSON
    filename = strip_title_to_filename(title)
    logging.info("Saving results to file: %s", filename)
    await build_and_write_json(filename, probabilities, is_woc, aggregations)
    
    logging.info("=== Main pipeline completed successfully for question: %s ===", title[:100] + "..." if len(title) > 100 else title)

    return final_answer, summarization
