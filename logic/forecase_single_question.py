from typing import Tuple, Dict, List

from agents.agent_creator import create_experts_analyzer_assistant, create_summarization_assistant
from agents.experts_extractor import run_expert_extractor, expert_creator, multiple_questions_expert_creator
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from logic.summarization import run_summarization_phase
from utils.PROMPTS import SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE, \
    NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE
from utils.config import get_gpt_config
from utils.utils import normalize_and_average


async def forecast_single_binary_question(question: str, news: str, cache_seed: int = 42) -> Tuple[int, str]:
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    expert_identifier = create_experts_analyzer_assistant(config=config)

    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(expert_identifier,
                                                                                               question)

    all_professional_experts = expert_creator(experts=professional_expertise, config=config,
                                              frameworks_specialties=specialty)
    all_academic_experts = expert_creator(experts=academic_disciplines, config=config,
                                          frameworks_specialties=frameworks)

    all_experts = all_professional_experts + all_academic_experts

    results = run_first_stage_forecasters(all_experts,
                                          question)

    news_analysis_results = run_second_stage_forecasters(all_experts, news)

    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(first_phase_results=results, news_analysis_results=news_analysis_results,
                                            question=question, summarization_assistant=summarization_assistant)

    mean_result_probabilities = [int(result['revised_probability']) for result in news_analysis_results.values()]
    final_proba = sum(mean_result_probabilities) / len(mean_result_probabilities)

    return final_proba, summarization


async def forecast_single_multiple_choice_question(question: str, options: List[str], news: str,
                                                   cache_seed: int = 42) -> Tuple[Dict[str, float], str]:
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    expert_identifier = create_experts_analyzer_assistant(config=config)

    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(expert_identifier,
                                                                                               question)

    all_professional_experts = multiple_questions_expert_creator(experts=professional_expertise, config=config,
                                                                 frameworks_specialties=specialty,
                                                                 prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE,
                                                                 options=options)
    all_academic_experts = multiple_questions_expert_creator(experts=academic_disciplines, config=config,
                                                             frameworks_specialties=frameworks,
                                                             prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, options=options)
    all_experts = all_professional_experts + all_academic_experts

    results = run_first_stage_forecasters(all_experts,
                                          question)

    news_analysis_results = run_second_stage_forecasters(all_experts, news,
                                                         prompt=NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE.format(options=options),
                                                         output_format=NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE)

    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(first_phase_results=results, news_analysis_results=news_analysis_results,
                                            question=question, summarization_assistant=summarization_assistant)

    result_probabilities = [result['revised_distribution'] for result in news_analysis_results.values()]
    normalized_result_probabilities = normalize_and_average(result_probabilities, options = options)
    fractioned_result_probabilities = {key:value/100 for key, value in normalized_result_probabilities.items()}


    return fractioned_result_probabilities, summarization
