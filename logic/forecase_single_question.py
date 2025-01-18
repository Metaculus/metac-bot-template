from typing import Tuple

from agents.agent_creator import create_experts_analyzer_assistant, create_summarization_assistant
from agents.asknews_scrapper import AskNewsScrapper
from agents.experts_extractor import run_expert_extractor, expert_creator
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from logic.summarization import run_summarization_phase
from utils.config import get_gpt_config



async def forecast_single_binary_question(question:str,news:str) -> Tuple[int,str]:
    config = get_gpt_config(42, 0.7, "gpt-4o", 120)

    expert_identifier = create_experts_analyzer_assistant(config=config)

    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(expert_identifier,
                                                                                               question)

    all_professional_experts = expert_creator(experts=professional_expertise, config=config,
                                              frameworks_specialties=specialty)
    all_academic_experts = expert_creator(experts=academic_disciplines, config=config,
                                          frameworks_specialties=frameworks)
    results = run_first_stage_forecasters(all_professional_experts + all_academic_experts,
                                          question)

    all_experts = all_professional_experts + all_academic_experts

    news_analysis_results = run_second_stage_forecasters(all_experts, news)

    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(first_phase_results=results, news_analysis_results=news_analysis_results,question=question,summarization_assistant=summarization_assistant)

    mean_result_probabilities = [int(result['revised_probability']) for result in news_analysis_results.values()]
    final_proba = sum(mean_result_probabilities) / len(mean_result_probabilities)

    return final_proba, summarization



async def forecast_single_multiple_choice_question(question:str,news:str) -> Tuple[int,str]:
    config = get_gpt_config(42, 0.7, "gpt-4o", 120)

    expert_identifier = create_experts_analyzer_assistant(config=config)

    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(expert_identifier,
                                                                                               question)

    all_professional_experts = expert_creator(experts=professional_expertise, config=config,
                                              frameworks_specialties=specialty, )
    all_academic_experts = expert_creator(experts=academic_disciplines, config=config,
                                          frameworks_specialties=frameworks)
    results = run_first_stage_forecasters(all_professional_experts + all_academic_experts,
                                          question)

    all_experts = all_professional_experts + all_academic_experts

    news_analysis_results = run_second_stage_forecasters(all_experts, news)

    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(first_phase_results=results, news_analysis_results=news_analysis_results,question=question,summarization_assistant=summarization_assistant)

    mean_result_probabilities = [int(result['revised_probability']) for result in news_analysis_results.values()]
    final_proba = sum(mean_result_probabilities) / len(mean_result_probabilities)

    return final_proba, summarization