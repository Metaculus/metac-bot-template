import datetime
import json
import os
import re
from typing import Tuple, Dict, List

import numpy as np

from agents.agent_creator import (
    create_experts_analyzer_assistant,
    create_summarization_assistant,
)
from agents.experts_extractor import (
    run_expert_extractor,
    expert_creator,
    multiple_questions_expert_creator,
)
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from logic.summarization import run_summarization_phase
from utils.config import get_gpt_config
from utils.utils import normalize_and_average


def strip_title_to_filename(title: str) -> str:

    filename = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(' ', '_'))
    return filename[:100]  # Limit to 100 chars


async def forecast_single_binary_question(
    question_details: dict,
    news: str,
    cache_seed: int = 42
) -> Tuple[int, str]:

    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    forecast_date = datetime.datetime.now().isoformat()

    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    expert_identifier = create_experts_analyzer_assistant(config=config)
    academic_disciplines, frameworks, professional_expertise, specialty = await run_expert_extractor(
        expert_identifier,
        title
    )

    all_professional_experts = await expert_creator(
        experts=professional_expertise, config=config, frameworks_specialties=specialty
    )
    all_academic_experts = await expert_creator(
        experts=academic_disciplines, config=config, frameworks_specialties=frameworks
    )
    all_experts = all_professional_experts + all_academic_experts

    phase_1_results = await run_first_stage_forecasters(all_experts, title)

    #todo put into function
    # Collect probabilities
    initial_probabilities = [
        int(res["initial_probability"]) for res in phase_1_results.values()
    ]
    final_probabilities_phase1 = [
        int(res["final_probability"]) for res in phase_1_results.values()
    ]

    # Calculate Phase 1 stats
    if len(initial_probabilities) > 1:
        initial_mean = float(np.mean(initial_probabilities))
        initial_std = float(np.std(initial_probabilities, ddof=1))
    else:
        initial_mean, initial_std = (0.0, 0.0)

    if len(final_probabilities_phase1) > 1:
        final_mean_phase1 = float(np.mean(final_probabilities_phase1))
        final_std_phase1 = float(np.std(final_probabilities_phase1, ddof=1))
    else:
        final_mean_phase1, final_std_phase1 = (0.0, 0.0)

    # 5) Phase 2: incorporate news, revise forecasts
    phase_2_results = await run_second_stage_forecasters(all_experts, news)

    #todo put into function
    revised_probabilities = [
        int(res["revised_probability"]) for res in phase_2_results.values()
    ]
    if len(revised_probabilities) > 1:
        revised_mean = float(np.mean(revised_probabilities))
        revised_std = float(np.std(revised_probabilities, ddof=1))
    else:
        revised_mean, revised_std = (0.0, 0.0)

    # 6) Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(
        first_phase_results=phase_1_results,
        news_analysis_results=phase_2_results,
        question=title,
        summarization_assistant=summarization_assistant,
    )

    # 7) Final probability (simple mean)
    final_proba = int(round(revised_mean))

    # 8) Build final JSON in the requested structure
    final_json = {
        # General question data
        "question": title,
        "description": description,
        "fine_print": fine_print,
        "date": forecast_date,
        "news": news,
    }

    #todo wrap into function
    # Forecasters detail
    forecasters_list = []
    for agent_name, p1_data in phase_1_results.items():
        # p1_data has the Phase 1 keys
        p2_data = phase_2_results.get(agent_name, {})
        # We'll rename "analysis_updates" -> "revised_reasoning" for clarity
        revised_reasoning = p2_data.get("analysis_updates", "")

        forecasters_list.append({
            "agent_name": agent_name,
            "initial_reasoning": p1_data.get("initial_reasoning", ""),
            "initial_probability": p1_data.get("initial_probability", None),
            "perspective_derived_factors": p1_data.get("perspective_derived_factors", ""),
            "phase_1_final_probability": p1_data.get("final_probability", None),
            "revised_reasoning": revised_reasoning,
            "revised_probability": p2_data.get("revised_probability", None)
        })

    final_json["forecasters"] = forecasters_list

    #todo wrap into function
    # Statistics
    stats_dict = {
        "mean_initial_probability": initial_mean,
        "sd_initial_probability": initial_std,
        "mean_phase_1_final_probability": final_mean_phase1,
        "sd_phase_1_final_probability": final_std_phase1,
        "mean_revised_probability": revised_mean,
        "sd_revised_probability": revised_std
    }
    final_json["statistics"] = stats_dict

    #todo wrap into function
    # 9) Write the JSON file
    filename = strip_title_to_filename(title)
    os.makedirs("forecasts", exist_ok=True)
    filepath = os.path.join("forecasts", f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    # Return final probability + summarization
    return final_proba, summarization


async def forecast_single_multiple_choice_question(
    question_details: dict,
    options: List[str],
    news: str,
    cache_seed: int = 42
) -> Tuple[Dict[str, float], str]:
    # Extract question info
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    forecast_date = datetime.datetime.now().isoformat()


    # 1) Model/LLM config
    config = get_gpt_config(cache_seed=cache_seed, temperature=, model=, timeout=)

    # 2) Identify relevant experts
    expert_identifier = create_experts_analyzer_assistant(config=config)
    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(
        expert_identifier, title
    )

    # 3) Create specialized multiple-choice agents
    # We pass the "options" into multiple_questions_expert_creator so each agent can forecast each choice
    all_professional_experts = await multiple_questions_expert_creator(
        experts=professional_expertise,
        config=config,
        frameworks_specialties=specialty,
        options=options
    )
    all_academic_experts = await multiple_questions_expert_creator(
        experts=academic_disciplines,
        config=config,
        frameworks_specialties=frameworks,
        options=options
    )
    all_experts = all_professional_experts + all_academic_experts

    # 4) Phase 1: initial forecasts
    phase_1_results = await run_first_stage_forecasters(all_experts, title)

    # 4a) Collect the final distributions from Phase 1
    final_distributions_phase1 = [
        res["final_distribution"] for res in phase_1_results.values()
    ]
    # Aggregate them
    phase_1_aggregated_distribution = normalize_and_average(final_distributions_phase1, options)

    # 5) Phase 2: news integration
    phase_2_results = await run_second_stage_forecasters(all_experts, news)


    # Collect the revised distributions
    revised_distributions = [
        res["revised_distribution"] for res in phase_2_results.values()
    ]
    phase_2_aggregated_distribution = normalize_and_average(revised_distributions, options)

    # 6) Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(
        first_phase_results=phase_1_results,
        news_analysis_results=phase_2_results,
        question=title,
        summarization_assistant=summarization_assistant
    )

    # 7) Final fractioned probabilities (0..1)
    final_fractions_dict = {opt: val / 100.0 for opt, val in phase_2_aggregated_distribution.items()}

    # 8) Build final JSON
    final_json = {
        "question": title,
        "description": description,
        "fine_print": fine_print,
        "date": forecast_date,
        "news": news,
        "options": options
    }

    # Forecasters detail
    forecasters_list = []
    for agent_name, p1_data in phase_1_results.items():
        p2_data = phase_2_results.get(agent_name, {})
        revised_reasoning = p2_data.get("analysis_updates", "")

        forecasters_list.append({
            "agent_name": agent_name,
            "initial_reasoning": p1_data.get("initial_reasoning", ""),
            "initial_distribution": p1_data.get("initial_distribution", {}),
            "perspective_derived_factors": p1_data.get("perspective_derived_factors", ""),
            "phase_1_final_distribution": p1_data.get("final_distribution", {}),
            "revised_reasoning": revised_reasoning,
            "revised_distribution": p2_data.get("revised_distribution", {})
        })

    final_json["forecasters"] = forecasters_list

    # Stats: aggregated distributions
    final_json["statistics"] = {
        "phase_1_aggregated_distribution": phase_1_aggregated_distribution,
        "phase_2_aggregated_distribution": phase_2_aggregated_distribution
    }

    # 9) Save the JSON
    filename = strip_title_to_filename(title)
    os.makedirs("forecasts", exist_ok=True)
    filepath = os.path.join("forecasts", f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    # Return final distribution + summarization
    return final_fractions_dict, summarization
