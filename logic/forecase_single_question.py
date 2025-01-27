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

##############################################
# Helper function: create a file-friendly name
##############################################
def strip_title_to_filename(title: str) -> str:
    """
    Convert a title into a file-friendly string:
    - Replace spaces with underscores.
    - Remove non-alphanumeric characters except underscores.
    - Limit length to avoid filesystem issues.
    """
    filename = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(' ', '_'))
    return filename[:100]  # Limit to 100 chars


############################################################
# forecast_single_binary_question
############################################################
async def forecast_single_binary_question(
    question_details: dict,
    news: str,
    cache_seed: int = 42
) -> Tuple[int, str]:
    """
    Forecast a binary (Yes/No) question using a multi-phase (expert-based) process.

    The output JSON includes:
      1) question
      2) description
      3) fine_print
      4) date
      5) news
      Then for each forecaster:
        - initial_reasoning
        - initial_probability
        - perspective_derived_factors
        - phase_1_final_probability
        - revised_reasoning
        - revised_probability
      Lastly, overall statistics:
        - mean_initial_probability
        - sd_initial_probability
        - mean_phase_1_final_probability
        - sd_phase_1_final_probability
        - mean_revised_probability
        - sd_revised_probability

    Returns:
      (final_proba, summarization):
        - final_proba: integer, mean of revised probabilities (rounded)
        - summarization: a text summarizing the entire forecast process
    """

    # Extract question info
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    # We'll use "date" to store the time we did the forecast
    forecast_date = datetime.datetime.now().isoformat()

    # 1) Model/LLM config
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    # 2) Identify relevant experts
    expert_identifier = create_experts_analyzer_assistant(config=config)
    # We'll feed the question title (or a concatenation) to the expert finder
    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(
        expert_identifier,
        title
    )

    # 3) Create expert agents
    all_professional_experts = expert_creator(
        experts=professional_expertise, config=config, frameworks_specialties=specialty
    )
    all_academic_experts = expert_creator(
        experts=academic_disciplines, config=config, frameworks_specialties=frameworks
    )
    all_experts = all_professional_experts + all_academic_experts

    # 4) Phase 1: initial forecasts from each agent
    phase_1_results = run_first_stage_forecasters(all_experts, title)
    # Example phase_1_results[agent_name] = {
    #   "initial_reasoning": "...",
    #   "initial_probability": 45,
    #   "perspective_derived_factors": "...",
    #   "final_probability": 50
    # }

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
    phase_2_results = run_second_stage_forecasters(all_experts, news)
    # Example phase_2_results[agent_name] = {
    #   "prior_probability": 50,
    #   "analysis_updates": "...",
    #   "revised_probability": 55
    # }

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
    summarization = run_summarization_phase(
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

    # 9) Write the JSON file
    filename = strip_title_to_filename(title)
    os.makedirs("forecasts", exist_ok=True)
    filepath = os.path.join("forecasts", f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    # Return final probability + summarization
    return final_proba, summarization


############################################################
# forecast_single_multiple_choice_question
############################################################
async def forecast_single_multiple_choice_question(
    question_details: dict,
    options: List[str],
    news: str,
    cache_seed: int = 42
) -> Tuple[Dict[str, float], str]:
    """
    Forecast a multiple-choice question using a multi-phase (expert-based) process.

    The output JSON includes:
      1) question
      2) description
      3) fine_print
      4) date
      5) news
      Then for each forecaster:
        - initial_reasoning
        - initial_distribution
        - perspective_derived_factors
        - phase_1_final_distribution
        - revised_reasoning
        - revised_distribution
      Lastly, overall statistics (aggregated distribution).

    Returns:
      (final_fractions_dict, summarization):
        - final_fractions_dict: final normalized distribution across options (0-1 range)
        - summarization: textual summary
    """

    # Extract question info
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    forecast_date = datetime.datetime.now().isoformat()

    # 1) Model/LLM config
    config = get_gpt_config(
        seed=cache_seed,
        temperature=0.7,
        model_name="gpt-4o",
        max_tokens=120
    )

    # 2) Identify relevant experts
    expert_identifier = create_experts_analyzer_assistant(config=config)
    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(
        expert_identifier, title
    )

    # 3) Create specialized multiple-choice agents
    # We pass the "options" into multiple_questions_expert_creator so each agent can forecast each choice
    all_professional_experts = multiple_questions_expert_creator(
        experts=professional_expertise,
        config=config,
        frameworks_specialties=specialty,
        options=options
    )
    all_academic_experts = multiple_questions_expert_creator(
        experts=academic_disciplines,
        config=config,
        frameworks_specialties=frameworks,
        options=options
    )
    all_experts = all_professional_experts + all_academic_experts

    # 4) Phase 1: initial forecasts
    phase_1_results = run_first_stage_forecasters(all_experts, title)
    # Example: phase_1_results[agent_name] might have:
    # {
    #   "initial_reasoning": "...",
    #   "initial_distribution": {"OptionA": 20, "OptionB": 80, ...},
    #   "perspective_derived_factors": "...",
    #   "final_distribution": {"OptionA": 25, "OptionB": 75, ...}
    # }

    # 4a) Collect the final distributions from Phase 1
    final_distributions_phase1 = [
        res["final_distribution"] for res in phase_1_results.values()
    ]
    # Aggregate them
    phase_1_aggregated_distribution = normalize_and_average(final_distributions_phase1, options)

    # 5) Phase 2: news integration
    phase_2_results = run_second_stage_forecasters(all_experts, news)
    # Example: phase_2_results[agent_name] might have:
    # {
    #   "prior_distribution": {...},
    #   "analysis_updates": "...",
    #   "revised_distribution": {...}
    # }

    # Collect the revised distributions
    revised_distributions = [
        res["revised_distribution"] for res in phase_2_results.values()
    ]
    phase_2_aggregated_distribution = normalize_and_average(revised_distributions, options)

    # 6) Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(
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
