from typing import Tuple, Dict, List
import re
import numpy as np
import os
import json

from agents.agent_creator import create_experts_analyzer_assistant, create_summarization_assistant
from agents.experts_extractor import run_expert_extractor, expert_creator, multiple_questions_expert_creator
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from logic.summarization import run_summarization_phase
from utils.PROMPTS import SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE, \
    NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE
from utils.config import get_gpt_config
from utils.utils import normalize_and_average

def strip_title_to_filename(title: str) -> str:
    """
    Convert a title into a file-friendly string:
    - Replace spaces with underscores.
    - Remove non-alphanumeric characters except underscores.
    - Limit length to avoid filesystem errors.
    """
    filename = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(' ', '_'))
    return filename[:100]  # Limit filename length to 100 characters


async def forecast_single_binary_question(
    question: str,
    news: str,
    cache_seed: int = 42
) -> Tuple[int, str]:
    """
    Extended version of forecast_single_binary_question that:
    1. Captures detailed stats and reasoning for Phase 1 and Phase 2.
    2. Saves results as a JSON file named after the question title.

    Returns:
        - final_proba (int): Final probability as a percentage.
        - summarization (str): Summarized reasoning.
    """
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    # Step 1: Identify Experts
    expert_identifier = create_experts_analyzer_assistant(config=config)
    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(
        expert_identifier, question
    )

    # Step 2: Create expert agents
    all_professional_experts = expert_creator(
        experts=professional_expertise, config=config, frameworks_specialties=specialty
    )
    all_academic_experts = expert_creator(
        experts=academic_disciplines, config=config, frameworks_specialties=frameworks
    )
    all_experts = all_professional_experts + all_academic_experts

    # Step 3: Phase 1 - Initial forecasting
    results = run_first_stage_forecasters(all_experts, question)

    # Extract initial probabilities
    initial_probabilities = [int(agent_result['initial_probability']) for agent_result in results.values()]
    initial_mean = float(np.mean(initial_probabilities)) if initial_probabilities else 0.0
    initial_std = float(np.std(initial_probabilities, ddof=1)) if len(initial_probabilities) > 1 else 0.0

    # Extract Phase 1 final probabilities
    final_probabilities_phase1 = [int(agent_result['final_probability']) for agent_result in results.values()]
    final_mean_phase1 = float(np.mean(final_probabilities_phase1)) if final_probabilities_phase1 else 0.0
    final_std_phase1 = float(np.std(final_probabilities_phase1, ddof=1)) if len(final_probabilities_phase1) > 1 else 0.0

    # Collect reasoning from Phase 1
    phase_1_reasoning = {
        agent_name: {
            "initial_reasoning": agent_result["initial_reasoning"],
            "perspective_derived_factors": agent_result["perspective_derived_factors"],
            "final_probability": agent_result["final_probability"]
        }
        for agent_name, agent_result in results.items()
    }

    # Step 4: Phase 2 - News integration and updated forecasting
    news_analysis_results = run_second_stage_forecasters(all_experts, news)

    # Extract revised probabilities
    revised_probabilities = [int(agent_result['revised_probability']) for agent_result in news_analysis_results.values()]
    revised_mean = float(np.mean(revised_probabilities)) if revised_probabilities else 0.0
    revised_std = float(np.std(revised_probabilities, ddof=1)) if len(revised_probabilities) > 1 else 0.0

    # Collect reasoning from Phase 2
    phase_2_reasoning = {
        agent_name: {
            "prior_probability": agent_result["prior_probability"],
            "analysis_updates": agent_result["analysis_updates"],
            "revised_probability": agent_result["revised_probability"]
        }
        for agent_name, agent_result in news_analysis_results.items()
    }

    # Compute final probability (mean of revised probabilities)
    final_proba = int(round(revised_mean))

    # Step 5: Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(
        first_phase_results=results,
        news_analysis_results=news_analysis_results,
        question=question,
        summarization_assistant=summarization_assistant,
    )

    # Step 6: Prepare detailed data for saving
    detailed_data = {
        "title": question.split('\n\n')[0].replace("Question title: ", "").strip(),
        "description": question.split('\n\n')[1].replace("Question description: ", "").strip(),
        "fine_print": question.split('Fine print: ')[-1].strip() if "Fine print: " in question else "",
        "initial_predictions": initial_probabilities,
        "initial_mean": initial_mean,
        "initial_std": initial_std,
        "final_predictions_phase1": final_probabilities_phase1,
        "final_mean_phase1": final_mean_phase1,
        "final_std_phase1": final_std_phase1,
        "phase_1_reasoning": phase_1_reasoning,
        "news": news,
        "revised_predictions": revised_probabilities,
        "revised_mean": revised_mean,
        "revised_std": revised_std,
        "phase_2_reasoning": phase_2_reasoning
    }

    # Step 7: Save detailed data to a JSON file
    # Extract title for filename
    title_for_filename = detailed_data["title"]
    filename = strip_title_to_filename(title_for_filename)
    os.makedirs("forecasts", exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join("forecasts", f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        print("Saving detailed data to:", filepath)
        json.dump(detailed_data, f, indent=4)

    # Return original outputs for compatibility
    return final_proba, summarization


async def forecast_single_multiple_choice_question(
    question: str,
    options: List[str],
    news: str,
    cache_seed: int = 42
) -> Tuple[Dict[str, float], str]:
    """
    Extended multiple-choice forecast function that:
      1. Gathers Phase 1 & Phase 2 reasoning.
      2. Computes aggregated distributions.
      3. Saves detailed data to a JSON file.
      4. Returns (fractioned_result_probabilities, summarization) as before.
    """
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)

    # 1) Identify relevant experts
    expert_identifier = create_experts_analyzer_assistant(config=config)
    academic_disciplines, frameworks, professional_expertise, specialty = run_expert_extractor(
        expert_identifier, question
    )

    # 2) Create specialized multiple-choice agents
    all_professional_experts = multiple_questions_expert_creator(
        experts=professional_expertise,
        config=config,
        frameworks_specialties=specialty,
        prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE,
        options=options
    )
    all_academic_experts = multiple_questions_expert_creator(
        experts=academic_disciplines,
        config=config,
        frameworks_specialties=frameworks,
        prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE,
        options=options
    )
    all_experts = all_professional_experts + all_academic_experts

    # 3) Phase 1: initial forecasts
    results = run_first_stage_forecasters(all_experts, question)
    # Each agent result is expected to have:
    # {
    #   "initial_reasoning": str,
    #   "initial_distribution": { "OptionA": int, "OptionB": int, ... },
    #   "perspective_derived_factors": [...],
    #   "final_distribution": { "OptionA": int, "OptionB": int, ... }
    # }

    # 3a) Extract final distributions from Phase 1
    #   e.g. each agent's "final_distribution": { "OptionA": int, "OptionB": int, ... }
    phase_1_final_distributions = [agent_result["final_distribution"] for agent_result in results.values()]

    # Compute aggregated (average) distribution for Phase 1 final forecasts
    #   We'll use the same normalize_and_average function, but that function
    #   expects a list of dictionaries with numeric ints for each option.
    #   We'll use the same "options" list to ensure consistent ordering.
    phase_1_aggregated_distribution = normalize_and_average(phase_1_final_distributions, options=options)

    # 3b) Collect Phase 1 reasoning
    #   This captures the initial reasoning, perspective factors, and final distribution for each agent
    phase_1_reasoning = {
        agent_name: {
            "initial_reasoning": agent_result["initial_reasoning"],
            "perspective_derived_factors": agent_result["perspective_derived_factors"],
            "final_distribution": agent_result["final_distribution"]
        }
        for agent_name, agent_result in results.items()
    }

    # 4) Phase 2: news integration / revised forecasts
    news_analysis_results = run_second_stage_forecasters(
        all_experts,
        news,
        prompt=NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE.format(options=options),
        output_format=NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE
    )
    # Each agent result is expected to have:
    # {
    #   "prior_distribution": { "OptionA": int, "OptionB": int, ... },
    #   "points_reinforcing_prior_analysis": str,
    #   "points_challenging_prior_analysis": str,
    #   "overall_effect_on_forecast": { "OptionA": "+int%" or "-int%", ... },
    #   "revised_distribution": { "OptionA": int, "OptionB": int, ... }
    # }

    # 4a) Gather all revised distributions
    revised_distributions = [agent_result["revised_distribution"] for agent_result in news_analysis_results.values()]

    # Compute aggregated (average) distribution for Phase 2 revised forecasts
    phase_2_aggregated_distribution = normalize_and_average(revised_distributions, options=options)

    # 4b) Collect Phase 2 reasoning
    phase_2_reasoning = {
        agent_name: {
            "prior_distribution": agent_result["prior_distribution"],
            "points_reinforcing_prior_analysis": agent_result["points_reinforcing_prior_analysis"],
            "points_challenging_prior_analysis": agent_result["points_challenging_prior_analysis"],
            "overall_effect_on_forecast": agent_result["overall_effect_on_forecast"],
            "revised_distribution": agent_result["revised_distribution"]
        }
        for agent_name, agent_result in news_analysis_results.items()
    }

    # 5) Summarization (Phase 1 + Phase 2)
    summarization_assistant = create_summarization_assistant(config)
    summarization = run_summarization_phase(
        first_phase_results=results,
        news_analysis_results=news_analysis_results,
        question=question,
        summarization_assistant=summarization_assistant
    )

    # 6) Compute final distribution (already done internally).
    #    In your existing code, you do this to produce the final output:
    result_probabilities = [result["revised_distribution"] for result in news_analysis_results.values()]
    normalized_result_probabilities = normalize_and_average(result_probabilities, options=options)
    fractioned_result_probabilities = {key: value / 100 for key, value in normalized_result_probabilities.items()}

    # 7) Prepare the data to be saved
    # Parse question "title", "description", "fine_print" if they are included in the question string
    # (You can adjust parsing logic if your question text format differs)
    question_title = question.split('\n\n')[0].replace("Question title: ", "").strip()
    question_description = ""
    question_fine_print = ""
    if len(question.split('\n\n')) > 1:
        question_description = question.split('\n\n')[1].replace("Question description: ", "").strip()
    if "Fine print: " in question:
        question_fine_print = question.split('Fine print: ')[-1].strip()

    # Build the final dictionary with all details
    detailed_data = {
        "title": question_title,
        "description": question_description,
        "fine_print": question_fine_print,
        # Phase 1
        "phase_1_reasoning": phase_1_reasoning,
        "phase_1_aggregated_distribution": phase_1_aggregated_distribution,  # e.g. {"OptionA": 30, "OptionB": 70}
        # Phase 2
        "phase_2_reasoning": phase_2_reasoning,
        "phase_2_aggregated_distribution": phase_2_aggregated_distribution,  # e.g. {"OptionA": 25, "OptionB": 75}
        # The final fractioned distribution you return from this function
        "final_fractioned_result_probabilities": fractioned_result_probabilities,
        "news": news
    }

    # 8) Save to JSON file
    filename = strip_title_to_filename(question_title)
    os.makedirs("forecasts", exist_ok=True)
    filepath = os.path.join("forecasts", f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        print("Saving detailed data to:", filepath)
        json.dump(detailed_data, f, indent=4)

    # 9) Return the same values as before for backward compatibility
    return fractioned_result_probabilities, summarization
