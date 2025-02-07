import datetime
import random

import asyncio
import os
from typing import List, Tuple
import json

import numpy as np

from agents.agent_creator import create_summarization_assistant
from agents.experts_extractor import expert_creator, multiple_questions_expert_creator
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from logic.forecase_single_question import strip_title_to_filename
from logic.summarization import run_summarization_phase
from utils.PROMPTS import SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE, \
    NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE
from utils.config import get_gpt_config
from utils.utils import normalize_and_average

FORECASTS_PATH = "../forecasts/"
EXPERTS_PATH = "../experts.json"
WISDOM_OF_CROWDS_PATH = f"{FORECASTS_PATH}wisdom_of_crowds_forecasts/"
WHITELIST = [ 'Will_Russia_have_control_of_Chasiv_Yar_on_February_28_2025.json', 'How_many_tornadoes_will_NOAA_report_for_February_2025.json', 'Before_March_31_2025_will_Israel_or_Hamas_accuse_the_other_of_violating_the_ceasefire_agreed_to_on_J.json', 'How_many_tornadoes_will_NOAA_report_for_March_2025.json', 'Will_Mike_Johnson_cease_to_be_Speaker_of_the_US_House_of_Representatives_before_April_1_2025.json', 'Will_the_USDAposted_recall_by_Yu_Shang_Food_Inc_of_ReadyToEat_Meat_and_Poultry_Products_issued_Novem.json', 'Before_2030_how_many_new_AI_labs_will_be_leading_labs_within_2_years_of_their_founding.json', 'Will_humans_go_extinct_before_2100.json', 'Will_the_United_States_accuse_North_Korea_of_invading_South_Korea_before_April_1_2025.json', 'How_many_Grammy_awards_will_Kendrick_Lamar_win_in_2025.json', 'Will_Franois_Bayrou_step_down_or_be_removed_from_his_position_as_Prime_Minister_of_France_before_Mar.json', 'Will_the_World_Health_Organization_prequalify_moxidectin_before_April_1_2025.json', 'Will_the_debt_ceiling_be_raised_or_suspended_in_the_US_before_March_17_2025.json', 'Will_North_Korea_test_a_nuclear_weapon_before_April_1_2025.json', 'Will_Donald_Trump_attend_the_Super_Bowl_in_2025.json', 'Will_the_Federal_Reserve_cut_interest_rates_before_April_1_2025.json', 'Will_Justin_Trudeau_cease_to_be_Prime_Minister_of_Canada_before_April_1_2025.json', 'What_will_be_the_return_of_the_Egyptian_stock_market_index_the_EGX_30_in_February_2025.json', 'Will_the_Euro_Area_Inflation_Rate_be_above_24_for_February_2025.json', 'How_many_Mediterranean_migrants_and_refugees_will_enter_Europe_in_February_2025.json', 'Will_the_government_of_Greenland_officially_announce_a_date_for_an_independence_referendum_before_Ap.json', 'Will_Elon_Musk_attend_the_Super_Bowl_in_2025.json', 'Will_Intuitive_Machines_land_with_fully_working_payloads_on_the_Moon_before_April_1_2025.json', 'Will_the_eighth_Starship_integrated_flight_test_reach_an_altitude_of_160_kilometers_before_March_10_.json', 'Will_DeepSeek_be_ranked_higher_than_ChatGPT_on_the_AppStore_on_April_1_2025.json', 'Between_them_how_many_Grammy_awards_will_Chappell_Roan_and_Charli_XCX_win_in_2025.json', 'How_many_Grammy_awards_will_Sabrina_Carpenter_win_in_2025.json', 'Will_Elon_Musk_confirm_that_he_is_Adrian_Dittman_before_April_1_2025.json', 'Will_Geico_State_Farm_or_Progressive_run_an_ad_at_the_Super_Bowl_in_2025.json', 'How_many_movies_will_be_new_on_Netflixs_global_top_10_movies_list_for_the_week_ending_February_9_202.json', 'Will_Apple_run_an_ad_at_the_Super_Bowl_in_2025.json', 'Will_Joe_Biden_andor_Kamala_Harris_attend_the_Super_Bowl_in_2025.json', 'Will_Russia_have_control_of_Chasiv_Yar_on_January_30_2025.json', 'Will_OpenAI_publicly_release_the_full_o3_model_before_March_28_2025.json', 'Will_Rihanna_JayZ_andor_Beyonce_appear_at_the_Super_Bowl_halftime_show_in_2025.json', 'Will_TikTok_become_available_in_the_US_on_both_the_App_Store_and_Google_Play_before_April_5_2025.json', 'Will_the_Euro_Area_Inflation_Rate_be_above_24_for_January_2025.json', 'Will_the_Federal_Reserve_cut_interest_rates_before_February_1_2025.json', 'Will_OpenAI_Anthropic_or_Perplexity_run_an_ad_at_the_Super_Bowl_in_2025.json', 'What_will_be_the_return_of_the_Egyptian_stock_market_index_the_EGX_30_in_January_2025.json']
#'Which_of_the_five_largest_companies_in_the_world_will_see_the_highest_stock_price_growth_in_February.json', 'Will_William_Ruto_cease_to_be_President_of_Kenya_before_April_1_2025.json',
async def run_binary_wisdom_of_crowds(file:str) -> Tuple[int, str]:
    question_details = json.load(open(f"{FORECASTS_PATH}{file}"))

    # Gather question info
    title = question_details.get("question", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    news = question_details.get("news", "")
    forecast_date = datetime.datetime.now().isoformat()
    num_of_experts = len(question_details.get("forecasters", []))

    # 1) Model/LLM config
    config = get_gpt_config(42, 0.7, "gpt-4o", 120)

    chosen_experts, chosen_frameworks = sample_experts_and_frameworks(k = num_of_experts)
    # 3) Create specialized agents
    all_experts = await expert_creator(
        experts=chosen_experts, config=config, frameworks_specialties=chosen_frameworks
    )


    # 4) Phase 1: initial forecasts
    phase_1_results = await run_first_stage_forecasters(all_experts, title)

    # Collect initial and final probabilities from Phase 1, with safer error handling:
    try:
        initial_probabilities = [
            int(res["initial_probability"])
            for res in phase_1_results.values()
            if "initial_probability" in res
        ]
    except Exception as e:
        print("Error extracting initial probabilities:", e)
        initial_probabilities = []

    try:
        final_probabilities_phase1 = [
            int(res["final_probability"])
            for res in phase_1_results.values()
            if "final_probability" in res
        ]
    except Exception as e:
        print("Error extracting final probabilities (phase 1):", e)
        final_probabilities_phase1 = []

    # Calculate Phase 1 statistics
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

    # Collect revised probabilities, with safer error handling:
    try:
        revised_probabilities = [
            int(res["revised_probability"])
            for res in phase_2_results.values()
            if "revised_probability" in res
        ]
    except Exception as e:
        print("Error extracting revised probabilities:", e)
        revised_probabilities = []

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

    # 7) Final probability (simple mean, rounded)
    final_proba = int(round(revised_mean))

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
        "sd_revised_probability": revised_std,
    }
    final_json["statistics"] = stats_dict

    # 9) Write the JSON file
    filename = strip_title_to_filename(title)
    filepath = os.path.join(WISDOM_OF_CROWDS_PATH, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)
        f.close()

    # Return final probability + summarization
    return final_proba, summarization

async def run_multiple_choice_wisdom_of_crowds(file:str) -> Tuple[int, str]:
    question_details = json.load(open(f"{FORECASTS_PATH}{file}"))

    # Extract question info
    title = question_details.get("question", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    news = question_details.get("news", "")
    options = question_details.get("options", [])
    if not options:
        options = list(question_details.get("statistics", {}).get("phase_2_aggregated_distribution", {}).keys())
    forecast_date = datetime.datetime.now().isoformat()
    num_of_experts = len(question_details.get("forecasters", []))


    # 1) Model/LLM config
    config = get_gpt_config(42, 0.7, "gpt-4o", 120)

    chosen_experts, chosen_frameworks = sample_experts_and_frameworks(k = num_of_experts)
    # 3) Create specialized agents
    all_experts = await multiple_questions_expert_creator(
        experts=chosen_experts, config=config, frameworks_specialties=chosen_frameworks,prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, options=options
    )

    # 4) Phase 1: initial forecasts
    phase_1_results = await run_first_stage_forecasters(all_experts, title)

    # Collect final distributions from Phase 1, with safer error handling
    try:
        final_distributions_phase1 = [
            res["final_probability"]
            for res in phase_1_results.values()
            if "final_probability" in res
        ]
    except Exception as e:
        print("Error extracting phase 1 final distributions:", e)
        final_distributions_phase1 = []

    if not final_distributions_phase1:
        raise ValueError("No valid phase 1 final distributions were extracted; got an empty list.")

    # Normalize and average Phase 1 distributions
    try:
        phase_1_aggregated_distribution = normalize_and_average(final_distributions_phase1, options=options)
    except Exception as e:
        print("Error during normalization of phase 1 distributions:", e)
        raise

    # 5) Phase 2: incorporate news updates (new script approach with specialized MC prompt)
    news_prompt = NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE.format(options=options)
    phase_2_results = await run_second_stage_forecasters(
        all_experts,
        news,
        prompt=news_prompt,
        output_format=NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE
    )

    # Collect revised distributions from Phase 2, with safer error handling
    try:
        revised_distributions = [
            res["revised_distribution"]
            for res in phase_2_results.values()
            if "revised_distribution" in res
        ]
    except Exception as e:
        print("Error extracting revised distributions:", e)
        revised_distributions = []

    if not revised_distributions:
        raise ValueError("No valid revised distributions were extracted; got an empty list.")

    # Normalize and average Phase 2 distributions
    try:
        phase_2_aggregated_distribution = normalize_and_average(revised_distributions, options=options)
    except Exception as e:
        print("Error during normalization of phase 2 distributions:", e)
        raise

    # 6) Summarization
    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(
        first_phase_results=phase_1_results,
        news_analysis_results=phase_2_results,
        question=title,
        summarization_assistant=summarization_assistant
    )

    # 7) Final fractioned probabilities (0..1)
    final_fractions_dict = {
        opt: val / 100.0
        for opt, val in phase_2_aggregated_distribution.items()
    }

    # 8) Build final JSON with old script's comprehensive structure
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

    # Store aggregated distributions in statistics
    final_json["statistics"] = {
        "phase_1_aggregated_distribution": phase_1_aggregated_distribution,
        "phase_2_aggregated_distribution": phase_2_aggregated_distribution
    }

    # 9) Write the JSON file
    filename = strip_title_to_filename(title)
    filepath = os.path.join(WISDOM_OF_CROWDS_PATH, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)
        f.close()

    # Return final distribution + summarization
    return final_fractions_dict, summarization


def sample_experts_and_frameworks(k=15) -> Tuple[List[str],List[str]]:
    all_experts = json.load(open(EXPERTS_PATH))
    chosen_disciplines = random.sample(all_experts, k)
    chosen_experts = []
    chosen_frameworks = []
    for discipline in chosen_disciplines:
        chosen_experts.append(discipline["Discipline"])
        chosen_frameworks.append(random.sample(discipline["Framework/Specialty"],1))
    return chosen_experts, chosen_frameworks



def check_question_type(file:str)->str:
    if file.startswith("will"):
        return "binary"
    else:
        file_contents = json.load(open(f"{FORECASTS_PATH}{file}"))
        if "options" in file_contents:
            return "multiple_choice"
        elif file_contents.get("statistics",{}).get("phase_2_aggregated_distribution",{}):
            return "multiple_choice"
        else:
            return "binary"


def check_run_wisdom_of_crowds(all_files:List[str],wisdom_of_crowds_files:List[str])->bool:
    should_run = False
    if len(all_files) == 0:
        print("No forecast files found.")
    if len(wisdom_of_crowds_files) < len(all_files):
        should_run = True
        print("Running wisdom of crowds")
        print(f"files missing: {set(all_files) - set(wisdom_of_crowds_files)}")

    return should_run


async def main():
    all_files = [file for file in os.listdir(FORECASTS_PATH) if file.endswith(".json")]
    all_files = [file for file in all_files if file not in WHITELIST]
    wisdom_of_crowds_files = [file for file in os.listdir(WISDOM_OF_CROWDS_PATH) if file.endswith(".json")]
    should_run = check_run_wisdom_of_crowds(all_files,wisdom_of_crowds_files)
    if should_run:
        print("Running wisdom of crowds")
    else:
        print("No need to run wisdom of crowds")
        return

    # Run wisdom of crowds
    files_needed = set(all_files) - set(wisdom_of_crowds_files)
    for file in files_needed:
        print(f"Running wisdom of crowds for {file}")
        question_type = check_question_type(file)
        if question_type == "binary":
            print("Running binary wisdom of crowds")
            await run_binary_wisdom_of_crowds(file)
        else:
            print("Running multiple choice wisdom of crowds")
            await run_multiple_choice_wisdom_of_crowds(file)



if __name__ == "__main__":
    asyncio.run(main())

