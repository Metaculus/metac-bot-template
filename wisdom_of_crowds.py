import datetime
import json
import os
import random
from typing import List, Tuple

import asyncio
import numpy as np

from agents.agent_creator import create_summarization_assistant
from agents.experts_extractor import expert_creator, multiple_questions_expert_creator
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from logic.forecase_single_question import strip_title_to_filename
from logic.summarization import run_summarization_phase
from main import forecast_individual_question
from utils.PROMPTS import SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE, \
    NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE
from utils.config import get_gpt_config
from utils.utils import normalize_and_average

FORECASTS_PATH = "forecasts/"
WISDOM_OF_CROWDS_PATH = f"{FORECASTS_PATH}wisdom_of_crowds_forecasts/"
WHITELIST = ['Will_Russia_have_control_of_Chasiv_Yar_on_February_28_2025.json',
             'How_many_tornadoes_will_NOAA_report_for_February_2025.json',
             'Before_March_31_2025_will_Israel_or_Hamas_accuse_the_other_of_violating_the_ceasefire_agreed_to_on_J.json',
             'How_many_tornadoes_will_NOAA_report_for_March_2025.json',
             'Will_Mike_Johnson_cease_to_be_Speaker_of_the_US_House_of_Representatives_before_April_1_2025.json',
             'Will_the_USDAposted_recall_by_Yu_Shang_Food_Inc_of_ReadyToEat_Meat_and_Poultry_Products_issued_Novem.json',
             'Before_2030_how_many_new_AI_labs_will_be_leading_labs_within_2_years_of_their_founding.json',
             'Will_the_United_States_accuse_North_Korea_of_invading_South_Korea_before_April_1_2025.json',
             'How_many_Grammy_awards_will_Kendrick_Lamar_win_in_2025.json',
             'Will_Franois_Bayrou_step_down_or_be_removed_from_his_position_as_Prime_Minister_of_France_before_Mar.json',
             'Will_the_World_Health_Organization_prequalify_moxidectin_before_April_1_2025.json',
             'Will_the_debt_ceiling_be_raised_or_suspended_in_the_US_before_March_17_2025.json',
             'Will_North_Korea_test_a_nuclear_weapon_before_April_1_2025.json',
             'Will_Donald_Trump_attend_the_Super_Bowl_in_2025.json',
             'Will_the_Federal_Reserve_cut_interest_rates_before_April_1_2025.json',
             'Will_Justin_Trudeau_cease_to_be_Prime_Minister_of_Canada_before_April_1_2025.json',
             'What_will_be_the_return_of_the_Egyptian_stock_market_index_the_EGX_30_in_February_2025.json',
             'Will_the_Euro_Area_Inflation_Rate_be_above_24_for_February_2025.json',
             'How_many_Mediterranean_migrants_and_refugees_will_enter_Europe_in_February_2025.json',
             'Will_the_government_of_Greenland_officially_announce_a_date_for_an_independence_referendum_before_Ap.json',
             'Will_Elon_Musk_attend_the_Super_Bowl_in_2025.json',
             'Will_Intuitive_Machines_land_with_fully_working_payloads_on_the_Moon_before_April_1_2025.json',
             'Will_the_eighth_Starship_integrated_flight_test_reach_an_altitude_of_160_kilometers_before_March_10_.json',
             'Will_DeepSeek_be_ranked_higher_than_ChatGPT_on_the_AppStore_on_April_1_2025.json',
             'Between_them_how_many_Grammy_awards_will_Chappell_Roan_and_Charli_XCX_win_in_2025.json',
             'How_many_Grammy_awards_will_Sabrina_Carpenter_win_in_2025.json',
             'Will_Elon_Musk_confirm_that_he_is_Adrian_Dittman_before_April_1_2025.json',
             'Will_Geico_State_Farm_or_Progressive_run_an_ad_at_the_Super_Bowl_in_2025.json',
             'How_many_movies_will_be_new_on_Netflixs_global_top_10_movies_list_for_the_week_ending_February_9_202.json',
             'Will_Apple_run_an_ad_at_the_Super_Bowl_in_2025.json',
             'Will_Joe_Biden_andor_Kamala_Harris_attend_the_Super_Bowl_in_2025.json',
             'Will_Russia_have_control_of_Chasiv_Yar_on_January_30_2025.json',
             'Will_OpenAI_publicly_release_the_full_o3_model_before_March_28_2025.json',
             'Will_Rihanna_JayZ_andor_Beyonce_appear_at_the_Super_Bowl_halftime_show_in_2025.json',
             'Will_TikTok_become_available_in_the_US_on_both_the_App_Store_and_Google_Play_before_April_5_2025.json',
             'Will_the_Euro_Area_Inflation_Rate_be_above_24_for_January_2025.json',
             'Will_the_Federal_Reserve_cut_interest_rates_before_February_1_2025.json',
             'Will_OpenAI_Anthropic_or_Perplexity_run_an_ad_at_the_Super_Bowl_in_2025.json',
             'What_will_be_the_return_of_the_Egyptian_stock_market_index_the_EGX_30_in_January_2025.json']



def check_run_wisdom_of_crowds(all_files: List[str], wisdom_of_crowds_files: List[str]) -> bool:
    should_run = False
    if len(all_files) == 0:
        print("No forecast files found.")
    if len(wisdom_of_crowds_files) < len(all_files):
        should_run = True
        print("Running wisdom of crowds")
        print(f"files missing: {set(all_files) - set(wisdom_of_crowds_files)}")

    return should_run


def get_question_details(file: str) -> Tuple[int, int, int, str]:
    details = json.load(open(f"{FORECASTS_PATH}{file}"))
    question_details = details.get("question_details", {})
    num_of_experts = len(details.get("results", []))
    news = details.get("news", "")

    question_id = question_details.get("id", None)
    post_id = question_details.get("post_id", None)

    return question_id, post_id, num_of_experts, news


async def main():
    all_files = [file for file in os.listdir(FORECASTS_PATH) if file.endswith(".json")]
    all_files = [file for file in all_files if file not in WHITELIST]
    wisdom_of_crowds_files = [file for file in os.listdir(WISDOM_OF_CROWDS_PATH) if file.endswith(".json")]
    should_run = check_run_wisdom_of_crowds(all_files, wisdom_of_crowds_files)
    if should_run:
        print("Running wisdom of crowds")
    else:
        print("No need to run wisdom of crowds")
        return

    # Run wisdom of crowds
    files_needed = set(all_files) - set(wisdom_of_crowds_files)
    for file in files_needed:
        print(f"Running wisdom of crowds for {file}")

        question_id, post_id, num_of_experts, news = get_question_details(file)
        if not question_id or not post_id:
            print(f"Question ID or Post ID not found in file: {file}\nYou can run it manually if needed")
            continue
        await forecast_individual_question(question_id=question_id, post_id=post_id, submit_prediction=False,
                                           skip_previously_forecasted_questions=False, is_woc=True, cache_seed=42,
                                           num_of_experts=num_of_experts, news = news)


if __name__ == "__main__":
    asyncio.run(main())
