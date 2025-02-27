import datetime
import json
import os
import re
import random
from typing import Tuple, List, Dict

import asyncio

from agents.agent_creator import create_experts_analyzer_assistant
from agents.experts_extractor import multiple_questions_expert_creator, expert_creator, run_expert_extractor
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from utils.PROMPTS import SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE
EXPERTS_PATH = "experts.json"


def extract_question_details(question_details: dict) -> Tuple[str, str, str, str, str]:
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    forecast_date = datetime.datetime.now().isoformat()
    return title, description, fine_print, resolution_criteria, forecast_date


def create_prompt(question_details: Dict[str, str]) -> str:
    title, description, fine_print, resolution_criteria, forecast_date = extract_question_details(question_details)
    full_prompt = f"Forecast Date: {forecast_date}\n\n{title}\n\nDescription:\n{description}\n\nFine Print:\n{fine_print}\n\nResolution Criteria:\n{resolution_criteria}\n\n"
    return full_prompt


async def create_experts(professional_expertise, academic_disciplines, specialty, frameworks, config,
                         is_multiple_choice=False, options=None):
    if is_multiple_choice:
        return (
            await multiple_questions_expert_creator(experts=professional_expertise, config=config,
                                                    frameworks_specialties=specialty,
                                                    prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, options=options),
            await multiple_questions_expert_creator(experts=academic_disciplines, config=config,
                                                    frameworks_specialties=frameworks,
                                                    prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, options=options),
        )
    else:
        return (
            await expert_creator(experts=professional_expertise, config=config, frameworks_specialties=specialty),
            await expert_creator(experts=academic_disciplines, config=config, frameworks_specialties=frameworks),
        )


def strip_title_to_filename(title: str) -> str:
    """
    Helper function to create a safe filename from the question title.
    """
    filename = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(' ', '_'))
    return filename[:100]  # Limit to 100 characters.


async def perform_forecasting_phase(experts, question_details: Dict[str, str], news=None, is_multiple_choice=False,
                                    options=None) -> Dict[str, Dict[str, Dict[str, str]]]:
    question_formatted = create_prompt(question_details)
    results = {}

    async def forecast_for_expert(expert):
        phase_1 = await run_first_stage_forecasters([expert], question=question_formatted,
                                                    question_title=question_details['title'], system_message="",
                                                    options=options)
        news_prompt = NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE.format(options=options) if is_multiple_choice else None
        phase_2 = await run_second_stage_forecasters([expert], news, prompt=news_prompt, options=options)
        return expert, phase_1, phase_2

    tasks = [forecast_for_expert(expert) for expert in experts]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    for expert, phase_1_result, phase_2_result in results_list:
        results[expert.name] = {
            "phase_1_result": phase_1_result,
            "phase_2_result": phase_2_result
        }

    return results


async def identify_experts(config: dict, title: str):
    expert_identifier = create_experts_analyzer_assistant(config=config)
    return await run_expert_extractor(expert_identifier, title)


def sample_experts_and_frameworks(k=15) -> Tuple[List[str], List[str]]:
    all_experts = json.load(open(EXPERTS_PATH))
    chosen_disciplines = random.sample(all_experts, k)
    chosen_experts = []
    chosen_frameworks = []
    for discipline in chosen_disciplines:
        chosen_experts.append(discipline["Discipline"])
        chosen_frameworks.append(random.sample(discipline["Framework/Specialty"], 1))
    return chosen_experts, chosen_frameworks


async def get_all_experts(config: dict, question_details: dict, is_multiple_choice=False, options=None,is_woc=False,num_of_experts=None):
    if is_woc:
        experts,specialties = sample_experts_and_frameworks(num_of_experts)
        all_experts,_ = await create_experts(experts,[],specialties,[], config, is_multiple_choice, options)
        return all_experts

    else:
        title = question_details.get("title", "")
        academic_disciplines, frameworks, professional_expertise, specialty = await identify_experts(config, title)
        all_professional_experts, all_academic_experts = await create_experts(
            professional_expertise, academic_disciplines, specialty, frameworks, config, is_multiple_choice, options
        )
        all_experts = all_professional_experts + all_academic_experts
        return all_experts

def extract_probabilities(results, first_step_key: str, second_step_key: str) -> Tuple[List[int], List[int]]:
    first_step_probabilities = []
    second_step_probabilities = []
    for expert, values in results.items():
        first_step_probabilities.append(values["phase_1_result"][first_step_key])
        second_step_probabilities.append(values["phase_2_result"][second_step_key])
    return first_step_probabilities, second_step_probabilities


def build_and_write_json(filename, data, is_woc=False):
    path = "forecasts"
    if is_woc:
        path = f"forecasts/wisdom_of_crowds_forecasts"
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
