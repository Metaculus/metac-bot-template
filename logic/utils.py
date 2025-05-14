import asyncio
import datetime
import json
import random
import re
from typing import Tuple, List, Dict

import aiofiles
import aiofiles.os
from autogen_agentchat.agents import AssistantAgent

from agents.agent_creator import create_experts_analyzer_assistant
from agents.experts_extractor import expert_creator, run_expert_extractor
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters, run_revised_stage_forecasters
from utils.PROMPTS import FIRST_PHASE_INSTRUCTIONS, REVISED_OUTPUT_FORMAT

EXPERTS_PATH = "experts.json"


def extract_question_details(question_details: dict) -> Tuple[str, str, str, str, str]:
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    forecast_date = datetime.datetime.now().isoformat()
    return title, description, fine_print, resolution_criteria, forecast_date


def create_prompt(question_details: Dict[str, str], news: str) -> str:
    title, description, fine_print, resolution_criteria, forecast_date = extract_question_details(question_details)
    full_prompt = FIRST_PHASE_INSTRUCTIONS + (
        f"##Forecast Date: {forecast_date}\n\n##Question:\n{title}\n\n##Description:\n{description}\n\n##Fine Print:\n"
        f"{fine_print}\n\n##Resolution Criteria:\n{resolution_criteria}\n\n##News Articles:\n{news}")
    return full_prompt


async def create_experts(professional_expertise, academic_disciplines, specialty, frameworks, config,
                         is_multiple_choice=False, options=None):
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
    results = {}
    prompt = create_prompt(question_details, news)
    tasks = [forecast_for_expert(expert, run_first_stage_forecasters, prompt=prompt,
                                 options=options) for expert in experts]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    for expert, result in results_list:
        results[expert.name] = result

    return results

async def perform_revised_forecasting_step(experts, question_details: Dict[str, str], news=None, is_multiple_choice=False,
                                    options=None):

    results = {}
    tasks = [forecast_for_expert(expert, run_revised_stage_forecasters, prompt = REVISED_OUTPUT_FORMAT,
                                 options=options) for expert in experts]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    for expert, result in results_list:
        results[expert.name] = result

    return results

async def forecast_for_expert(expert: AssistantAgent, stage_function=run_first_stage_forecasters, options=None,
                              prompt = ""):

    results = await stage_function([expert], prompt = prompt,
                                   system_message="",
                                   options=options)
    return expert, results


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


async def get_all_experts(config: dict, question_details: dict, is_multiple_choice=False, options=None, is_woc=False,
                          num_of_experts=None):
    if is_woc:
        experts, specialties = sample_experts_and_frameworks(num_of_experts)
        all_experts, _ = await create_experts(experts, [], specialties, [], config, is_multiple_choice, options)
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


async def build_and_write_json(filename, data, is_woc=False):
    path = "forecasts/wisdom_of_crowds_forecasts" if is_woc else "forecasts/q2"
    await aiofiles.os.makedirs(path, exist_ok=True)

    filepath = f"{path}/{filename}.json"

    async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=4))


def format_phase1_results_to_string(phase1_results_input_dict: Dict[str, Dict]) -> str:
    if not phase1_results_input_dict:
        return "No Phase 1 forecasts were available to display.\n"

    phase1_summary_str = ""
    for agent_name, p1_result_obj in phase1_results_input_dict.items():
        phase1_summary_str += f"\n--- Forecast from Forecaster: {agent_name} ---\n"
        try:
            phase1_summary_str += f"{json.dumps(p1_result_obj, indent=2)}\n"
        except TypeError as e:
            phase1_summary_str += f"Could not fully serialize forecast data for {agent_name}. Raw data: {str(p1_result_obj)}. Error: {e}\n"
        except Exception as e:
            phase1_summary_str += f"An unexpected error occurred while serializing forecast for {agent_name}. Raw data: {str(p1_result_obj)}. Error: {e}\n"

    phase1_summary_str += "\n---\n"
    return phase1_summary_str
