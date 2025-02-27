import asyncio
import datetime
import json
import os
import re
from typing import Tuple, List, Dict
import tqdm.asyncio

from agents.agent_creator import create_experts_analyzer_assistant
from agents.experts_extractor import multiple_questions_expert_creator, expert_creator, run_expert_extractor
from logic.chat import run_first_stage_forecasters, run_second_stage_forecasters
from utils.PROMPTS import SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE


def extract_question_details(question_details: dict) -> Tuple[str, str, str, str, str]:
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    forecast_date = datetime.datetime.now().isoformat()
    return title, description, fine_print, resolution_criteria, forecast_date

def create_prompt(question_details:Dict[str,str])->str:
    title, description, fine_print, resolution_criteria, forecast_date = extract_question_details(question_details)
    full_prompt = f"Forecast Date: {forecast_date}\n\n{title}\n\nDescription:\n{description}\n\nFine Print:\n{fine_print}\n\nResolution Criteria:\n{resolution_criteria}\n\n"
    return full_prompt


async def create_experts(professional_expertise, academic_disciplines, specialty, frameworks, config, is_multiple_choice=False, options=None):
    if is_multiple_choice:
        return (
            await multiple_questions_expert_creator(experts=professional_expertise,config= config, frameworks_specialties=specialty, prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, options=options),
            await multiple_questions_expert_creator(experts=academic_disciplines, config=config, frameworks_specialties=frameworks, prompt=SPECIFIC_EXPERTISE_MULTIPLE_CHOICE, options=options),
        )
    else:
        return (
            await expert_creator(experts=professional_expertise,config= config,frameworks_specialties= specialty),
            await expert_creator(experts=academic_disciplines,config= config,frameworks_specialties= frameworks),
        )



def strip_title_to_filename(title: str) -> str:
    """
    Helper function to create a safe filename from the question title.
    """
    filename = re.sub(r'[^a-zA-Z0-9_]', '', title.replace(' ', '_'))
    return filename[:100]  # Limit to 100 characters.


async def perform_forecasting_phase(experts, question_details: Dict[str, str], news=None, is_multiple_choice=False, options=None)->Dict[str, Dict[str, Dict[str, str]]]:
    question_formatted = create_prompt(question_details)
    results = {}

    async def forecast_for_expert(expert):
        phase_1 = await run_first_stage_forecasters([expert], question=question_formatted, question_title=question_details['title'], system_message="", options=options)
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


# async def perform_forecasting_phase(experts, question_details:Dict[str,str], phase, news=None, is_multiple_choice=False, options=None):
#     question_formatted = create_prompt(question_details)
#     if phase == 1:
#         return await run_first_stage_forecasters(experts, question=question_formatted, question_title=question_details['title'], system_message="", options=options)
#     elif phase == 2:
#         news_prompt = NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE.format(options=options) if is_multiple_choice else None
#         return await run_second_stage_forecasters(experts, news, prompt=news_prompt,options=options)

async def identify_experts(config: dict, title: str):
    expert_identifier = create_experts_analyzer_assistant(config=config)
    return await run_expert_extractor(expert_identifier, title)

def extract_probabilities(results, first_step_key: str,second_step_key:str) -> Tuple[List[int], List[int]]:
    first_step_probabilities = []
    second_step_probabilities = []
    for expert, values in results.items():
        first_step_probabilities.append(values["phase_1_result"][first_step_key])
        second_step_probabilities.append(values["phase_2_result"][second_step_key])
    return first_step_probabilities, second_step_probabilities


def build_and_write_json(filename, data):
    os.makedirs("forecasts", exist_ok=True)
    filepath = os.path.join("forecasts", f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

