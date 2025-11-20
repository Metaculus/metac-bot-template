import asyncio
import datetime
import json
import random
import re
from typing import Tuple, List, Dict, Any

import aiofiles
import aiofiles.os
import numpy as np
from autogen_agentchat.agents import AssistantAgent

from agents.agent_creator import create_experts_analyzer_assistant
from agents.experts_extractor import expert_creator, run_expert_extractor
from logic.chat import run_first_stage_forecasters, run_revised_stage_forecasters
from utils.PROMPTS import FIRST_PHASE_INSTRUCTIONS, REVISED_OUTPUT_FORMAT
from utils.utils import normalize_and_average

EXPERTS_PATH = "experts.json"


def extract_question_details(question_details: dict) -> Tuple[str, str, str, str, str, str]:
    title = question_details.get("title", "")
    description = question_details.get("description", "")
    fine_print = question_details.get("fine_print", "")
    resolution_criteria = question_details.get("resolution_criteria", "")
    forecast_date = datetime.datetime.now().isoformat()
    aggregations = question_details.get("aggregations", "")
    return title, description, fine_print, resolution_criteria, forecast_date, aggregations


def create_prompt(question_details: Dict[str, str], news: str,system_message = FIRST_PHASE_INSTRUCTIONS) -> str:
    title, description, fine_print, resolution_criteria, forecast_date, aggregations = extract_question_details(question_details)
    full_prompt = system_message + (
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
                                    options=None, system_message = FIRST_PHASE_INSTRUCTIONS) -> Dict[str, Dict[str, Dict[str, str]]]:
    results = {}
    prompt = create_prompt(question_details, news, system_message)
    tasks = [forecast_for_expert(expert, run_first_stage_forecasters, prompt=prompt,
                                 options=options) for expert in experts]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    for expert, result in results_list:
        key = getattr(expert, "display_name", expert.name)
        results[key] = result

    return results


async def perform_revised_forecasting_step(experts, question_details: Dict[str, str], news=None,
                                           is_multiple_choice=False,
                                           options=None):
    results = {}
    tasks = [forecast_for_expert(expert, run_revised_stage_forecasters, prompt=REVISED_OUTPUT_FORMAT,
                                 options=options) for expert in experts]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    for expert, result in results_list:
        key = getattr(expert, "display_name", expert.name)
        results[key] = result

    return results


async def forecast_for_expert(expert: AssistantAgent, stage_function=run_first_stage_forecasters, options=None,
                              prompt=""):
    results = await stage_function([expert], prompt=prompt,
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


async def build_and_write_json(filename, data, is_woc=False, subdirectory=None):
    if is_woc:
        path = "forecasts/wisdom_of_crowds_forecasts"
    else:
        base_path = "forecasts/fall"
        path = f"{base_path}/{subdirectory}" if subdirectory else base_path

    await aiofiles.os.makedirs(path, exist_ok=True)

    filepath = f"{path}/{filename}.json"

    async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=4))


def get_relevant_contexts_to_group_discussion(first_run_results: Dict[str, Dict]) -> str:
    output = {}
    for expert, values in first_run_results.items():
        output[expert] = {
            'reasoning': values['final_reasoning'],
            'probability': values['final_probability'],
        }
    output = json.dumps(output, indent=4)
    return output




def get_first_phase_probabilities(first_step_results, is_multiple_choice, options):
    deliberation_step_probability = [result['final_probability'] for result in first_step_results.values() if
                                     'final_probability' in result]
    bound_bottom_to_one_hundred = lambda x: min(max(x, 1), 100)
    deliberation_step_probability = [bound_bottom_to_one_hundred(prob) for prob in deliberation_step_probability]

    deliberation_step_probability_result = int(
        round(np.mean(deliberation_step_probability))) if not is_multiple_choice else {
        opt: val / 100.0 for opt, val in normalize_and_average(deliberation_step_probability, options=options).items()
    }

    sd_deliberation_step = np.std(deliberation_step_probability, ddof=1)
    mean_deliberation_step = np.mean(deliberation_step_probability)

    # Build final JSON
    probability_json = {
        "deliberation_results": first_step_results,
        "deliberation_probability": deliberation_step_probability,
        "deliberation_mean_probability": mean_deliberation_step,
        "deliberation_sd": sd_deliberation_step,
        "deliberation_probability_result": deliberation_step_probability_result,
    }

    return probability_json


def get_probabilities(first_step_results, revision_results, group_results, is_multiple_choice, options, probabilities) -> Dict[
    str, Any]:
    revision_step_probability = [result['revised_probability'] for result in revision_results.values() if
                                 'revised_probability' in result]
    bound_bottom_to_one_hundred = lambda x: min(max(x, 1), 100)
    revision_step_probability = [bound_bottom_to_one_hundred(prob) for prob in revision_step_probability]

    revision_step_probability_result = int(round(np.mean(revision_step_probability))) if not is_multiple_choice else {
        opt: val / 100.0 for opt, val in normalize_and_average(revision_step_probability, options=options).items()
    }

    sd_revision_step = np.std(revision_step_probability, ddof=1)
    mean_revision_step = np.mean(revision_step_probability)

    # Build final JSON
    probabilities.update({
        "group_results": group_results,
        "revision_results": revision_results,
        "revision_probability": revision_step_probability,
        "revision_mean_probability": mean_revision_step,
        "revision_sd": sd_revision_step,
        "revision_probability_result": revision_step_probability_result,
    })

    return probabilities


def enrich_probabilities(probabilities, question_details, news, forecast_date, summarization, forecasters):
    probabilities["question_details"] = question_details
    probabilities["news"] = news
    probabilities["date"] = forecast_date
    probabilities["summary"] = summarization
    probabilities["forecasters"] = forecasters
