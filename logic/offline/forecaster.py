import json
from typing import List, Dict, Tuple, Union

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from agents.agent_creator import create_group, create_summarization_assistant
from logic.chat import validate_and_parse_response
from logic.summarization import run_summarization_phase
from logic.utils import (
    extract_question_details,
    perform_forecasting_phase,
    perform_revised_forecasting_step,
    strip_title_to_filename,
    build_and_write_json,
    get_probabilities,
    enrich_probabilities,
    get_first_phase_probabilities,
    get_relevant_contexts_to_group_discussion,
)
from utils.PROMPTS import SPECIFIC_META_MESSAGE_EXPERTISE_DISPASSION, \
    SPECIFIC_META_MESSAGE_EXPERTISE_SLOWLY, FIRST_PHASE_INSTRUCTIONS_SLOWLY, GROUP_INSTRUCTIONS_DISPASSION, \
    SPECIFIC_META_MESSAGE_EXPERTISE, GROUP_INSTRUCTIONS
from utils.config import get_gpt_config

EXP_NAME_DISPASSION = "_dispassion"
EXP_NAME_SLOWLY = "_slowly"


def _create_offline_agent(name: str, chosen_system_message: str) -> AssistantAgent:
    client = OpenAIChatCompletionClient(model="gpt-4.1", temperature=0.7)
    system_message = chosen_system_message.format(expertise=name)

    camel_name = _to_camel_case(name)
    # Ensure the name doesn't exceed 63 characters
    if len(camel_name) > 63:
        camel_name = _smart_truncate_agent_name(camel_name, name)

    agent = AssistantAgent(name=camel_name, system_message=system_message, model_client=client)
    agent.display_name = name
    return agent


import re


def _smart_truncate_agent_name(camel_name: str, original_name: str) -> str:
    """Simple truncation to stay under 64 characters."""
    return camel_name[:63]


def _to_camel_case(text: str) -> str:
    # Remove parentheses and their contents, then add back the contents as separate words
    parenthetical = re.findall(r'\((.*?)\)', text)
    no_parens = re.sub(r'\(.*?\)', '', text)

    # Combine the main text and the parenthetical content
    words = no_parens.split() + [word for phrase in parenthetical for word in phrase.split()]

    # Clean up any stray punctuation and make all lowercase first
    words = [re.sub(r'\W+', '', word) for word in words if word.strip()]

    # Convert to camelCase
    if not words:
        return ''
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])


async def dispassion(
        question_details: dict,
        news: str,
        expert_names: List[str],
        cache_seed: int = 42,
        is_multiple_choice: bool = False,
        options: List[str] | None = None,
        is_woc: bool = False,
) -> Tuple[Union[int, Dict[str, float]], str]:
    title, description, fine_print, resolution_criteria, forecast_date, aggregations = extract_question_details(question_details)
    config = get_gpt_config(cache_seed, 1, "gpt-4.1", 120)

    experts = [_create_offline_agent(name, SPECIFIC_META_MESSAGE_EXPERTISE_DISPASSION) for name in expert_names]
    group_chat = create_group(experts)

    results = await perform_forecasting_phase(experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options,
                                              system_message = FIRST_PHASE_INSTRUCTIONS_SLOWLY)

    group_contextualization = get_relevant_contexts_to_group_discussion(results)
    probabilities = get_first_phase_probabilities(results, is_multiple_choice, options)

    group_results = await group_chat.run(
        task=GROUP_INSTRUCTIONS_DISPASSION.format(phase1_results_json_string=group_contextualization,
                                       forecasters_list=expert_names))

    parsed_group_results = {
        answer.source: validate_and_parse_response(answer.content)
        for answer in group_results.messages if answer.source != "user"
    }

    revision_results = await perform_revised_forecasting_step(
        experts, question_details, news=news,
        is_multiple_choice=is_multiple_choice, options=options
    )

    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(results, question_details, summarization_assistant)

    probabilities = get_probabilities(results, revision_results, parsed_group_results,
                                      is_multiple_choice, options, probabilities)

    enrich_probabilities(probabilities, question_details, news, forecast_date, summarization, expert_names)

    final_answer = probabilities["revision_probability_result"]

    filename = strip_title_to_filename(title) + EXP_NAME_DISPASSION
    await build_and_write_json(filename, probabilities, is_woc, subdirectory="dispassion")

    return final_answer, summarization


async def slowly(
        question_details: dict,
        news: str,
        expert_names: List[str],
        cache_seed: int = 42,
        is_multiple_choice: bool = False,
        options: List[str] | None = None,
        is_woc: bool = False,
) -> Tuple[Union[int, Dict[str, float]], str]:
    title, description, fine_print, resolution_criteria, forecast_date, aggregations = extract_question_details(question_details)
    config = get_gpt_config(cache_seed, 1, "gpt-4.1", 120)

    experts = [_create_offline_agent(name, SPECIFIC_META_MESSAGE_EXPERTISE_SLOWLY) for name in expert_names]

    results = await perform_forecasting_phase(experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options)

    probabilities = get_first_phase_probabilities(results, is_multiple_choice, options)


    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(results, question_details, summarization_assistant)

    final_answer = probabilities["deliberation_probability_result"]

    filename = strip_title_to_filename(title) + EXP_NAME_SLOWLY
    await build_and_write_json(filename, probabilities, is_woc, subdirectory="slowly")

    return final_answer, summarization



async def main_pipeline(
        question_details: dict,
        news: str,
        expert_names: List[str],
        cache_seed: int = 42,
        is_multiple_choice: bool = False,
        options: List[str] | None = None,
        is_woc: bool = False,
) -> Tuple[Union[int, Dict[str, float]], str]:
    title, description, fine_print, resolution_criteria, forecast_date, aggregations = extract_question_details(question_details)
    config = get_gpt_config(cache_seed, 1, "gpt-4.1", 120)

    experts = [_create_offline_agent(name, SPECIFIC_META_MESSAGE_EXPERTISE) for name in expert_names]
    group_chat = create_group(experts)

    results = await perform_forecasting_phase(experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options,
                                              system_message = FIRST_PHASE_INSTRUCTIONS_SLOWLY)

    group_contextualization = get_relevant_contexts_to_group_discussion(results)
    probabilities = get_first_phase_probabilities(results, is_multiple_choice, options)

    group_results = await group_chat.run(
        task=GROUP_INSTRUCTIONS.format(phase1_results_json_string=group_contextualization,
                                       forecasters_list=expert_names))

    parsed_group_results = {
        answer.source: validate_and_parse_response(answer.content)
        for answer in group_results.messages if answer.source != "user"
    }

    revision_results = await perform_revised_forecasting_step(
        experts, question_details, news=news,
        is_multiple_choice=is_multiple_choice, options=options
    )

    summarization_assistant = create_summarization_assistant(config)
    summarization = await run_summarization_phase(results, question_details, summarization_assistant)

    probabilities = get_probabilities(results, revision_results, parsed_group_results,
                                      is_multiple_choice, options, probabilities)

    enrich_probabilities(probabilities, question_details, news, forecast_date, summarization, expert_names)

    final_answer = probabilities["revision_probability_result"]

    filename = strip_title_to_filename(title) + "recreated"
    await build_and_write_json(filename, probabilities, is_woc, subdirectory="recreation")

    return final_answer, summarization




async def forecast_from_json(forecasting_function, path: str, is_woc: bool = False, cache_seed: int = 42) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_details = data.get("question_details", {})
    news = data.get("news", "")
    expert_names = data.get("forecasters", [])
    try:
        await forecasting_function(question_details=question_details, news=news, expert_names=expert_names,
                               cache_seed=cache_seed,
                               is_multiple_choice=question_details.get("type") == "multiple_choice",
                               options=question_details.get("options"), is_woc=is_woc)
    except Exception as e:
        print(f"Error processing question '{question_details.get('title', 'Unknown Title')}': {e}")
