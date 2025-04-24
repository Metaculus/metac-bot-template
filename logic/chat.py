import datetime
import json
import re
from typing import List, Dict

from autogen import ConversableAgent

from utils.PROMPTS import NEWS_STEP_INSTRUCTIONS, NEWS_OUTPUT_FORMAT


async def run_first_stage_forecasters(forecasters: List[ConversableAgent], question: str,
                                      system_message: str = "", options: List[str] = "") -> Dict[str, dict]:
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    phase_one_introduction = f"Today's date is {today_date} Your forecasting question is: '{question}'\n\n"

    if options:
        phase_one_introduction += f"\n\nOptions:\n\n{', '.join(options)}\n"

    analyses = await gather_forecasts(forecasters, system_message, phase_one_introduction)
    return analyses


async def run_second_stage_forecasters(forecasters: List[ConversableAgent],
                                       prompt: str = NEWS_STEP_INSTRUCTIONS) -> Dict[str, dict]:
    phase_two_instruction_news_analysis = f"Please revise your answer based on previous steps."
    analyses = await gather_forecasts(forecasters, prompt, phase_two_instruction_news_analysis)
    return analyses


async def forecast(forecaster: ConversableAgent, phase_instructions: str, phase_introduction: str) -> Dict[str, dict]:
    result = await forecaster.a_generate_reply(
        messages=[
            {"role": "assistant", "content": phase_instructions},
            {"role": "user", "content": phase_introduction},
        ]
    )
    return validate_and_parse_response(result['content'])


async def gather_forecasts(forecasters: List[ConversableAgent], system_message: str, phase_introduction: str) -> Dict[
    str, dict]:
    result = {}
    for forecaster in forecasters:
        if not system_message:
            system_message = forecaster.system_message
        try:
            result = await forecast(forecaster, system_message, phase_introduction)
            return result
        except Exception as e:
            print(f"Error with {forecaster.name}: {e}\n\n")
            print(f"Skipping forecaster:\n{forecaster.name}")
    return result


def validate_and_parse_response(response):
    try:
        response = response.replace("json", "").replace("```", "").replace("\n", "")
        if not response.endswith("}"):
            ind = find_last_curly_brace_position(response)
            response = response[:ind + 1]

        return json.loads(response)

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON response: {response}")


def find_last_curly_brace_position(text):
    matches = list(re.finditer(r'}', text))  # Find all closing curly braces
    if matches:
        last_position = matches[-1].start()  # Get the start position of the last match
        return last_position
    else:
        return -1  # Return -1 if no curly brace is found
