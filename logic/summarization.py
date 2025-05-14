from typing import Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import BaseChatMessage

from logic.chat import validate_and_parse_response
from logic.utils import create_prompt


async def run_summarization_phase(results: Dict[str, dict],
                                  question: Dict[str, str], summarization_assistant: AssistantAgent) -> str:
    full_question = create_prompt(question,news = "No News Articles")
    first_phase_results_to_string_indented = "\n\n".join(
        [f"{key}:\n{value}" for key, value in results.items()])
    prompt = (f"The question and question details are:\n\n{full_question}\n\n"
              f"Below are the results of the two phases for each expert:\n\n{first_phase_results_to_string_indented}\n\n")

    result = await summarization_assistant.run(task=prompt)
    summary = validate_and_parse_response(result)

    return summary['summary']
