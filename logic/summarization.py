from typing import Dict

from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from flaml.autogen import ConversableAgent

from logic.chat import validate_and_parse_response
from logic.utils import create_prompt


async def run_summarization_phase(results: Dict[str, dict],
                                  question: Dict[str, str], summarization_assistant: ConversableAgent, news:str) -> str:
    full_question = create_prompt(question, news=news)
    first_phase_results_to_string_indented = "\n\n".join(
        [f"{key}:\n{value}" for key, value in results.items()])
    prompt = (f"The question and question details are:\n\n{full_question}\n\n"
              f"Below are the results of the two phases for each expert:\n\n{first_phase_results_to_string_indented}\n\n")
    result = await summarization_assistant.a_generate_reply(messages=[{"role": "system", "content": prompt}])
    summary = validate_and_parse_response(result)

    return summary['summary']
