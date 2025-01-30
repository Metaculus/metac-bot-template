from typing import Dict

from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from logic.chat import validate_and_parse_response


async def run_summarization_phase(first_phase_results:Dict[str,dict],news_analysis_results:Dict[str,dict],question:str,summarization_assistant:GPTAssistantAgent)-> str:
    first_phase_results_to_string_indented = "\n\n".join([f"{key}:\n{value}" for key,value in first_phase_results.items()])
    news_results_to_string_indented = "\n\n".join([f"{key}:\n{value}" for key,value in news_analysis_results.items()])
    prompt = (f"Below are the results from the first phase of the analysis (Phase 1):\n\n{first_phase_results_to_string_indented}\n\n{question}"
              f"Below are the news analysis results from the professional experts (Phase 2):\n\n{news_results_to_string_indented}\n\n{question}")
    result = await summarization_assistant.a_generate_reply(messages=[{"role":"system","content":prompt}])
    summary = validate_and_parse_response(result)

    return summary['summary']