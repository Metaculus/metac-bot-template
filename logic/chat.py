import datetime
import json
from typing import List, Dict

from asknews_sdk.dto import SearchResponse
from autogen import ConversableAgent

from utils.PROMPTS import NEWS_STEP_INSTRUCTIONS, NEWS_OUTPUT_FORMAT


def run_first_stage_forecasters(forecasters: List[ConversableAgent], question: str, prompt:str = "") -> Dict[str, dict]:
    analyses = {}
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    phase_one_introduction = f"Welcome to Phase 1. Today's date is {todays_date} Your forecasting question is: '{question}'"
    for forecaster in forecasters:
        if prompt == "":
            prompt = forecaster.system_message
        result = forecast(forecaster, forecaster.system_message,phase_one_introduction)
        analyses[forecaster.name] = result
    return analyses

def run_second_stage_forecasters(forecasters: List[ConversableAgent], news: str, prompt:str = NEWS_STEP_INSTRUCTIONS, output_format:str = NEWS_OUTPUT_FORMAT) -> Dict[str, dict]:
    analyses = {}

    phase_two_instruction_news_analysis = f"Welcome to Phase 2: News Analysis. Below are the news articles you'll need to take into consideration:\n\n{news}"
    for forecaster in forecasters:
        result = forecast(forecaster, prompt,phase_two_instruction_news_analysis+output_format)
        analyses[forecaster.name] = result
    return analyses


def forecast(forecaster: ConversableAgent, phase_instructions:str, phase_introduction: str) -> Dict[str, dict]:
    result = forecaster.generate_reply(
        messages=[
            {"role": "assistant", "content": phase_instructions},
            {"role": "user", "content": phase_introduction},
        ]
    )
    return validate_and_parse_response(result['content'])


def validate_and_parse_response(response):
    try:
        response = response.replace("json","").replace("```","")
        return json.loads(response)

    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON response: {response}")


# def initiate_dialogue(forecasters: List[ConversableAgent], history: Dict[str, Any], news: SearchResponse, question: str,
#                       admin_message: str) -> Tuple[ChatResult, ChatResult]:
#     llm_config = get_gpt_config(42, 0.7, "gpt-4o", 120)
#     introduction_message = generate_introduction_message(history, question)
#     news_message = generate_news_message(news)
#     groupchat = create_chat(forecasters, list(history.values()), "round_robin", (len(forecasters) + 1))
#     groupchat_manager = create_chat_manager(groupchat, llm_config)
#     admin = create_admin(admin_message)
#     second_phase_results = admin.initiate_chat(groupchat_manager, message=introduction_message,
#                                                summary_method="reflection_with_llm")
#     third_phase_results = admin.initiate_chat(groupchat_manager, message=news_message, clear_history=False,
#                                               summary_method="reflection_with_llm")
#
#     return second_phase_results, third_phase_results
#
#
# def generate_news_message(news: SearchResponse):
#     third_step_instructions = THIRD_STEP_INSTRUCTIONS+news.as_string+"\n\n"+FINAL_OUTPUT_FORMAT
#     return third_step_instructions
#
#
# # def generate_introduction_message(initial_analyses: Dict[str, Any], question: str):
#     introduction_message = (SECOND_STEP_INSTRUCTIONS+
#             "Welcome to Phase 2: group deliberation. Below are the initial predictions made by the forecasters in the group:\n\n" +
#             "\n".join(
#                 f"**{name}:**\n"
#                 f"Initial Probability: {analysis['initial_probability']}% (Reasoning: {analysis['initial_reasoning']})\n"
#                 f"Factors:\n" +
#                 "\n".join(
#                     f"- {factor['name']}: {factor['reasoning']} (Effect: {factor['effect']})"
#                     for factor in analysis["factors"]
#                 ) +
#                 f"\nFinal Probability: {analysis['final_probability']}%\n"
#                 for name, analysis in initial_analyses.items()
#             ) +
#             "\n\nDeliberate on the following question:\n"
#             f"**'{question}'**\n\n"
#             "Discuss each other's forecasts, critique points where necessary, and refine your predictions accordingly.\n"
#             "You are participants on a prediction platform such as Metaculus, PredictIt or Polymarket.\n\n"+REFINED_OUTPUT_FORMAT
#
#     )
#     return introduction_message
