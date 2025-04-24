from typing import List, Dict, Union, Tuple

import numpy as np

from agents.agent_creator import create_chat, create_chat_manager, create_admin
from logic.call_asknews import run_research
from logic.utils import extract_question_details, get_all_experts, perform_forecasting_phase, \
    perform_revised_forecasting_step
from utils.config import get_gpt_config
from utils.utils import normalize_and_average


async def chat_group_single_question(
        question_details: dict,
        cache_seed: int = 42,
        is_multiple_choice: bool = False,
        options: List[str] = None,
        is_woc:bool=False,
        num_of_experts:str|None = None,
        news: str = None
) -> Tuple[Union[int, Dict[str, float]], str]:
    title, description, fine_print, resolution_criteria, forecast_date = extract_question_details(question_details)
    config = get_gpt_config(cache_seed, 0.7, "gpt-4o", 120)


    news = await run_research(question_details)

    # Identify and create experts
    all_experts = await get_all_experts(config, question_details , is_multiple_choice, options,is_woc, num_of_experts)

    # Forecasting
    results = await perform_forecasting_phase(all_experts, question_details, news=news,
                                              is_multiple_choice=is_multiple_choice, options=options)

    processed_results = [{"content":f"{content}","name":name,"role":"assistant"} for name,content in results.items()]

    # Extract probabilities
    final_probability = [result['final_probability'] for result in results.values() if 'final_probability' in result]
    initial_probability = [result['initial_probability'] for result in results.values() if 'initial_probability' in result]

    final_result = int(round(np.mean(final_probability))) if not is_multiple_choice else {
        opt: val / 100.0 for opt, val in normalize_and_average(final_probability, options=options).items()
    }

    sd_initial_step = np.std(initial_probability, ddof=1)
    sd_deliberation_step = np.std(final_probability, ddof=1)
    mean_initial_step = np.mean(initial_probability)
    mean_deliberation_step = np.mean(final_probability)

    group_chat = create_chat(all_experts,processed_results,"random", len(all_experts)+1)

    chat_manager = create_chat_manager(groupchat=group_chat, config=config)

    admin = create_admin(system_message="message")
    chat_results = admin.initiate_chat(chat_manager,message = f"Hello, I am the admin of this chat. This is th latest history of messages you've provided me:\n\n {processed_results}.\n\n tell me what was your opinion on the subject?",clear_history=False)

    final_step_results = await perform_revised_forecasting_step(all_experts)

















