from typing import Dict, Any, Literal, List

from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from utils.PROMPTS import SPECIFIC_META_MESSAGE_EXPERTISE, EXPERTISE_ANALYZER_PROMPT, SUMMARIZATION_PROMPT, \
    SPECIFIC_EXPERTISE_MULTIPLE_CHOICE
from utils.utils import to_camel_case

def create_agent(name:str, expertise:str, config:Dict[str,Any], human_input: Literal["ALWAYS", "NEVER"] = "NEVER") -> ConversableAgent:
    system_message = SPECIFIC_META_MESSAGE_EXPERTISE.format(expertise=expertise)
    return ConversableAgent(name = name, system_message=system_message, llm_config=config, human_input_mode=human_input)

async def create_gpt_assistant(config:Dict[str,Any], expertise: str, specialty_expertise: str, prompt:str = SPECIFIC_META_MESSAGE_EXPERTISE) -> GPTAssistantAgent:
    expertise_and_specialty_framework = f"{expertise} ({specialty_expertise})"
    name = f'{to_camel_case(expertise)}{to_camel_case(specialty_expertise)}Agent'
    system_message = prompt.format(expertise=expertise_and_specialty_framework)
    return GPTAssistantAgent(name=name, instructions=system_message, llm_config=config)

async def create_gpt_assistant_multiple_choices(config:Dict[str,Any], expertise: str, specialty_expertise: str, options:List[str], prompt:str = SPECIFIC_EXPERTISE_MULTIPLE_CHOICE) -> GPTAssistantAgent:
    expertise_and_specialty_framework = f"{expertise} ({specialty_expertise})"
    name = f'{to_camel_case(expertise)}{to_camel_case(specialty_expertise)}Agent'
    system_message = prompt.format(expertise=expertise_and_specialty_framework, options=options)
    return GPTAssistantAgent(name=name, instructions=system_message, llm_config=config)


def create_admin(system_message, code_execution_config: Literal[False] = False) -> UserProxyAgent:
    return UserProxyAgent(name="Admin", system_message=system_message, code_execution_config=code_execution_config)


def create_chat(agents: List[ConversableAgent], messages: List[dict],selection_method:Literal, max_rounds: int) -> GroupChat:
    return GroupChat(messages=messages, agents=agents, max_round=max_rounds, speaker_selection_method=selection_method)


def create_chat_manager(groupchat: GroupChat, config: Dict[str, Any]) -> GroupChatManager:
    return GroupChatManager(groupchat=groupchat, llm_config=config)

def create_experts_analyzer_assistant(config:Dict[str,Any], prompt:str = EXPERTISE_ANALYZER_PROMPT) -> ConversableAgent:
    return ConversableAgent(name="ExpertsAnalyzerAgent",system_message=prompt, llm_config=config, human_input_mode="NEVER")

def create_summarization_assistant(config:Dict[str,Any]) -> ConversableAgent:
    return ConversableAgent(name="SummarizationAgent",system_message=SUMMARIZATION_PROMPT, llm_config=config, human_input_mode="NEVER")