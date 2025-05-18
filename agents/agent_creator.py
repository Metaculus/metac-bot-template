import os
from typing import Dict, Any, Literal, List

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.agents.openai import OpenAIAssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

from utils.PROMPTS import SPECIFIC_META_MESSAGE_EXPERTISE, EXPERTISE_ANALYZER_PROMPT, SUMMARIZATION_PROMPT
from utils.utils import to_camel_case

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY")
OPEN_AI_CLIENT = AsyncOpenAI(api_key=OPENAI_API_KEY)


def create_agent(config: Dict[str, Any], expertise: str, specialty_expertise: str,
                 prompt: str = SPECIFIC_META_MESSAGE_EXPERTISE) -> AssistantAgent:
    client = OpenAIChatCompletionClient(model="gpt-4.1", temperature=1)
    expertise_and_specialty_framework = f"{expertise} ({specialty_expertise})"
    name = f'{to_camel_case(expertise)}{to_camel_case(specialty_expertise)}'
    name = name[:63] # Limit to 63 characters for autogen purposes
    system_message = prompt.format(expertise=expertise_and_specialty_framework)
    return AssistantAgent(name=name, system_message=system_message, model_client=client)


def create_openai_agent(config: Dict[str, Any], expertise: str, specialty_expertise: str,
                       prompt: str = SPECIFIC_META_MESSAGE_EXPERTISE) -> OpenAIAssistantAgent:
    expertise_and_specialty_framework = f"{expertise} ({specialty_expertise})"
    name = f'{to_camel_case(expertise)}{to_camel_case(specialty_expertise)}'
    system_message = prompt.format(expertise=expertise_and_specialty_framework)
    return OpenAIAssistantAgent(client=OPEN_AI_CLIENT, name=name, description="You are an expert forecaster",
                                instructions=system_message, model="gpt-4.1", temperature=config["temperature"])


def create_group(agents: List) -> RoundRobinGroupChat:
    return RoundRobinGroupChat(participants=agents, max_turns=len(agents))


def create_admin(system_message, code_execution_config: Literal[False] = False) -> UserProxyAgent:
    return UserProxyAgent(name="Admin", system_message=system_message, code_execution_config=code_execution_config)


def create_summarization_assistant(config: Dict[str, Any]) -> OpenAIAssistantAgent:
    return OpenAIAssistantAgent(name="SummarizationAgent", description="You are a summarizer",
                                instructions=SUMMARIZATION_PROMPT, model="gpt-4.1", temperature=config["temperature"],
                                client=OPEN_AI_CLIENT)


def create_experts_analyzer_assistant(config: Dict[str, Any],
                                      prompt: str = EXPERTISE_ANALYZER_PROMPT) -> OpenAIAssistantAgent:
    return OpenAIAssistantAgent(name="ExpertsAnalyzerAgent", instructions=prompt, model="gpt-4.1",description="You identify well-established areas of expertise to answer a forecasting question",
                                temperature=config["temperature"], client=OPEN_AI_CLIENT)
