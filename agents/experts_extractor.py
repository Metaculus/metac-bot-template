from typing import Dict, List, Any

from autogen_agentchat.agents import AssistantAgent

from agents.agent_creator import create_agent
from logic.chat import validate_and_parse_response
from utils.PROMPTS import SPECIFIC_META_MESSAGE_EXPERTISE


async def run_expert_extractor(expert_identifier: AssistantAgent, question: str):
    experts = await expert_identifier.run(task=question)

    parsed_experts = validate_and_parse_response(experts.messages[1].content)
    academic_disciplines = []
    frameworks = []
    professional_expertise = []
    specialty = []
    academic_disciplines += [discipline["discipline"] for discipline in parsed_experts['academic_disciplines']]
    frameworks += [discipline['frameworks'] for discipline in parsed_experts["academic_disciplines"]]
    professional_expertise += [expertise['expertise'] for expertise in parsed_experts["professional_expertise"]]
    specialty += [expertise['specialty'] for expertise in parsed_experts["professional_expertise"]]

    return academic_disciplines, frameworks, professional_expertise, specialty


async def expert_creator(experts: List[str], frameworks_specialties: List[List[str]], config: Dict[str, Any], prompt: str = SPECIFIC_META_MESSAGE_EXPERTISE) -> List[
    AssistantAgent]:
    all_agents = []
    for expert, specialties in zip(experts, frameworks_specialties):
        for specialty in specialties:
            agent = create_agent(expertise=expert, config=config, specialty_expertise=specialty, prompt=prompt)
            all_agents.append(agent)
    return all_agents
