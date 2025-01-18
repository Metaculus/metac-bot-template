from typing import Dict, List, Any

from flaml.autogen import ConversableAgent

from agents.agent_creator import create_gpt_assistant
from logic.chat import validate_and_parse_response
from utils.PROMPTS import SPECIFIC_EXPERTISE


def run_expert_extractor(expert_identifier: ConversableAgent, question: str):
    experts = expert_identifier.generate_reply(
        messages=[{"role": "system", "content": expert_identifier.system_message},
                  {"role": "user", "content": question}])

    experts = validate_and_parse_response(experts)
    academic_disciplines = []
    frameworks = []
    professional_expertise = []
    specialty = []
    academic_disciplines += [discipline["discipline"] for discipline in experts['academic_disciplines']]
    frameworks += [discipline['frameworks'] for discipline in experts["academic_disciplines"]]
    professional_expertise += [expertise['expertise'] for expertise in experts["professional_expertise"]]
    specialty += [expertise['specialty'] for expertise in experts["professional_expertise"]]

    return academic_disciplines, frameworks, professional_expertise, specialty


def expert_creator(experts: List[str], frameworks_specialties: List[List[str]], config: Dict[str, Any], prompt: str = SPECIFIC_EXPERTISE) -> List[
    ConversableAgent]:
    all_agents = []
    for expert, specialties in zip(experts, frameworks_specialties):
        for specialty in specialties:
            agent = create_gpt_assistant(expertise=expert, config=config, specialty_expertise=specialty, prompt=prompt)
            all_agents.append(agent)
    return all_agents

