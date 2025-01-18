import json

from autogen import AssistantAgent

from utils.PROMPTS import KEYWORDS_PROMPT
from utils.config import get_gpt_config


class QuestionToQuery:
    def __init__(self):

        self._agent = AssistantAgent(name="KeywordAgent", system_message=KEYWORDS_PROMPT, llm_config=get_gpt_config(42, 0.7, "gpt-4o", 120),
                                     human_input_mode="NEVER")

    def _create_messages(self,question):
        return [
            {"role": "system",
             "content": self._agent.system_message},
            {"role": "user",
             "content": f"Question Title:{question['title']}\n\nQuestion Description:{question['description']}"},
        ]

    def generate_query(self, question):
        messages = self._create_messages(question)
        result = self._agent.generate_reply(messages=messages)
        parsed_json = self._validate_and_parse_response(result)
        return parsed_json

    @staticmethod
    def _validate_and_parse_response(response):
        response = response.replace("json", "").replace("```", "")
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {response}")