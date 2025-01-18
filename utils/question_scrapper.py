import json
import os
import random
from typing import Any, Dict

import warnings
import requests


class QuestionScrapper:
    def __init__(self,tournament_id:int):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self._api_base_url = "https://www.metaculus.com/api2"
        self._tournament_id = tournament_id
        self._auth = self._create_auth_headers()

    def _create_auth_headers(self):
        metaculus_token = os.environ.get("METACULUS_API_KEY")
        auth_headers = {"headers": {"Authorization": f"Token {metaculus_token}"}}
        return auth_headers

    def post_question_comment(self,question_id: int, comment_text: str) -> None:
        """
        Post a comment on the question page as the bot user.
        """

        response = requests.post(
            f"{self._api_base_url}/comments/",
            json={
                "comment_text": comment_text,
                "submit_type": "N",
                "include_latest_prediction": True,
                "question": question_id,
            },
            **self._auth,
        )
        if not response.ok:
            raise Exception(response.text)

    def post_question_prediction(self,question_id: int, prediction_percentage: float) -> None:
        """
        Post a prediction value (between 1 and 100) on the question.
        """
        assert 1 <= prediction_percentage <= 100, "Prediction must be between 1 and 100"
        url = f"{self._api_base_url}/questions/{question_id}/predict/"
        response = requests.post(
            url,
            json={"prediction": float(prediction_percentage) / 100},
            **self._auth,
        )
        if not response.ok:
            raise Exception(response.text)

    def get_question_details(self,question_id: int) -> dict:
        """
        Get all details about a specific question.
        """
        url = f"{self._api_base_url}/questions/{question_id}/"
        response = requests.get(
            url,
            **self._auth,
        )
        if not response.ok:
            raise Exception(response.text)
        return json.loads(response.content)

    def list_questions(self, offset=0, count=10,status = "open") -> Dict[str, Any]:
        """
        List (all details) {count} questions from the {tournament_id}
        """
        url_qparams = {
            "limit": count,
            "offset": offset,
            "has_group": "false",
            "order_by": "-activity",
            "forecast_type": "binary",
            "project": self._tournament_id,
            "status": status,
            "type": "forecast",
            "include_description": "true",
        }
        url = f"{self._api_base_url}/questions/"
        response = requests.get(url, **self._auth, params=url_qparams)
        if not response.ok:
            raise Exception(response.text)
        data = json.loads(response.content)
        return data

    def get_random_question(self, status = "closed") -> Dict[str, Any]:
        questions = self.list_questions(status=status)
        questions = questions["results"]
        return random.choice(questions)

