from forecasting_tools import MetaculusQuestion

from datetime import datetime
import json


def get_latest_data(question: MetaculusQuestion) -> dict:
    data = json.loads(question.model_dump_json())
    return (
        data["api_json"]["question"]["aggregations"]["recency_weighted"]["latest"] or {}
    )


def get_latest_forecast_values(question: MetaculusQuestion) -> list:
    return get_latest_data(question).get("forecast_values", [])


def verify_community_prediction_exists(question: MetaculusQuestion) -> bool:
    res = json.loads(question.model_dump_json())
    try:
        reveal_time = datetime.fromisoformat(res["cp_reveal_time"])
        return reveal_time < datetime.now()
    except (KeyError, TypeError, ValueError):
        return False


def get_binary_community_prediction(question: MetaculusQuestion) -> float | None:
    prediction = get_latest_data(question).get("centers", [])
    print(f">>> binary_comm_prediction: {prediction}")
    return prediction[0] if prediction else None


def get_prompt_context(question: MetaculusQuestion) -> dict:
    context = {
        "question_text": question.question_text,
        "resolution_criteria": question.resolution_criteria,
        "fine_print": question.fine_print
    }
    return context
