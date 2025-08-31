from forecasting_tools import (
    MetaculusQuestion
)

import json

def get_latest_data(question: MetaculusQuestion) -> dict:
    data = json.loads(question.model_dump_json())
    return data["api_json"]["question"]["aggregations"]["recency_weighted"]["latest"] or {}

def get_latest_forecast_values(question: MetaculusQuestion) -> list:
    return get_latest_data(question).get("forecast_values", [])

def get_binary_community_prediction(question: MetaculusQuestion) -> float | None:
    prediction = get_latest_data(question).get("centers", [])
    print(f">>> binary_comm_prediction: {prediction}")
    return prediction[0] if prediction else None
