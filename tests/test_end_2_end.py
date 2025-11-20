import os
import datetime

import pytest

from logic.offline.forecaster import forecast_from_json, slowly, dispassion
from main import forecast_questions

TEST_FORECAST_DIR = "forecasts/fall"

@pytest.mark.asyncio
async def test_e2e_binary():
    open_question_id_post_id = [(578, 578)]
    now = datetime.datetime.now()
    await forecast_questions(
            open_question_id_post_id,
            submit_prediction=True,
            skip_previously_forecasted_questions=False,
            use_hyde=False,
            cache_seed=42
        )
    print(f"time taken to run: {datetime.datetime.now() - now}")


@pytest.mark.parametrize("forecasting_function", [dispassion, slowly])
@pytest.mark.asyncio
async def test_offline_e2e_binary(forecasting_function):
    files = [f for f in os.listdir(TEST_FORECAST_DIR) if f.endswith(".json")]
    for file in files:
        await forecast_from_json(forecasting_function=forecasting_function,path = os.path.join(TEST_FORECAST_DIR, file), is_woc=False)

