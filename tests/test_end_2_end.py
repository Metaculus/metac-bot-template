import os
import random
import unittest
import datetime

from logic.offline.forecaster import forecast_from_json
from main import forecast_questions

TEST_FORECAST_DIR = "tests/forecasts/q2"




class TestEnd2End(unittest.IsolatedAsyncioTestCase):

    async def test_e2e_binary(self):
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


    async def test_offline_e2e_binary(self):
        files = [f for f in os.listdir(TEST_FORECAST_DIR) if f.endswith(".json")]
        for file in files:
            await forecast_from_json(os.path.join(TEST_FORECAST_DIR, file), is_woc=False)

if __name__ == "__main__":
    unittest.main()