import asyncio
import os

from logic.offline.forecaster import forecast_from_json, dispassion, slowly

FORECAST_DIR = "forecasts/q2"

async def main_dispassion() -> None:
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    for file in files:
        await forecast_from_json(forecasting_function=dispassion,path = os.path.join(FORECAST_DIR, file), is_woc=False)

async def main_slowly() -> None:
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    for file in files:
        await forecast_from_json(forecasting_function=slowly, path = os.path.join(FORECAST_DIR, file), is_woc=True)


if __name__ == "__main__":
    asyncio.run(main_dispassion())
    asyncio.run(main_slowly())
