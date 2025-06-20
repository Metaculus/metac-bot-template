import asyncio
import os

from logic.offline.forecaster import forecast_from_json

FORECAST_DIR = "forecasts"


async def main() -> None:
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    for file in files:
        await forecast_from_json(os.path.join(FORECAST_DIR, file), is_woc=False)


if __name__ == "__main__":
    asyncio.run(main())
