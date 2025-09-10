import asyncio
import logging
import os

from logic.offline.forecaster import forecast_from_json, dispassion, slowly

# Configure logging to display INFO messages to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Silence third-party library logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

FORECAST_DIR = "forecasts/fall"

async def main_dispassion() -> None:
    logging.info("=== Starting offline dispassion forecasting ===")
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    logging.info("Found %s JSON files in %s", len(files), FORECAST_DIR)
    
    for file in files:
        logging.info("Processing dispassion forecast for: %s", file)
        await forecast_from_json(forecasting_function=dispassion, path=os.path.join(FORECAST_DIR, file), is_woc=False)
    
    logging.info("=== Completed offline dispassion forecasting ===")

async def main_slowly() -> None:
    logging.info("=== Starting offline slowly forecasting ===")
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    logging.info("Found %s JSON files in %s", len(files), FORECAST_DIR)
    
    for file in files:
        logging.info("Processing slowly forecast for: %s", file)
        await forecast_from_json(forecasting_function=slowly, path=os.path.join(FORECAST_DIR, file), is_woc=False)
    
    logging.info("=== Completed offline slowly forecasting ===")

async def main() -> None:
    """Run both forecasting methods concurrently"""
    logging.info("=== Starting offline forecasting pipeline ===")
    
    # Run both functions concurrently
    await asyncio.gather(
        main_dispassion(),
        main_slowly()
    )
    
    logging.info("=== All offline forecasting completed ===")

if __name__ == "__main__":
    asyncio.run(main())
