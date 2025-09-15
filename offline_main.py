import asyncio
import json
import logging
import os

from logic.offline.forecaster import forecast_from_json, dispassion, slowly
from logic.utils import strip_title_to_filename

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


def get_question_title_from_json(file_path: str) -> str:
    """Extract question title from JSON file to determine output filenames."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("question_details", {}).get("title", "")
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logging.warning("Could not extract title from %s: %s", file_path, e)
        return ""


def should_skip_file(input_file_path: str) -> bool:
    """Check if both dispassion and slowly forecasts already exist for this question."""
    title = get_question_title_from_json(input_file_path)
    if not title:
        return False

    base_filename = strip_title_to_filename(title)
    slowly_path = f"forecasts/fall/slowly/{base_filename}_slowly.json"
    dispassion_path = f"forecasts/fall/dispassion/{base_filename}_dispassion.json"

    both_exist = os.path.exists(slowly_path) and os.path.exists(dispassion_path)

    if both_exist:
        logging.info("Skipping %s - both forecasts already exist", os.path.basename(input_file_path))

    return both_exist

async def main_dispassion() -> None:
    logging.info("=== Starting offline dispassion forecasting ===")
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    logging.info("Found %s JSON files in %s", len(files), FORECAST_DIR)

    processed_count = 0
    skipped_count = 0

    for file in files:
        file_path = os.path.join(FORECAST_DIR, file)

        if should_skip_file(file_path):
            skipped_count += 1
            continue

        logging.info("Processing dispassion forecast for: %s", file)
        await forecast_from_json(forecasting_function=dispassion, path=file_path, is_woc=False)
        processed_count += 1

    logging.info("=== Completed offline dispassion forecasting - processed: %d, skipped: %d ===",
                processed_count, skipped_count)

async def main_slowly() -> None:
    logging.info("=== Starting offline slowly forecasting ===")
    files = [f for f in os.listdir(FORECAST_DIR) if f.endswith(".json")]
    logging.info("Found %s JSON files in %s", len(files), FORECAST_DIR)

    processed_count = 0
    skipped_count = 0

    for file in files:
        file_path = os.path.join(FORECAST_DIR, file)

        if should_skip_file(file_path):
            skipped_count += 1
            continue

        logging.info("Processing slowly forecast for: %s", file)
        await forecast_from_json(forecasting_function=slowly, path=file_path, is_woc=False)
        processed_count += 1

    logging.info("=== Completed offline slowly forecasting - processed: %d, skipped: %d ===",
                processed_count, skipped_count)

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
