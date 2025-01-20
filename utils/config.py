import os
from typing import Dict, Any



def get_gpt_config(cache_seed: int, temprature: float, model: str, timeout: int) -> Dict[str, Any]:
    return {
        "cache_seed": cache_seed,
        "temperature": temprature,
        "config_list": [
            {"model": model, "api_key": os.environ.get("OPENAI_API_KEY"),
             "response_format": {"type": "json_object"},
             }
        ],
        "timeout": timeout,
    }
