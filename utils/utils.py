import json
import os
import re
from collections import defaultdict
from typing import List


def to_camel_case(text: str) -> str:
    # 1. Remove parentheses (your original logic)
    processed_text = text.replace("(", "").replace(")", "")

    # 2. Remove specific problematic characters: apostrophes (both kinds), commas, and hyphens.
    #    Replaces them with an empty string, effectively deleting them.
    processed_text = re.sub(r"[’',-]", "", processed_text)

    # 3. Your original splitting and capitalization logic
    words = processed_text.split()

    if not words:
        return ""  # Your original fallback

    # Your original concatenation logic.
    # WARNING: If 'words[0].capitalize()' starts with a digit after the above processing
    # (e.g., if an input was "1st-thing" -> "1stthing" -> "1stthing"),
    # this will still result in a part of the agent name that is an invalid Python identifier.
    # You explicitly asked to remove the first digit solution.
    return words[0].capitalize() + ''.join(word.capitalize() for word in words[1:])


def set_env_vars(path: str) -> None:
    with open(path) as f:
        env_vars = json.load(f)
    for key, value in env_vars.items():
        os.environ[key] = value



def normalize_and_average(probability_dicts: list[dict],options: List[str]) -> dict:
    # Initialize a dictionary to store cumulative probabilities
    cumulative_probabilities = defaultdict(int)
    num_dicts = len(probability_dicts)

    # Sum up all probabilities for each option
    for prob_dict in probability_dicts:
        for key, value in prob_dict.items():
            if key not in options:
                continue
            cumulative_probabilities[key] += value

    # Compute average probabilities
    average_probabilities = {key: cumulative_probabilities[key] / num_dicts for key in cumulative_probabilities}

    # Normalize to ensure sum is exactly 100%
    total = sum(average_probabilities.values())
    normalized_probabilities = {key: round((value / total) * 100) for key, value in average_probabilities.items()}

    # Adjust rounding error if needed
    rounding_error = 100 - sum(normalized_probabilities.values())
    if rounding_error != 0:
        # Find the key with the largest fractional part and adjust its value
        largest_key = max(normalized_probabilities, key=lambda k: (average_probabilities[k] % 1))
        normalized_probabilities[largest_key] += rounding_error




    return normalized_probabilities

