import json
import os
from collections import defaultdict

def to_camel_case(expertise: str) -> str:
    # Split the string on whitespace and parentheses
    words = expertise.replace("(", "").replace(")", "").split()
    # Combine words in camel case
    return words[0].capitalize() + ''.join(word.capitalize() for word in words[1:])


def set_env_vars(path: str) -> None:
    with open(path) as f:
        env_vars = json.load(f)
    for key, value in env_vars.items():
        os.environ[key] = value



def normalize_and_average(probability_dicts: list[dict]) -> dict:
    # Initialize a dictionary to store cumulative probabilities
    cumulative_probabilities = defaultdict(int)
    num_dicts = len(probability_dicts)

    # Sum up all probabilities for each option
    for prob_dict in probability_dicts:
        for key, value in prob_dict.items():
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

    return normalized_probabilities/100
