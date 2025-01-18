import json
import os
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
