import yaml
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))

def load_prompt(prompt_file: str, **kwargs) -> str:
    with open(PROMPTS_DIR / prompt_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    template = _env.from_string(data["template"])
    return template.render(**kwargs)
