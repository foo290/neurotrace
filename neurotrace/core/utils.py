import json
import os
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """
    Load a prompt file from the prompts directory.

    Args:
        name (str): Name of the prompt file (without .md)

    Returns:
        str: Prompt text
    """
    path = os.path.join(PROMPT_DIR, "tools", f"{name}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file {path} does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_json_loads(json_string: str, return_type: type = dict) -> type:
    """
    Safely load a JSON string, returning an empty dictionary on failure.

    Args:
        json_string (str): The JSON string to load.

    Returns:
        :param json_string:
        :param return_type:
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return return_type()


def strip_json_code_block(text: str) -> str:
    """
    Strip code block formatting from a JSON string.

    Args:
        text (str): The input text containing a JSON code block.

    Returns:
        str: The cleaned JSON string without code block formatting.
    """
    return text.strip("```json\n").strip("```").strip()
