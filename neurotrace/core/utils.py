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
