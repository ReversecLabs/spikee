"""
spikee/targets/deepseek.py

Unified Deepseek target that invokes models by a simple key.

Keys:
  - "deepseek-r1" → "DeepSeek-R1-0528"
  - "deepseek-v3" → "DeepSeek-V3-0324"

Usage:
    target_options: str key from get_available_option_values(); default is "deepseek-r1".

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional

# Load environment variables (DEEPSEEK_API_KEY)
load_dotenv()

# Map keys to actual Deepseek model identifiers
_OPTION_MAP: Dict[str, str] = {
    "deepseek-r1": "deepseek-reasoner",
    "deepseek-v3": "deepseek-chat",
}

# Default key
_DEFAULT_KEY = "deepseek-v3"


def get_available_option_values() -> List[str]:
    """Return supported keys; first option is default."""
    options = [_DEFAULT_KEY]  # Default first
    options.extend([key for key in _OPTION_MAP if key != _DEFAULT_KEY])
    return options


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
) -> str:
    """
    Send messages to a Deepseek model by key.

    Raises:
        ValueError if target_options is provided but invalid.
    """
    # Determine key or default
    key = target_options if target_options is not None else _DEFAULT_KEY
    if key not in _OPTION_MAP:
        valid = ", ".join(get_available_option_values())
        raise ValueError(f"Unknown Deepseek key '{key}'. Valid keys: {valid}")

    # Resolve to actual model identifier
    model_name = _OPTION_MAP[key]

    # Initialize Deepseek client
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Build messages
    messages = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("user", input_text))

    # Invoke model
    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print(f"Error during Deepseek completion ({model_name}): {e}")
        raise


if __name__ == "__main__":
    print("Supported Deepseek keys:", get_available_option_values())
    try:
        print(process_input("Hello!", target_options="deepseek-r1"))
    except Exception as err:
        print("Error:", err)
