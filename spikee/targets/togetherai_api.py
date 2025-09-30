"""
spikee/targets/togetherai.py

A unified TogetherAI target that invokes models based on a simple string key.

Usage:
    target_options: str, one of the keys returned by get_available_option_values().
    If None, the default key is used.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response
"""

from typing import List, Dict
from dotenv import load_dotenv
from langchain_together import ChatTogether

# Load environment variables
load_dotenv()

# Map of shorthand keys to TogetherAI model identifiers
OPTION_MAP: Dict[str, str] = {
    "gemma2-8b": "google/gemma-2-9b-it",
    "gemma2-27b": "google/gemma-2-27b-it",
    "llama4-maverick-fp8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama31-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama31-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama31-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "llama33-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "qwen3-235b-fp8": "Qwen/Qwen3-235B-A22B-fp8-tput",
}

# Default shorthand key
DEFAULT_KEY = "llama31-8b"


def get_available_option_values() -> List[str]:
    """Return supported keys; first option is default."""
    options = [DEFAULT_KEY]  # Default first
    options.extend([key for key in OPTION_MAP if key != DEFAULT_KEY])
    return options


def _resolve_model(key: str) -> str:
    """
    Convert a shorthand key to the full model identifier.
    Raises ValueError for unknown keys.
    """
    if key not in OPTION_MAP:
        valid = ", ".join(OPTION_MAP.keys())
        raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
    return OPTION_MAP[key]


def process_input(
    input_text: str, system_message: str = None, target_options: str = None
) -> str:
    """
    Send messages to TogetherAI based on the provided key.

    Args:
        input_text: the user's prompt or document.
        system_message: optional system context.
        target_options: shorthand key for model; uses default if None.

    Returns:
        The model's text response.
    Raises:
        ValueError if target_options is provided but invalid.
    """

    # Choose the model key
    key = target_options if target_options is not None else DEFAULT_KEY
    # Resolve to full model name or raise
    model_name = _resolve_model(key)
    # Initialize the TogetherAI client
    llm = ChatTogether(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Build the message list
    messages = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("user", input_text))

    # Invoke the model
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error during TogetherAI completion ({model_name}): {e}")
        raise


if __name__ == "__main__":
    # Display available keys and test a default run
    print("Supported keys:", get_available_option_values())
    try:
        output = process_input("Hello!", target_options="llama31-8b")
        print(output)
    except Exception as e:
        print("Error in process_input:", e)
