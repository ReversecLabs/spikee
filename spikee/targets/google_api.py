"""
spikee/targets/google.py

Unified Google Generative AI target that invokes models by model name.

Usage:
    target_options: str, one of the model names returned by get_available_option_values().
    If None, DEFAULT_MODEL is used.

Exposed:
    get_available_option_values() -> list of supported model names (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Supported model names
_SUPPORTED_MODELS: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-exp-1206",
]

# Default model name
DEFAULT_MODEL = "gemini-2.0-flash"


def get_available_option_values() -> List[str]:
    """Return supported model names; first option is default."""
    options = [DEFAULT_MODEL]  # Default first
    options.extend([model for model in _SUPPORTED_MODELS if model != DEFAULT_MODEL])
    return options


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
) -> str:
    """
    Send messages to a Google Generative AI model by model name.

    Raises:
        ValueError if target_options is provided but invalid.
    """
    # Determine model to use
    model_name = target_options if target_options is not None else DEFAULT_MODEL
    if model_name not in _SUPPORTED_MODELS:
        valid = ", ".join(_SUPPORTED_MODELS)
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}")

    # Initialize the client
    llm = ChatGoogleGenerativeAI(
        model=model_name,
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
        print(ai_msg)
        return ai_msg.content
    except Exception as e:
        print(f"Error during Google completion ({model_name}): {e}")
        raise


if __name__ == "__main__":
    print("Supported Google models:", get_available_option_values())
    try:
        response = process_input("What is 5=5 elevated to the power of 6?")
        print(response)
    except Exception as err:
        print("Error:", err)
