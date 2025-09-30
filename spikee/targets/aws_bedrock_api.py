"""
spikee/targets/aws_bedrock_api.py

Unified AWS Bedrock target that invokes Anthropic Claude models based on a simple key.

Keys:
  - "claude35-haiku" → "us.anthropic.claude-3-5-haiku-20241022-v1:0"
  - "claude35-sonnet" → "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
  - "claude37-sonnet" → "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

Usage:
    target_options: str key from get_available_option_values(); default is "claude35-haiku".

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from typing import List, Dict, Optional

# Load environment variables (e.g., AWS credentials)
load_dotenv()

# Map shorthand keys to AWS Bedrock model identifiers
_OPTION_MAP: Dict[str, str] = {
    "claude35-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude35-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude37-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
}

# Default key
_DEFAULT_KEY = "claude35-haiku"


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
    Send messages to an AWS Bedrock model by key.

    Raises:
        ValueError if target_options is provided but invalid.
    """
    # Determine key or default
    key = target_options if target_options is not None else _DEFAULT_KEY
    if key not in _OPTION_MAP:
        valid = ", ".join(get_available_option_values())
        raise ValueError(f"Unknown AWS Bedrock key '{key}'. Valid keys: {valid}")

    model_id = _OPTION_MAP[key]

    # Initialize Bedrock client
    llm = ChatBedrock(
        model_id=model_id,
        model_kwargs={"temperature": 0},
    )

    # Build messages
    messages: List[tuple] = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("user", input_text))

    # Invoke model
    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print(f"Error during AWS Bedrock completion ({model_id}): {e}")
        raise


if __name__ == "__main__":
    print("Supported Bedrock keys:", get_available_option_values())
    try:
        print(process_input("Hello!", target_options="claude35-sonnet"))
    except Exception as err:
        print("Error:", err)
