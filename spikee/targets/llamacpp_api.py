"""
spikee/targets/llamaccp_api.py

Unified OpenAI target that can invoke any supported OpenAI model based on a simple key.

Usage:
    target_options: str key returned by get_available_option_values(); defaults to DEFAULT_KEY.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None, logprobs=False) ->
        - For models supporting logprobs: returns (content, logprobs)
        - Otherwise: returns content only
"""

from typing import Optional, List, Tuple, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "http://localhost:8080/"

# load environment variables
load_dotenv()


def get_available_option_values() -> List[str]:
    """Return supported keys; first option is default."""
    options = [DEFAULT_BASE_URL, "http://hostname:port"]  # Default first
    return options


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
    logprobs: bool = False,
) -> Union[str, Tuple[str, any]]:
    """
    Send messages to an OpenAI model based on a simple key.

    Returns:
        - (content, logprobs) if model supports logprobs
        - content otherwise
    """
    base_url = DEFAULT_BASE_URL if target_options is None else target_options

    llm = ChatOpenAI(
        base_url=base_url,
        api_key="abc",
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    messages = []
    if system_message:
        messages.append(("system", system_message))
    messages.append(("user", input_text))

    try:
        ai_msg = llm.invoke(messages)
        return ai_msg.content
    except Exception as e:
        print(f"Error during OpenAI completion: {e}")
        raise


if __name__ == "__main__":
    print(process_input("Hello!", target_options=""))
