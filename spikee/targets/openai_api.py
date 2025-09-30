"""
spikee/targets/openai.py

Unified OpenAI target that can invoke any supported OpenAI model based on a simple key.

Usage:
    target_options: str key returned by get_available_option_values(); defaults to DEFAULT_KEY.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None, logprobs=False) ->
        - For models supporting logprobs: returns (content, logprobs)
        - Otherwise: returns content only
"""

from typing import Optional, List, Dict, Tuple, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load environment variables
load_dotenv()

# shorthand to model identifier map
_OPTION_MAP: Dict[str, str] = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1": "gpt-4.1",
    "o1-mini": "o1-mini",
    "o1": "o1",
    "o3-mini": "o3-mini",
    "o3": "o3",
    "o4-mini": "o4-mini",
}

# default key
DEFAULT_KEY = "gpt-4o"

# which full models support logprobs
_LOGPROBS_MODELS = {"gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"}
# which models do NOT support system messages
_NO_SYSTEM_MODELS = {"o1-mini", "o1", "o3-mini", "o3", "o4-mini"}


def get_available_option_values() -> List[str]:
    """Return supported keys; first option is default."""
    options = [DEFAULT_KEY]  # Default first
    options.extend([key for key in _OPTION_MAP if key != DEFAULT_KEY])
    return options


def _resolve_model(key: Optional[str]) -> str:
    """Convert shorthand key to full model id or error."""
    chosen = key if key is not None else DEFAULT_KEY
    if chosen not in _OPTION_MAP:
        valid = ", ".join(_OPTION_MAP.keys())
        raise ValueError(f"Unknown OpenAI key '{chosen}'. Valid keys: {valid}")
    return _OPTION_MAP[chosen]


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
    model_id = _resolve_model(target_options)
    llm = ChatOpenAI(
        model=model_id,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    # attach logprobs when supported
    if model_id in _LOGPROBS_MODELS:
        llm = llm.bind(logprobs=logprobs)

    # build messages
    if model_id in _NO_SYSTEM_MODELS:
        prompt = input_text
        if system_message:
            prompt = f"{system_message}\n{input_text}"
        messages = [("user", prompt)]
    else:
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("user", input_text))

    try:
        ai_msg = llm.invoke(messages)
        if model_id in _LOGPROBS_MODELS:
            return ai_msg.content, ai_msg.response_metadata.get("logprobs")
        return ai_msg.content
    except Exception as e:
        print(f"Error during OpenAI completion ({model_id}): {e}")
        raise


if __name__ == "__main__":
    print("Supported OpenAI keys:", get_available_option_values())
    # example without logprobs
    print(process_input("Hello!", target_options="gpt4o"))
    # example with logprobs
    print(process_input("Hello!", target_options="gpt4o", logprobs=True))
