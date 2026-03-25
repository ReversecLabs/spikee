from spikee.utilities.modules import load_module_from_path
from spikee.templates.provider import Provider
from spikee.list import list_modules

from typing import Dict, List, Any, Union


def get_supported_providers() -> List[str]:
    """Return a list of supported LLM providers."""
    return list_modules("providers")


def validate_llm_provider(option: str) -> bool:
    """Validate if the provided option corresponds to a supported LLM provider."""
    try:
        provider_name, _ = option.split("/", 1)
        provider = load_module_from_path(provider_name, "providers")
        return True
    except (ValueError, ImportError):
        return False


def get_llm(options: str = "", max_tokens: Union[int, None] = 8, temperature: float = 0, additional_kwargs=None) -> Union[Provider, None]:
    """
    Returns an Provider.

    Arguments:
        options (str): The LLM model option string.
        max_tokens (int): Maximum tokens for the LLM response (Default: 8 for LLM Judging).
        temperature (float): Sampling temperature for the LLM (Default: 0).
    """

    if options == "offline":
        return None

    elif "/" in options:
        provider_name, model_name = options.split("/", 1)

    else:
        raise ValueError(
            f"Invalid options format: '{options}' - expected format 'provider/model', for example 'lang-bedrock/claude35-haiku'."
        )

    try:
        provider = load_module_from_path(provider_name, "providers")

    except (ImportError, ValueError) as e:
        raise ImportError(f"Error loading provider '{provider_name}': {str(e)}")

    # TODO: support additional kwargs for providers that require them (e.g. API keys, etc.)

    if model_name == "":
        model_name = provider.default_model

    provider.setup(model=model_name, max_tokens=max_tokens, temperature=temperature)
    return provider
