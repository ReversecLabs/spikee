"""
spikee/targets/azure.py

Unified Azure Chat target that invokes Azure OpenAI deployments based on a simple string.

Note: `target_options` here is the **deployment name**, not the underlying model.

Usage:
    target_options: str, one of the deployment names returned by get_available_option_values().
    If None, DEFAULT_DEPLOYMENT is used.

Exposed:
    get_available_option_values() -> list of supported deployment names (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import List, Optional

# Load environment variables (e.g., AZURE credentials, API_VERSION)
load_dotenv()

# Supported Azure deployment names
#
# !!! EDIT TO MATCH YOUR DEPLOYMENTS !!!
#
_SUPPORTED_DEPLOYMENTS: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
]

# Default deployment
DEFAULT_DEPLOYMENT = _SUPPORTED_DEPLOYMENTS[0]


def get_available_option_values() -> List[str]:
    """Return supported deployment names; first option is default."""
    options = [DEFAULT_DEPLOYMENT]  # Default first
    options.extend([d for d in _SUPPORTED_DEPLOYMENTS if d != DEFAULT_DEPLOYMENT])
    return options


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
) -> str:
    """
    Send messages to an Azure OpenAI deployment specified by target_options.

    Raises:
        ValueError if target_options is provided but invalid.
    """
    # deployment name selection
    deployment = target_options if target_options is not None else DEFAULT_DEPLOYMENT
    if deployment not in _SUPPORTED_DEPLOYMENTS:
        valid = ", ".join(_SUPPORTED_DEPLOYMENTS)
        raise ValueError(
            f"Unknown Azure deployment '{deployment}'. Valid options: {valid}"
        )

    # Initialize the Azure Chat client
    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=os.getenv("API_VERSION", "2024-05-01-preview"),
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
        print(f"Error during Azure completion ({deployment}): {e}")
        raise


if __name__ == "__main__":
    print("Supported Azure deployments:", get_available_option_values())
    try:
        out = process_input("Hello!", target_options="gpt-4o-mini")
        print(out)
    except Exception as err:
        print("Error:", err)
