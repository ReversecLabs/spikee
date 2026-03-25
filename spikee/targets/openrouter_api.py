"""
spikee/targets/openrouter_api.py

Unified OpenRouter target, extending ProviderTarget, that invokes models based on a simple string key.

Usage:
    target_options: str, one of the model IDs returned by get_available_option_values() or any valid OpenRouter model.
    If None, DEFAULT_MODEL is used.

Exposed:
    get_available_option_values() -> list of supported model IDs (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""
from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class OpenRouterTarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="openrouter")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'openrouter' provider."


if __name__ == "__main__":
    target = OpenRouterTarget()
    print("Supported models:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="model=openrouter/google/gemini-2.5-flash"))
    except Exception as err:
        print("Error:", err)
