"""
spikee/targets/ollama.py

Unified Ollama target that invokes models based on a simple string key.

Usage:
    target_options: str, one of the keys returned by get_available_option_values().
    If None, DEFAULT_KEY is used.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from typing import Any, List, Optional, Tuple, Union
from dotenv import load_dotenv


class OllamaTarget(ProviderTarget):
    # OLLAMA URL defined in .env as OLLAMA_URL

    def __init__(self):
        super().__init__(provider="ollama")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'ollama' provider."


if __name__ == "__main__":
    load_dotenv()
    target = OllamaTarget()
    print("Supported Ollama models:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!"))
    except Exception as err:
        print("Error:", err)
