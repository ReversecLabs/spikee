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


from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class OpenAITarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="openai")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'openai' provider."


if __name__ == "__main__":
    load_dotenv()
    target = OpenAITarget()
    print("Supported OpenAI keys:", target.get_available_option_values())

    # example without logprobs
    print(target.process_input("Hello!", target_options="openai/gpt-4o"))
    # example with logprobs
    print(target.process_input("Hello!", target_options="openai/gpt-4o", logprobs=True))
