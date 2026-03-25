"""
spikee/targets/llamaccp_api.py

Unified OpenAI target, extending ProviderTarget, that can invoke any supported OpenAI model based on a simple key.
"""


from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class LlamacppAPITarget(ProviderTarget):
    # LLAMACPP URL defined in .env as LLAMACPP_URL

    def __init__(self):
        super().__init__(provider="llamacpp_api")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'llamacpp_api' provider."


if __name__ == "__main__":
    load_dotenv()
    target = LlamacppAPITarget()
    print(target.process_input("Hello!"))
