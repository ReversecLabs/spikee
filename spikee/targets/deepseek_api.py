"""
spikee/targets/deepseek.py

Unified Deepseek target, extending ProviderTarget, that invokes models by a simple key.

Keys:
  - "deepseek-r1" → "DeepSeek-R1-0528"
  - "deepseek-v3" → "DeepSeek-V3-0324"
"""

from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class DeepseekTarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="deepseek")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'deepseek' provider."


if __name__ == "__main__":
    load_dotenv()
    target = DeepseekTarget()
    print("Supported Deepseek keys:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="model=deepseek/deepseek-chat"))
    except Exception as err:
        print("Error:", err)
