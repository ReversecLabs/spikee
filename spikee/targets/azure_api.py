"""
spikee/targets/azure.py

Unified Azure Chat target, extending ProviderTarget, that invokes Azure OpenAI deployments based on a simple string.

Note: `target_options` here is the **deployment name**, not the underlying model.
"""

from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class AzureAPITarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="azure")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'azure' provider."


if __name__ == "__main__":
    load_dotenv()
    target = AzureAPITarget()
    print("Supported Azure deployments:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="model=azure/gpt-4o"))
    except Exception as err:
        print("Error:", err)
