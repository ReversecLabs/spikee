"""
spikee/targets/azure.py

Unified Azure Chat target, extending ProviderTarget, that invokes Azure OpenAI deployments based on a simple string.

Note: `target_options` here is the **deployment name**, not the underlying model.
"""

from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.llm import AZURE_MODEL_LIST

from dotenv import load_dotenv


class AzureAPITarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="azure", default_model="gpt-4o-mini", models=AZURE_MODEL_LIST)


if __name__ == "__main__":
    load_dotenv()
    target = AzureAPITarget()
    print("Supported Azure deployments:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="model=gpt-4o-mini"))
    except Exception as err:
        print("Error:", err)
