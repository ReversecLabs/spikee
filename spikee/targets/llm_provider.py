from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class LLMProviderTargetModule(ProviderTarget):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "Generic LLM target for supported LLM providers - see 'spikee list providers' => '--target-options \"model=<provider>/<model>\"'."


if __name__ == "__main__":
    load_dotenv()
    target = LLMProviderTargetModule()
    print("Supported provider keys:", target.get_available_option_values())
    try:

        print(target.process_input("Hello!", target_options="model=bedrock/claude37-sonnet"))
    except Exception as err:
        print("Error:", err)
