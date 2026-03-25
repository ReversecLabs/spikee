from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class ProviderTargetModule(ProviderTarget):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "Generic LLM target for supporting LLM providers."


if __name__ == "__main__":
    load_dotenv()
    target = ProviderTargetModule()
    print("Supported provider keys:", target.get_available_option_values())
    try:

        print(target.process_input("Hello!", target_options="bedrock-claude35-sonnet"))
    except Exception as err:
        print("Error:", err)
