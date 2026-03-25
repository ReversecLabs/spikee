"""
spikee/targets/google.py

Unified Google Generative AI target, extending ProviderTarget, that invokes models by model name.
"""

from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class GoogleAPITarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="google", default_model="gemini-2.5-flash")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'google' provider."


if __name__ == "__main__":
    load_dotenv()
    target = GoogleAPITarget()
    print("Supported Google models:", target.get_available_option_values())
    try:
        print(target.process_input("What is 5=5 elevated to the power of 6?"))
    except Exception as err:
        print("Error:", err)
