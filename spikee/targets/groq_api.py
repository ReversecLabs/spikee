"""
spikee/targets/groq.py

Unified Groq target, extending ProviderTarget, that invokes models based on a simple string key.

Supported production models:
  - distil-whisper-large-v3-en
  - gemma2-9b-it
  - llama-3.1-8b-instant
  - llama-3.3-70b-versatile
  - meta-llama/llama-guard-4-12b
  - whisper-large-v3
  - whisper-large-v3-turbo
"""

from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
from typing import List, Tuple


class GroqApiTarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="groq", default_model="gemma2-9b-it")

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Target for the 'groq' provider."


if __name__ == "__main__":
    load_dotenv()
    target = GroqApiTarget()
    print("Supported Groq models:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="model=gemma2-9b-it"))
    except Exception as err:
        print("Error:", err)
