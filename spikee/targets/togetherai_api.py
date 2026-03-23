"""
spikee/targets/togetherai.py

A unified TogetherAI target, extending ProviderTarget, that invokes models based on a simple string key.

Usage:
    target_options: str, one of the keys returned by get_available_option_values().
    If None, the default key is used.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response
"""
from spikee.templates.provider_target import ProviderTarget
from spikee.utilities.llm import TOGETHER_AI_MODEL_MAP

from dotenv import load_dotenv


class TogetherAITarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="togetherai", default_model="gemma2-9b-it", models=TOGETHER_AI_MODEL_MAP)


if __name__ == "__main__":
    load_dotenv()
    target = TogetherAITarget()
    print("Supported keys:", target.get_available_option_values())

    try:
        print(target.process_input("Hello!", target_options="model=llama31-8b"))
    except Exception as e:
        print("Error:", e)
