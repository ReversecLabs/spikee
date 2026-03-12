"""
spikee/targets/openrouter_api.py

Unified OpenRouter target that invokes models based on a simple string key.

Usage:
    target_options: str, one of the model IDs returned by get_available_option_values() or any valid OpenRouter model.
    If None, DEFAULT_MODEL is used.

Exposed:
    get_available_option_values() -> list of supported model IDs (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""
from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, OPENROUTER_MODEL_LIST

import os
from dotenv import load_dotenv
from typing import List, Optional


class OpenRouterTarget(Target):
    # Default model ID
    DEFAULT_MODEL = "google/gemini-2.5-flash"

    def get_available_option_values(self) -> List[str]:
        """Return supported model IDs; first option is default."""
        options = [self.DEFAULT_MODEL]  # Default first
        options.extend([m for m in OPENROUTER_MODEL_LIST if m != self.DEFAULT_MODEL])
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to OpenRouter based on the designated model name.
        """
        # Model selection
        model_id = target_options if target_options is not None else self.DEFAULT_MODEL
        
        if model_id.startswith("openrouter-"):
            model_id = model_id.replace("openrouter-", "")
        
        llm = get_llm(f"openrouter-{model_id}", max_tokens=None, temperature=0)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        return llm.invoke(messages, content_only=True).strip()

if __name__ == "__main__":
    target = OpenRouterTarget()
    print("Supported models:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="google/gemini-2.5-flash"))
    except Exception as err:
        print("Error:", err)
