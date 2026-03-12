"""
spikee/targets/deepseek.py

Unified Deepseek target that invokes models by a simple key.

Keys:
  - "deepseek-r1" → "DeepSeek-R1-0528"
  - "deepseek-v3" → "DeepSeek-V3-0324"

Usage:
    target_options: str key from get_available_option_values(); default is "deepseek-r1".

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, DEEPSEEK_MODEL_LIST

from dotenv import load_dotenv
from typing import List, Dict, Optional


class DeepseekTarget(Target):
    # Default key
    _DEFAULT_KEY = "deepseek-chat"

    def get_available_option_values(self) -> List[str]:
        """Return supported keys; first option is default."""
        options = [self._DEFAULT_KEY]  # Default first
        options.extend([model for model in DEEPSEEK_MODEL_LIST if model != self._DEFAULT_KEY])
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to a Deepseek model by key.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        # Determine key or default
        model_id = target_options if target_options is not None else self._DEFAULT_KEY

        if model_id.startswith("deepseek-deepseek-"):
            model_id = model_id.replace("deepseek-", "")

        # Initialize Deepseek client
        llm = get_llm(f"deepseek-{model_id}", max_tokens=None, temperature=0)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            return llm.invoke(messages, content_only=True)
        
        except Exception as e:
            print(f"Error during Deepseek completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = DeepseekTarget()
    print("Supported Deepseek keys:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="deepseek-r1"))
    except Exception as err:
        print("Error:", err)
