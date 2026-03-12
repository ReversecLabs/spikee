"""
spikee/targets/google.py

Unified Google Generative AI target that invokes models by model name.

Usage:
    target_options: str, one of the model names returned by get_available_option_values().
    If None, DEFAULT_MODEL is used.

Exposed:
    get_available_option_values() -> list of supported model names (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, GOOGLE_MODEL_LIST

from dotenv import load_dotenv
from typing import List, Optional

class GoogleAPITarget(Target):
    # Default model name
    DEFAULT_MODEL = "gemini-2.5-flash"

    def get_available_option_values(self) -> List[str]:
        """Return supported model names; first option is default."""
        options = [self.DEFAULT_MODEL]  # Default first
        options.extend(
            [model for model in GOOGLE_MODEL_LIST if model != self.DEFAULT_MODEL]
        )
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to a Google Generative AI model by model name.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        model_id = target_options if target_options is not None else self.DEFAULT_MODEL
        
        if model_id.startswith("google-"):
            model_id = model_id.replace("google-", "")

        # Initialize the client
        llm = get_llm(f"google-{model_id}", max_tokens=None, temperature=0)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            return llm.invoke(messages, content_only=True)
        
        except Exception as e:
            print(f"Error during Google completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = GoogleAPITarget()
    print("Supported Google models:", target.get_available_option_values())
    try:
        print(target.process_input("What is 5=5 elevated to the power of 6?"))
    except Exception as err:
        print("Error:", err)
