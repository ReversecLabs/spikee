"""
spikee/targets/groq.py

Unified Groq target that invokes models based on a simple string key.

Supported production models:
  - distil-whisper-large-v3-en
  - gemma2-9b-it
  - llama-3.1-8b-instant
  - llama-3.3-70b-versatile
  - meta-llama/llama-guard-4-12b
  - whisper-large-v3
  - whisper-large-v3-turbo

Usage:
    target_options: str, one of the model IDs returned by get_available_option_values().
    If None, DEFAULT_MODEL is used.

Exposed:
    get_available_option_values() -> list of supported model IDs (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, GROK_MODEL_LIST

from dotenv import load_dotenv
from typing import List, Optional


class GroqApiTarget(Target):
    # Default model ID
    DEFAULT_MODEL = "gemma2-9b-it"

    def get_available_option_values(self) -> List[str]:
        """Return supported model names; first option is default."""
        options = [self.DEFAULT_MODEL]  # Default first
        options.extend(
            [model for model in GROK_MODEL_LIST if model != self.DEFAULT_MODEL]
        )
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to a Groq model by ID.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        # Determine which model to use
        model_id = target_options if target_options is not None else self.DEFAULT_MODEL
        
        if model_id.startswith("groq-"):
            model_id = model_id.replace("groq-", "")

        # Initialize the ChatGroq client
        llm = get_llm(f"groq-{model_id}", max_tokens=None, temperature=0)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            return llm.invoke(messages, content_only=True)
        
        except Exception as e:
            print(f"Error during Groq completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = GroqApiTarget()
    print("Supported Groq models:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="gemma2-9b-it"))
    except Exception as err:
        print("Error:", err)
