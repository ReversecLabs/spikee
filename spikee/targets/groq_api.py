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

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import List, Optional


class GroqApiTarget(Target):

    # Supported Groq model IDs
    _SUPPORTED_MODELS: List[str] = [
        "distil-whisper-large-v3-en",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-guard-4-12b",
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    ]

    # Default model ID
    DEFAULT_MODEL = "gemma2-9b-it"

    def get_available_option_values(self) -> List[str]:
        """Return supported model names; first option is default."""
        options = [self.DEFAULT_MODEL]  # Default first
        options.extend([model for model in self._SUPPORTED_MODELS if model != self.DEFAULT_MODEL])
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
        if model_id not in self._SUPPORTED_MODELS:
            valid = ", ".join(self._SUPPORTED_MODELS)
            raise ValueError(f"Unknown Groq model '{model_id}'. Valid models: {valid}")

        # Initialize the ChatGroq client
        llm = ChatGroq(
            model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Build messages
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("user", input_text))

        # Invoke model
        try:
            ai_msg = llm.invoke(messages)
            return ai_msg.content
        except Exception as e:
            print(f"Error during Groq completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = GroqApiTarget()
    print("Supported Groq models:", target.get_available_option_values())
    try:
        output = target.process_input("Hello!", target_options="gemma2-9b-it")
        print(output)
    except Exception as err:
        print("Error:", err)
