"""
spikee/targets/ollama.py

Unified Ollama target that invokes models based on a simple string key.

Supported models:
  - "phi4-mini"
  - "gemma3"
  - "llama3.2"

Usage:
    target_options: str, one of the keys returned by get_available_option_values().
    If None, DEFAULT_KEY is used.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target

from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

import requests  # needed to progromatically list available Ollama models


class OllamaTarget(Target):
    # Map shorthand keys to Ollama model identifiers
    _OPTION_MAP: Dict[str, str] = {
        "phi4-mini": "phi4-mini",
        "gemma3": "gemma3",
        "llama3.2": "llama3.2",
    }

    # Default key
    _DEFAULT_KEY = "phi4-mini"

    def get_available_ollama_models(
        self, baseurl="http://localhost:11434"
    ) -> List[str]:
        """Progromatically gather the list of local models see: ollama list"""
        try:
            response = requests.get(f"{baseurl}/api/tags")
            data = response.json()
            return [model["model"] for model in data["models"]]
        except Exception as e:
            # Something went wrong, we should fallback to the priority list already defined
            print(
                f"Error fetching Ollama models: {e}"
            )  # More informative error message
            return []

    def get_available_option_values(self) -> List[str]:
        """Return supported keys; first option is default."""
        local_models = self.get_available_ollama_models()
        if local_models:
            # Sucessfully returned list of models for ollama local instance.
            options = local_models
        else:
            options = [self._DEFAULT_KEY]  # Default first
            options.extend(
                [key for key in self._OPTION_MAP if key != self._DEFAULT_KEY]
            )
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to an Ollama model by key.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        # Determine key or default
        key = target_options if target_options is not None else self._DEFAULT_KEY
        if key not in self._OPTION_MAP:
            valid = ", ".join(self.get_available_option_values())
            raise ValueError(f"Unknown Ollama key '{key}'. Valid keys: {valid}")

        model_name = self._OPTION_MAP[key]

        # Initialize the Ollama client
        llm = ChatOllama(
            model=model_name,
            num_predict=None, #maximum number of tokens to predict
            client_kwargs={"timeout":30} #timeout in seconds (None = not configured)
        ).with_retry(
            stop_after_attempt=3,          # total attempts (1 initial + 2 retries)
            wait_exponential_jitter=True,  # backoff with jitter
        )

        # Build messages (no separate system role)
        prompt = input_text
        if system_message:
            prompt = f"{system_message}\n{input_text}"
        messages = [("user", prompt)]

        # Invoke model
        try:
            ai_msg = llm.invoke(messages)
            return ai_msg.content
        except Exception as e:
            print(f"Error during Ollama completion ({model_name}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = OllamaTarget()
    print("Supported Ollama keys:", target.get_available_option_values())
    try:
        out = target.process_input("Hello!", target_options="llama3.2")
        print(out)
    except Exception as err:
        print("Error:", err)
