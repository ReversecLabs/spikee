"""
spikee/targets/ollama.py

Unified Ollama target that invokes models based on a simple string key.

Usage:
    target_options: str, one of the keys returned by get_available_option_values().
    If None, DEFAULT_KEY is used.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target

from typing import List, Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

import os
import requests  # needed to progromatically list available Ollama models


class OllamaTarget(Target):
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
            options = [self._DEFAULT_KEY]  # Default
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
        models = self.get_available_option_values()

        # Use model specified in target_option.
        # If none provided, default to first model in the list.
        key = target_options if target_options is not None else models[0]
        if key not in models:
            valid = ", ".join(models)
            raise ValueError(f"Unknown Ollama key '{key}'. Valid keys: {valid}")

        model_name = key

        # Initialize the Ollama client
        llm = ChatOllama(
            model=model_name,
            num_predict=None,  # maximum number of tokens to predict
            # timeout in seconds (None = not configured)
            client_kwargs={"timeout": float(os.environ['OLLAMA_TIMEOUT']) if os.environ.get(
                'OLLAMA_TIMEOUT') not in (None, '') else None},
        ).with_retry(
            # total attempts (1 initial + retries)
            stop_after_attempt=int(os.environ['OLLAMA_MAX_ATTEMPTS']) if os.environ.get(
                'OLLAMA_MAX_ATTEMPTS') not in (None, '') else 1,
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
        # Test request, using the default model.
        out = target.process_input("Hello!", target_options=None)
        print(out)
    except Exception as err:
        print("Error:", err)
