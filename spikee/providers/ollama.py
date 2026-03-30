from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage

from any_llm import AnyLLM
from typing import List, Tuple, Dict, Union, Any
import os
import requests


class AnyLLMOllamaProvider(Provider):
    """AnyLLM provider for Ollama models"""

    BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

    stored_models: Union[Dict[str, str], None] = None

    @property
    def default_model(self) -> str:
        if self.stored_models is not None and len(self.stored_models) > 0:
            return list(self.stored_models.keys())[0]

        else:
            return "none"

    @property
    def models(self) -> Dict[str, str]:
        if self.stored_models is None:
            models = self.get_ollama_models()
            if "error" in models:
                self.stored_models = {models["error"]: models["error"]}
            else:
                self.stored_models = models

        return self.stored_models

    def get_ollama_models(self) -> Dict[str, str]:
        """Programmatically gather the list of local models see: ollama list"""
        try:
            response = requests.get(f"{self.BASE_URL}/api/tags")
            data = response.json()
            return {model["model"]: model["model"] for model in data["models"]}

        except Exception as e:
            return {"error": "Unable to fetch models from Ollama API: " + str(e)}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            self.llm = AnyLLM.create("ollama", api_base=self.BASE_URL)
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'ollama' is missing required packages for Ollama. Please run `pip install spikee[ollama]` to install them."
            )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options = options_kwargs

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for Ollama models via any-llm."

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke AnyLLM Ollama LLM with the provided messages."""

        formatted_messages = format_messages(messages)

        response = self.llm.completion(
            model=self.model, messages=formatted_messages, **self.options
        )

        return AIMessage(
            content=response.choices[0].message.content, original_response=response
        )
