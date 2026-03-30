from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage

from any_llm import AnyLLM
import os
from typing import List, Tuple, Dict, Union, Any


class AnyLLMAzureOpenAIProvider(Provider):
    """AnyLLM provider for Azure OpenAI models"""

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        }

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
            api_ver = os.getenv(
                "AZURE_OPENAI_API_VERSION",
                os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
            )
            self.llm = AnyLLM.create("azureopenai", api_version=api_ver)
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'azure_openai' is missing required packages for Azure OpenAI. Please run `pip install spikee[azure]` to install them."
            )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options = options_kwargs

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for Azure OpenAI models via any-llm."

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke AnyLLM Azure OpenAI LLM with the provided messages."""

        formatted_messages = format_messages(messages)

        response = self.llm.completion(
            model=self.model, messages=formatted_messages, **self.options
        )

        return AIMessage(
            content=response.choices[0].message.content, original_response=response
        )
