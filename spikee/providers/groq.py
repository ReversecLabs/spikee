from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage

from any_llm import AnyLLM
from typing import List, Tuple, Dict, Union, Any


class AnyLLMGroqProvider(Provider):
    """AnyLLM provider for Groq models"""

    @property
    def default_model(self) -> str:
        return "llama-3.1-8b-instant"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "distil-whisper-large-v3-en": "distil-whisper-large-v3-en",
            "gemma2-9b-it": "gemma2-9b-it",
            "llama-3.1-8b-instant": "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
            "meta-llama/llama-guard-4-12b": "meta-llama/llama-guard-4-12b",
            "whisper-large-v3": "whisper-large-v3",
            "whisper-large-v3-turbo": "whisper-large-v3-turbo",
        }

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        timeout = kwargs.get("timeout", self.default_timeout)
        llm_kwargs = {}
        if timeout is not None:
            llm_kwargs["timeout"] = timeout

        try:
            self.llm = AnyLLM.create("groq", **llm_kwargs)
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'groq' is missing required packages for Groq. Please run `pip install spikee[groq]` to install them."
            )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options = options_kwargs

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for Groq models via any-llm."

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke AnyLLM Groq LLM with the provided messages."""

        formatted_messages = format_messages(messages)

        response = self.llm.completion(
            model=self.model, messages=formatted_messages, **self.options
        )

        return AIMessage(
            content=response.choices[0].message.content, original_response=response
        )
