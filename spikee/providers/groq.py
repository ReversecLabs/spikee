import os

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class AgentFrameworkGroqProvider(Provider):
    """Agent Framework provider for Groq models"""

    @property
    def default_model(self) -> str:
        return "distil-whisper-large-v3-en"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "distil-whisper-large-v3-en": "groq/distil-whisper-large-v3-en",
            "gemma2-9b-it": "groq/gemma2-9b-it",
            "llama-3.1-8b-instant": "groq/llama-3.1-8b-instant",
            "llama-3.3-70b-versatile": "groq/llama-3.3-70b-versatile",
            "meta-llama/llama-guard-4-12b": "groq/meta-llama/llama-guard-4-12b",
            "whisper-large-v3": "groq/whisper-large-v3",
            "whisper-large-v3-turbo": "groq/whisper-large-v3-turbo"
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual Groq model identifiers
        self.model = self.models.get(self.model, self.model)

        api_key = os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY environment variable not set. Please set it to your Groq API key.")

        self.llm = OpenAIChatClient(
            model_id=self.model,
            base_url=GROQ_BASE_URL,
            api_key=api_key
        )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_completion_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for Groq models via Agent Framework (Custom)."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework Groq LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        return AIMessage(content=response.messages[0].text, original_response=response)
