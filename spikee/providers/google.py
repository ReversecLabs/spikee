import os

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class AgentFrameworkGoogleProvider(Provider):
    """Agent Framework provider for Google models"""

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def models(self) -> Dict[str, str]:
        return {
            # Gemini 3 (Latest)
            "gemini-3.1-pro": "gemini-3.1-pro",
            "gemini-3.1-flash": "gemini-3.1-flash",
            "gemini-3.0-pro": "gemini-3.0-pro",
            "gemini-3.0-flash": "gemini-3.0-flash",

            # Gemini 2.5
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash",

            # Gemini Older
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens  # Beware, 'finish_reason' may be 'length' if max_tokens is hit during thinking / no response
        self.temperature = temperature

        # Map user-friendly model names to actual Google model identifiers
        self.model = self.models.get(self.model, self.model)

        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to your Google API key.")

        self.llm = OpenAIChatClient(
            model_id=self.model,
            base_url=GEMINI_BASE_URL,
            api_key=api_key
        )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_completion_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for Google models via Agent Framework (Custom)."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework Gemini LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        return AIMessage(content=response.messages[0].text, original_response=response)
