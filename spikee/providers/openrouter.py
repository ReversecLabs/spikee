import os

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class AgentFrameworkOpenRouterProvider(Provider):
    """Agent Framework provider for OpenRouter models"""

    @property
    def default_model(self) -> str:
        return "google/gemini-2.5-flash"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "google/gemini-2.5-flash": "google/gemini-2.5-flash",
            "anthropic/claude-3.5-haiku": "anthropic/claude-3.5-haiku",
            "meta-llama/llama-3.1-8b-instruct": "meta-llama/llama-3.1-8b-instruct",
            "openai/gpt-4o-mini": "openai/gpt-4o-mini",
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual OpenRouter model identifiers
        self.model = self.models.get(self.model, self.model)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it to your OpenRouter API key.")

        self.llm = OpenAIChatClient(
            model_id=self.model,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key
        )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_completion_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for OpenRouter models via Agent Framework (Custom)."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework OpenRouter LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        return AIMessage(content=response.messages[0].text, original_response=response)
