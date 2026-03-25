import os

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio

TOGETHER_AI_BASE_URL = "https://api.together.xyz/v1"


class AgentFrameworkTogetherAIProvider(Provider):
    """Agent Framework provider for TogetherAI models"""

    @property
    def default_model(self) -> str:
        return "gemma2-8b"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gemma2-8b": "google/gemma-2-9b-it",
            "gemma2-27b": "google/gemma-2-27b-it",
            "llama4-maverick-fp8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "llama31-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama31-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "llama31-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "llama33-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "qwen3-235b-fp8": "Qwen/Qwen3-235B-A22B-fp8-tput",
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual TogetherAI model identifiers
        self.model = self.models.get(self.model, self.model)

        api_key = os.getenv("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError("TOGETHER_API_KEY environment variable not set. Please set it to your TogetherAI API key.")

        self.llm = OpenAIChatClient(
            model_id=self.model,
            base_url=TOGETHER_AI_BASE_URL,
            api_key=api_key
        )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_completion_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for TogetherAI models via Agent Framework (Custom)."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework TogetherAI LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        return AIMessage(content=response.messages[0].text, original_response=response)
