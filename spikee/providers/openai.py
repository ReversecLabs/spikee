from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio


class AgentFrameworkOpenAIProvider(Provider):
    """Agent Framework provider for OpenAI models"""

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def models(self) -> Dict[str, str]:
        return {
            # GPT-5.4 (Latest)
            "gpt-5.4": "gpt-5.4",
            "gpt-5.4-pro": "gpt-5.4-pro",
            "gpt-5.4-mini": "gpt-5.4-mini",
            "gpt-5.4-nano": "gpt-5.4-nano",

            # GPT-4 Series
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini",
            "gpt-4.1-nano": "gpt-4.1-nano",

            # GPT-4o Series (Omni)
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",

            # o-series Reasoning Models
            "o4-mini": "o4-mini",
            "o3": "o3",
            "o3-mini": "o3-mini",
            "o1": "o1",
            "o1-mini": "o1-mini",

            # Specialized - Coding
            "gpt-5-codex": "gpt-5-codex",
            "gpt-5.3-codex": "gpt-5.3-codex",
        }

    @property
    def logprobs_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.llm = OpenAIChatClient(model_id=self.model)

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        if self.model in self.logprobs_models:
            options_kwargs["logprobs"] = True
            options_kwargs["top_logprobs"] = 5

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for OpenAI models via Agent Framework."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework OpenAI LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        if self.model in self.logprobs_models:
            return AIMessage(
                content=response.messages[0].text,
                original_response=response,
                logprobs=response.additional_properties.get("logprobs", None)
            )

        else:
            return AIMessage(content=response.messages[0].text, original_response=response)
