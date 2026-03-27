from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.azure import AzureOpenAIChatClient, AzureOpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio


class AgentFrameworkAzureOpenAIProvider(Provider):
    """Agent Framework provider for Azure OpenAI models"""

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.llm = AzureOpenAIChatClient(deployment_name=self.model)

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: AzureOpenAIChatOptions = AzureOpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for Azure OpenAI models via Agent Framework."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework Azure OpenAI LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        return AIMessage(content=response.messages[0].text, original_response=response)
