import os

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import upgrade_messages, agent_framework_message_translation, Message, AIMessage

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio
import os


class AgentFrameworkCustomProvider(Provider):
    """Custom Agent Framework provider, providing an OpenAI based API provider"""

    @property
    def default_model(self) -> str:
        return list(self.models.keys())[0]  # Return the first model as the default

    @property
    def models(self) -> Dict[str, str]:
        return {"none": "none"}

    @property
    def name(self) -> str:
        return "Custom"

    @property
    def base_url(self) -> Union[str, None]:
        return os.getenv("CUSTOM_API_URL", None)

    @property
    def api_key(self) -> Union[str, None]:
        return os.getenv("CUSTOM_API_KEY", None)

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual Google model identifiers
        self.model = self.models.get(self.model, self.model)

        # Validate that necessary environment variables are set
        if self.base_url is None or self.api_key is None:
            raise ValueError(f"URL and API key variables must be set for the {self.name} provider.")

        self.llm = OpenAIChatClient(
            model_id=self.model,
            base_url=self.base_url,
            api_key=self.api_key
        )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_completion_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], f"LLM Provider for {self.name} (OpenAI based API) via Agent Framework."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework, for OpenAI based API LLM with the provided messages."""

        upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        return AIMessage(content=response.messages[0].text, original_response=response)
