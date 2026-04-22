from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import (
    Message,
    AIMessage,
)

from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from typing import List, Tuple, Dict, Union, Any

BASE_URL = "https://example.com/openai/v1"


class AnyLLMSampleProvider(Provider):
    """Sample AnyLLM provider"""

    @property
    def default_model(self) -> str:
        return "example1"

    @property
    def models(self) -> Dict[str, str]:
        return {"example1": "example1"}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual Groq model identifiers
        self.model = self.models.get(self.model, self.model)

        self.llm = OpenAIChatClient(
            model_id=self.model, base_url=BASE_URL, api_key="example API key"
        )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_completion_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options: OpenAIChatOptions = OpenAIChatOptions(**options_kwargs)

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "Sample provider for OpenAI API based AnyLLM providers."

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Return Mock Message"""

        # upgraded_messages = agent_framework_message_translation(upgrade_messages(messages))

        # response = asyncio.run(self.llm.get_response(messages=upgraded_messages, options=self.options))

        # return AIMessage(content=response.messages[0].text, original_response=response)

        return AIMessage(
            content="This is a sample response from the Sample Provider.",
            original_response={"mock": "response"},
        )
