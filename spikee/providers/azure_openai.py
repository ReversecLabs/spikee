import asyncio
import os
from typing import Any

from any_llm import AnyLLM

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint
from spikee.utilities.llm_message import AIMessage, MessageHint, format_messages


class AnyLLMAzureOpenAIProvider(Provider):
    """AnyLLM provider for Azure OpenAI models"""

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def models(self) -> dict[str, str]:
        return {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        }

    def setup(
        self,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        api_ver = os.getenv(
            "AZURE_OPENAI_API_VERSION",
            os.getenv("OPENAI_API_VERSION", "v1"),
        )

        timeout = kwargs.get("timeout", None)
        llm_kwargs = {"api_version": api_ver}
        if timeout is not None:
            llm_kwargs["timeout"] = timeout

        self.llm_kwargs = llm_kwargs

        options_kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options = options_kwargs

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.LLM], "LLM Provider for Azure OpenAI models via any-llm."

    def _invoke(self, messages: MessageHint) -> AIMessage:
        """Invoke AnyLLM Azure OpenAI LLM with the provided messages."""

        formatted_messages = format_messages(messages)

        async def complete():
            llm = self.create_client()
            try:
                return await llm.acompletion(
                    model=self.model, messages=formatted_messages, **self.options
                )
            finally:
                await llm.client.close()

        response = asyncio.run(complete())

        return AIMessage(
            content=response.choices[0].message.content, original_response=response
        )

    def create_client(self):
        """Keep the Azure client on the event loop that owns its connections."""
        return AnyLLM.create("azureopenai", **self.llm_kwargs)
