import asyncio
from typing import Any

from any_llm import AnyLLM

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint
from spikee.utilities.llm_message import AIMessage, MessageHint, format_messages


class AnyLLMOpenAIProvider(Provider):
    """AnyLLM provider for OpenAI models"""

    @property
    def default_model(self) -> str:
        return "gpt-4o"

    @property
    def models(self) -> dict[str, str]:
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
    def logprobs_models(self) -> list[str]:
        return ["gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]

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

        timeout = kwargs.get("timeout", self.default_timeout)
        llm_kwargs = {}
        if timeout is not None:
            llm_kwargs["timeout"] = timeout

        self.llm_kwargs = llm_kwargs

        options_kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        if self.model in self.logprobs_models:
            options_kwargs["logprobs"] = True
            options_kwargs["top_logprobs"] = 5

        self.options = options_kwargs

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.LLM], "LLM Provider for OpenAI models via any-llm."

    def _invoke(self, messages: MessageHint) -> AIMessage:
        """Invoke AnyLLM OpenAI LLM with the provided messages."""

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

        if self.model in self.logprobs_models:
            logprobs = None
            if (
                hasattr(response.choices[0], "logprobs")
                and response.choices[0].logprobs
            ):
                logprobs = response.choices[0].logprobs
            return AIMessage(
                content=response.choices[0].message.content,
                original_response=response,
                logprobs=logprobs,
            )

        else:
            return AIMessage(
                content=response.choices[0].message.content, original_response=response
            )

    def create_client(self):
        """Create the HTTP client on the event loop that will close it."""
        return AnyLLM.create("openai", **self.llm_kwargs)
