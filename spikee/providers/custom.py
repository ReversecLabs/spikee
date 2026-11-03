import asyncio
import os
from typing import Any

from any_llm import AnyLLM

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint
from spikee.utilities.llm_message import AIMessage, MessageHint, format_messages


class AnyLLMCustomProvider(Provider):
    """Custom AnyLLM provider, providing an OpenAI based API provider"""

    @property
    def default_model(self) -> str:
        return next(iter(self.models.keys()))  # Return the first model as the default

    @property
    def models(self) -> dict[str, str]:
        return {"none": "none"}

    @property
    def name(self) -> str:
        return "Custom"

    @property
    def base_url(self) -> str | None:
        return os.getenv("CUSTOM_API_URL", None)

    @property
    def api_key(self) -> str | None:
        return os.getenv("CUSTOM_API_KEY", None)

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

        # Map user-friendly model names to actual Google model identifiers
        self.model = self.models.get(self.model, self.model)

        # Validate that necessary environment variables are set
        if self.base_url is None or self.api_key is None:
            raise ValueError(
                f"URL and API key variables must be set for the {self.name} provider."
            )

        timeout = kwargs.get("timeout", self.default_timeout)
        llm_kwargs = {"api_base": self.base_url, "api_key": self.api_key}
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
        return [
            ModuleTag.LLM
        ], f"LLM Provider for {self.name} (OpenAI based API) via any-llm."

    def _invoke(self, messages: MessageHint) -> AIMessage:
        """Invoke AnyLLM, for OpenAI based API LLM with the provided messages."""

        formatted_messages = format_messages(messages)

        async def complete():
            # A provider can be reused by sequential calls or worker threads.
            # Keep its HTTP client and cleanup on this invocation's event loop.
            llm = self.create_client()
            try:
                return await llm.acompletion(
                    model=self.model, messages=formatted_messages, **self.options
                )
            finally:
                await llm.client.close()

        response = asyncio.run(complete())

        response = AIMessage(
            content=response.choices[0].message.content, original_response=response
        )

        self.response_validation(messages, response)

        return response

    def create_client(self):
        """Create an invocation-local OpenAI-compatible client."""
        try:
            return AnyLLM.create("openai", **self.llm_kwargs)
        except ImportError as error:
            raise ImportError(
                f"Provider '{self.name}' requires the OpenAI-compatible API dependencies."
            ) from error

    def response_validation(self, messages: MessageHint, response: AIMessage) -> None:
        """Abstract validation method for subclasses to implement specific response validation logic. Otherwise validations common OpenAI compatible API errors."""
