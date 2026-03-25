from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage

import litellm
from typing import List, Tuple, Dict, Union, Any


class LiteLLMBedrockProvider(Provider):
    """LiteLLM provider for Bedrock models"""

    @property
    def default_model(self) -> str:
        return "claude45-sonnet"

    @property
    def models(self) -> Dict[str, str]:
        return {
            # Claude 3.5
            "claude35-us-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "claude35-us-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",

            # Claude 3.7
            "claude37-us-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",

            # Claude 4.5 -- Global
            "claude45-haiku": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude45-sonnet": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude45-opus": "global.anthropic.claude-opus-4-5-20251101-v1:0",

            # Claude 4.5 -- US
            "claude45-us-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "claude45-us-sonnet": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude45-us-opus": "us.anthropic.claude-opus-4-5-20251101-v1:0",

            # Deepseek
            "deepseek-v3": "deepseek.v3-v1:0",
            "deepseek-v3.2": "deepseek.v3.2",

            # Qwen
            "qwen3-coder-30b": "qwen.qwen3-coder-30b-a3b-v1:0",
            "qwen3-next-80b": "qwen.qwen3-next-80b-a3b",
        }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual Bedrock model identifiers
        self.model = self.models.get(self.model, self.model)

        # Initialize the LiteLLM kwargs with the appropriate model and parameters
        self._kwargs = {
            "model": f"bedrock/{self.model}",
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            self._kwargs["max_tokens"] = self.max_tokens

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for AWS Bedrock models via LiteLLM."

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke LiteLLM Bedrock LLM with the provided messages."""

        messages = format_messages(messages)

        response = litellm.completion(messages=messages, drop_params=True, **self._kwargs)

        return AIMessage(content=response.choices[0].message.content.strip(), original_response=response)
