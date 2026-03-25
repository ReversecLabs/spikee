from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import standardise_messages, Message, AIMessage

import litellm
from typing import List, Tuple, Dict, Union, Any


class LiteLLMBedrockProvider(Provider):
    """LiteLLM provider for Bedrock models"""

    BEDROCK_MODEL_MAP: Dict[str, str] = {
        "claude35-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude45-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude35-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude37-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude45-opus": "global.anthropic.claude-opus-4-5-20251101-v1:0",
        "deepseek-v3": "deepseek.v3-v1:0",
        "qwen3-coder-30b-a3b-v1": "qwen.qwen3-coder-30b-a3b-v1:0",
    }

    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual Bedrock model identifiers
        self.model = self.BEDROCK_MODEL_MAP.get(self.model, self.model)

        # Initialize the LiteLLM kwargs with the appropriate model and parameters
        self._kwargs = {
            "model": f"bedrock/{self.model}",
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            self._kwargs["max_tokens"] = self.max_tokens

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for AWS Bedrock models via LiteLLM."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]."""
        return [model for model in self.BEDROCK_MODEL_MAP.keys()], True

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke LiteLLM Bedrock LLM with the provided messages."""

        messages = standardise_messages(messages)

        response = litellm.completion(messages=messages, drop_params=True, **self._kwargs)

        return AIMessage(content=response.choices[0].message.content.strip(), original_response=response)
