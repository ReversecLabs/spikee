from spikee.templates.provider import Provider
from spikee.utilities.providers import resolve_model_map, standardise_messages, Message

import litellm
from typing import List, Tuple, Dict, Union


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

    def setup(self, model: str, max_tokens: int = None, temperature: float = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Map user-friendly model names to actual Bedrock model identifiers
        self.model = resolve_model_map(self.model, self.BEDROCK_MODEL_MAP)

        # Initialize the LiteLLM kwargs with the appropriate model and parameters
        self._kwargs = {
            "model": f"bedrock/{self.model}",
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            self._kwargs["max_tokens"] = self.max_tokens

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]."""
        return [model for model in self.BEDROCK_MODEL_MAP.keys()], True

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]], content_only: bool = False):
        """Invoke LiteLLM Bedrock LLM with the provided messages."""

        messages = standardise_messages(messages)

        response = litellm.completion(messages=messages, drop_params=True, **self._kwargs)

        if content_only:
            return response.choices[0].message.content
        else:
            return response
