from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import standardise_messages, Message, AIMessage

from agent_framework.amazon import BedrockChatClient, BedrockChatOptions
from typing import List, Tuple, Dict, Union, Any
import asyncio


class AgentFrameworkBedrockProvider(Provider):
    """Agent Framework provider for Bedrock models"""

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

        self.llm = BedrockChatClient()
        self.options: BedrockChatOptions = BedrockChatOptions(
            model_id=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for AWS Bedrock models via Agent Framework."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]."""
        return [model for model in self.BEDROCK_MODEL_MAP.keys()], True

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke Agent Framework Bedrock LLM with the provided messages."""

        messages = standardise_messages(messages)

        response = asyncio.run(self.llm.run(messages=messages, options=self.options))

        return AIMessage(content=response.content.strip(), original_response=response)
