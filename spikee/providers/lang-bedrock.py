from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage

from langchain_aws import BedrockLLM, ChatBedrock
from typing import List, Tuple, Dict, Union, Any


class LangChainBedrockProvider(Provider):
    """LangChain provider for Bedrock models"""

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

        # Determine provider type based on model name and initialize the appropriate LLM wrapper
        if "claude" in self.model:
            self.bedrock_provider = "bedrock"

            self.llm = BedrockLLM(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        else:
            self.bedrock_provider = "bedrockcv"

            self.llm = ChatBedrock(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for AWS Bedrock models via LangChain."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]."""
        return [model for model in self.models.keys()], True

    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        """Invoke LangChain Bedrock LLM with the provided messages."""

        messages = format_messages(messages)

        response = self.llm.invoke(messages)

        return AIMessage(content=response.content.strip(), original_response=response)
