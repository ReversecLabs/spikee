import os
import logging
from any_llm import AnyLLM
from any_llm.logging import logger as any_llm_logger
from typing import Union, Any, Dict, Sequence

from spikee.templates.provider import Provider
from spikee.utilities.hinting import ModuleDescriptionHint, Content
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage


class AnyLLMBedrockProvider(Provider):
    """
    AnyLLM provider for Bedrock models

    AWS Authentication, can be performed via the following methods:
    - AWS Keys: Set the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_REGION` environment variables.
    - AWS Profiles: Configure an AWS profile and set the `AWS_PROFILE` and `AWS_REGION` environment variables.
        - https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html
        - SSO Configuration:
            1. Ensure you have installed AWS CLI: https://aws.amazon.com/cli/
            2. Using `aws configure sso` configure your AWS profile, setting a profile name.
            3. Using `aws sso login --profile <profile_name>` log in to your AWS account via SSO. (Also for revalidating expired credentials)
            4. Validate profile using `aws sts get-caller-identity --profile <profile_name>`.
            5. Set the `AWS_PROFILE` environment variable to your profile name, and `AWS_DEFAULT_REGION` to your desired region (e.g. `us-east-2`).
    """

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

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **kwargs,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.model = self.models.get(self.model, self.model)

        timeout = kwargs.get("timeout", self.default_timeout)
        llm_kwargs = {}
        if timeout is not None:
            llm_kwargs["timeout"] = timeout

        try:
            if os.getenv("AWS_PROFILE"):  # Extract Keys for AWS Profiles
                import boto3
                session = boto3.Session(profile_name=os.getenv("AWS_PROFILE"))
                frozen = session.get_credentials().get_frozen_credentials()

                # Inject as env vars so awscrt picks them up via the default chain
                os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
                if frozen.token:
                    os.environ["AWS_SESSION_TOKEN"] = frozen.token

            self.llm = AnyLLM.create("bedrock", **llm_kwargs)
            any_llm_logger.setLevel(logging.ERROR)
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'bedrock' is missing required packages for AWS Bedrock. Please run `pip install spikee[bedrock]` to install them."
            )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options = options_kwargs

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.LLM], "LLM Provider for AWS Bedrock models via any-llm."

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke AnyLLM Bedrock LLM with the provided messages."""

        formatted_messages = format_messages(messages)

        response = self.async_call(self.llm.acompletion, model=self.model, messages=formatted_messages, **self.options)

        return AIMessage(
            content=response.choices[0].message.content, original_response=response
        )
