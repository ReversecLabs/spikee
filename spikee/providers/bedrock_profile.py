from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import format_messages, Message, AIMessage

import os
from typing import List, Tuple, Dict, Union, Any


class BedrockProfileProvider(Provider):
    """
    Provider for Bedrock models, via an AWS profile.

    https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html

    SSO Configuration:
    1. Ensure you have installed AWS CLI: https://aws.amazon.com/cli/
    2. Using `aws configure sso` configure your AWS profile, setting a profile name.
    3. Using `aws sso login --profile <profile_name>` log in to your AWS account via SSO.
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
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.model = self.models.get(self.model, self.model)

        try:
            import boto3

            self.profile = os.getenv("AWS_PROFILE") or "default"
            self.region = os.getenv("AWS_DEFAULT_REGION")

            if self.profile is None or self.region is None:
                raise ValueError(
                    "AWS Profile or AWS Default Region not found. Please set the AWS_PROFILE and AWS_DEFAULT_REGION environment variables."
                )

            self.session = boto3.Session(profile_name=self.profile)
            self.client = self.session.client(service_name="bedrock-runtime", region_name=self.region)
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'bedrock_sso' is missing required packages for AWS Bedrock ('boto3'). Please run `pip install spikee[bedrock]` to install them."
            )

        options_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            options_kwargs["max_tokens"] = self.max_tokens

        if self.temperature is not None:
            options_kwargs["temperature"] = self.temperature

        self.options = options_kwargs

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "LLM Provider for AWS Bedrock models via an AWS profile. (Requires `AWS_PROFILE` and `AWS_DEFAULT_REGION` env to be set)"

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke Bedrock LLM, via an AWS profile, with the provided messages."""

        formatted_messages = format_messages(messages, bedrock_format=True)

        response = self.client.converse(
            modelId=self.model,
            messages=formatted_messages,
        )

        # import json
        # print("Raw Bedrock Response:", json.dumps(response, indent=2))

        return AIMessage(
            content=response['output']['message']['content'][0]['text'], original_response=response
        )
