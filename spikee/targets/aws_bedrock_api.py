"""
spikee/targets/aws_bedrock_api.py

Unified AWS Bedrock target, extending ProviderTarget, that invokes Anthropic Claude models based on a simple key.

Example Keys:
  - "claude35-haiku" → "us.anthropic.claude-3-5-haiku-20241022-v1:0"
  - "claude35-sonnet" → "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
  - "claude37-sonnet" → "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
"""

from spikee.templates.provider_target import ProviderTarget

from dotenv import load_dotenv


class AWSBedrockTarget(ProviderTarget):
    def __init__(self):
        super().__init__(provider="bedrock", default_model="claude45-haiku")


if __name__ == "__main__":
    load_dotenv()
    target = AWSBedrockTarget()
    print("Supported Bedrock keys:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="model=claude45-haiku"))

    except Exception as err:
        print("Error:", err)
