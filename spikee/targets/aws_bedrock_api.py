"""
spikee/targets/aws_bedrock_api.py

Unified AWS Bedrock target that invokes Anthropic Claude models based on a simple key.

Example Keys:
  - "claude35-haiku" → "us.anthropic.claude-3-5-haiku-20241022-v1:0"
  - "claude35-sonnet" → "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
  - "claude37-sonnet" → "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

Usage:
    target_options: str key from get_available_option_values(); default is "claude35-haiku".

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, BEDROCK_MODEL_MAP

from dotenv import load_dotenv
from typing import List, Optional


class AWSBedrockTarget(Target):
    # Default key
    _DEFAULT_KEY = "claude35-haiku"

    def get_available_option_values(self) -> List[str]:
        """Return supported keys; first option is default."""
        options = [self._DEFAULT_KEY]  # Default first
        options.extend([key for key in BEDROCK_MODEL_MAP if key != self._DEFAULT_KEY])
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to an AWS Bedrock model by key.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        model_id = target_options if target_options is not None else self._DEFAULT_KEY
        
        if model_id.startswith("bedrock-"):
            model_id = model_id.replace("bedrock-", "")
            
        # Initialize Bedrock client
        llm = get_llm(f"bedrock-{model_id}", max_tokens=None, temperature=0.7)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            return llm.invoke(messages, content_only=True)

        except Exception as e:
            print(f"Error during AWS Bedrock completion ({model_id}): {e}")
            raise

if __name__ == "__main__":
    load_dotenv()
    target = AWSBedrockTarget()
    print("Supported Bedrock keys:", target.get_available_option_values())
    try:

        print(target.process_input("Hello!", target_options="claude35-sonnet"))
    except Exception as err:
        print("Error:", err)
