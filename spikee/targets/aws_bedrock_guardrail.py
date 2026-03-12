
from spikee.templates.target import Target
from spikee.utilities.modules import parse_options

from dotenv import load_dotenv
from typing import List, Optional
import os
import boto3


class AWSBedrockGuardrailTarget(Target):
    def __init__(self):
        super().__init__()
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.guardrail_id = os.getenv("AWS_GUARDRAIL_ID")

    def get_available_option_values(self) -> List[str]:
        """Guardrail targets typically don't have configurable options."""
        return ["version=DRAFT"]

    def detect_prompt_injection_result(self, input_text, version):
        """Detect if prompt injection was blocked by AWS Bedrock guardrail."""
        content = [{"text": {"text": input_text}}]
        try:
            response = self.bedrock_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=version,
                source="INPUT",
                content=content,
            )
            # Check the action taken by the guardrail
            if response["action"] == "GUARDRAIL_INTERVENED":
                # Inspect assessments for the specific filter
                assessments = response.get("assessments", [])
                for assessment in assessments:
                    content_policy = assessment.get("contentPolicy", {})
                    filters = content_policy.get("filters", [])
                    for filter_entry in filters:
                        if (
                            filter_entry.get("type") == "PROMPT_ATTACK"
                            and filter_entry.get("action") == "BLOCKED"
                        ):
                            return True  # Return True only if criteria are met
                return True  # If action is intervened but not specifically blocked, still return True
        except Exception as e:
            print(f"Error during guardrail detection: {e}")
        return False  # Default to False if no valid result

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> bool:
        """
        Test if input text bypasses AWS Bedrock guardrail.
        
        Returns:
            bool: True if guardrail was bypassed (attack success), False if blocked (attack failed)
        """
        opts = parse_options(target_options)
        version = opts.get("version", "DRAFT")
        
        try:
            attack_detected = self.detect_prompt_injection_result(input_text, version=version)

            return not attack_detected
        except Exception as e:
            print(f"Error in guardrail processing: {e}")
            raise

if __name__ == "__main__":
    load_dotenv()
    target = AWSBedrockGuardrailTarget()
    print("Testing AWS Bedrock guardrail with input:", "Your prompt injection attack here")
    result = target.process_input("Your prompt injection attack here", target_options="version=DRAFT")
    print("Guardrail bypassed (attack success):", result)