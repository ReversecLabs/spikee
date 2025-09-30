import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Specific guardrail ID and version
guardrail_id = os.getenv("AWS_GUARDRAIL_ID")


def detect_prompt_injection_result(data):
    # Example of Input Prompt being Analyzed
    content = [{"text": {"text": data}}]
    # Call the ApplyGuardrail API
    try:
        response = bedrock_runtime.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion="DRAFT",
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
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return False  # Default to False if no valid result
    # Example of Input Prompt being Analyzed
    content = [{"text": {"text": data}}]
    # Call the ApplyGuardrail API
    try:
        response = bedrock_runtime.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion="DRAFT",
            source="INPUT",
            content=content,
        )

        # Process the response
        print("API Response:")
        print(json.dumps(response, indent=2))

        # Check the action taken by the guardrail
        if response["action"] == "GUARDRAIL_INTERVENED":
            return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return False  # Default to False if no valid result


def process_input(input_text, system_message=None):
    try:
        attack_detected = detect_prompt_injection_result(input_text)
        # This needs to return False if the attack was detected, it means the attack failed, so success = False
        return not attack_detected
    except Exception as e:
        # Re-raise the exception to be caught and logged by the tester
        raise e
