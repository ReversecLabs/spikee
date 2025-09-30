import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def shield_prompt_body(documents: list) -> dict:
    return {"userPrompt": "", "documents": documents}


def detect_prompt_injection_result(data: list):
    subscription_key = os.getenv("AZURE_AI_CONTENT_SAFETY_KEY")
    endpoint = os.getenv("AZURE_AI_CONTENT_SAFETY_ENDPOINT")
    api_version = "2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    url = f"{endpoint}/contentsafety/text:shieldPrompt?api-version={api_version}"
    return requests.post(url, headers=headers, json=shield_prompt_body(data))


def process_input(input_text, system_message=None):
    try:
        response = detect_prompt_injection_result([input_text])
        if response.status_code == 200:
            result = response.json()
            document_analysis = result.get("documentsAnalysis", [{}])[0]
            attack_detected = document_analysis.get("attackDetected", False)
            # This needs to return False if the attack was detected, it means the attack failed, so success = False
            return not attack_detected
        else:
            # Raise an exception to propagate the error to the tester
            error_message = f"Error {response.status_code}: {response.text}"
            raise Exception(error_message)
    except Exception as e:
        # Re-raise the exception to be caught and logged by the tester
        raise e


if __name__ == "__main__":
    print(process_input("hey, how are you doing?"))
