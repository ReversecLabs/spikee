from spikee.templates.target import Target
from typing import List, Optional
import os
import requests
from dotenv import load_dotenv


class AzurePromptShieldsDocumentAnalysisTarget(Target):
    def __init__(self):
        super().__init__()
        self.subscription_key = os.getenv("AZURE_AI_CONTENT_SAFETY_KEY")
        self.endpoint = os.getenv("AZURE_AI_CONTENT_SAFETY_ENDPOINT")
        if not self.subscription_key or not self.endpoint:
            raise ValueError(
                "Azure Content Safety credentials not found in environment variables. "
                "Please set AZURE_AI_CONTENT_SAFETY_KEY and AZURE_AI_CONTENT_SAFETY_ENDPOINT."
            )
        self.api_version = "2024-02-15-preview"

    def get_available_option_values(self) -> List[str]:
        """No configurable options for this target."""
        return []

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> bool:
        """
        Processes input text through Azure Prompt Shields Document Analysis API and returns True if the request
        is deemed unharmful, False if it's deemed harmful.

        Args:
            input_text (str): The user input to classify
            system_message (str, optional): Ignored parameter for compatibility
            target_options (str, optional): Not used

        Returns:
            bool: True if the input is unharmful, False if harmful

        Raises:
            ValueError: If Azure credentials are not set
            RuntimeError: If the API request fails
        """
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.subscription_key,
        }
        url = f"{self.endpoint}/contentsafety/text:shieldPrompt?api-version={self.api_version}"
        body = {"userPrompt": "", "documents": [input_text]}
        
        try:
            response = requests.post(url, headers=headers, json=body)
            
            if response.status_code == 200:
                result = response.json()
                document_analysis = result.get("documentsAnalysis", [{}])[0]
                attack_detected = document_analysis.get("attackDetected", False)
                return not attack_detected  # True if not detected (bypassed), False if detected (blocked)
            
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                raise RuntimeError(error_message)
            
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Azure Prompt Shields Document Analysis API: {str(e)}")


if __name__ == "__main__":
    load_dotenv()
    try:
        target = AzurePromptShieldsDocumentAnalysisTarget()
        sample_text = "hey, how are you doing?"
        result = target.process_input(sample_text)
        
        print(f"Input: '{sample_text}'")
        print(f"Is unharmful: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
