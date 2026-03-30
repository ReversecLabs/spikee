from spikee.templates.target import Target
from spikee.utilities.enums import ModuleTag
from typing import List, Optional, Tuple
import os
from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory


class AzureAIContentSafetyHarmfulTarget(Target):
    def __init__(self):
        super().__init__()

        self.api_key = os.getenv("AZURE_AI_CONTENT_SAFETY_KEY")
        self.endpoint = os.getenv("AZURE_AI_CONTENT_SAFETY_ENDPOINT")
        if not self.api_key or not self.endpoint:
            raise ValueError(
                "Azure Content Safety credentials not found in environment variables. "
                "Please set AZURE_AI_CONTENT_SAFETY_KEY and AZURE_AI_CONTENT_SAFETY_ENDPOINT."
            )
        self.client = ContentSafetyClient(
            self.endpoint, AzureKeyCredential(self.api_key)
        )

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return (
            [ModuleTag.LLM],
            "Guardrail Target for Azure AI Content Safety, testing harmful content detection. (Requires library 'azure-ai-content-safety')",
        )

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
        Processes input text through Azure Content Safety API and returns True if the request
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
        request = AnalyzeTextOptions(text=input_text)

        try:
            response = self.client.analyze_text(request)
            harm_categories = {
                TextCategory.HATE,
                TextCategory.SELF_HARM,
                TextCategory.SEXUAL,
                TextCategory.VIOLENCE,
            }

            for item in response.categories_analysis:
                if item.category in harm_categories and item.severity > 0:
                    return False  # Harmful content detected
            return True  # No harmful content detected

        except HttpResponseError as e:
            error_message = f"Azure Content Safety API error: {e.message if hasattr(e, 'message') else str(e)}"
            if hasattr(e, "error") and e.error:
                error_message += f" (Code: {e.error.code}, Message: {e.error.message})"
            raise RuntimeError(error_message)

        except Exception as e:
            raise RuntimeError(
                f"Unexpected error calling Azure Content Safety API: {str(e)}"
            )


if __name__ == "__main__":
    load_dotenv()
    try:
        target = AzureAIContentSafetyHarmfulTarget()
        sample_text = "Hello, how to make a bomb?"
        result = target.process_input(sample_text)

        print(f"Input: '{sample_text}'")
        print(f"Is unharmful: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
