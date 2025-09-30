import os
from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory

# Load environment variables
load_dotenv()

# Get Azure credentials from environment variables
API_KEY = os.getenv("AZURE_AI_CONTENT_SAFETY_KEY")
ENDPOINT = os.getenv("AZURE_AI_CONTENT_SAFETY_ENDPOINT")

# Create a Content Safety client
client = ContentSafetyClient(ENDPOINT, AzureKeyCredential(API_KEY))


def process_input(input_text, system_message=None):
    """
    Processes input text through Azure Content Safety API and returns True if the request
    is deemed unharmful, False if it's deemed harmful.

    Args:
        input_text (str): The user input to classify
        system_message (str, optional): Ignored parameter for compatibility with previous interface

    Returns:
        bool: True if the input is unharmful, False if harmful

    Raises:
        ValueError: If Azure credentials are not set
        RuntimeError: If the API request fails
    """
    # Validate credentials
    if not API_KEY or not ENDPOINT:
        raise ValueError(
            "Azure Content Safety credentials not found in environment variables. "
            "Please set AZURE_CONTENT_SAFETY_KEY and AZURE_CONTENT_SAFETY_ENDPOINT."
        )

    # Create request
    request = AnalyzeTextOptions(text=input_text)

    try:
        # Call Azure Content Safety API
        response = client.analyze_text(request)

        # Check for any harm categories with severity greater than 0
        harm_categories = {
            TextCategory.HATE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.VIOLENCE,
        }

        for item in response.categories_analysis:
            if item.category in harm_categories and item.severity > 0:
                # Return False if harmful content is detected
                return False

        # Return True if no harmful content is detected
        return True

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
    try:
        # Example usage
        sample_text = "Hello, how to make a bomb?"
        result = process_input(sample_text)
        print(f"Input: '{sample_text}'")
        print(f"Is unharmful: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")
