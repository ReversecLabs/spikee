import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", "")
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

client = ChatCompletionsClient(
    endpoint=os.getenv("AZURE_FOUNDARY_PHI35_MINI_ENDPOINT", ""),
    credential=AzureKeyCredential(api_key),
)


def process_input(input_text, system_message=None):
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    messages.append(UserMessage(content=input_text))

    try:
        response = client.complete(
            messages=messages, max_tokens=2048, stop=["<|endoftext|>"], temperature=0
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during LLM completion: {e}")
        raise
