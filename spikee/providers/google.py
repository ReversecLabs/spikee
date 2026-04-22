from spikee.providers.custom import AnyLLMCustomProvider
from typing import Dict, Union
import os


class AnyLLMGoogleProvider(AnyLLMCustomProvider):
    """AnyLLM provider for Google models (via Custom provider with OpenAI compatibility)"""

    # Beware max tokens, 'finish_reason' may be 'length' if max_tokens is hit during thinking, resulting in no response

    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"

    @property
    def models(self) -> Dict[str, str]:
        return {
            # Gemini 3 (Latest)
            "gemini-3.1-pro": "gemini-3.1-pro-preview",
            "gemini-3.1-flash": "gemini-3.1-flash-lite-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
            # Gemini 2.5
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash",
            # Gemini Older
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-1.5-flash-latest": "gemini-1.5-flash-latest",
        }

    @property
    def name(self) -> str:
        return "Google"

    @property
    def base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta/openai/"

    @property
    def api_key(self) -> Union[str, None]:
        return os.getenv("GOOGLE_API_KEY", None)
