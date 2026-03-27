from spikee.providers.custom import AgentFrameworkCustomProvider
from typing import Dict, Union
import os


class AgentFrameworkGroqProvider(AgentFrameworkCustomProvider):
    """Agent Framework provider for Groq models (via Custom provider with OpenAI compatibility)"""

    @property
    def default_model(self) -> str:
        return "llama-3.1-8b-instant"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "distil-whisper-large-v3-en": "distil-whisper-large-v3-en",
            "gemma2-9b-it": "gemma2-9b-it",
            "llama-3.1-8b-instant": "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
            "meta-llama/llama-guard-4-12b": "meta-llama/llama-guard-4-12b",
            "whisper-large-v3": "whisper-large-v3",
            "whisper-large-v3-turbo": "whisper-large-v3-turbo"
        }

    @property
    def name(self) -> str:
        return "Groq"

    @property
    def base_url(self) -> str:
        return "https://api.groq.com/openai/v1"

    @property
    def api_key(self) -> Union[str, None]:
        return os.getenv("GROQ_API_KEY", None)
