from spikee.providers.custom import AnyLLMCustomProvider
import os


class AnyLLMLLAMACPPProvider(AnyLLMCustomProvider):
    """AnyLLM provider for LLAMA CPP models (via Custom provider with OpenAI compatibility)"""

    @property
    def name(self) -> str:
        return "LLAMA CPP"

    @property
    def base_url(self) -> str:
        return os.getenv("LLAMACPP_URL", "http://localhost:8080/")

    @property
    def api_key(self) -> str:
        return "abc"
