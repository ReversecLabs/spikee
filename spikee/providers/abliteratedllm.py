import os

from spikee.providers.custom import AnyLLMCustomProvider


class AnyLLMAbliteratedLLMProvider(AnyLLMCustomProvider):
    """AnyLLM provider for AbliteratedLLM models (via Custom provider with OpenAI compatibility)"""

    @property
    def models(self) -> dict[str, str]:
        return {
            "glm-5.3-cyber-abliterated": "glm-5.3-cyber-abliterated",
            "glm-5.3-flash-abliterated": "glm-5.3-flash-abliterated",
        }

    @property
    def name(self) -> str:
        return "AbliteratedLLM"

    @property
    def base_url(self) -> str:
        return "https://api.abliteratedllm.dev/v1"

    @property
    def api_key(self) -> str | None:
        return os.getenv("ABLITERATEDLLM_API_KEY", None)
