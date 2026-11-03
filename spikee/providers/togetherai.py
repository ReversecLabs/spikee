import os

from spikee.providers.custom import AnyLLMCustomProvider


class AnyLLMTogetherAIProvider(AnyLLMCustomProvider):
    """AnyLLM provider for TogetherAI models (via Custom provider with OpenAI compatibility)"""

    @property
    def default_model(self) -> str:
        return "MiniMaxAI/MiniMax-M3.1-Flash"

    @property
    def models(self) -> dict[str, str]:
        return {
            "MiniMaxAI/MiniMax-M3.1-Flash": "MiniMaxAI/MiniMax-M3.1-Flash",
            "gemma2-8b": "google/gemma-2-9b-it",
            "gemma2-27b": "google/gemma-2-27b-it",
            "llama4-maverick-fp8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "llama31-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama31-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "llama31-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "llama33-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "qwen3-235b-fp8": "Qwen/Qwen3-235B-A22B-fp8-tput",
        }

    @property
    def name(self) -> str:
        return "TogetherAI"

    @property
    def base_url(self) -> str:
        return "https://api.together.ai/v1"

    @property
    def api_key(self) -> str | None:
        return os.getenv("TOGETHER_API_KEY")
