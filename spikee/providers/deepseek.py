from spikee.providers.custom import AgentFrameworkCustomProvider
from typing import Dict, Union
import os


class AgentFrameworkDeepSeekProvider(AgentFrameworkCustomProvider):
    """Agent Framework provider for DeepSeek models (via Custom provider with OpenAI compatibility)"""

    @property
    def models(self) -> Dict[str, str]:
        return {
            "deepseek-chat": "deepseek-chat",  # deepseek-v3.2 non-thinking
            "deepseek-reasoner": "deepseek-reasoner",  # deepseek-v3.2 thinking
        }

    @property
    def name(self) -> str:
        return "DeepSeek"

    @property
    def base_url(self) -> str:
        return "https://api.deepseek.com/v1"

    @property
    def api_key(self) -> Union[str, None]:
        return os.getenv("DEEPSEEK_API_KEY", None)
