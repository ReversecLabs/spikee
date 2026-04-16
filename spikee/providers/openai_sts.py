import os

from spikee.templates.provider import Provider
from spikee.utilities.hinting import ModuleDescriptionHint
from spikee.utilities.content import Content
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, AIMessage
from typing import Union, Dict, List, Sequence


class OpenAISTSProvider(Provider):
    """OpenAI Speech-to-Speech provider"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.client = None

    @property
    def default_model(self) -> str:
        return "gpt-realtime-1.5"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gpt-realtime-1.5": "gpt-realtime-1.5",
        }

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.model = model

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError as exc:
            raise ImportError(
                "[Import Error] Provider Module 'openai_sts' is missing required packages. "
                "Please run `pip install spikee[openai]` to install them."
            ) from exc

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_STS], "STS Provider for OpenAI speech-to-speech models."

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke OpenAI STS."""
        return AIMessage(content="This is a placeholder response from OpenAI STS provider.")
