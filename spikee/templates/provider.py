from spikee.templates.module import Module
from spikee.utilities.llm_message import Message, AIMessage

from abc import ABC, abstractmethod
from typing import List, Union, Any


class Provider(Module, ABC):

    @property
    def default_model(self) -> Union[str, None]:
        """Override in subclass to specify a default model key."""
        return None

    @property
    def models(self) -> Union[dict, list, None]:
        """Override in subclass to specify a mapping of user-friendly keys to actual model identifiers."""
        return None

    @property
    def logprobs_models(self) -> List[str]:
        """Override in subclass to specify which models support logprobs."""
        return []

    @abstractmethod
    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        pass

    @abstractmethod
    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        pass
