from spikee.templates.module import Module
from spikee.utilities.llm_message import Message, AIMessage

from abc import ABC, abstractmethod
from typing import List, Union, Any


class Provider(Module, ABC):

    @abstractmethod
    def setup(self, model: str, max_tokens: Union[int, None] = None, temperature: Union[float, None] = None):
        pass

    @abstractmethod
    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> AIMessage:
        pass
