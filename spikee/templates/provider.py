from spikee.templates.module import Module
from spikee.utilities.providers import Message

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union


class Provider(Module, ABC):

    @abstractmethod
    def setup(self, model: str, max_tokens: int = None, temperature: float = None):
        pass

    @abstractmethod
    def invoke(self, messages: Union[str, List[Union[Message, dict, tuple, str]]], content_only: bool = False):
        pass
