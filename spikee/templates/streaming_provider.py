from spikee.templates.provider import Provider
from spikee.utilities.llm_message import Message

from abc import ABC, abstractmethod
from typing import Callable, List, Union


class StreamingProvider(Provider, ABC):
    @abstractmethod
    def invoke_streaming(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]], callback: Callable
    ) -> None:
        """Invoke the provider with the given messages and stream the response using the callback."""
        pass
