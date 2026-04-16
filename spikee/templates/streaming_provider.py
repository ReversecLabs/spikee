from spikee.templates.provider import Provider
from spikee.utilities.llm_message import Message
from spikee.utilities.content import Content

from abc import ABC, abstractmethod
from typing import Callable, List, Union, Sequence


class StreamingProvider(Provider, ABC):
    @abstractmethod
    def invoke_streaming(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]], callback: Callable
    ) -> None:
        """Invoke the provider with the given messages and stream the response using the callback."""
        pass
