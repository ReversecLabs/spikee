from spikee.templates.provider import Provider
from spikee.utilities.llm_message import Message, MessageHint
from spikee.utilities.hinting import Content

from abc import ABC, abstractmethod
from typing import Callable, Union, Sequence


class StreamingProvider(Provider, ABC):
    @abstractmethod
    def invoke_streaming(
        self,
        messages: MessageHint,
        callback: Callable,
    ) -> None:
        """Invoke the provider with the given messages and stream the response using the callback."""
