from spikee.templates.provider import Provider
from spikee.utilities.llm_message import MessageHint

from abc import ABC, abstractmethod
from typing import Callable


class StreamingProvider(Provider, ABC):
    @abstractmethod
    def invoke_streaming(
        self,
        messages: MessageHint,
        callback: Callable,
    ) -> None:
        """Invoke the provider with the given messages and stream the response using the callback."""
