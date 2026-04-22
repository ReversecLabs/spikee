from spikee.templates.module import Module
from spikee.utilities.llm_message import Message, AIMessage

from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import os


class Provider(Module, ABC):
    @property
    def default_timeout(self) -> Union[float, None]:
        """Global fallback for provider timeouts, reads from SPIKEE_API_TIMEOUT."""
        val = os.getenv("SPIKEE_API_TIMEOUT")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        return None

    @property
    def default_model(self) -> Union[str, None]:
        """Override in subclass to specify a default model key."""
        return None

    @property
    def models(self) -> Union[dict, None]:
        """Override in subclass to specify a mapping of user-friendly keys to actual model identifiers."""
        return None

    @property
    def logprobs_models(self) -> List[str]:
        """Override in subclass to specify which models support logprobs."""
        return []

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]."""
        if self.models is not None:
            return [model for model in self.models.keys()], True

        else:
            return [], True

    @abstractmethod
    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        """Sets up the provider with the specified model and parameters."""
        pass

    @abstractmethod
    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke the provider with the given messages and return an AIMessage response."""
        pass
