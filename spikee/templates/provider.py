from abc import ABC, abstractmethod
from typing import Any, List, Union, Callable
import os
import asyncio
import gc

from spikee.templates.module import Module
from spikee.utilities.llm_message import AIMessage, MessageHint
from spikee.utilities.hinting import ModuleOptionsHint

class ProviderError(Exception):
    """Custom exception for provider-related errors."""
    def __init__(self, message, prompt: MessageHint = "", response: Union[AIMessage, None] = None, metadata: dict = {}):
        super().__init__(message)
        self.prompt = prompt
        self.response = response
        self.metadata = metadata

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
    
    @property
    def debug_prompt(self) -> bool:
        """Global flag to enable debug prompt logging, reads from PROVIDER_DEBUG."""
        val = os.getenv("PROVIDER_DEBUG")
        if val:
            return val.lower() in ("1", "true", "yes")
        return False

    def get_available_option_values(self) -> ModuleOptionsHint:
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

    def invoke(
        self, messages: MessageHint
    ) -> AIMessage:
        """
        Invokes the provider with the given messages returning an AIMessage response, and implements management logic.
        
        Legacy providers 
        """

        if self.debug_prompt:
            print(f"[{self.__class__.__name__} DEBUG] Prompt: {messages}")

        try:
            response = self._invoke(messages)
        except ProviderError as e:
            raise e
        
        except Exception as e:
            raise ProviderError(
                f"A generic error occurred while invoking the provider: {str(e)}",
                prompt=messages,
                response=None,
                metadata={"error_type": type(e).__name__, "error_message": str(e)},
            ) from e

        if self.debug_prompt:
            print(f"[{self.__class__.__name__} DEBUG] Response: {response.content if response else 'No response'}")

        return response


    def _invoke(
        self, messages: MessageHint
    ) -> AIMessage:
        """Internal abstract method to invoke the provider; should be implemented by subclasses."""
        raise ProviderError(
            "The '_invoke' method must be implemented by subclasses of Provider.",
            prompt=messages,
            response=None,
            metadata={"error_type": "NotImplementedError"},
        )
        

    def async_call(self, fun: Callable, **params) -> Any:

        async def run_async_call(fun: Callable, **params) -> Any:
            result = await fun(**params)

            # Force GC to trigger any pending __del__ finalizers (e.g. httpx AsyncClient)
            # so their cleanup coroutines are scheduled before we gather and the loop closes.
            gc.collect()

            # Drain pending httpx cleanup tasks to avoid "Event loop is closed" on Python 3.12+
            pending = [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            return result

        return asyncio.run(run_async_call(fun, **params))
    
