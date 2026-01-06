from abc import ABC, abstractmethod
from typing import List, Optional

from .target import Target
from spikee.utilities.enums import Turn


class MultiTarget(Target, ABC):
    def __init__(self, turn_types: List[Turn] = [Turn.SINGLE], backtrack: bool = False):
        """Define target capabilities and initialize shared dictionary for multi-turn data."""
        super().__init__(
            turn_types=turn_types,
            backtrack=backtrack
        )

        self.__shared_dict = None

    def add_managed_dict(self, shared_dict):
        """Adds a managed dictionary for multi-turn session data.

        Args:
            shared_dict: A multiprocessing managed dictionary to store session data.
        """
        self.__shared_dict = shared_dict

    def _get_spikee_session_data(self, spikee_session_id: str) -> object:
        """Retrieves or initializes session data for a given Spikee session ID.

        Args:
            spikee_session_id (str): The unique identifier for the Spikee session.
        """
        if spikee_session_id not in self.__shared_dict:
            return None
        return self.__shared_dict[spikee_session_id]

    def update_spikee_session_data(self, spikee_session_id: str, data: object):
        """Updates the session data for a given Spikee session ID.

        Args:
            spikee_session_id (str): The unique identifier for the Spikee session.
            data (object): The session data to store.
        """
        self.__shared_dict[spikee_session_id] = data

    @abstractmethod
    def get_available_option_values(self) -> List[str]:
        """Returns supported option values.

        Returns:
            List[str]: List of supported options; first is default.
        """
        return None

    @abstractmethod
    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        spikee_session_id: Optional[str] = None,
        backtrack: Optional[bool] = False,
    ) -> object:
        """Sends prompts to the defined target

        Args:
            input_text(str): User Prompt
            system_message(Optional[str], optional): System Prompt. Defaults to None.
            target_options(Optional[str], optional): Target options. Defaults to None.

        Returns:
            str: Response from the target
            throws tester.GuardrailTrigger: Indicates guardrail was triggered
            throws Exception: Raises exception on failure
        """
        pass
