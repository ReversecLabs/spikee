from abc import ABC, abstractmethod
from typing import List, Optional

from .target import Target
from spikee.utilities.enums import Turn


class MultiTarget(Target, ABC):
    def __init__(self, turn_types: List[Turn] = [Turn.MULTI], backtrack: bool = False):
        """Define target capabilities and initialize shared dictionary for multi-turn data."""
        super().__init__(turn_types=turn_types, backtrack=backtrack)

        self.__target_data = None

    def add_managed_dicts(self, target_data, add_dicts: List[str] = []):
        """Adds managed dictionaries for multi-turn session data.

        Args:
            target_data: A multiprocessing managed dictionary to store generic data.
            add_dicts (List[str], optional): List of dictionary keys to add. Defaults to {}.
        """
        self.__target_data = target_data

        for dict_key in add_dicts:
            self.__target_data[dict_key] = {}

    def _get_target_data(self, id: str) -> object:
        """Retrieves or initializes session data for a given ID. (Generic Implementation)

        Args:
            id (str): The unique identifier for the session.
        """
        if id is None:
            raise ValueError("id cannot be None")
        if id not in self.__target_data:
            return None
        return self.__target_data[id]

    def _update_target_data(self, id: str, data: object):
        """Updates the session data for a given ID. (Generic Implementation)

        Args:
            id (str): The unique identifier for the session.
            data (object): The session data to store.
        """
        if id is None:
            raise ValueError("id cannot be None")
        self.__target_data[id] = data

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
