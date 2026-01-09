from abc import ABC, abstractmethod
from typing import List, Optional

from .target import Target
from spikee.utilities.enums import Turn


class MultiTarget(Target, ABC):
    def __init__(self, turn_types: List[Turn] = [Turn.MULTI], backtrack: bool = False):
        """Define target capabilities and initialize shared dictionary for multi-turn data."""
        super().__init__(
            turn_types=turn_types,
            backtrack=backtrack
        )

        self.__session_dict = None
        self.__id_dict = None

    def add_managed_dicts(self, session_data, id_data):
        """Adds managed dictionaries for multi-turn session data.

        Args:
            history_data: A multiprocessing managed dictionary to store session history data.
            correlation_data: A multiprocessing managed dictionary to store session ID correlation data.
        """
        self.__session_dict = session_data
        self.__id_dict = id_data

    def _get_spikee_session_data(self, spikee_session_id: str) -> object:
        """Retrieves or initializes session data for a given Spikee session ID.

        Args:
            spikee_session_id (str): The unique identifier for the Spikee session.
        """
        if spikee_session_id is None:
            raise ValueError("spikee_session_id cannot be None")
        if spikee_session_id not in self.__session_dict:
            return None
        return self.__session_dict[spikee_session_id]

    def _update_spikee_session_data(self, spikee_session_id: str, data: object):
        """Updates the session data for a given Spikee session ID.

        Args:
            spikee_session_id (str): The unique identifier for the Spikee session.
            data (object): The session data to store.
        """
        if spikee_session_id is None:
            raise ValueError("spikee_session_id cannot be None")
        self.__session_dict[spikee_session_id] = data

    def get_target_session_id(self, spikee_session_id: str) -> str:
        """Retrieves the target-specified ID for a given Spikee session ID.

        Args:
            spikee_session_id (str): The unique identifier for the Spikee session.
        """
        if spikee_session_id is None:
            raise ValueError("spikee_session_id cannot be None")
        if spikee_session_id not in self.__id_dict:
            return None
        return self.__id_dict[spikee_session_id]

    def set_target_session_id(self, spikee_session_id: str, target_session_id: str):
        """Sets the correlation between the Spikee session ID and the target-specified ID.
        This overwrites any existing mapping for this spikee_session_id.

        Args:
            spikee_session_id (str): The unique identifier for the Spikee session.
            target_session_id (str): The target-specified identifier to correlate.
        """
        if spikee_session_id is None:
            raise ValueError("spikee_session_id cannot be None")
        if target_session_id is None:
            raise ValueError("target_session_id cannot be None")

        self.__id_dict[spikee_session_id] = target_session_id

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
