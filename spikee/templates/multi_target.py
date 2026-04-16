from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union

from .target import Target
from spikee.utilities.enums import Turn
from spikee.utilities.hinting import ContentHint


class MultiTarget(Target, ABC):
    def __init__(self, turn_types: List[Turn] = [Turn.MULTI], backtrack: bool = False):
        """Define target capabilities and initialize shared dictionary for multi-turn data."""
        super().__init__(turn_types=turn_types, backtrack=backtrack)

        self.__target_data: Dict[Any, Any] = {}

    def add_managed_dicts(self, target_data, add_dicts: List[str] = []):
        """Adds managed dictionaries for multi-turn session data.

        Args:
            target_data: A multiprocessing managed dictionary to store generic data.
            add_dicts (List[str], optional): List of dictionary keys to add. Defaults to {}.
        """
        self.__target_data = target_data

        for dict_key in add_dicts:
            self.__target_data[dict_key] = {}

    def _get_target_data(self, uid: str) -> Any:
        """Retrieves or initializes session data for a given ID. (Generic Implementation)

        Args:
            uid (str): The unique identifier for the session.
        """
        if uid not in self.__target_data:
            return None

        return self.__target_data[uid]

    def _update_target_data(self, uid: str, data: Any):
        """Updates the session data for a given ID. (Generic Implementation)

        Args:
            uid (str): The unique identifier for the session.
            data (Any): The session data to store.
        """
        self.__target_data[uid] = data

    @abstractmethod
    def process_input(
        self,
        input_text: ContentHint,
        system_message: Optional[ContentHint] = None,
        target_options: Optional[str] = None,
        spikee_session_id: Optional[str] = None,
        backtrack: Optional[bool] = False,
    ) -> Union[ContentHint, bool, Tuple[Union[ContentHint, bool], Any]]:
        """Sends prompts to the defined target

        Args:
            input_text(ContentHint): User Prompt
            system_message(Optional[ContentHint], optional): System Prompt. Defaults to None.
            target_options(Optional[str], optional): Target options. Defaults to None.

        Returns:
            ContentHint: Response from the target
            bool: Whether the target's response indicates a successful attack (if applicable)
            Tuple[Union[ContentHint, bool], Any]: Optionally return additional metadata along with the response and success status
            throws tester.GuardrailTrigger: Indicates guardrail was triggered
            throws Exception: Raises exception on failure
        """
        pass
