from abc import ABC, abstractmethod
from typing import List, Optional

from .multi_target import MultiTarget
from spikee.utilities.enums import Turn


class SimpleMultiTarget(MultiTarget, ABC):
    __SIMPLIFIED_CONVERSATION_KEY = "conversation_data"
    __SIMPLIFIED_ID_MAP_KEY = "id_map"

    def __init__(self, turn_types: List[Turn] = [Turn.MULTI], backtrack: bool = False):
        """Define target capabilities and initialize shared dictionary for multi-turn data."""
        super().__init__(turn_types=turn_types, backtrack=backtrack)

    def add_managed_dicts(self, target_data):
        """Adds managed dictionaries for multi-turn session data.

        Args:
            target_data: A multiprocessing managed dictionary to store generic data.
        """
        super().add_managed_dicts(
            target_data,
            [self.__SIMPLIFIED_CONVERSATION_KEY, self.__SIMPLIFIED_ID_MAP_KEY],
        )

    def _get_conversation_data(self, session_id: str) -> object:
        """Retrieves or initializes conversation data for a given session ID. (Simplified Implementation - Conversation)
           Abstraction layer over the generic implementation, that stores conversation data within a nested dictionary.

        Args:
            session_id (str): The unique identifier for the conversation session.
        """
        if session_id is None:
            raise ValueError("session_id cannot be None")

        # Get all conversation data for a specific key
        conversation_data = self._get_target_data(self.__SIMPLIFIED_CONVERSATION_KEY)

        # Create new blank conversation
        if conversation_data.get(session_id, None) is None:
            conversation_data[session_id] = []
            self._update_target_data(
                self.__SIMPLIFIED_CONVERSATION_KEY, conversation_data
            )

        return conversation_data[session_id]

    def _update_conversation_data(self, session_id: str, conversation_data: object):
        """Updates a conversation for a given session ID. (Simplified Implementation - Conversation)
           Abstraction layer over the generic implementation, that stores conversation data within a nested dictionary.

        Args:
            session_id (str): The unique identifier for the conversation session.
            message (object): The message to append to the conversation data.
        """
        if session_id is None:
            raise ValueError("session_id cannot be None")

        # Update all conversation data for a specific key
        conversations = self._get_target_data(self.__SIMPLIFIED_CONVERSATION_KEY)
        conversations[session_id] = conversation_data
        self._update_target_data(self.__SIMPLIFIED_CONVERSATION_KEY, conversations)

    def _append_conversation_data(self, session_id: str, role: str, content: str):
        """Appends a message to the conversation data for a given session ID. (Simplified Implementation - Conversation)
           Abstraction layer over the generic implementation, that stores conversation data within a nested dictionary.

        Args:
            session_id (str): The unique identifier for the conversation session.
            message (object): The message to append to the conversation data.
        """
        if session_id is None:
            raise ValueError("session_id cannot be None")

        # Append message to conversation data for a specific key
        conversation = self._get_conversation_data(session_id=session_id)
        conversation.append({"role": role, "content": content})
        self._update_conversation_data(
            session_id=session_id, conversation_data=conversation
        )

    def _get_id_map(self, spikee_session_id: str):
        """Returns the ID mapping for a given parent ID. (Simplified Implementation - ID Map)
           Abstraction layer over the generic implementation, that stores ID mappings within a nested dictionary.

        Args:
            spikee_session_id (str): The unique identifier for the parent session.
        """
        if spikee_session_id is None:
            raise ValueError("spikee_session_id cannot be None")

        # Get existing ID map or initialize a new one
        id_map = self._get_target_data(self.__SIMPLIFIED_ID_MAP_KEY)

        if id_map.get(spikee_session_id, None) is None:
            id_map[spikee_session_id] = None
            self._update_target_data(self.__SIMPLIFIED_ID_MAP_KEY, id_map)
        return id_map[spikee_session_id]

    def _update_id_map(self, spikee_session_id: str, associated_ids: object):
        """Updates the ID mapping for a given parent ID (Simplified Implementation - ID Map)
           Abstraction layer over the generic implementation, that stores ID mappings within a nested dictionary.

        Args:
            spikee_session_id (str): The unique identifier for the parent session.
            associated_ids (object): A list of associated child session IDs.
        """
        if spikee_session_id is None:
            raise ValueError("spikee_session_id cannot be None")
        if associated_ids is None:
            raise ValueError("associated_ids cannot be None")

        # Update ID map for a specific parent ID
        id_map = self._get_target_data(self.__SIMPLIFIED_ID_MAP_KEY)
        id_map[spikee_session_id] = associated_ids
        self._update_target_data(self.__SIMPLIFIED_ID_MAP_KEY, id_map)

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
