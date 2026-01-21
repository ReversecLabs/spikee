from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from spikee.utilities.enums import Turn
from spikee.templates.standardised_conversation import StandardisedConversation


class Attack(ABC):
    def __init__(self, turn_type: Turn = Turn.SINGLE):
        super().__init__()

        self.turn_type = turn_type

    @staticmethod
    def standardised_input_return(input: str, conversation: StandardisedConversation = None, objective: str = None) -> Dict[str, Any]:
        """Standardise the return format for attacks."""
        standardised_return = {
            "input": str(input)
        }

        if conversation:
            standardised_return["conversation"] = str(conversation.conversation)

        if objective:
            standardised_return["objective"] = str(objective)

        return standardised_return

    @abstractmethod
    def get_available_option_values(self) -> List[str]:
        """Return supported attack options; first option is default."""
        return None

    @abstractmethod
    def attack(
        self,
        entry: Dict[str, Any],
        target_module: Any,
        call_judge: callable,
        max_iterations: int,
        attempts_bar=None,
        bar_lock=None,
    ) -> Tuple[int, bool, object, str]:
        """
        Performs attack on the target module.

        Returns:
            Tuple[int, bool, object, str]: A tuple containing:
                - Total number of messages in the conversation (int)
                - Success status of the attack (bool)
                - Input (Str or Dict) - Use standardised_input_return to format Dict
                - Last response from the target module (str)
        """
        pass
