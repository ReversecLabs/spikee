from abc import ABC, abstractmethod
import json
from typing import Dict, Any, Union, Callable, Optional

from spikee.utilities.enums import Turn
from spikee.templates.module import Module
from spikee.templates.standardised_conversation import StandardisedConversation
from spikee.utilities.hinting import Content, AttackResponseHint


class Attack(Module, ABC):
    def __init__(self, turn_type: Turn = Turn.SINGLE):
        super().__init__()

        self.turn_type = turn_type

    @staticmethod
    def standardised_input_return(
        input: Content,
        conversation: Union[StandardisedConversation, None] = None,
        objective: Optional[Content] = None,
    ) -> Dict[str, Any]:
        """Standardise the return format for attacks."""
        standardised_return = {"input": input if isinstance(input, Content) else str(input)}

        if conversation:
            standardised_return["conversation"] = json.dumps(conversation.conversation)

        if objective:
            standardised_return["objective"] = str(objective)

        return standardised_return

    @abstractmethod
    def attack(
        self,
        entry: Dict[str, Any],
        target_module: Any,
        call_judge: Callable,
        max_iterations: int,
        attempts_bar=None,
        bar_lock=None,
        attack_options=None,
    ) -> AttackResponseHint:
        """
        Performs attack on the target module.

        Returns:
            AttackResponseHint / Tuple[int, bool, Union[Content, Dict[str, Any]], Content]: A tuple containing:
                - Total number of messages in the conversation (int)
                - Success status of the attack (bool)
                - Input (Str or Dict) - Use standardised_input_return to format Dict
                - Last response from the target module (str)
        """
