from abc import ABC, abstractmethod
from typing import List, Optional

from spikee.utilities.enums import Turn


class Target(ABC):
    def __init__(self, turn_types: List[Turn] = [Turn.SINGLE], backtrack: bool = False):
        super().__init__()

        self.config = {
            "single-turn": Turn.SINGLE in turn_types,
            "multi-turn": Turn.MULTI in turn_types,
            "backtrack": backtrack
        }

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
    ) -> object:
        """Sends prompts to the defined target

        Args:
            input_text (str): User Prompt
            system_message (Optional[str], optional): System Prompt. Defaults to None.
            target_options (Optional[str], optional): Target options. Defaults to None.

        Returns:
            str: Response from the target
            throws tester.GuardrailTrigger: Indicates guardrail was triggered
            throws Exception: Raises exception on failure
        """
        pass
