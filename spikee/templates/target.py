from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from spikee.utilities.enums import Turn
from spikee.templates.module import Module


class Target(Module, ABC):
    def __init__(self, turn_types: List[Turn] = [Turn.SINGLE], backtrack: bool = False):
        super().__init__()

        self.config = {
            "single-turn": Turn.SINGLE in turn_types,
            "multi-turn": Turn.MULTI in turn_types,
            "backtrack": backtrack,
        }

    @abstractmethod
    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> Union[str, bool, Tuple[Union[str, bool], Any]]:
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
