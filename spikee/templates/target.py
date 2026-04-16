from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

from spikee.utilities.enums import Turn
from spikee.templates.module import Module
from spikee.utilities.hinting import ContentHint


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
        input_text: ContentHint,
        system_message: Optional[ContentHint] = None,
        target_options: Optional[str] = None,
    ) -> Union[ContentHint, bool, Tuple[Union[ContentHint, bool], Any]]:
        """Sends prompts to the defined target

        Args:
            input_text (ContentHint): User Prompt
            system_message (Optional[ContentHint], optional): System Prompt. Defaults to None.
            target_options (Optional[str], optional): Target options. Defaults to None.

        Returns:
            ContentHint: Response from the target
            bool: Whether the target's response indicates a successful attack (if applicable)
            Tuple[Union[ContentHint, bool], Any]: Optionally return additional metadata along with the response and success status
            throws tester.GuardrailTrigger: Indicates guardrail was triggered
            throws Exception: Raises exception on failure
        """
        pass
