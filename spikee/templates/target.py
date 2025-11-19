from abc import ABC, abstractmethod
from typing import List, Optional


class Target(ABC):
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
