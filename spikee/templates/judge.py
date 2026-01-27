from abc import ABC, abstractmethod
from typing import List, Tuple
import random
import string

from spikee.utilities.enums import JudgeType


class Judge(ABC):
    @abstractmethod
    def get_description(self) -> Tuple[JudgeType, str]:
        """Returns the type and a short description of the judge."""
        pass

    @abstractmethod
    def get_available_option_values(self) -> List[str]:
        """Returns supported option values.

        Returns:
            List[str]: List of supported options; first is default.
        """
        return None

    @abstractmethod
    def judge(self, llm_input, llm_output, judge_args="", judge_options=None) -> bool:
        pass

    def _generate_random_token(self, length=8):
        """
        Generate a random alphanumeric token.
        """
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))
