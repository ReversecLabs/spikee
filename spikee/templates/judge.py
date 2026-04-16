from abc import ABC, abstractmethod
import random
import string

from spikee.templates.module import Module
from spikee.utilities.hinting import ContentHint


class Judge(Module, ABC):
    @abstractmethod
    def judge(self, llm_input: ContentHint, llm_output: ContentHint, judge_args="", judge_options="") -> bool:
        pass

    def _generate_random_token(self, length=8):
        """
        Generate a random alphanumeric token.
        """
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))
