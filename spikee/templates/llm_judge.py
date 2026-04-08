from typing import Tuple, List, Union

from .judge import Judge
from spikee.utilities.llm import get_llm
from spikee.templates.provider import Provider


class LLMJudge(Judge):
    DEFAULT_LLM_MODEL = "openai/gpt-4o"

    def __init__(self, max_tokens=None):
        super().__init__()
        self.max_tokens = max_tokens

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """
        Returns the list of supported judge_options; first option is default.
        """
        return [], True

    def _get_llm(self, judge_options="") -> Union[Provider, None]:
        """
        Initialize and return the appropriate LLM based on judge_options.
        """
        if judge_options is None or judge_options.strip() == "":
            judge_options = self.DEFAULT_LLM_MODEL

        return get_llm(judge_options, max_tokens=self.max_tokens)
