from typing import Tuple, List

from .judge import Judge
from spikee.utilities.llm import get_llm


class LLMJudge(Judge):
    DEFAULT_LLM_MODEL = "openai-gpt-4.1-mini"

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """
        Returns the list of supported judge_options; first option is default.
        """
        return [], True

    def _get_llm(self, judge_options):
        """
        Initialize and return the appropriate LLM based on judge_options.
        """
        try:
            llm = get_llm(judge_options)
            return llm
        except ValueError as e:
            print(f"[Error] {e}")
            return None
