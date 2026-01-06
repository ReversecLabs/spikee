from typing import Tuple, List

from .judge import Judge
from spikee.utilities.llm import get_llm, get_supported_llm_models, get_supported_prefixes


class LLMJudge(Judge):
    DEFAULT_LLM_MODEL = "openai-gpt-4.1-mini"

    def get_available_option_values(self):
        """
        Returns the list of supported judge_options; first option is default.
        """
        options = [self.DEFAULT_LLM_MODEL]  # Default first
        options.extend(
            [
                model
                for model in get_supported_llm_models()
                if model != self.DEFAULT_LLM_MODEL
            ]
        )
        return options

    def get_available_prefixes(self) -> Tuple[bool, List[str]]:
        """
        Returns the list of supported LLM model prefixes.
        """
        return True, get_supported_prefixes()

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
