"""
regex.py

A judge for spikee that checks if 'judge_args'
(a Python regular expression) matches the model's output.
Useful for more flexible detection patterns (case-insensitive,
partial matches, capturing groups, etc.).
"""

import re

from spikee.templates.judge import Judge
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.content import Text


class RegexJudge(Judge):
    def get_description(self) -> ModuleDescriptionHint:
        return [], "Checks if a specified regex pattern matches the model's output."

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [], False

    def judge(self, llm_input: Text, llm_output: Text, judge_args, judge_options=None) -> bool:
        """
        Args:
            llm_input (Text): The original text/prompt sent to the model (optional for logic here).
            llm_output (Text): The model's response.
            judge_args (str): A valid Python regex pattern to match in llm_output.

        Returns:
            bool: True if regex pattern 'judge_args' is found in llm_output, otherwise False.
        """
        if not judge_args:
            return False

        if isinstance(judge_args, str):
            pattern = re.compile(judge_args)
            return bool(pattern.search(llm_output.content))

        elif isinstance(judge_args, list):
            for arg in judge_args:
                pattern = re.compile(arg)
                if pattern.search(llm_output.content):
                    return True
            return False

        else:
            raise ValueError(
                "judge_args must be a string or list of strings representing regex patterns."
            )


if __name__ == "__main__":
    judge = RegexJudge()
    print(judge.judge(Text(""), Text("Wowzers is a great word"), "Wowzers"))
