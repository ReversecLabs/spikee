"""Judge that only accepts Audio content."""
from typing import Union, List, Optional

from spikee.templates.judge import Judge
from spikee.utilities.hinting import ModuleOptionsHint, Audio


class AudioOnlyJudge(Judge):
    """Judge with strict Audio type requirement."""

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def judge(
        self,
        llm_input: Audio,
        llm_output: Audio,
        judge_args: Union[str, List[str]],
        judge_options: Optional[str] = None
    ) -> bool:
        """Check if audio output contains expected content."""
        from spikee.utilities.hinting import get_content

        # Extract raw audio content
        output_text = get_content(llm_output)

        # Check if expected string is in output
        expected = judge_args if isinstance(judge_args, str) else judge_args[0]

        return expected in output_text
