"""Judge that validates content types."""
from typing import Union, List, Optional

from spikee.templates.judge import Judge
from spikee.utilities.hinting import ModuleOptionsHint, Content, get_content


class ContentTypeJudge(Judge):
    """Judge that checks if output contains expected content type marker."""

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def judge(
        self,
        llm_input: Content,
        llm_output: Content,
        judge_args: Union[str, List[str]],
        judge_options: Optional[str] = None
    ) -> bool:
        """Check if output contains the expected type marker from judge_args."""

        # Extract raw content from both input and output
        output_text = get_content(llm_output)

        # judge_args contains the expected marker (e.g., "AUDIO_ECHO", "IMAGE_ECHO")
        expected_marker = judge_args if isinstance(judge_args, str) else judge_args[0]

        return expected_marker in output_text
