"""Mock target that accepts and returns Audio content."""
from typing import Optional

from spikee.templates.target import Target
from spikee.utilities.hinting import ModuleOptionsHint, Audio


class MockAudioTarget(Target):
    """Target that accepts Audio input and returns Audio output."""

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def process_input(
        self,
        input_text: Audio,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> Audio:
        """Echo back audio with a prefix."""
        from spikee.utilities.hinting import get_content, get_content_type

        # Get raw content
        raw = get_content(input_text)
        content_type = get_content_type(input_text)

        # Add prefix indicating content type received
        response = f"AUDIO_ECHO[{content_type}]:{raw}"

        return Audio(response)
