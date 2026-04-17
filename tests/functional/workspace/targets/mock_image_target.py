"""Mock target that accepts and returns Image content."""
from typing import Optional

from spikee.templates.target import Target
from spikee.utilities.hinting import ModuleOptionsHint, Image


class MockImageTarget(Target):
    """Target that accepts Image input and returns Image output."""

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def process_input(
        self,
        input_text: Image,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> Image:
        """Echo back image with a prefix."""
        from spikee.utilities.hinting import get_content, get_content_type

        # Get raw content
        raw = get_content(input_text)
        content_type = get_content_type(input_text)

        # Add prefix indicating content type received
        response = f"IMAGE_ECHO[{content_type}]:{raw}"

        return Image(response)
