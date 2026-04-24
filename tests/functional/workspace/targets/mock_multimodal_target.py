"""Mock target that accepts any Content type and returns matching type."""
from typing import Optional

from spikee.templates.target import Target
from spikee.utilities.hinting import ModuleOptionsHint, Content


class MockMultimodalTarget(Target):
    """Target that accepts any Content type and echoes with same type."""

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def process_input(
        self,
        input_text: Content,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> Content:
        """Echo back content with same type as input."""
        from spikee.utilities.hinting import get_content, get_content_type, content_factory

        # Get raw content and type
        raw = get_content(input_text)
        content_type = get_content_type(input_text)

        # Add prefix and return same type
        response = f"MULTIMODAL_ECHO[{content_type}]:{raw}"

        return content_factory(response, content_type)
