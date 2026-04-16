from typing import Optional

from spikee.templates.target import Target
from spikee.utilities.hinting import ModuleOptionsHint
from spikee.utilities.content import Text


class MockBooleanTarget(Target):
    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def process_input(
        self,
        input_text: Text,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> bool:
        # Return boolean True directly
        return True
