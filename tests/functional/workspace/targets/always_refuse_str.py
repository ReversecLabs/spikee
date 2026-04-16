from typing import Optional

from spikee.templates.target import Target
from spikee.utilities.hinting import ModuleOptionsHint


class AlwaysRefuseTarget(Target):
    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> str:
        return str("Sorry, I can't answer that.")
