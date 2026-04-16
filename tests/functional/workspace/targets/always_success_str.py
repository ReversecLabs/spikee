from typing import Optional

from spikee.templates.target import Target
from spikee.utilities.hinting import ModuleOptionsHint


CANARY_RESPONSES = "EN-CHECK IT-CHECK restricted-check"


class AlwaysSuccessTarget(Target):
    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> str:
        return str(CANARY_RESPONSES)
