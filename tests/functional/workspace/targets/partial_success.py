from typing import List, Optional

from spikee.templates.target import Target


class PartialSuccessTarget(Target):
    CANARY_RESPONSES = "EN-CHECK IT-CHECK restricted-check"
    SUCCESS_THRESHOLD = 3

    def __init__(self) -> None:
        self._call_count = 0

    def get_available_option_values(self) -> List[str]:
        return []

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> str:
        self._call_count += 1
        if self._call_count <= self.SUCCESS_THRESHOLD:
            return self.CANARY_RESPONSES
        return "Sorry, I can't answer that."
