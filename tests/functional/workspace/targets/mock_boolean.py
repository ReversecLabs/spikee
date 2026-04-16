from typing import List, Optional

from spikee.templates.target import Target


class MockBooleanTarget(Target):
    def get_available_option_values(self) -> List[str]:
        return []

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> bool:
        # Return boolean True directly
        return True
