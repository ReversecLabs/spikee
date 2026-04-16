from typing import List, Optional

from spikee.templates.target import Target
from spikee.tester import GuardrailTrigger


class AlwaysGuardrailTarget(Target):
    def get_available_option_values(self) -> List[str]:
        return []

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> str:
        raise GuardrailTrigger("This is a guardrail trigger. The response should be checked for the canary response to confirm that the guardrail was triggered correctly.")
