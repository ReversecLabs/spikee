from typing import Optional


CANARY_RESPONSES = "EN-CHECK IT-CHECK restricted-check"


def get_available_option_values():
    return None


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
    logprobs: bool = False,
):
    return CANARY_RESPONSES
