from typing import Optional

CANARY_RESPONSES = "EN-CHECK IT-CHECK restricted-check"

_CALL_COUNT = 0
_SUCCESS_THRESHOLD = 3


def get_available_option_values():
    return None


def process_input(
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
    logprobs: bool = False,
):
    global _CALL_COUNT
    _CALL_COUNT += 1
    if _CALL_COUNT <= _SUCCESS_THRESHOLD:
        return CANARY_RESPONSES
    return "Sorry, I can't answer that."
