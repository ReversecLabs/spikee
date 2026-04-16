from typing import Optional
from spikee.utilities.content import Text


def get_available_option_values():
    return None


def process_input(
    input_text: Text,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
    logprobs: bool = False,
):
    return Text("Sorry, I can't answer that.")
