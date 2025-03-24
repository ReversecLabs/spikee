"""
sample_target.py

This is an example target for spikee that returns a pre-canned (mock) response.
Use it as a template for writing real targets that call APIs or local models.

Usage:
    1. Place this file in your local `targets/` folder.
    2. Run the spikee test command, pointing to this target, e.g.:
         spikee test --dataset datasets/example.jsonl --target sample_target
"""

import os
from dotenv import load_dotenv

# Load environment variables, if you need them (e.g., for API keys).
load_dotenv()

def process_input(input_text, system_message=None):
    """
    Mock target function required by spikee.

    This function simulates an LLM or guardrail by returning a fixed string.
    In a real target:
      - You might instantiate a model client here (e.g., OpenAI, Azure, etc.)
      - You'd pass `input_text` (and optionally `system_message`) to the model
      - Return the raw text or a decision (True/False) for guardrails

    Args:
        input_text (str): The main prompt or text to be processed.
        system_message (str, optional): A system or meta-prompt, if applicable.

    Returns:
        str or bool: For spikee's "canary" success criteria, return a string.
                     For the "boolean" success criteria, return True/False.
    """

    # If you wanted to do something with system_message:
    if system_message:
        # Possibly parse it, log it, or combine with input_text.
        pass

    # Return a pre-canned response to mimic an LLM answer
    mock_response = "Hello! I am a mock LLM. "

    return mock_response
