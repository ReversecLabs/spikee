"""
Base64 Encoding Plugin

This plugin transforms the input text into Base64 encoding.

Usage:
    spikee generate --plugins base64

Parameters:
    text (str): The input text to be transformed.

Returns:
    str: The transformed text in Base64 encoding.
"""

import base64
from typing import List


def transform(text: str, exclude_patterns: List[str] = None) -> str:
    """
    Transforms the input text into Base64 encoding.

    Args:
        text (str): The input text.

    Returns:
        str: The transformed text in Base64 encoding.
    """
    return base64.b64encode(text.encode()).decode()
