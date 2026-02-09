"""
1337 Plugin

This plugin transforms the input text into "1337 speak" (leetspeak) by replacing
certain letters with numerals according to a fixed dictionary. Any substring that
exactly matches one of the user-supplied exclusion regex patterns (passed via the
exclude_patterns parameter) is left unchanged.

Usage:
    spikee generate --plugins 1337

Reference:
    https://mindgard.ai/blog/bypassing-azure-ai-content-safety-guardrails

Parameters:
    text (str): The input text to be transformed.
    exclude_patterns (List[str], optional): A list of regex patterns. Any substring
        that exactly matches one of these patterns will be preserved as-is.

Returns:
    str: The transformed text.
"""

from typing import List, Tuple
import re


from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag


class LeetspeekPlugin(BasicPlugin):
    LEET_DICT = {
        "A": "4",
        "a": "4",
        "E": "3",
        "e": "3",
        "I": "1",
        "i": "1",
        "O": "0",
        "o": "0",
        "T": "7",
        "t": "7",
        "S": "5",
        "s": "5",
        "B": "8",
        "b": "8",
        "G": "6",
        "g": "6",
        "Z": "2",
        "z": "2",
    }

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Transforms text into 1337 speak."

    def get_available_option_values(self) -> List[str]:
        return None

    def plugin_transform(self, text: str, plugin_option: str = None) -> str:
        """
        Transforms the input text into 1337 speak while preserving any substring that
        exactly matches one of the exclusion regex patterns.

        If an exclusion list is provided, the plugin creates a compound regex by joining
        the patterns. It then splits the text using re.split() so that any substring that
        exactly matches one of the patterns is isolated and left unmodified. All other parts
        are transformed using the leet dictionary.

        Args:
            text (str): The input text.
            plugin_option (str, optional): An optional plugin option.
        Returns:
            str: The transformed text.
        """

        return "".join(self.LEET_DICT.get(c, c) for c in text)
