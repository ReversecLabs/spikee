"""
Hex Encoding Plugin

This plugin transforms the input text into hexadecimal encoding. Any substring that
exactly matches one of the user-supplied exclusion regex patterns (passed via the
exclude_patterns parameter) is left unchanged.

Usage:
    spikee generate --plugins hex

Parameters:
    text (str): The input text to be transformed.
    exclude_patterns (List[str], optional): A list of regex patterns. Any substring
        that exactly matches one of these patterns will be preserved as-is.

Returns:
    str: The transformed text in hexadecimal encoding.
"""

import re
from typing import List, Tuple

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag


class HexPlugin(BasicPlugin):

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Transforms text into hexadecimal encoding."

    def get_available_option_values(self) -> List[str]:
        return None

    def plugin_transform(self, text: str, plugin_option: str = None) -> str:
        """
        Transforms the input text into hexadecimal encoding while preserving any substring
        that exactly matches one of the exclusion regex patterns.

        Args:
            text (str): The input text.
            plugin_option (str, optional): An optional plugin option.

        Returns:
            str: The transformed text in hexadecimal encoding.
        """

        return text.encode().hex()
