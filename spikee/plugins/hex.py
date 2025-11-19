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
from typing import List

from spikee.templates.plugin import Plugin


class HexPlugin(Plugin):
    def get_available_option_values(self) -> List[str]:
        return None

    def hex_encode(self, text: str) -> str:
        """
        Encodes the input text into hexadecimal format.

        Args:
            text (str): The input text.

        Returns:
            str: The hex-encoded representation of the text.
        """
        return text.encode().hex()

    def transform(self, text: str, exclude_patterns: List[str] = None) -> str:
        """
        Transforms the input text into hexadecimal encoding while preserving any substring
        that exactly matches one of the exclusion regex patterns.

        Args:
            text (str): The input text.
            exclude_patterns (List[str], optional): A list of regex patterns to exclude from transformation.

        Returns:
            str: The transformed text in hexadecimal encoding.
        """
        if exclude_patterns:
            compound = "(" + "|".join(exclude_patterns) + ")"
            compound_re = re.compile(compound)
            chunks = re.split(compound, text)
        else:
            chunks = [text]
            compound_re = None

        result_chunks = []
        for chunk in chunks:
            if compound_re and compound_re.fullmatch(chunk):
                # Leave excluded substrings untouched.
                result_chunks.append(chunk)
            else:
                transformed = self.hex_encode(chunk)
                result_chunks.append(transformed)

        return " ".join(result_chunks)
