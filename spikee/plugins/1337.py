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

from typing import List
import re

from spikee.templates.plugin import Plugin


class LeetspeekPlugin(Plugin):
    def get_available_option_values(self) -> List[str]:
        return None

    def transform(self, text: str, exclude_patterns: List[str] = None) -> str:
        """
        Transforms the input text into 1337 speak while preserving any substring that
        exactly matches one of the exclusion regex patterns.

        If an exclusion list is provided, the plugin creates a compound regex by joining
        the patterns. It then splits the text using re.split() so that any substring that
        exactly matches one of the patterns is isolated and left unmodified. All other parts
        are transformed using the leet dictionary.

        Args:
            text (str): The input text.
            exclude_patterns (List[str], optional): A list of regex patterns to exclude from transformation.

        Returns:
            str: The transformed text.
        """
        leet_dict = {
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
                transformed = "".join(leet_dict.get(c, c) for c in chunk)
                result_chunks.append(transformed)
        return "".join(result_chunks)
