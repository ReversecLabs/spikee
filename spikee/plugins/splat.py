"""
Splats Obfuscation Plugin

This plugin obfuscates the input text using splat-based techniques (e.g., asterisks '*', special characters,
and spacing tricks) while maintaining readability for an LLM. The goal is to create a transformation that
keeps the message understandable but harder to detect with basic filters.

Usage:
    spikee generate --plugins splats

Parameters:
    text (str): The input text to be obfuscated.
    exclude_patterns (List[str], optional): A list of regex patterns. Any substring
        that exactly matches one of these patterns will be preserved as-is.

Returns:
    str: The transformed text using splat-based obfuscation techniques.
"""

import random
import re
from typing import List

from spikee.templates.plugin import Plugin


class SplatPlugin(Plugin):
    def get_available_option_values(self) -> List[str]:
        return None

    def obfuscate_with_splats(self, text: str) -> str:
        """
        Obfuscates the input text using splat-based techniques such as inserting asterisks,
        breaking words with special characters, and spacing tricks while maintaining readability.

        Args:
            text (str): The input text.

        Returns:
            str: The obfuscated text.
        """
        obfuscated_text = []
        for word in text.split():
            obfuscated_word = word

            # Randomly insert asterisks within words
            if random.random() < 0.6:  # 60% chance
                split_point = random.randint(1, len(word) - 1) if len(word) > 1 else 0
                obfuscated_word = word[:split_point] + "*" + word[split_point:]

            # Randomly pad words with splats
            if random.random() < 0.4:  # 40% chance
                obfuscated_word = "*" + obfuscated_word + "*"

            obfuscated_text.append(obfuscated_word)

        return " * ".join(obfuscated_text)  # Separating words with splats

    def transform(self, text: str, exclude_patterns: List[str] = None) -> str:
        """
        Transforms the input text using splat-based obfuscation while preserving substrings that match
        the exclusion regex patterns.

        Args:
            text (str): The input text.
            exclude_patterns (List[str], optional): A list of regex patterns to exclude from transformation.

        Returns:
            str: The obfuscated text using splat-based techniques.
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
                transformed = self.obfuscate_with_splats(chunk)
                result_chunks.append(transformed)

        return " ".join(result_chunks)
