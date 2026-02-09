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
from typing import List, Tuple

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.modules import parse_options


class SplatPlugin(BasicPlugin):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Transforms text using splat-based obfuscation techniques."

    def get_available_option_values(self) -> List[str]:
        return ["character=*", "insert_rand=0.6", "pad_rand=0.4"]

    def plugin_transform(self, text: str, plugin_option: str = None) -> str:
        """
        Obfuscates the input text using splat-based techniques such as inserting asterisks,
        breaking words with special characters, and spacing tricks while maintaining readability.

        Args:
            text (str): The input text.

        Returns:
            str: The obfuscated text.
        """
        options = parse_options(plugin_option)
        character = options.get("character", "*")
        insert_rand = float(options.get("insert_rand", 0.6))
        pad_rand = float(options.get("pad_rand", 0.4))

        obfuscated_text = []
        for word in text.split():
            obfuscated_word = word

            # Randomly insert asterisks within words
            if random.random() < insert_rand:  # 60% chance
                split_point = random.randint(1, len(word) - 1) if len(word) > 1 else 0
                obfuscated_word = word[:split_point] + character + word[split_point:]

            # Randomly pad words with splats
            if random.random() < pad_rand:  # 40% chance
                obfuscated_word = character + obfuscated_word + character

            obfuscated_text.append(obfuscated_word)

        return f" {character} ".join(obfuscated_text)  # Separating words with splats
