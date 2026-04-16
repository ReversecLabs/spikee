"""
Caesar Cipher Plugin

This plugin transforms the input text using a simple Caesar cipher encryption.
By default, it shifts letters forward by a 3 number of positions in the alphabet.

Usage:
    spikee generate --plugins caesar

Parameters:
    text (str): The input text to be transformed.
    shift (int): The number of positions to shift each letter (default is 3).

Returns:
    str: The encrypted text using the Caesar cipher.
"""

from typing import List

from spikee.templates.plugin import Plugin
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.content import Text
from spikee.utilities.enums import ModuleTag


class CeasarPlugin(Plugin):
    DEFAULT_SHIFT = 3

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING], "Transforms text using a Caesar cipher encryption."

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [
            "shift=3",
            "shift=N (1-26)",
        ], False

    def _parse_shift_option(self, option: str) -> int:
        """Parse shift option string like 'shift=3' and return the number."""
        if option and option.startswith("shift="):
            try:
                n = int(option.split("=")[1])
                if 0 <= n <= 26:
                    return n
            except (ValueError, IndexError):
                pass
        return self.DEFAULT_SHIFT

    def caesar_cipher(self, text: str, shift: int = 3) -> str:
        """
        Encrypts the input text using a Caesar cipher with the given shift value.

        Args:
            text (str): The input text.
            shift (int): The number of positions to shift each letter.

        Returns:
            str: The encrypted text.
        """
        result = []
        for char in text:
            if char.isalpha():
                shift_base = ord("A") if char.isupper() else ord("a")
                result.append(chr((ord(char) - shift_base + shift) % 26 + shift_base))
            else:
                result.append(char)
        return "".join(result)

    def transform(
        self,
        content: Text,
        exclude_patterns: List[str] = [],
        plugin_option: str = ""
    ) -> Text:
        """
        Transforms the input text using the Caesar cipher.

        Args:
            text (str): The input text.
            shift (int): The number of positions to shift each letter (default is 3).

        Returns:
            str: The encrypted text using the Caesar cipher.
        """
        shift = self._parse_shift_option(plugin_option)

        return Text(self.caesar_cipher(content.content, shift))
