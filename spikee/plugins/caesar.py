"""
Caesar Cipher Plugin

This plugin transforms the input text using a simple Caesar cipher.
By default, it shifts letters forward by a 3 number of positions in the alphabet.
"Hello" with shift=3 > "Khoor"
Use shift=13 for rot13, which is its own reverse.

Usage:
    spikee generate --plugins caesar
    spikee generate --plugins caesar --plugin-options "caesar:shift=5"

Parameters:
    text (str): The input text to be transformed.
    shift (int): The number of positions to shift each letter (default is 3).
Returns:
    str: The encrypted text using the Caesar cipher.
"""

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint


class CaesarPlugin(BasicPlugin):
    DEFAULT_SHIFT = 3

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING], "Transforms text using a Caesar cipher encryption."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return ["shift=3", "shift=N (1-26)", "shift=13 for rot13"], False

    def plugin_transform(self, content: str, plugin_option: str = "") -> str:
        shift = self._parse_shift_option(plugin_option)
        return self.caesar_cipher(content, shift)

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

    def caesar_cipher(self, content: str, shift: int = 3) -> str:
        """
        Encrypts the input text using a Caesar cipher with the given shift value.

        Args:
            content (str): The input text.
            shift (int): The number of positions to shift each letter.

        Returns:
            str: The encrypted text.
        """
        result = []
        for char in content:
            if char.isalpha():
                shift_base = ord("A") if char.isupper() else ord("a")
                result.append(chr((ord(char) - shift_base + shift) % 26 + shift_base))
            else:
                result.append(char)
        return "".join(result)
