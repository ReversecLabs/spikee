"""
BinaryInAscii Plugin

Encodes text as space-separated ASCII/Unicode binary codepoint values (8 bits each).
"Spikee plugin" > "01010011 01110000 01101001 01101011 01100101 01100101 00100000
01110000 01101100 01110101 01100111 01101001 01101110"

Usage:
    spikee generate --plugins binary_in_ascii

Parameters:
    text (str): Input string to transform.
    exclude_patterns (List[str], optional): Supplied by the framework. Substrings matching
        these regex patterns are preserved as-is.

Returns:
    str: Space-separated 8-bit binary codepoint values.
"""

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint


class BinaryInAsciiPlugin(BasicPlugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [
            ModuleTag.ENCODING
        ], "Encodes text as 8-bit binary codepoint representation."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [], False

    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        return self._apply_binary(text)

    def _apply_binary(self, text: str) -> str:
        """Return space-separated 8-bit binary Unicode codepoints for each character."""
        return " ".join(f"{ord(c):08b}" for c in text)
