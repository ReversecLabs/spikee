"""
ASCII Smuggler Plugin

This plugin converts ASCII text into a series of Unicode tags that are generally invisible
in most UI elements. This technique is useful for bypassing certain content filters.
The encoding is done by mapping each ASCII character (in the range 0x20 to 0x7E) to a
corresponding Unicode code point in the Private Use Area (starting at 0xE0000). Optionally,
start and end markers (0xE0001 and 0xE007F) can be added.

If an exclusion list is provided via the parameter `exclude_patterns` (a list of regex strings),
the plugin will split the input text using a compound regex. Any substring that exactly matches
one of these regex patterns will be preserved (left unchanged), while all other parts will be encoded.

Usage:
    spikee generate --plugins ascii-smuggler

Reference:
    https://embracethered.com/blog/ascii-smuggler.html

Parameters:
    text (str): The input text to encode.
    exclude_patterns (List[str], optional): A list of regex patterns. Any substring that exactly
        matches one of these patterns is left untransformed.

Returns:
    str: The encoded text.
"""

from typing import List, Tuple

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.modules import parse_options


class AsciiSmuggler(BasicPlugin):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Transforms ASCII text into Unicode tags using the ASCII Smuggler technique."

    def get_available_option_values(self) -> List[str]:
        return None

    def plugin_transform(self, text: str, plugin_option: str = None) -> str:
        """
        Converts ASCII text to Unicode tags using the ASCII Smuggler technique.

        Encodes the message by converting each ASCII character (0x20 to 0x7E) to a corresponding
        Unicode character in the Private Use Area (starting at 0xE0000). If use_unicode_tags is True,
        start and end markers (U+E0001 and U+E007F) are added.

        Args:
            text (str): The input text.
            plugin_option (str, optional): An optional plugin option.

        Returns:
            str: The transformed text.
        """
        encoded = []
        code_points = []
        invalid_chars = ""

        options = parse_options(plugin_option)
        use_unicode_tags = options.get("use-unicode-tags", "true").lower() == "true"

        if use_unicode_tags:
            encoded.append(chr(0xE0001))
            code_points.append("U+E0001")

        for char in text:
            if 0x20 <= ord(char) <= 0x7E:
                code_point = 0xE0000 + ord(char)
                encoded.append(chr(code_point))
                code_points.append(f"U+{code_point:X}")
            else:
                invalid_chars += char
                encoded.append(char)

        if use_unicode_tags:
            encoded.append(chr(0xE007F))
            code_points.append("U+E007F")

        status_message = (
            f"Invalid characters detected: {invalid_chars}" if invalid_chars else ""
        )

        full_response = {
            "code_points": " ".join(code_points),
            "encoded": "".join(encoded),
            "status": status_message,
        }

        return "".join(encoded)
