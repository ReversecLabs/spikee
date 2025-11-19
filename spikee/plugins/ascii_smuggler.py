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

from typing import List
import re

from spikee.templates.plugin import Plugin


class AsciiSmuggler(Plugin):
    def get_available_option_values(self) -> List[str]:
        return None

    def encode_message(self, message: str, use_unicode_tags: bool = True) -> dict:
        """
        Encodes the message by converting each ASCII character (0x20 to 0x7E) to a corresponding
        Unicode character in the Private Use Area (starting at 0xE0000). If use_unicode_tags is True,
        start and end markers (U+E0001 and U+E007F) are added.

        Args:
            message (str): The input message.
            use_unicode_tags (bool): Whether to include start/end tags.

        Returns:
            dict: A dictionary with keys "encoded" (the encoded message), "code_points" (a string listing
                the code points used), and "status" (information about any invalid characters).
        """
        encoded = []
        code_points = []
        invalid_chars = ""

        if use_unicode_tags:
            encoded.append(chr(0xE0001))
            code_points.append("U+E0001")

        for char in message:
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
        return {
            "code_points": " ".join(code_points),
            "encoded": "".join(encoded),
            "status": status_message,
        }

    def transform(self, text: str, exclude_patterns: List[str] = None) -> str:
        """
        Converts ASCII text to Unicode tags using the ASCII Smuggler technique.

        If exclude_patterns is provided, the plugin first combines the list into a compound regex
        and uses re.split() to break the text into chunks. Any chunk that exactly matches one of the
        provided exclusion regex patterns is left unmodified; all other chunks are encoded.

        Args:
            text (str): The input text.
            exclude_patterns (List[str], optional): A list of regex strings. Substrings that exactly match
                any of these patterns will be preserved.

        Returns:
            str: The encoded text.
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
                # This chunk exactly matches one of the exclusion patterns—leave it unchanged.
                result_chunks.append(chunk)
            else:
                result = self.encode_message(chunk)
                result_chunks.append(result["encoded"])
        return "".join(result_chunks)
