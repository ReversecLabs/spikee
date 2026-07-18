"""
Base32 Encoding Plugin

This plugin transforms the input text into Base32 encoding.

Base32 uses a different alphabet (A-Z and 2-7) from Base64, so it tokenises
differently and can slip past guardrails or filters that only account for
Base64-style payloads.

Usage:
    spikee generate --plugins base32

Parameters:
    text (str): The input text to be transformed.

Returns:
    str: The transformed text in Base32 encoding.
"""

import base64
from typing import List, Optional

from spikee.templates.plugin import Plugin
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.enums import ModuleTag


class Base32(Plugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING], "Transforms text into Base32 encoding."

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [], False

    def transform(
        self, content: str, exclude_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Transforms the input text into Base32 encoding.

        Args:
            content (str): The input text.

        Returns:
            str: The transformed text in Base32 encoding.
        """
        return base64.b32encode(content.encode()).decode()
