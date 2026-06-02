"""
DecimalEncoder Plugin

Encodes text as space-separated ASCII/Unicode decimal codepoint values.
"Spikee plugin" > "decimal 83 112 105 107 101 101 32 112 108 117 103 105 110"

Usage:
    spikee generate --plugins decimal
    spikee generate --plugins decimal --plugin-options "decimal:hint=false"

Parameters:
    text (str): Input string to transform.
    exclude_patterns (List[str], optional): Supplied by the framework. Substrings matching
        these regex patterns are preserved as-is.

Returns:
    str: Space-separated decimal codepoint values, prefixed with 'decimal ' by default.
"""

from typing import List, Optional

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.modules import parse_options


class DecimalEncoderPlugin(BasicPlugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING], "Encodes text as decimal codepoint representation."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [
            "hint=true",
            "hint=true/false (hint=true prepends the literal string 'decimal ' to the full output)",
        ], False

    def plugin_transform(self, content: str, plugin_option: str = "") -> str:
        return self._apply_decimal(content)

    def transform(
        self,
        content: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: str = "",
    ) -> str:
        """
        Overrides BasicPlugin.transform solely to apply the optional hint prefix.

        Args:
            content (str): The full input text.
            exclude_patterns (List[str], optional): Passed through to BasicPlugin for exclusion handling.
            plugin_option (str, optional): Plugin options string.

        Returns:
            str: The transformed text, with optional 'decimal ' prefix.
        """
        opts = parse_options(plugin_option)
        hint = opts.get("hint", "true").lower() == "true"

        transformed = super().transform(
            content, exclude_patterns=exclude_patterns, plugin_option=plugin_option
        )

        return "decimal " + transformed if hint else transformed

    def _apply_decimal(self, content: str) -> str:
        """Return space-separated decimal Unicode codepoints for each character."""
        return " ".join(str(ord(c)) for c in content)
