"""
OctalEncoder Plugin

Encodes text as space-separated ASCII/Unicode octal codepoint values.
"Spikee plugin" > "octal 123 160 151 153 145 145 40 160 154 165 147 151 156"

Usage:
    spikee generate --plugins octal
    spikee generate --plugins octal --plugin-options "octal:hint=false"

Parameters:
    text (str): Input string to transform.
    exclude_patterns (List[str], optional): Supplied by the framework. Substrings matching
        these regex patterns are preserved as-is.

Returns:
    str: Space-separated octal codepoint values, prefixed with 'octal ' by default.
"""

from typing import List, Optional

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.modules import parse_options


class OctalEncoderPlugin(BasicPlugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING], "Encodes text as octal codepoint representation."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [
            "hint=true",
            "hint=true/false (hint=true prepends the literal string 'octal ' to the full output)",
        ], False

    def plugin_transform(self, content: str, plugin_option: str = "") -> str:
        return self._apply_octal(content)

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
            str: The transformed text, with optional 'octal ' prefix.
        """
        opts = parse_options(plugin_option)
        hint = opts.get("hint", "true").lower() == "true"

        transformed = super().transform(
            content, exclude_patterns=exclude_patterns, plugin_option=plugin_option
        )

        return "octal " + transformed if hint else transformed

    def _apply_octal(self, content: str) -> str:
        """Return space-separated octal Unicode codepoints for each character."""
        return " ".join(f"{ord(c):o}" for c in content)
