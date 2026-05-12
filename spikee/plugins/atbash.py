"""
Atbash Plugin

This basic plugin transforms the input text with the Atbash transformation, which swaps letters
with their counterpart on the other side of the alphabet. A becomes Z, B becomes Y, etc.,
until Y becomes B and Z becomes A.
Case of the original letter is preserved.


Usage:
    spikee generate --plugins atbash
    spikee generate --plugins atbash --plugin-options "atbash:hint=false"

Reference:
    https://mindgard.ai/blog/bypassing-azure-ai-content-safety-guardrails

Parameters:
    text (str): The input text to be transformed.
    exclude_patterns (List[str], optional): Supplied by the framework. Substrings matching
        these regex patterns are preserved as-is.

Returns:
    str: The transformed text.
"""

from typing import List, Optional

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.modules import parse_options


class AtbashPlugin(BasicPlugin):
    _ALPHA_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _ALPHA_LOWER = "abcdefghijklmnopqrstuvwxyz"
    ATBASH_TABLE = str.maketrans(
        _ALPHA_UPPER + _ALPHA_LOWER,
        _ALPHA_UPPER[::-1] + _ALPHA_LOWER[::-1],
    )

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.ENCODING], "Applies Atbash cipher with optional hint."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [
            "hint=true",
            "hint=true/false (hint=true prepends the literal string 'atbash ' to the full output)",
        ], False

    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        """
        Transforms the input text using the Atbash cipher.

        Args:
            text (str): The input text (or chunk thereof).
            plugin_option (str, optional): Plugin options string.

        Returns:
            str: The Atbash-transformed text.
        """
        return self._apply_atbash(text)

    def transform(
        self,
        text: str,
        exclude_patterns: Optional[List[str]] = None,
        plugin_option: str = "",
    ) -> str:
        """
        Overrides BasicPlugin.transform solely to apply the optional hint prefix.

        Args:
            text (str): The full input text.
            exclude_patterns (List[str], optional): Passed through to BasicPlugin for exclusion handling.
            plugin_option (str, optional): Plugin options string.

        Returns:
            str: The transformed text, with optional 'atbash ' prefix.
        """
        opts = parse_options(plugin_option)
        hint = opts.get("hint", "true").lower() == "true"

        transformed = super().transform(
            text, exclude_patterns=exclude_patterns, plugin_option=plugin_option
        )

        return "atbash " + transformed if hint else transformed

    def _apply_atbash(self, text: str) -> str:
        return text.translate(self.ATBASH_TABLE)
