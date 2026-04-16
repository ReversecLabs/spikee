"""
sample_plugin.py

This file shows a simple example plugin for Spikee.

Plugins must define a `transform(text: str, exclude_patterns: List[str] = []) -> Union[str, List[str]]` function.
Spikee will call this function for each input prompt, passing in the original text and, optionally, a list of regex
patterns that should be excluded from transformation.

Key Concepts:
- Exclusion Support: If `exclude_patterns` is provided, any substring that exactly matches one of the regex patterns
  should be preserved as-is.
- Multiple Variants: Plugins can return a single transformed string or a list of strings to test multiple variants.

Usage within Spikee:
    spikee generate --plugins sample_plugin

This sample plugin simply transforms the input text to uppercase.
"""

from typing import List, Union
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.content import Text
import re

from spikee.templates.plugin import Plugin


class SamplePlugin(Plugin):
    def get_description(self) -> ModuleDescriptionHint:
        return [], "A sample plugin that transforms text to uppercase, preserving excluded patterns."

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [], False

    def transform(
        self,
        content: Text,  # specify specific content types using Text, Audio, Image subclasses of Content
        exclude_patterns: List[str] = []
    ) -> Union[Text, List[Text]]:
        """
        Transforms the input text to uppercase, preserving any substrings that match the given exclusion patterns.

        Args:
            text (str): The input prompt to transform.
            exclude_patterns (List[str], optional): Regex patterns for substrings to preserve.

        Returns:
            str: The transformed text in uppercase.
        """
        text = content.content

        if exclude_patterns:
            compound = "(" + "|".join(exclude_patterns) + ")"
            compound_re = re.compile(compound)
            chunks = re.split(compound, text)

            result_chunks = []
            for chunk in chunks:
                if compound_re.fullmatch(chunk):
                    result_chunks.append(chunk)
                else:
                    result_chunks.append(chunk.upper())
            return Text("".join(result_chunks))
        else:
            return Text(text.upper())
