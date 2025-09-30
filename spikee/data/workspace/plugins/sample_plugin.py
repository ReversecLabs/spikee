"""
sample_plugin.py

This file shows a simple example plugin for Spikee.

Plugins must define a `transform(text: str, exclude_patterns: List[str] = None) -> Union[str, List[str]]` function.
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
import re


def transform(text: str, exclude_patterns: List[str] = None) -> Union[str, List[str]]:
    """
    Transforms the input text to uppercase, preserving any substrings that match the given exclusion patterns.

    Args:
        text (str): The input prompt to transform.
        exclude_patterns (List[str], optional): Regex patterns for substrings to preserve.

    Returns:
        str: The transformed text in uppercase.
    """
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
        return "".join(result_chunks)
    else:
        return text.upper()
