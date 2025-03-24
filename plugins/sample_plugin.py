"""
sample_plugin.py

This file shows a simple example plugin for spikee.
Plugins must define a `transform(text: str) -> str` function,
which takes the original text and returns a modified version.

Usage within spikee:
    spikee generate --plugins sample_plugin

Then spikee will call `transform(text)` for each prompt,
replacing it with the plugin-transformed text.
"""

def transform(text: str) -> str:
    """
    A minimal plugin that just converts the text to uppercase.
    Customize this function to implement any transformation or obfuscation 
    you’d like to test against your target LLM or guardrail.

    """
    return text.upper()
