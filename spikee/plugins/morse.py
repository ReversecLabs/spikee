"""
Morse Code Plugin

This plugin transforms the input text into Morse code. Letters and numbers are
converted using the international Morse code standard, while unsupported characters
are left unchanged. Any substring that exactly matches one of the user-supplied
exclusion regex patterns (passed via the exclude_patterns parameter) is left unchanged.

Usage:
    spikee generate --plugins morse

Parameters:
    text (str): The input text to be transformed.
    exclude_patterns (List[str], optional): A list of regex patterns. Any substring
        that exactly matches one of these patterns will be preserved as-is.

Returns:
    str: The transformed text in Morse code.
"""

from typing import List
import re

MORSE_CODE_DICT = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
    " ": "/",
}


def transform(text: str, exclude_patterns: List[str] = None) -> str:
    """
    Transforms the input text into Morse code while preserving any substring that
    exactly matches one of the exclusion regex patterns.

    If an exclusion list is provided, the plugin creates a compound regex by joining
    the patterns. It then splits the text using re.split() so that any substring that
    exactly matches one of the patterns is isolated and left unmodified. All other parts
    are transformed into Morse code.

    Args:
        text (str): The input text.
        exclude_patterns (List[str], optional): A list of regex patterns to exclude from transformation.

    Returns:
        str: The transformed text in Morse code.
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
            # Leave excluded substrings untouched.
            result_chunks.append(chunk)
        else:
            transformed = " ".join(MORSE_CODE_DICT.get(c.upper(), c) for c in chunk)
            result_chunks.append(transformed)

    return " ".join(result_chunks)
