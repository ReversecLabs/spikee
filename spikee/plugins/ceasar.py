"""
Caesar Cipher Plugin

This plugin transforms the input text using a simple Caesar cipher encryption.
By default, it shifts letters forward by a 3 number of positions in the alphabet.

Usage:
    spikee generate --plugins caesar

Parameters:
    text (str): The input text to be transformed.
    shift (int): The number of positions to shift each letter (default is 3).

Returns:
    str: The encrypted text using the Caesar cipher.
"""

from typing import List


def caesar_cipher(text: str, shift: int = 3) -> str:
    """
    Encrypts the input text using a Caesar cipher with the given shift value.

    Args:
        text (str): The input text.
        shift (int): The number of positions to shift each letter.

    Returns:
        str: The encrypted text.
    """
    result = []
    for char in text:
        if char.isalpha():
            shift_base = ord("A") if char.isupper() else ord("a")
            result.append(chr((ord(char) - shift_base + shift) % 26 + shift_base))
        else:
            result.append(char)
    return "".join(result)


def transform(text: str, exclude_patterns: List[str] = None) -> str:
    """
    Transforms the input text using the Caesar cipher.

    Args:
        text (str): The input text.
        shift (int): The number of positions to shift each letter (default is 3).

    Returns:
        str: The encrypted text using the Caesar cipher.
    """
    return caesar_cipher(text, 3)
