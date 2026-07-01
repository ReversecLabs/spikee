"""
Zero-Width Plugin

This plugin interleaves invisible zero-width Unicode characters between the
visible characters of the text. To a human reader (and in most UIs) the text
looks unchanged ("ignore previous instructions" still reads normally), but the
underlying byte sequence no longer contains any contiguous ASCII substring, which
defeats naive keyword/regex filters, substring denylists, and byte-level matching.

This is distinct from the existing plugins:
    - `ascii_smuggler` *replaces* the text with invisible Unicode tag characters
      (U+E00xx), so the visible content disappears entirely.
    - `homoglyph` *substitutes* Latin glyphs with look-alike confusables from other
      scripts, changing every visible code point.
Zero-width interspersing keeps the original visible glyphs intact and simply
injects invisible separators between them.

By default a separator is inserted in every gap (``ratio=1.0``), which is fully
deterministic. A lower ``ratio`` fills only that fraction of gaps, producing a
lighter obfuscation; pass ``seed`` for reproducible partial output. The invisible
character used is configurable via ``char``.

Usage:
    spikee generate --plugins zero_width
    spikee generate --plugins zero_width --plugin-options "zero_width:char=zwnj"
    spikee generate --plugins zero_width --plugin-options "zero_width:ratio=0.5,seed=42"

Reference:
    https://en.wikipedia.org/wiki/Zero-width_space
    https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/

Parameters:
    text (str): The input text to be transformed.
    char (str): Which zero-width character to insert. One of ``zwsp`` (U+200B,
        default), ``zwnj`` (U+200C), ``zwj`` (U+200D), ``wj`` (U+2060), or
        ``zwnbsp``/``bom`` (U+FEFF).
    ratio (float): Fraction of inter-character gaps to fill, in [0.0, 1.0]
        (default: 1.0). At 1.0 a separator is inserted in every gap; below 1.0 a
        random subset of gaps is chosen.
    seed (int, optional): Seed for the random subset when ``ratio`` < 1.0, for
        reproducible output.
    exclude_patterns (List[str], optional): Supplied by the framework. Substrings
        matching these regex patterns are preserved as-is.

Returns:
    str: The transformed text with zero-width characters interspersed.
"""

import random
from typing import Dict

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.modules import parse_options


class ZeroWidthPlugin(BasicPlugin):
    DEFAULT_RATIO = 1.0
    DEFAULT_CHAR = "zwsp"

    # Supported invisible separators. Keys are the option values users pass.
    ZERO_WIDTH_CHARS: Dict[str, str] = {
        "zwsp": "​",    # ZERO WIDTH SPACE
        "zwnj": "‌",    # ZERO WIDTH NON-JOINER
        "zwj": "‍",     # ZERO WIDTH JOINER
        "wj": "⁠",      # WORD JOINER
        "zwnbsp": "﻿",  # ZERO WIDTH NO-BREAK SPACE (BOM)
        "bom": "﻿",     # alias for zwnbsp
    }

    def get_description(self) -> ModuleDescriptionHint:
        return (
            [ModuleTag.ENCODING, ModuleTag.OBFUSCATION],
            "Intersperses invisible zero-width characters between visible characters.",
        )

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [
            "char=zwsp",
            "char=zwsp/zwnj/zwj/wj/zwnbsp (which zero-width character to insert)",
            "ratio=N (0.0-1.0, fraction of inter-character gaps to fill)",
            "seed=N (optional integer seed for reproducible partial output)",
        ], False

    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        """
        Intersperses zero-width characters between the characters of ``text``.

        Args:
            text (str): The input text (or chunk thereof).
            plugin_option (str, optional): Plugin options string, e.g.
                "char=zwnj,ratio=0.5,seed=42".

        Returns:
            str: The transformed text.
        """
        zw_char, ratio, seed = self._parse_options(plugin_option)

        if len(text) < 2 or ratio <= 0.0:
            return text

        if ratio >= 1.0:
            return zw_char.join(text)

        rng = random.Random(seed)
        result = [text[0]]
        for char in text[1:]:
            if rng.random() < ratio:
                result.append(zw_char)
            result.append(char)
        return "".join(result)

    def _parse_options(self, plugin_option: str):
        """Parse the ``char``, ``ratio`` and ``seed`` options, falling back to defaults."""
        opts = parse_options(plugin_option)

        char_key = str(opts.get("char", self.DEFAULT_CHAR)).lower()
        zw_char = self.ZERO_WIDTH_CHARS.get(char_key, self.ZERO_WIDTH_CHARS[self.DEFAULT_CHAR])

        ratio = self.DEFAULT_RATIO
        if "ratio" in opts:
            try:
                ratio = max(0.0, min(1.0, float(opts["ratio"])))
            except ValueError:
                ratio = self.DEFAULT_RATIO

        seed = None
        if "seed" in opts:
            try:
                seed = int(opts["seed"])
            except ValueError:
                seed = None

        return zw_char, ratio, seed
