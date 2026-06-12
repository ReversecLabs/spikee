"""
Homoglyph Plugin

This plugin substitutes Latin (ASCII) characters with visually-identical Unicode
"confusables" drawn from the Cyrillic and Greek scripts. To a human reader the text
looks unchanged ("paypal" still reads as "paypal"), but every substituted character has
a different code point, which defeats naive keyword/regex filters and byte-level
denylists that assume ASCII.

This is distinct from the existing plugins:
    - `ascii_smuggler` hides text in invisible Unicode tag characters (U+E00xx).
    - `1337` replaces letters with numerals/symbols, which is visually obvious.
Homoglyph substitution keeps the text human-legible while changing the underlying bytes.

By default every mappable character is substituted (``ratio=1.0``), which is fully
deterministic. A lower ``ratio`` substitutes only that fraction of mappable characters,
producing mixed-script text; pass ``seed`` for reproducible partial substitution.

Usage:
    spikee generate --plugins homoglyph
    spikee generate --plugins homoglyph --plugin-options "homoglyph:ratio=0.5"
    spikee generate --plugins homoglyph --plugin-options "homoglyph:ratio=0.5,seed=42"

Reference:
    https://www.unicode.org/Public/security/latest/confusables.txt
    https://en.wikipedia.org/wiki/IDN_homograph_attack

Parameters:
    text (str): The input text to be transformed.
    ratio (float): Fraction of mappable characters to substitute, in [0.0, 1.0]
        (default: 1.0). At 1.0 substitution is deterministic; below 1.0 a random
        subset is chosen.
    seed (int, optional): Seed for the random subset when ``ratio`` < 1.0, for
        reproducible output.
    exclude_patterns (List[str], optional): Supplied by the framework. Substrings
        matching these regex patterns are preserved as-is.

Returns:
    str: The transformed text with Latin characters replaced by confusables.
"""

import random
from typing import Dict

from spikee.templates.basic_plugin import BasicPlugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint
from spikee.utilities.modules import parse_options


class HomoglyphPlugin(BasicPlugin):
    DEFAULT_RATIO = 1.0

    # Latin -> visually-identical Cyrillic/Greek confusable.
    # Only high-confidence, single-character mappings are included; unmapped
    # characters are passed through unchanged.
    CONFUSABLES: Dict[str, str] = {
        # lowercase
        "a": "а",  # CYRILLIC SMALL LETTER A
        "c": "с",  # CYRILLIC SMALL LETTER ES
        "d": "ԁ",  # CYRILLIC SMALL LETTER KOMI DE
        "e": "е",  # CYRILLIC SMALL LETTER IE
        "g": "ɡ",  # LATIN SMALL LETTER SCRIPT G
        "h": "һ",  # CYRILLIC SMALL LETTER SHHA
        "i": "і",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
        "j": "ј",  # CYRILLIC SMALL LETTER JE
        "o": "о",  # CYRILLIC SMALL LETTER O
        "p": "р",  # CYRILLIC SMALL LETTER ER
        "s": "ѕ",  # CYRILLIC SMALL LETTER DZE
        "v": "ѵ",  # CYRILLIC SMALL LETTER IZHITSA
        "x": "х",  # CYRILLIC SMALL LETTER HA
        "y": "у",  # CYRILLIC SMALL LETTER U
        # uppercase
        "A": "А",  # CYRILLIC CAPITAL LETTER A
        "B": "В",  # CYRILLIC CAPITAL LETTER VE
        "C": "С",  # CYRILLIC CAPITAL LETTER ES
        "E": "Е",  # CYRILLIC CAPITAL LETTER IE
        "H": "Н",  # CYRILLIC CAPITAL LETTER EN
        "I": "Ι",  # GREEK CAPITAL LETTER IOTA
        "J": "Ј",  # CYRILLIC CAPITAL LETTER JE
        "K": "К",  # CYRILLIC CAPITAL LETTER KA
        "M": "М",  # CYRILLIC CAPITAL LETTER EM
        "N": "Ν",  # GREEK CAPITAL LETTER NU
        "O": "О",  # CYRILLIC CAPITAL LETTER O
        "P": "Р",  # CYRILLIC CAPITAL LETTER ER
        "S": "Ѕ",  # CYRILLIC CAPITAL LETTER DZE
        "T": "Т",  # CYRILLIC CAPITAL LETTER TE
        "X": "Х",  # CYRILLIC CAPITAL LETTER HA
        "Y": "Ү",  # CYRILLIC CAPITAL LETTER STRAIGHT U
    }

    def get_description(self) -> ModuleDescriptionHint:
        return (
            [ModuleTag.ENCODING, ModuleTag.OBFUSCATION],
            "Substitutes Latin characters with visually-identical Unicode confusables.",
        )

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [
            "ratio=1.0",
            "ratio=N (0.0-1.0, fraction of mappable characters to substitute)",
            "seed=N (optional integer seed for reproducible partial substitution)",
        ], False

    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        """
        Substitutes Latin characters with Unicode confusables.

        Args:
            text (str): The input text (or chunk thereof).
            plugin_option (str, optional): Plugin options string, e.g.
                "ratio=0.5,seed=42".

        Returns:
            str: The transformed text.
        """
        ratio, seed = self._parse_options(plugin_option)

        if ratio >= 1.0:
            return "".join(self.CONFUSABLES.get(c, c) for c in text)

        if ratio <= 0.0:
            return text

        rng = random.Random(seed)
        return "".join(
            self.CONFUSABLES[c] if (c in self.CONFUSABLES and rng.random() < ratio) else c
            for c in text
        )

    def _parse_options(self, plugin_option: str):
        """Parse the ``ratio`` and ``seed`` options, falling back to defaults."""
        opts = parse_options(plugin_option)

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

        return ratio, seed
