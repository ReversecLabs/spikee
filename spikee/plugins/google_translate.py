"""
Google Translate Plugin

Requires: pip install "spikee[google-translate]"
"""

from typing import List, Tuple
import asyncio

from spikee.templates.plugin import Plugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.modules import parse_options

DEFAULT_SOURCE_LANGUAGE = "en"  # Default source language for translation
DEFAULT_TARGET_LANGUAGE = "zh-cn"  # Default target language for translation


class GoogleTranslator(Plugin):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return (
            [],
            'Transforms text using Google Translate. (Requires: `pip install "spikee[google-translate]"`)',
        )

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [
            "source-lang=<language_code>",
            "target-lang=<language_code>",
            "en",
            "es",
            "fr",
            "de",
            "zh-cn",
            "ja",
            "ru",
            "ar",
            "hi",
            "pt",
        ], False

    def transform(
        self, text: str, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> str:
        """
        Transforms the input text into another language using google translate.

        Args:
            text (str): The input text.
            exclude_patterns (List[str], optional): Patterns to exclude from translation. Defaults to None.
            plugin_option (str, optional): Plugin options as a string. Defaults to None.

        Returns:
            str: The translated text.
        """
        try:
            from googletrans import Translator
        except ImportError as e:
            raise ImportError(
                'Missing required packages for Google Translate. Please install it with: `pip install "spikee[google-translate]"`'
            ) from e

        options = parse_options(plugin_option)
        source_lang = options.get("source-lang", DEFAULT_SOURCE_LANGUAGE)
        target_lang = options.get("target-lang", DEFAULT_TARGET_LANGUAGE)

        translator = Translator()
        translated = asyncio.run(
            translator.translate(text, src=source_lang, dest=target_lang)
        )
        return translated.text
