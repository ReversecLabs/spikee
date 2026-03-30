import re
from typing import List, Tuple

from spikee.templates.plugin import Plugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.modules import parse_options


class FlipPlugin(Plugin):
    """
    Flip Attack Plugin

    Supports three modes:
            - FWO: Flip Word Order
            - FCW: Flip Chars in Word
            - FCS: Flip Chars in Sentence
    """

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Transforms text using the flip attack."

    def get_available_option_values(self) -> List[str]:
        return ["mode=FWO", "FCW", "FCS", "resp_exc=false"]

    def transform(
        self, text: str, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> str:
        opts = parse_options(plugin_option)

        mode = opts.get("mode", "FWO").upper()
        respect_exclude = opts.get("resp_exc", "false").lower() == "true"

        if respect_exclude:
            combined_pattern = "(" + "|".join(exclude_patterns) + ")"
            chunks = re.split(combined_pattern, text)
            transformed_chunks = []
            for i, chunk in enumerate(chunks):
                if i % 2 == 0:
                    transformed_chunks.append(self._apply_flip(chunk, mode)[0])
                else:
                    transformed_chunks.append(chunk)

            return "".join(transformed_chunks)

        else:
            return self._apply_flip(text, mode)

    def _apply_flip(self, text: str, mode: str) -> str:
        if mode == "FWO":
            return self._flip_word_order(text)
        elif mode == "FCW":
            return self._flip_char_in_word(text)
        elif mode == "FCS":
            return self._flip_char_in_sentence(text)

        raise ValueError(
            f"Invalid mode: {mode}. Supported modes are FWO, FCW, FCS, ALL."
        )

    def _flip_word_order(self, input_str: str) -> str:
        return " ".join(input_str.split()[::-1])

    def _flip_char_in_word(self, input_str: str) -> str:
        return " ".join([word[::-1] for word in input_str.split()])

    def _flip_char_in_sentence(self, input_str: str) -> str:
        return input_str[::-1]
