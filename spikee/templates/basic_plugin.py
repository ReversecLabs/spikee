from abc import ABC, abstractmethod
import re
from typing import List, Union

from spikee.templates.plugin import Plugin


class BasicPlugin(Plugin, ABC):
    @abstractmethod
    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        """Transform the input text according to the plugin's functionality."""
        pass

    def transform(
        self, text: str, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> Union[str, List[str]]:

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
                result_chunks.append(chunk)
            else:
                transformed = self.plugin_transform(chunk, plugin_option)
                result_chunks.append(transformed)

        return "".join(result_chunks)
