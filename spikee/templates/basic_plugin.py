from abc import ABC, abstractmethod
import re
from typing import List, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.content import Text


class BasicPlugin(Plugin, ABC):
    @abstractmethod
    def plugin_transform(self, text: str, plugin_option: str = "") -> str:
        """Transform the input text according to the plugin's functionality."""
        pass

    def transform(
        self, content: Text, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> Union[Text, List[Text]]:

        text = content.content

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

        return Text("".join(result_chunks))
