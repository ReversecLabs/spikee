from abc import ABC, abstractmethod
from typing import List, Union, overload

from spikee.templates.module import Module
from spikee.utilities.hinting import ContentHint


class Plugin(Module, ABC):
    @abstractmethod
    @overload
    def transform(
        self, content: ContentHint, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> Union[ContentHint, List[ContentHint]]:
        pass

    @abstractmethod
    @overload
    def transform(
        self, text: str, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> Union[str, List[str]]:
        pass
