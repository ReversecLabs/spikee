from abc import ABC, abstractmethod
from typing import List, Union

from spikee.templates.module import Module


class Plugin(Module, ABC):
    @abstractmethod
    def transform(
        self, text: str, exclude_patterns: List[str] = [], plugin_option: str = ""
    ) -> Union[str, List[str]]:
        pass
