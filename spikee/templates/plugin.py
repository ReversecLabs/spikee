from abc import ABC, abstractmethod
from typing import List, Union, overload, Optional

from spikee.templates.module import Module
from spikee.utilities.hinting import Content


class Plugin(Module, ABC):
    @abstractmethod
    @overload
    def transform(
        self, content: Content, exclude_patterns: Optional[List[str]] = None, plugin_option: str = ""
    ) -> Union[Content, List[Content]]:
        pass

    @abstractmethod
    @overload
    def transform(
        self, text: str, exclude_patterns: Optional[List[str]] = None, plugin_option: str = ""
    ) -> Union[str, List[str]]:
        pass
