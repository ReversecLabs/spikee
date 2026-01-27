from abc import ABC, abstractmethod
from typing import List, Union, Tuple


class Plugin(ABC):
    @abstractmethod
    def get_description(self) -> Tuple[str, str]:
        """Return type and a short description of the plugin."""
        pass

    @abstractmethod
    def get_available_option_values(self) -> List[str]:
        """Return supported plugin options; first option is default."""
        return None

    @abstractmethod
    def transform(
        self, text: str, exclude_patterns: List[str] = None
    ) -> Union[str, List[str]]:
        pass
