from spikee.utilities.enums import ModuleTag

from abc import ABC
from typing import List, Tuple


class Module(ABC):
    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "No Module description available."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]
        e.g., (["mode=aggressive"], True) - Had an option mode, and requires llm 'model' to operate."""
        return [], False
