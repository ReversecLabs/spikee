from abc import ABC
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint


class Module(ABC):
    def get_description(self) -> ModuleDescriptionHint:
        return [], "No Module description available."

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]
        e.g., (["mode=aggressive"], True) - Had an option mode, and requires llm 'model' to operate."""
        return [], False
