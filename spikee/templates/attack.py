from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from spikee.utilities.enums import Turn


class Attack(ABC):
    def __init__(self, turn_type: Turn = Turn.SINGLE):
        super().__init__()

        self.turn_type = turn_type

    @abstractmethod
    def get_available_option_values(self) -> List[str]:
        """Return supported attack options; first option is default."""
        return None

    @abstractmethod
    def attack(
        self,
        entry: Dict[str, Any],
        target_module: Any,
        call_judge: callable,
        max_iterations: int,
        attempts_bar=None,
        bar_lock=None,
    ) -> Tuple[int, bool, str, str]:
        pass
