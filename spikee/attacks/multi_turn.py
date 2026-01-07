import uuid
from typing import Tuple
import traceback


from spikee.templates.attack import Attack
from spikee.utilities.enums import Turn


class MultiTurnAttack(Attack):
    def __init__(self):
        """Define multi-turn capabilities for attack."""
        super().__init__(turn_type=Turn.MULTI)

    @property
    def __name__(self):
        return "multi_turn"

    def get_available_option_values(self) -> str:
        return None

    def attack(
        self,
        entry: dict,
        target_module: object,
        call_judge: callable,
        max_iterations: int,
        attempts_bar=None,
        bar_lock=None,
        attack_option: str = None,
    ) -> Tuple[int, bool, str, str]:
        if (
            "text" not in entry
            or not isinstance(entry["text"], list)
            or not all(isinstance(item, str) for item in entry["text"])
        ):
            raise ValueError("Entry must contain a valid 'text' field for manual multi-turn attack.")

        try:
            system_message = entry.get("system_message", None)
            session_id = str(uuid.uuid4())
            conversation = []

            for message in entry["text"]:
                conversation.append({"role": "user", "content": message})
                response = target_module.process_input(
                    input_text=message,
                    system_message=system_message,
                    spikee_session_id=session_id
                )
                conversation.append({"role": "assistant", "content": response})

            success = call_judge(entry, response)

            return len(entry["text"]), success, {"conversation": conversation}, response
        except Exception as e:
            traceback.print_exc()
            return 0, False, f"Error during multi-turn attack: {str(e)}", ""
