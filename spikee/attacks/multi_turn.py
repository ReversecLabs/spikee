import uuid
from typing import Callable
import traceback


from spikee.templates.attack import Attack
from spikee.tester import AdvancedTargetWrapper
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, AttackResponseHint, process_target_content
from spikee.utilities.enums import Turn, ModuleTag


class MultiTurnAttack(Attack):
    def __init__(self):
        """Define multi-turn capabilities for attack."""
        super().__init__(turn_type=Turn.MULTI)

    def get_description(self) -> ModuleDescriptionHint:
        return [
            ModuleTag.MULTI
        ], "Performs a manual multi-turn attack by sending a defined series of messages"

    def get_available_option_values(self) -> ModuleOptionsHint:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return [], False

    def attack(
        self,
        entry: dict,
        target_module: AdvancedTargetWrapper,
        call_judge: Callable[[dict, str], bool],
        max_iterations: int,
        attempts_bar=None,
        bar_lock=None,
        attack_options: str = "",
    ) -> AttackResponseHint:
        original_text = entry.get("content", entry.get("text", ""))
        if entry.get("content_type", "text") != "text":
            raise ValueError("MultiTurn Attack only supports text content type.")

        if not isinstance(original_text, list) or not all(isinstance(item, str) for item in original_text):
            raise ValueError("For MultiTurn Attack, 'text' field must be a list of strings representing the conversation turns.")

        # Attempt multi-turn attack
        try:
            system_message = entry.get("system_message", None)
            session_id = str(uuid.uuid4())
            conversation = []

            count = 0
            for message in original_text:
                # Send message and handle history
                conversation.append({"role": "user", "content": message})
                response = process_target_content(target_module.process_input(
                    input_text=message,
                    system_message=system_message,
                    spikee_session_id=session_id,
                ))

                conversation.append({"role": "assistant", "content": response})

                # Implement Max Iteration
                count += 1
                if count > max_iterations:
                    break

                # Update attempts bar
                if attempts_bar:
                    with bar_lock:
                        attempts_bar.update(1)

            success = call_judge(entry, response)

            # Finalize attempts bar
            if attempts_bar:
                with bar_lock:
                    remaining = max_iterations - count
                    attempts_bar.total = attempts_bar.total - remaining

            return len(original_text), success, {"conversation": conversation}, response
        except Exception as e:
            traceback.print_exc()
            return 0, False, f"Error during multi-turn attack: {str(e)}", ""
