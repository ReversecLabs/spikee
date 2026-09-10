import traceback
import uuid
from collections.abc import Callable
from copy import deepcopy

from spikee.templates.attack import Attack
from spikee.tester import AdvancedTargetWrapper
from spikee.utilities.enums import ModuleTag, Turn
from spikee.utilities.hinting import (
    AttackAttempt,
    AttackResponseHint,
    ModuleDescriptionHint,
    ModuleOptionsHint,
    process_target_content,
)


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
        return_all_attempts: bool = False,
    ) -> AttackResponseHint:
        original_text = entry.get("content", entry.get("text", ""))
        if entry.get("content_type", "text") != "text":
            raise ValueError("MultiTurn Attack only supports text content type.")

        if not isinstance(original_text, list) or not all(
            isinstance(item, str) for item in original_text
        ):
            raise ValueError(
                "For MultiTurn Attack, 'text' field must be a list of strings representing the conversation turns."
            )

        history = []
        current = None
        # Attempt multi-turn attack
        try:
            system_message = entry.get("system_message", None)
            session_id = str(uuid.uuid4())
            conversation = []

            count = 0
            response = ""
            for message in original_text[:max_iterations]:
                # Send message and handle history
                conversation.append({"role": "user", "content": message})
                if return_all_attempts:
                    current = AttackAttempt(
                        {"input": message, "conversation": deepcopy(conversation)},
                        "",
                        None,
                    )
                    history.append(current)
                response = process_target_content(
                    target_module.process_input(
                        input_text=message,
                        system_message=system_message,
                        spikee_session_id=session_id,
                    )
                )

                conversation.append({"role": "assistant", "content": response})

                if return_all_attempts:
                    current.response = response
                    current.input["conversation"] = deepcopy(conversation)

                # Implement Max Iteration
                count += 1

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
                    attempts_bar.refresh()

            if return_all_attempts:
                if history:
                    history[-1].success = success
                return history or [AttackAttempt("", response, success, attempts=0)]
            return count, success, {"conversation": conversation}, response
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            if return_all_attempts:
                if current is not None:
                    current.error = str(e)
                else:
                    history.append(
                        AttackAttempt("", "", False, attempts=0, error=str(e))
                    )
                return history
            return 0, False, f"Error during multi-turn attack: {e!s}", ""
