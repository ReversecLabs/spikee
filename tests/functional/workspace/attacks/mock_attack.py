from typing import Any, Dict, List, Optional, Tuple

from spikee.templates.attack import Attack


class MockAttack(Attack):
    def get_available_option_values(self) -> List[str]:
        return []

    def attack(
        self,
        entry: Dict[str, Any],
        target_module: Any,
        call_judge: callable,
        max_iterations: int,
        attempts_bar=None,
        bar_lock=None,
        attack_option: Optional[str] = None,
    ) -> Tuple[int, bool, str, str]:
        last_input = ""
        last_response = ""
        success = False

        for iteration in range(max_iterations):
            candidate_input = f"[mock-attack-{iteration}] {entry['text']}"
            last_input = candidate_input

            result = target_module.process_input(
                candidate_input,
                entry.get("system_message"),
            )
            response = result[0] if isinstance(result, tuple) else result
            last_response = response
            success = call_judge(entry, response)

            if attempts_bar:
                if bar_lock:
                    with bar_lock:
                        attempts_bar.update(1)
                else:
                    attempts_bar.update(1)

            if success:
                return iteration + 1, True, last_input, last_response

        return max_iterations, False, last_input, last_response
