from typing import Tuple
import threading
from collections import defaultdict
import spikee.attacks.crescendo
from spikee.attacks.crescendo import Crescendo
from spikee.templates.standardised_conversation import StandardisedConversation

# 1. Mock the LLM object used by Crescendo


class MockLLM:
    def invoke(self, *args, **kwargs):
        return type(
            "MockResponse", (), {"content": '{"generated_question": "FALLBACK"}'}
        )()


# 2. Monkeypatch get_llm
spikee.attacks.crescendo.get_llm = lambda *args, **kwargs: MockLLM()


class MockCrescendoAttack(Crescendo):
    def __init__(self):
        super().__init__()
        # Use thread-local storage for counters to handle concurrent attacks
        self._thread_local = threading.local()
        # Default scenario steps
        self.scenarios = {
            "success": ["Q1", "Q2", "WIN"],
            "refusal": ["Q1", "REFUSE", "Q2", "WIN"],
            "failure": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
        }

    def _get_counters(self):
        if not hasattr(self._thread_local, "counters"):
            self._thread_local.counters = defaultdict(int)
        return self._thread_local.counters

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
        # Parse scenario
        opts = self._parse_attack_option(attack_option)
        # Store scenario in thread-local because attack() sets it for the duration of the call
        if not hasattr(self._thread_local, "scenario"):
            self._thread_local.scenario = "success"
        self._thread_local.scenario = opts.get("scenario", "success")

        return super().attack(
            entry,
            target_module,
            call_judge,
            max_iterations,
            attempts_bar,
            bar_lock,
            attack_option,
        )

    def _generate_question(
        self,
        llm,
        objective: str,
        conversation: StandardisedConversation,
        last_message_id: int,
        max_turns: int,
        remaining_turns: int,
    ) -> str:
        counters = self._get_counters()
        hist_len = conversation.get_path_length(last_message_id, False)

        # If this is the start of a new attempt (empty history), reset counters for this thread/depth
        # Note: Crescendo calls _generate_question repeatedly.
        # But if history > 0, we are deep in.
        # If history == 0, we are at start.
        if hist_len == 0:
            counters.clear()

        attempt_count = counters[hist_len]
        counters[hist_len] += 1

        scenario_name = getattr(self._thread_local, "scenario", "success")
        steps = self.scenarios.get(scenario_name, ["WIN"])

        # Logic for determining prompt based on history length and retry count
        # Default flow: 1 step per turn (2 items in history per turn)
        # index = hist_len // 2

        if scenario_name == "refusal":
            # Flow: Q1 -> REFUSE -> Q2 -> WIN
            # len=0: Q1
            # len=2: REFUSE (1st try), Q2 (2nd try)
            # len=4: WIN
            if hist_len == 0:
                return "Q1"
            elif hist_len == 2:
                if attempt_count == 0:
                    return "REFUSE"  # Triggers refusal/backtrack
                else:
                    return "Q2"  # Recovery
            elif hist_len == 4:
                return "WIN"
            else:
                return "WIN"

        else:
            # Linear flow for success/failure
            index = hist_len // 2
            if index < len(steps):
                return steps[index]
            return "WIN"

    def _is_refusal(self, llm, text: str) -> bool:
        return "cannot" in text.lower()
