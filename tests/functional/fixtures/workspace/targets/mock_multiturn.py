
import uuid
from typing import List, Optional
from spikee.templates.multi_target import MultiTarget
from spikee.utilities.enums import Turn


class MockMultiTurnTarget(MultiTarget):
    def __init__(self):
        super().__init__(turn_types=[Turn.MULTI], backtrack=True)
        # Initialize internal dicts for standalone functional testing
        self.add_managed_dicts({}, {})

    def get_available_option_values(self) -> List[str]:
        return ["default"]

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        spikee_session_id: Optional[str] = None,
        backtrack: Optional[bool] = False,
    ) -> str:
        # Retrieve current session state
        if spikee_session_id is not None:
            session_state = self._get_target_data(spikee_session_id)
            if session_state is None:
                # Initialize new session state
                session_state = {"internal_id": None, "history": []}
                self._update_target_data(spikee_session_id, session_state)

            if session_state['internal_id'] is None:
                session_state['internal_id'] = str(uuid.uuid4())
                self._update_target_data(spikee_session_id, session_state)
        else:
            session_state = {"internal_id": str(uuid.uuid4()), "history": []}

        # Handle Backtracking
        if backtrack:
            history = session_state["history"]
            if history and len(history) >= 2:
                # Remove last turn (user + assistant)
                session_state["history"] = history[:-2]

                # Simulate creating a NEW internal session for the branch
                session_state["internal_id"] = str(uuid.uuid4())
                self._update_target_data(spikee_session_id, session_state)

        # Logic
        response = ""
        if "RESET" in input_text:
            session_state["history"] = []
            if spikee_session_id is not None:
                self._update_target_data(spikee_session_id, session_state)
            response = "History cleared"
        elif "REFUSE" in input_text:
            response = "I cannot do that"
        elif "WIN" in input_text:
            response = "Here is the flag"
        else:
            response = f"Response to {input_text}"

        # Update History
        if spikee_session_id is not None:
            history = session_state["history"]
            history.append({"role": "user", "content": input_text})
            history.append({"role": "assistant", "content": response})
            session_state["history"] = history
            self._update_target_data(spikee_session_id, session_state)

        return response
