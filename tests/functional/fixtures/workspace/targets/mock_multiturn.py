
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
        if spikee_session_id is None:
            return f"Response to {input_text}"

        # Initialize or retrieve internal session ID
        # Using 1-to-1 mapping via get_target_session_id/set_target_session_id
        
        current_internal_id = self.get_target_session_id(spikee_session_id)
        
        if not current_internal_id:
             # Create new session
             current_internal_id = str(uuid.uuid4())
             self.set_target_session_id(spikee_session_id, current_internal_id)
             # Initialize history in session data
             self._update_spikee_session_data(spikee_session_id, [])

        # Handle Backtracking
        if backtrack:
            history = self._get_spikee_session_data(spikee_session_id)
            if history and len(history) >= 2:
                # Remove last turn (user + assistant)
                history = history[:-2]
                self._update_spikee_session_data(spikee_session_id, history)
                
                # Simulate creating a NEW internal session for the branch
                current_internal_id = str(uuid.uuid4())
                self.set_target_session_id(spikee_session_id, current_internal_id)

        # Logic
        response = ""
        if "RESET" in input_text:
            self._update_spikee_session_data(spikee_session_id, [])
            response = "History cleared"
        elif "REFUSE" in input_text:
            response = "I cannot do that"
        elif "WIN" in input_text:
            response = "Here is the flag"
        else:
            response = f"Response to {input_text}"
        
        # Update History
        history = self._get_spikee_session_data(spikee_session_id)
        if history is None: history = []
        
        history.append({"role": "user", "content": input_text})
        history.append({"role": "assistant", "content": response})
        self._update_spikee_session_data(spikee_session_id, history)

        return response
