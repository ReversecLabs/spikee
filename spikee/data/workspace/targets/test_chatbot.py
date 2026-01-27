"""
test_chatbot.py

This is an example Multi-Turn target for the Spikee Test Chatbot (https://github.com/ReversecLabs/spikee-test-chatbot).
This uses HTTP(s) requests to communicate with the Chatbot API, and manages multi-turn conversations
using Spikee's MultiTarget template.

Usage:
    1. Place this file in your local `targets/` folder.
    2. Run the spikee test command, pointing to this target, e.g.:
        spikee test --dataset datasets/example.jsonl --target test_chatbot --attack <multi-turn capable attack>

Return values:
    - For typical LLM completion, return a string that represents the model's response.

References:
    - See `simple_test_chatbot.py` for a simplified version of this target using `SimpleMultiTarget`.
    - This file demonstrates manual session and history management using the raw `MultiTarget` interface.
"""

from spikee.templates.multi_target import (
    MultiTarget,
)  # MultiTarget, includes a series of functiona to manage conversation history and multiprocessing safe storage.
from spikee.utilities.enums import Turn
import traceback

import json
import uuid
import requests
from typing import Optional, List

from dotenv import load_dotenv


class TestChatbotTarget(MultiTarget):
    def __init__(self):
        super().__init__(
            turn_types=[
                Turn.SINGLE,
                Turn.MULTI,
            ],  # Specify that this target supports both single-turn and multi-turn interactions
            backtrack=True,  # Does the target + target application support backtracking
        )

    def get_available_option_values(self) -> List[str]:
        return ["http://localhost:8000"]

    def send_message(
        self,
        url: str,
        session_id: str,
        message: str,
        model: str = "together-qwen-next-80b",
    ) -> str:
        """Used to send messages to the Chatbot target, and update conversation history.

        Args:
            url (str): Chatbot API Base URL
            session_id (str): Session ID for conversation tracking
            message (str): Message to send
            model (str): Model to use (default: gpt-4o)

        Returns:
            str: Response from the Chatbot
        """

        # --------------------------------
        # Send request to the Chatbot API via POST /api/chat
        payload = {"message": message, "session_id": session_id, "model": model}

        # Ensure URL ends with / if not present, but avoid double slashes if user provided it
        # However, simplistic joining:
        api_url = f"{url.rstrip('/')}/api/chat"

        try:
            response = requests.post(
                url=api_url,
                headers={
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=30,
            )

            response.raise_for_status()

            try:
                resp_json = response.json()
                # Try common keys
                result = (
                    resp_json.get("response")
                    or resp_json.get("message")
                    or resp_json.get("content")
                )
                if result is None:
                    # Fallback if no obvious key
                    result = str(resp_json)
            except json.JSONDecodeError:
                result = response.text

        except requests.exceptions.RequestException as e:
            raise e
        # --------------------------------

        return result

    def get_new_conversation_id(self, url: str, spikee_session_id: str) -> str:
        """Generates a new conversation ID, ensuring it does not already exist."""
        session_id = str(uuid.uuid4())

        # Check collision
        while self.validate_conversation_id(url=url, conversation_id=session_id):
            session_id = str(uuid.uuid4())

        session_state = self._get_target_data(spikee_session_id)
        if session_state is None:
            session_state = {"target_session_id": session_id, "history": []}
        else:
            session_state["target_session_id"] = session_id

        self._update_target_data(spikee_session_id, session_state)
        return session_id

    def validate_conversation_id(self, url: str, conversation_id: str) -> bool:
        """Validates if a conversation ID exists by querying the session API."""
        api_url = f"{url.rstrip('/')}/api/sessions/{conversation_id}"

        try:
            response = requests.get(
                url=api_url, headers={"Content-Type": "application/json"}, timeout=10
            )

            # If 200 OK, it exists. If 404, it doesn't.
            if response.status_code == 200:
                return True
            elif response.status_code == 404:
                return False
            else:
                # Some other error, assume it doesn't exist or we can't use it?
                return False

        except requests.exceptions.RequestException:
            # If we can't connect, we can't validate. Assume False (not found/usable)?
            return False

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        spikee_session_id: Optional[str] = None,
        backtrack: Optional[bool] = False,
    ) -> str:
        # ---- Determine the URL based on target options ----
        url = "http://localhost:8000"

        # ---- Retrieve Session State ----
        session_state = None
        if spikee_session_id is not None:
            session_state = self._get_target_data(spikee_session_id)
            if session_state is None:
                # Initialize new session state
                session_state = {"target_session_id": None, "history": []}
                self._update_target_data(spikee_session_id, session_state)

        # ---- Validate new conversation ID for multi-turn sessions ----
        target_session_id = None
        if spikee_session_id is None:
            # print(f"[DEBUG] spikee_session_id is None. Creating ephemeral session without correlation.")
            target_session_id = str(uuid.uuid4())
            # Ensure unique
            while self.validate_conversation_id(
                url=url, conversation_id=target_session_id
            ):
                target_session_id = str(uuid.uuid4())
        else:
            target_session_id = session_state.get("target_session_id")
            if target_session_id is None:  # New conversation
                target_session_id = self.get_new_conversation_id(
                    url=url, spikee_session_id=spikee_session_id
                )
                # Note: get_new_conversation_id now updates session_state inside via _update_target_data,
                # but we should refresh our local copy if we want to be safe, or just trust ret value.
                # Actually, my previous edit to get_new_conversation_id updates the DB.
                # Let's just use the returned value.

        # ---- Backtracking ----
        if backtrack and spikee_session_id is not None:
            history = session_state.get("history", [])
            if history is not None and len(history) >= 2:
                # Remove last turn (user + assistant)
                history = history[:-2]

                # API doesn't support "reset to state", so we must create NEW session and replay
                # Note: This is expensive if history is long, but necessary if API is stateless/append-only
                new_target_session_id = self.get_new_conversation_id(
                    url=url, spikee_session_id=spikee_session_id
                )

                for entry in history:
                    if entry["role"] == "user":
                        self.send_message(
                            url=url,
                            session_id=new_target_session_id,
                            message=entry["content"],
                        )

                target_session_id = new_target_session_id

                # Update state
                session_state["target_session_id"] = target_session_id
                session_state["history"] = history
                self._update_target_data(spikee_session_id, session_state)

        # ---- Send the new message ----
        response = self.send_message(
            url=url,
            session_id=target_session_id,
            message=input_text,
        )

        # ---- Update History ----
        if spikee_session_id is not None:
            # Refresh state in case it changed (unlikely here but good practice)
            # session_state = self._get_target_data(spikee_session_id)
            history = session_state.get("history", [])

            history.append({"role": "user", "content": input_text})
            history.append({"role": "assistant", "content": response})

            session_state["history"] = history
            self._update_target_data(spikee_session_id, session_state)

        return response


if __name__ == "__main__":
    load_dotenv()
    try:
        target = TestChatbotTarget()
        # Initialize internal storage for standalone testing
        target.add_managed_dicts({}, {})

        # Define a mock session ID
        test_session_id = "manual-test-session"

        print(f"Sending message to target with session_id: {test_session_id}")
        response = target.process_input(
            "Hello, my name is Spikee", spikee_session_id=test_session_id
        )
        print("Response:", response)
        response = target.process_input(
            "What was my name?", spikee_session_id=test_session_id
        )
        print("Response:", response)

    except Exception:
        traceback.print_exc()
