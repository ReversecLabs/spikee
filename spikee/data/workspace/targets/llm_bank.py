"""
llm_bank.py

This is an example Multi-Turn target for Reversec's LLMBank (https://llmbank.fs-playground.com/).
This uses HTTP(s) requests to communicate with the LLMBank API, and manages multi-turn conversations 
using Spikee's MultiTarget template. (NB, this target has been designed for the LLMBank API, so other 
conversational LLM applications will require different implementations).

Usage:
    1. Place this file in your local `targets/` folder.
    2. Run the spikee test command, pointing to this target, e.g.:
        spikee test --dataset datasets/example.jsonl --target llm_bank --attack <multi-turn capable attack>

Return values:
    - For typical LLM completion, return a string that represents the model's response.
"""

from spikee.templates.multi_target import MultiTarget  # MultiTarget, includes a series of functiona to manage conversation history and multiprocessing safe storage.
from spikee.utilities.enums import Turn

import json
import uuid
import requests
from typing import Optional, List

SESSION_COOKIE = "change_me"


class LLMBankTarget(MultiTarget):

    def __init__(self):
        super().__init__(
            turn_types=[Turn.SINGLE, Turn.MULTI],  # Specify that this target supports both single-turn and multi-turn interactions (Target Default is SINGLE only, MultiTarget default is MULTI only)
            backtrack=True                        # Does the target + target application support backtracking (e.g., editing previous messages in the conversation)
        )

    def get_available_option_values(self) -> List[str]:
        return ["cloud", "<url>"]

    def send_message(
        self,
        url: str,
        session_cookie: str,
        session_id: str,
        message: str,
    ) -> str:
        """Used to send messages to the LLMBank target, and update conversation history.

        Args:
            url (str): LLMBank API URL
            session_cookie (str): Session cookie for authentication
            spikee_session_id (str): Spikee session ID for conversation tracking
            message (str): Message to send to the LLMBank

        Returns:
            str: Response from the LLMBank
        """

        # Attempt to get conversation history
        history = self._get_spikee_session_data(session_id)
        if history is None:
            history = []

        # --------------------------------
        # Send request to the LLMBank API, this section will be context dependent on the target application implmentation.
        # LLMBank, has a simple design allowing conversations to be created and continued via the `/api/chat` endpoint.
        payload = {
            "message": message,
            "thread_id": session_id,
        }

        try:
            response = requests.post(
                url=url+"chat",
                headers={
                    "Content-Type": "application/json",
                    "Cookie": f"session={session_cookie}",
                },
                data=json.dumps(payload),
                timeout=30
            )

            response.raise_for_status()
            result = response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with LLMBank: {str(e)}", response)
            raise
        # --------------------------------

        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result})
        self._update_spikee_session_data(session_id, history)

        return result

    def get_new_conversation_id(
        self,
        url: str,
        spikee_session_id: str
    ) -> str:
        """Generates a new conversation ID for LLMBank target, ensuring it does not already exist."""
        llmbank_session_id = str(uuid.uuid4())
        while self.validate_conversation_id(url=url, session_cookie=SESSION_COOKIE, conversation_id=llmbank_session_id):
            llmbank_session_id = str(uuid.uuid4())

        self._add_spikee_id_correlation(spikee_session_id, llmbank_session_id)
        return llmbank_session_id

    def validate_conversation_id(
        self,
        url: str,
        session_cookie: str,
        conversation_id: str
    ) -> bool:
        """Validates if a conversation ID exists in the LLMBank target."""

        try:
            response = requests.get(
                url=url+"conversations",
                headers={
                    "Content-Type": "application/json",
                    "Cookie": f"session={session_cookie}",
                },
                timeout=30
            )

            response.raise_for_status()
            results = [conversation.get("thread_id") for conversation in response.json()]

            return conversation_id in results

        except requests.exceptions.RequestException as e:
            print(f"Error validating conversation ID with LLMBank: {str(e)}", response)
            raise

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        spikee_session_id: Optional[str] = None,
        backtrack: Optional[bool] = False,
    ) -> str:
        # ---- Determine the URL based on target options ----
        if target_options is None or target_options == "cloud":
            url = "https://llmbank.fs-playground.com/api/"
        else:
            url = target_options.rstrip()

        # ---- Validate new conversation ID for multi-turn sessions ----
        id_correlation = self._get_spikee_id_correlation(spikee_session_id)
        if id_correlation is None:  # New conversation, generate a new LLMBank session ID
            llmbank_session_id = self.get_new_conversation_id(url=url, spikee_session_id=spikee_session_id)
        else:  # Existing conversation, retrieve the last LLMBank session ID
            llmbank_session_id = id_correlation[-1]

        # ---- Backtracking ----
        if backtrack:
            history = self._get_spikee_session_data(llmbank_session_id)
            if history is not None:  # Since the LLMBank API does not support backtracking, we are bodging it and creating a new conversation using the previous history.
                history = history[:-2]
                llmbank_session_id = self.get_new_conversation_id(url=url, spikee_session_id=spikee_session_id)

                for entry in history:
                    if entry["role"] == "user":
                        self.send_message(
                            url=url,
                            session_cookie=SESSION_COOKIE,
                            session_id=llmbank_session_id,
                            message=entry["content"],
                        )

        # ---- Send the new message to the LLMBank application ----
        response = self.send_message(
            url=url,
            session_cookie=SESSION_COOKIE,
            session_id=llmbank_session_id,
            message=input_text,
        )

        return response
