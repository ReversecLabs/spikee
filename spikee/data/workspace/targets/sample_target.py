"""
sample_target.py

This is an example HTTP(s) request target for spikee that calls an external API, based on target options.
Demonstrates the current class-based target interface (preferred going forward as the legacy function style will be deprecated in 1.0).
The example URLs are fictional and meant to illustrate how to structure such a target.

Usage:
    1. Place this file in your local `targets/` folder.
    2. Run the spikee test command, pointing to this target, e.g.:
         spikee test --dataset datasets/example.jsonl --target sample__target

Return values:
    - For typical LLM completion, return a string that represents the model's response.
    - For guardrail usage, return True or False:
        * True indicates the attack was successful (guardrail bypassed).
        * False indicates the guardrail blocked the attack.
"""

from spikee.templates.target import Target
from spikee.tester import GuardrailTrigger

from dotenv import load_dotenv
import json
import requests
from typing import Optional, Dict, List


class SampleRequestTarget(Target):

    _OPTIONS_MAP: Dict[str, str] = {
        "example1": "https://reversec.com/api/example1",
        "example2": "https://reversec.com/api/example2",
    }
    _DEFAULT_KEY = "example1"

    def get_available_option_values(self) -> List[str]:
        """Returns a list of supported option values, first is default. None if no options."""
        options = [self._DEFAULT_KEY]
        options.extend([key for key in self._OPTIONS_MAP if key != self._DEFAULT_KEY])
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:

        # Option Validation
        key = target_options if target_options is not None else self._DEFAULT_KEY

        if key not in self._OPTIONS_MAP:
            valid = ", ".join(self.get_available_option_values())
            raise ValueError(f"Unknown option value '{key}'. Valid options: {valid}")

        option = self._OPTIONS_MAP[key]

        # Example Request Logic
        url = option

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message if system_message else "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )

            response.raise_for_status()
            result = response.json()
            return result.get("answer", "No answer available.")

        except requests.exceptions.RequestException as e:
            if response.status_code == 400:  # Guardrail Triggered
                raise GuardrailTrigger(f"Guardrail was triggered by the target: {e}")

            else:
                print(f"Error during HTTP request: {e}")
                raise


if __name__ == "__main__":
    load_dotenv()
    try:
        target = SampleRequestTarget()
        response = target.process_input("Hello!")
        print(response)
    except Exception as err:
        print("Error:", err)
