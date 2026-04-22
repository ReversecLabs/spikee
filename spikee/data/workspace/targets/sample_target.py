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
from spikee.utilities.modules import parse_options
from spikee.utilities.enums import ModuleTag

from dotenv import load_dotenv
import json
import requests
from typing import Optional, List, Tuple, Union, Any


class SampleRequestTarget(Target):
    _DEFAULT_URL = "https://reversec.com/api/example1"

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [], "Sample Request Target - sends HTTP request to URL"

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""
        return ["url=" + self._DEFAULT_URL], False

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = "",
    ) -> Union[str, bool, Tuple[Union[str, bool], Any]]:
        # Option Validation `--target-options 'url=https://myapi.com/endpoint'` to override default URL
        options = parse_options(target_options)
        url = options.get("url", self._DEFAULT_URL)

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                    if system_message
                    else "You are a helpful assistant.",
                },
                {"role": "user", "content": input_text},
            ]
        }

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=30
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
