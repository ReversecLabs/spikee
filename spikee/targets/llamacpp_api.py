"""
spikee/targets/llamaccp_api.py

Unified OpenAI target that can invoke any supported OpenAI model based on a simple key.

Usage:
    target_options: str key returned by get_available_option_values(); defaults to DEFAULT_KEY.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None, logprobs=False) ->
        - For models supporting logprobs: returns (content, logprobs)
        - Otherwise: returns content only
"""


from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage

from dotenv import load_dotenv
from typing import Optional, List


class LlamacppAPITarget(Target):
    DEFAULT_BASE_URL = "http://localhost:8080/"

    def get_available_option_values(self) -> List[str]:
        """Return supported keys; first option is default."""
        options = [self.DEFAULT_BASE_URL, "http://hostname:port"]  # Default first
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to an OpenAI model based on a simple key.
        """
        base_url = self.DEFAULT_BASE_URL if target_options is None else target_options

        llm = get_llm(f"llamacpp-{base_url}", max_tokens=None, temperature=0)

        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        try:
            return llm.invoke(messages, content_only=True)
        
        except Exception as e:
            print(f"Error during OpenAI completion: {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = LlamacppAPITarget()
    print(target.process_input("Hello!", target_options=""))
