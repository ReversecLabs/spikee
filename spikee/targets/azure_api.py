"""
spikee/targets/azure.py

Unified Azure Chat target that invokes Azure OpenAI deployments based on a simple string.

Note: `target_options` here is the **deployment name**, not the underlying model.

Usage:
    target_options: str, one of the deployment names returned by get_available_option_values().
    If None, DEFAULT_DEPLOYMENT is used.

Exposed:
    get_available_option_values() -> list of supported deployment names (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response content
"""

from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, AZURE_MODEL_LIST

from dotenv import load_dotenv
from typing import List, Optional


class AzureAPITarget(Target):
    def get_available_option_values(self) -> List[str]:
        """Return supported deployment names; first option is default."""
        return AZURE_MODEL_LIST

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to an Azure OpenAI deployment specified by target_options.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        # deployment name selection
        model_id = target_options

        # Initialize the Azure Chat client
        llm = get_llm(f"azure-{model_id}", max_tokens=None, temperature=0)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            return llm.invoke(messages, content_only=True)
        
        except Exception as e:
            print(f"Error during Azure completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = AzureAPITarget()
    print("Supported Azure deployments:", target.get_available_option_values())
    try:
        print(target.process_input("Hello!", target_options="gpt-4o-mini"))
    except Exception as err:
        print("Error:", err)
