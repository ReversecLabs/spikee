"""
spikee/targets/togetherai.py

A unified TogetherAI target that invokes models based on a simple string key.

Usage:
    target_options: str, one of the keys returned by get_available_option_values().
    If None, the default key is used.

Exposed:
    get_available_option_values() -> list of supported keys (default marked)
    process_input(input_text, system_message=None, target_options=None) -> response
"""
from spikee.templates.target import Target
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, TOGETHER_AI_MODEL_MAP

from typing import List, Optional
from dotenv import load_dotenv


class TogetherAITarget(Target):
    # Default shorthand key
    DEFAULT_KEY = "llama31-8b"


    def get_available_option_values(self) -> List[str]:
        """Return supported keys; first option is default."""
        options = [self.DEFAULT_KEY]  # Default first
        options.extend([key for key in TOGETHER_AI_MODEL_MAP if key != self.DEFAULT_KEY])
        return options


    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to TogetherAI based on the provided key.

        Args:
            input_text: the user's prompt or document.
            system_message: optional system context.
            target_options: shorthand key for model; uses default if None.

        Returns:
            The model's text response.
        Raises:
            ValueError if target_options is provided but invalid.
        """

        model_id = target_options if target_options is not None else self.DEFAULT_KEY
        
        if model_id.startswith("together-"):
            model_id = model_id.replace("together-", "")
            
        # Initialize the OpenAI client pointing to Together API
        llm = get_llm(f"together-{model_id}", max_tokens=None, temperature=0)

        # Build the message list
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke the model
        try:    
            return llm.invoke(messages, content_only=True)
        except Exception as e:
            print(f"Error during TogetherAI completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = TogetherAITarget()
    print("Supported keys:", target.get_available_option_values())
    
    try:
        print(target.process_input("Hello!", target_options="llama31-8b"))
    except Exception as e:
        print("Error:", e)
