"""

spikee/targets/openai.py

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
from spikee.utilities.llm import get_llm, SystemMessage, HumanMessage, OPENAI_MODEL_LIST

from dotenv import load_dotenv
from typing import Optional, List, Tuple, Union


class OpenAITarget(Target):
    # default key
    DEFAULT_KEY = "gpt-4o"

    # which full models support logprobs
    _LOGPROBS_MODELS = {"gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"}
    # which models do NOT support system messages
    _NO_SYSTEM_MODELS = {"o1-mini", "o1", "o3-mini", "o3", "o4-mini"}

    def get_available_option_values(self) -> List[str]:
        """Return supported keys; first option is default."""
        options = [self.DEFAULT_KEY]  # Default first
        options.extend([key for key in OPENAI_MODEL_LIST if key != self.DEFAULT_KEY])
        return options

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> Union[str, Tuple[str, any]]:
        """
        Send messages to an OpenAI model based on a simple key.

        Returns:
            - (content, logprobs) if model supports logprobs
            - content otherwise
        """
        model_id = target_options if target_options is not None else self.DEFAULT_KEY
        
        if model_id.startswith("openai-"):
            model_id = model_id.replace("openai-", "")
        
        if model_id in self._LOGPROBS_MODELS and logprobs:
            llm = get_llm(f"openai-{model_id}", max_tokens=None, temperature=0, additional_kwargs={"logprobs": True, "top_logprobs": 5})
        else:
            llm = get_llm(f"openai-{model_id}", max_tokens=None, temperature=0)

        # build messages
        if model_id in self._NO_SYSTEM_MODELS:
            prompt = input_text
            if system_message:
                prompt = f"{system_message}\n{input_text}"
            messages = [HumanMessage(prompt)]
            
        else:
            messages = []
            if system_message:
                messages.append(SystemMessage(system_message))
            messages.append(HumanMessage(input_text))

        try:
            ai_msg = llm.invoke(messages)
            
            if model_id in self._LOGPROBS_MODELS and logprobs:
                return ai_msg.choices[0].message.content, ai_msg.choices[0].logprobs
            
            return ai_msg.choices[0].message.content
        except Exception as e:
            print(f"Error during OpenAI completion ({model_id}): {e}")
            raise


if __name__ == "__main__":
    load_dotenv()
    target = OpenAITarget()
    print("Supported OpenAI keys:", target.get_available_option_values())
    
    # example without logprobs
    print(target.process_input("Hello!", target_options="gpt-4o"))
    # example with logprobs
    print(target.process_input("Hello!", target_options="gpt-4o", logprobs=True))
