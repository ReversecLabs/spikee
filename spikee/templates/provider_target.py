from spikee.templates.target import Target
from spikee.utilities.llm import get_llm
from spikee.utilities.providers import HumanMessage, SystemMessage
from spikee.utilities.modules import parse_options

from typing import List, Optional, Tuple, Union


class ProviderTarget(Target):
    def __init__(self, provider=None, default_model: Union[str, None] = None, models: Union[dict, list, None] = None):
        self._provider = provider
        self._default_model = default_model
        self._models = models

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""

        if isinstance(self._models, dict):
            options = [key for key in self._models]
            return options, True

        elif isinstance(self._models, list):
            return self._models, True

        return [], True

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> str:
        """
        Send messages to a provider model by key.

        Raises:
            ValueError if target_options is provided but invalid.
        """
        options = parse_options(target_options)

        if len(options) == 0 and target_options is not None and len(target_options) > 0:
            print(f"Warning: target_options missing key 'model='. Attempting 'model={target_options}'")
            options["model"] = target_options

        model_id = options.get("model", None)
        max_tokens = options.get("max_tokens", None)
        temperature = options.get("temperature", 0.7)

        # If models are defined for this provider and no model_id is given, use the first/default model
        if model_id is None:
            if self._default_model is not None:
                model_id = self._default_model

            elif self._models is not None:
                if isinstance(self._models, dict):
                    model_id = list(self._models.keys())[0]

                elif isinstance(self._models, list):
                    model_id = self._models[0]

            else:
                raise ValueError(
                    "ProviderTarget requires a 'model' option to specify which provider/model to use."
                )

        # If a provider is set for this target and the model_id doesn't already include it, prepend it
        if self._provider is not None and not model_id.startswith(self._provider):
            model_id = f"{self._provider}-{model_id}"

        if self._provider == "deepseek" and not model_id.startswith("deepseek-deepseek-"):
            model_id = f"deepseek-{model_id}"

        # Initialize provider client
        llm = get_llm(model_id, max_tokens=max_tokens, temperature=temperature)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            return llm.invoke(messages, content_only=True)

        except Exception as e:
            print(f"Error during provider model completion ({model_id}): {e}")
            raise
