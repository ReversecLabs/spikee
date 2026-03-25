from spikee.templates.target import Target
from spikee.utilities.llm import get_llm
from spikee.utilities.llm_message import HumanMessage, SystemMessage
from spikee.utilities.modules import parse_options

from typing import List, Optional, Tuple, Union, Any


class ProviderTarget(Target):
    def __init__(
        self,
        provider=None,
        default_model: Union[str, None] = None,
        models: Union[dict, list, None] = None
    ):
        self._provider_name = provider

        self._default_model = default_model
        self._models = models

        if self._provider_name is not None and (self._default_model is None or self._models is None):
            self.set_defaults()

    def set_defaults(self):
        if self._provider_name is not None:
            provider = get_llm(f"{self._provider_name}/")

            if self._default_model is None:
                self._default_model = f"{self._provider_name}/{provider.default_model}"

            if self._models is None:
                self._models = provider.models

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported attack options; Tuple[options (default is first), llm_required]"""

        if isinstance(self._models, dict):
            options = [key for key, value in self._models.items()]
            return options, True

        elif isinstance(self._models, list):
            return self._models, True

        return [], True

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        logprobs: bool = False,
    ) -> Union[str, bool, Tuple[Union[str, bool], Any]]:
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

        if max_tokens is not None:
            max_tokens = int(max_tokens)

        if temperature is not None:
            temperature = float(temperature)

        if self._provider_name is None:

            if model_id is not None and '/' in model_id:
                self._provider_name, model = model_id.split('/', 1)

                if model is None or model == "":
                    self.set_defaults()

            else:
                raise ValueError(
                    "ProviderTarget requires a provider name to be specified in the model option (e.g. 'model=bedrock/claude45-sonnet') or as a default provider with model mappings."
                )

        if model_id is None:
            if self._default_model is not None:
                model_id = self._default_model

            elif self._models is not None:
                if isinstance(self._models, dict):
                    model_id = f"{self._provider_name}/{list(self._models.keys())[0]}"

                elif isinstance(self._models, list):
                    model_id = f"{self._provider_name}/{self._models[0]}"

            else:
                raise ValueError(
                    "ProviderTarget requires a 'model' option to specify which provider/model to use."
                )

        if '/' not in model_id:
            model_id = f"{self._provider_name}/{model_id}"

        # Initialize provider client
        llm = get_llm(model_id, max_tokens=max_tokens, temperature=temperature)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(system_message))
        messages.append(HumanMessage(input_text))

        # Invoke model
        try:
            response = llm.invoke(messages)

        except Exception as e:
            print(f"Error during provider model completion ({model_id}): {e}")
            raise

        if 'logprobs' in response.metadata and logprobs:
            return response.content, response.metadata['logprobs']

        else:
            return response.content
