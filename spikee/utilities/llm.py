from typing import Dict, List, Any, Union
import os
import json
import litellm
from litellm import NotFoundError
from filelock import FileLock

# region LLM Models/Prefixes
SUPPORTED_LLM_MODELS = [
    "llamaccp-server",
    "offline",
    "mock",
]

SUPPORTED_PREFIXES = [
    "openai-",
    "google-",
    "bedrock-",
    "ollama-",
    "llamaccp-server-",
    "together-",
    "groq-",
    "deepseek-",
    "openrouter-",
    "azure-",
    "mock-",
]


def get_supported_llm_models() -> List[str]:
    """Return the list of supported LLM models."""
    return SUPPORTED_LLM_MODELS


def get_supported_prefixes() -> List[str]:
    """Return the list of supported LLM model prefixes."""
    return SUPPORTED_PREFIXES

# endregion


# region LLM Model Maps
AZURE_MODEL_LIST: List[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
]

BEDROCK_MODEL_MAP: Dict[str, str] = {
    "claude35-haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude45-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude35-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude37-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude45-opus": "global.anthropic.claude-opus-4-5-20251101-v1:0",
    "deepseek-v3": "deepseek.v3-v1:0",
    "qwen3-coder-30b-a3b-v1": "qwen.qwen3-coder-30b-a3b-v1:0",
}

GOOGLE_MODEL_LIST: List[str] = [
    "gemini-3.0-pro",
    "gemini-3.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-exp-1206",
]

GROK_MODEL_LIST: List[str] = [
    "distil-whisper-large-v3-en",
    "gemma2-9b-it",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-guard-4-12b",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]

DEEPSEEK_MODEL_LIST: List[str] = [
    "deepseek-chat",  # deepseek-v3.2 non-thinking
    "deepseek-reasoner",  # deepseek-v3.2 thinking
]

OPENAI_MODEL_LIST: List[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o1-mini",
    "o1",
    "o3-mini",
    "o3",
    "o4-mini",
]

OPENROUTER_MODEL_LIST: List[str] = [
    "google/gemini-2.5-flash",
    "anthropic/claude-3.5-haiku",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4o-mini",
]

TOGETHER_AI_MODEL_MAP: Dict[str, str] = {
    "gemma2-8b": "google/gemma-2-9b-it",
    "gemma2-27b": "google/gemma-2-27b-it",
    "llama4-maverick-fp8": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama31-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama31-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama31-405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "llama33-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "qwen3-235b-fp8": "Qwen/Qwen3-235B-A22B-fp8-tput",
}


def _resolve_model_map(key: str, model_map: Dict[str, str]) -> str:
    """
    Convert a shorthand key to the full model identifier.
    """
    if key in model_map:
        return model_map[key]

    return key
# endregion

# region Wrappers


class LLMWrapper():
    """
    A wrapper class for LLM instances that provides a consistent interface and can be extended with additional functionality.
    """

    RETRY_ATTEMPTS = 2

    def __init__(self, model_name, llm_lite_kwargs):
        self.model_name = model_name
        self.llm_lite_kwargs = llm_lite_kwargs

        if self.llm_lite_kwargs is None:
            raise ValueError("LLMWrapper requires llm_lite_kwargs cannot be None.")

        litellm.success_callback = [self.__billing_callback]
        litellm.suppress_debug_info = True  # Prevent litellm from printing request/response info to console; we handle logging in the callback.
        # litellm._turn_on_debug()

        self.__billing_path = os.path.join(os.getcwd(), "billing.json")

        if os.path.exists(self.__billing_path):
            self.__billing = True
        else:
            self.__billing = None

    def __billing_callback(self, kwargs, completion_response, start_time, end_time):
        if self.__billing is not None:
            model = self.model_name
            cost = float(kwargs["response_cost"])
            input_tokens = completion_response.usage.get("prompt_tokens")
            output_tokens = completion_response.usage.get("completion_tokens")

            lock = FileLock(self.__billing_path + ".lock")
            with lock:
                with open(self.__billing_path, "r+") as f:
                    self.__billing = json.load(f)

                    self.__billing['total_cost'] = cost + self.__billing.get('total_cost', 0.0)

                    if model not in self.__billing['models']:
                        self.__billing['models'][model] = {
                            'cost': 0.0,
                            'input_tokens': 0,
                            'output_tokens': 0,
                        }

                    self.__billing['models'][model]['cost'] = cost + self.__billing['models'][model].get('cost', 0.0)
                    self.__billing['models'][model]['input_tokens'] = input_tokens + self.__billing['models'][model].get('input_tokens', 0)
                    self.__billing['models'][model]['output_tokens'] = output_tokens + self.__billing['models'][model].get('output_tokens', 0)

                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(self.__billing, indent=2))

    def __correct_messages(self, messages):
        corrected_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if ("role" in msg and "content" in msg):
                    corrected_messages.append(msg)
                else:
                    raise ValueError(f"Invalid message format: {msg}. Each message dict must contain 'role' and 'content' keys.")

            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                corrected_messages.append({"role": role, "content": content})

            elif isinstance(msg, Message) or isinstance(msg, SystemMessage) or isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                corrected_messages.append(msg.to_dict())

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

        return corrected_messages

    def __standard_invoke(self, messages):
        attempts = 0
        while attempts < self.RETRY_ATTEMPTS:
            attempts += 1
            try:
                return litellm.completion(
                    messages=messages,
                    drop_params=True,
                    **self.llm_lite_kwargs
                )
            except NotFoundError as e:
                print(f"[ERROR] Model Not Found: '{self.model_name}'")
                raise e
            except Exception as e:
                if attempts >= self.RETRY_ATTEMPTS:
                    raise e

    def invoke(self, messages, content_only: bool = False):
        corrected_messages = self.__correct_messages(messages)

        response = self.__standard_invoke(corrected_messages)

        if content_only:
            return response.choices[0].message.content
        else:
            return response


class MockWrapper():
    def __init__(self, model_name: str, max_tokens: Union[int, None], temperature: float):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.model_name != "mock":
            self._llm = get_llm(model_name, max_tokens=max_tokens, temperature=temperature)

    def invoke(self, messages, content_only: bool = False):
        if self.model_name == "mock":
            response = "This is a mock response from the LLM."

            if self.max_tokens is not None:
                response = response[:self.max_tokens]
        else:
            response = self._llm.invoke(messages, content_only=content_only)

        print("[Mock LLM] Message:", messages)
        print("[Mock LLM] Response:", response)
        print("--------------------------------")

        return response

# endregion

# region Messages


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__("system", content)


class HumanMessage(Message):
    def __init__(self, content: str):
        super().__init__("user", content)


class AIMessage(Message):
    def __init__(self, content: str):
        super().__init__("assistant", content)

# endregion


def validate_llm_option(option: str) -> bool:
    """
    Validate if the provided options correspond to a supported LLM model.
    """
    return option in SUPPORTED_LLM_MODELS or any(
        option.startswith(prefix) for prefix in SUPPORTED_PREFIXES
    )


def get_llm(options: str = "", max_tokens: Union[int, None] = 8, temperature: float = 0, additional_kwargs=None) -> Union[LLMWrapper, MockWrapper]:
    """
    Returns an LLMWrapper.

    Arguments:
        options (str): The LLM model option string.
        max_tokens (int): Maximum tokens for the LLM response (Default: 8 for LLM Judging).
        temperature (float): Sampling temperature for the LLM (Default: 0).
    """
    if not validate_llm_option(options):
        raise ValueError(
            f"Unsupported LLM option: '{options}'. "
            f"Supported Prefixes: {SUPPORTED_PREFIXES}, Supported Models: {SUPPORTED_LLM_MODELS}"
        )

    # Configure default kwargs
    kwargs: Dict[str, Any] = {
        "temperature": temperature,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if additional_kwargs:
        kwargs.update(additional_kwargs)

    # Model specific kwargs
    if options.startswith("openai-"):
        model_name = options.replace("openai-", "")
        kwargs["model"] = f"openai/{model_name}"
        kwargs["num_retries"] = 2

    elif options.startswith("google-"):
        # litellm expects gemini/ or vertex_ai/ prefix, we use gemini for Google AI Studio
        model_name = options.replace("google-", "")
        kwargs["model"] = f"gemini/{model_name}"
        kwargs["num_retries"] = 2

    elif options.startswith("bedrock-"):
        model_name_key = options.replace("bedrock-", "")
        model_name = _resolve_model_map(model_name_key, BEDROCK_MODEL_MAP)
        kwargs["model"] = f"bedrock/{model_name}"

    elif options.startswith("ollama-"):
        model_name = options.replace("ollama-", "")
        kwargs["model"] = f"ollama/{model_name}"

        timeout = os.environ.get("OLLAMA_TIMEOUT")
        if timeout:
            kwargs["timeout"] = float(timeout)
        attempts = os.environ.get("OLLAMA_MAX_ATTEMPTS")
        kwargs["num_retries"] = int(attempts) if attempts else 1

    elif options.startswith("llamaccp-server"):
        if options == "llamaccp-server":
            url = "http://localhost:8080/"
        else:
            try:
                port = int(options.split("llamaccp-server-")[-1])
                url = f"http://localhost:{port}/"
            except ValueError as e:
                raise ValueError(
                    f"Invalid port in options: '{options}'. Expected format 'llamaccp-server-[port]', for example 'llamaccp-server-8080'."
                ) from e

        kwargs["model"] = "openai/custom-model"  # Litellm routes via openai base api handling
        kwargs["api_base"] = url
        kwargs["api_key"] = "abc"
        kwargs["num_retries"] = 2

    elif options.startswith("together-"):
        model_name_key = options.replace("together-", "")
        model_name = _resolve_model_map(model_name_key, TOGETHER_AI_MODEL_MAP)

        kwargs["model"] = f"together_ai/{model_name}"
        kwargs["api_key"] = os.environ.get("TOGETHER_API_KEY")
        kwargs["num_retries"] = 2

    elif options.startswith("groq-"):
        model_name = options.replace("groq-", "")
        kwargs["model"] = f"groq/{model_name}"
        kwargs["num_retries"] = 2

    elif options.startswith("deepseek-"):
        model_name = options.replace("deepseek-", "", 1)
        kwargs["model"] = f"deepseek/{model_name}"
        kwargs["num_retries"] = 2

    elif options.startswith("openrouter-"):
        model_name = options.replace("openrouter-", "")
        kwargs["model"] = f"openrouter/{model_name}"
        kwargs["num_retries"] = 2

        if max_tokens is None:
            kwargs["max_tokens"] = 2048  # OpenRouter models often require explicit max_tokens

    elif options.startswith("azure-"):
        model_name = options.replace("azure-", "")

        kwargs["model"] = f"azure/{model_name}"
        kwargs["api_version"] = os.environ.get("API_VERSION", "2024-05-01-preview")
        kwargs["num_retries"] = 2

    elif options.startswith("offline"):
        return None

    elif options == "mock":
        return MockWrapper(model_name="mock", max_tokens=max_tokens, temperature=temperature)

    elif options.startswith("mock-"):
        return MockWrapper(model_name=options, max_tokens=max_tokens, temperature=temperature)

    else:
        raise ValueError(
            f"Invalid options format: '{options}' - review documentation for valid prefixes (e.g., 'bedrock-', 'google-', 'openai-')."
        )

    model_name = kwargs.get("model")
    return LLMWrapper(model_name=model_name, llm_lite_kwargs=kwargs)
