from typing import Dict, List

import os

EXAMPLE_LLM_MODELS = [
    "openai-gpt-4.1-mini",
    "openai-gpt-4o",
]

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
    "mock-",
]


def get_example_llm_models() -> List[str]:
    """Return the list of example LLM models."""
    return EXAMPLE_LLM_MODELS


def get_supported_llm_models() -> List[str]:
    """Return the list of supported LLM models."""
    return SUPPORTED_LLM_MODELS


def get_supported_prefixes() -> List[str]:
    """Return the list of supported LLM model prefixes."""
    return SUPPORTED_PREFIXES


# Map of shorthand keys to TogetherAI model identifiers
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

# Default shorthand key
DEFAULT_TOGETHER_AI_KEY = "llama31-8b"


def _resolve_togetherai_model(key: str) -> str:
    """
    Convert a shorthand key to the full model identifier.
    Raises ValueError for unknown keys.
    """
    if key not in TOGETHER_AI_MODEL_MAP:
        valid = ", ".join(TOGETHER_AI_MODEL_MAP.keys())
        raise ValueError(f"Unknown model key '{key}'. Valid keys: {valid}")
    return TOGETHER_AI_MODEL_MAP[key]


def validate_llm_option(option: str) -> bool:
    """
    Validate if the provided options correspond to a supported LLM model.
    """
    if option is None:
        raise ValueError("LLM option cannot be None, ensure than modules leveraging LLM utilities specify an LLM option.")

    return option in SUPPORTED_LLM_MODELS or any(option.startswith(prefix) for prefix in SUPPORTED_PREFIXES)


def get_llm(options=None, max_tokens=8):
    """
    Initialize and return the appropriate LLM based on options.

    Arguments:
        options (str): The LLM model option string.
        max_tokens (int): Maximum tokens for the LLM response (Default: 8 for LLM Judging).
    """
    if not validate_llm_option(options):
        raise ValueError(
            f"Unsupported LLM option: '{options}'. "
            f"Supported Prefixes: {SUPPORTED_PREFIXES}, Supported Models: {SUPPORTED_LLM_MODELS}"
        )

    if options.startswith("openai-"):
        from langchain_openai import ChatOpenAI

        model_name = options.replace("openai-", "")
        return ChatOpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0,
            timeout=None,
            max_retries=2,
        )

    elif options.startswith("google-"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = options.replace("google-", "")
        return ChatGoogleGenerativeAI(
            transport="rest",
            model=model_name,
            max_tokens=max_tokens,
            temperature=0,
            timeout=None,
            max_retries=2
        )

    elif options.startswith("bedrock-"):
        from langchain_aws import ChatBedrock

        model_name = options.replace("bedrock-", "")
        return ChatBedrock(model=model_name, max_tokens=max_tokens, temperature=0)

    elif options.startswith("ollama-"):
        from langchain_ollama import ChatOllama

        model_name = options.replace("ollama-", "")
        return ChatOllama(
            model=model_name,
            num_predict=max_tokens,  # maximum number of tokens to predict
            temperature=0,
            client_kwargs={"timeout": float(os.environ['OLLAMA_TIMEOUT']) if os.environ.get(
                'OLLAMA_TIMEOUT') not in (None, '') else None},
                # timeout in seconds (None = not configured)
        ).with_retry(
            stop_after_attempt=int(os.environ['OLLAMA_MAX_ATTEMPTS']) if os.environ.get(
                'OLLAMA_MAX_ATTEMPTS') not in (None, '') else 1,
                # total attempts (1 initial + retries)
            wait_exponential_jitter=True,  # backoff with jitter
        )

    elif options.startswith("llamaccp-server"):
        from langchain_openai import ChatOpenAI

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

        return ChatOpenAI(
            base_url=url,
            api_key="abc",
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    elif options.startswith("together"):
        from langchain_together import ChatTogether

        model_name_key = options.replace("together-", "")
        key = model_name_key if options is not None else DEFAULT_TOGETHER_AI_KEY
        model_name = _resolve_togetherai_model(key)

        return ChatTogether(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0,
            timeout=None,
            max_retries=2,
        )

    elif options.startswith("offline"):
        return None

    elif options == "mock":
        return MockLLM(max_tokens=max_tokens)

    elif options.startswith("mock"):
        return MockLLM(
            options[5:], max_tokens=max_tokens
        )  # Pass model name after 'mock'

    else:
        raise ValueError(
            f"Invalid options format: '{options}'. Expected prefix 'openai-', 'google-', 'ollama-', 'bedrock-', 'llamaccp-server', 'together-', or 'offline'."
        )


class MockLLM:
    # A mock LLM class for testing purposes

    def __init__(self, model_name=None, max_tokens=8):
        if model_name is None or model_name == "":
            print("[MockLLM] No model name provided; using default mock behavior.")
            self.model = None
            self.max_tokens = max_tokens

        else:
            print("[MockLLM] Initializing mock LLM with model name:", model_name)
            self.model = get_llm(model_name, max_tokens=max_tokens)

    def invoke(self, messages):
        if self.model:
            response = self.model.invoke(messages)

        else:
            response = "This is a mock response from the LLM."

            if self.max_tokens is not None:
                response = response[: self.max_tokens]

        print("[Mock LLM] Message:", messages)
        print(
            "[Mock LLM] Response:",
            response,
            (" ======== " + response.content) if hasattr(response, "content") else "",
        )
        print("--------------------------------")

        return response
