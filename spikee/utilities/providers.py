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
    "custom-",
    "mock-",
]


def get_supported_llm_models() -> List[str]:
    """Return the list of supported LLM models."""
    return SUPPORTED_LLM_MODELS


def get_supported_prefixes() -> List[str]:
    """Return the list of supported LLM model prefixes."""
    return SUPPORTED_PREFIXES

# endregion


# region LLM Model Maps - TEMPORARY - will be moved to respective provider modules
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

# endregion

# region provider helpers


def resolve_model_map(key: str, model_map: Dict[str, str]) -> str:
    """
    Convert a shorthand key to the full model identifier.
    """

    if key in model_map:
        return model_map[key]

    return key


def standardise_messages(messages):
    corrected_messages = []
    if isinstance(messages, str):
        # If a single string is provided, treat it as a user message
        corrected_messages.append({"role": "user", "content": messages})

    elif isinstance(messages, list):

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

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                corrected_messages.append({"role": "user", "content": msg})

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    return corrected_messages
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
