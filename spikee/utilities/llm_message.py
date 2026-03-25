from typing import Dict, List, Any, Union
from agent_framework import Message as AFMessage

# region LLM Model Maps - TEMPORARY - will be moved to respective provider modules


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


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.metadata = {}

    @property
    def contents(self):
        """For compatibility with Agent Framework's Message format"""
        return [self.content]

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__("system", content)


class HumanMessage(Message):
    def __init__(self, content: str):
        super().__init__("user", content)


class AIMessage(Message):
    def __init__(self, content: str, **kwargs):
        super().__init__("assistant", content)

        for key, value in kwargs.items():
            self.metadata[key] = value

    @property
    def original_response(self) -> Any:
        return self.metadata.get("original_response", None)


def format_messages(messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> List[Dict[str, str]]:
    """Convert various message formats (string, dict, tuple, Message objects) into a standardized list of dicts with 'role' and 'content' keys."""
    formatted_messages = []
    if isinstance(messages, str):
        # If a single string is provided, treat it as a user message
        formatted_messages.append({"role": "user", "content": messages})

    elif isinstance(messages, list):

        for msg in messages:
            if isinstance(msg, dict):
                if ("role" in msg and "content" in msg):
                    formatted_messages.append(msg)
                else:
                    raise ValueError(f"Invalid message format: {msg}. Each message dict must contain 'role' and 'content' keys.")

            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                formatted_messages.append({"role": role, "content": content})

            elif isinstance(msg, Message) or isinstance(msg, SystemMessage) or isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                formatted_messages.append(msg.to_dict())

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                formatted_messages.append({"role": "user", "content": msg})

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    return formatted_messages


def upgrade_messages(messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> List[Message]:
    """Upgrade various message formats (string, dict, tuple, Message objects) into a standardized list of Message objects."""
    upgraded_messages = []
    if isinstance(messages, str):
        # If a single string is provided, treat it as a user message
        upgraded_messages.append(Message(role="user", content=messages))

    elif isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                if ("role" in msg and "content" in msg):
                    upgraded_messages.append(Message(role=msg["role"], content=msg["content"]))
                else:
                    raise ValueError(f"Invalid message format: {msg}. Each message dict must contain 'role' and 'content' keys.")

            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                upgraded_messages.append(Message(role=role, content=content))

            elif isinstance(msg, Message) or isinstance(msg, SystemMessage) or isinstance(msg, HumanMessage) or isinstance(msg, AIMessage):
                upgraded_messages.append(msg)

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                upgraded_messages.append(Message(role="user", content=msg))

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    return upgraded_messages


def agent_framework_message_translation(messages: List[Message]) -> List[AFMessage]:
    """Translate our internal Message format to Agent Framework's Message format"""
    agent_framework_messages = []

    for msg in messages:
        agent_framework_messages.append(AFMessage(role=msg.role, contents=[msg.content]))

    return agent_framework_messages
