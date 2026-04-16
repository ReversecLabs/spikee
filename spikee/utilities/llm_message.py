from typing import Dict, List, Any, Union

from spikee.utilities.hinting import ContentHint
from spikee.utilities.content import Content, Text
from spikee.utilities.enums import ContentType


class Message:
    def __init__(self, role: str, content: ContentHint):
        self.role = role
        self.__content: Content = content if isinstance(content, Content) else Text(content)
        self.metadata = {}

    @property
    def content(self) -> str:
        return self.__content.content

    @property
    def content_type(self) -> ContentType:
        return self.__content.content_type

    @property
    def content_object(self) -> Content:
        return self.__content

    @property
    def contents(self) -> List[Content]:
        """For compatibility with list representation of contents"""
        return [self.__content]

    def to_dict(self) -> Dict[str, Union[str, Content]]:
        return {"role": self.role, "content": self.__content}

    def formatted_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": str(self.__content.content)}


class SystemMessage(Message):
    def __init__(self, content: ContentHint):
        super().__init__("system", content)


class HumanMessage(Message):
    def __init__(self, content: ContentHint):
        super().__init__("user", content)


class AIMessage(Message):
    def __init__(self, content: ContentHint, **kwargs):
        super().__init__("assistant", content)

        for key, value in kwargs.items():
            self.metadata[key] = value

    @property
    def original_response(self) -> Any:
        return self.metadata.get("original_response", None)


def format_messages(
    messages: Union[str, List[Union[Message, dict, tuple, str]]],
    bedrock_format: bool = False,
) -> List[Dict[str, Union[str, List[str]]]]:
    """Convert various message formats (string, dict, tuple, Message objects) into a standardized list of dicts with 'role' and 'content' keys."""
    formatted_messages = []
    if isinstance(messages, str):
        # If a single string is provided, treat it as a user message
        formatted_messages.append({"role": "user", "content": messages})

    elif isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                if "role" in msg and "content" in msg:
                    formatted_messages.append(msg)
                else:
                    raise ValueError(
                        f"Invalid message format: {msg}. Each message dict must contain 'role' and 'content' keys."
                    )

            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                formatted_messages.append({"role": role, "content": content})

            elif (
                isinstance(msg, Message)
                or isinstance(msg, SystemMessage)
                or isinstance(msg, HumanMessage)
                or isinstance(msg, AIMessage)
            ):
                formatted_messages.append(msg.formatted_dict())

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                formatted_messages.append({"role": "user", "content": msg})

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    if bedrock_format:
        # Bedrock expects messages in the format: {"role": "user", "content": ["message content"]}
        for msg in formatted_messages:
            if isinstance(msg["content"], str):
                msg["content"] = [{"text": msg["content"]}]

    return formatted_messages


def upgrade_messages(
    messages: Union[str, List[Union[Message, dict, tuple, str, Content]]],
) -> List[Message]:
    """Upgrade various message formats (string, dict, tuple, Message objects) into a standardized list of Message objects."""
    upgraded_messages = []
    if isinstance(messages, str):
        # If a single string is provided, treat it as a user message
        upgraded_messages.append(Message(role="user", content=messages))

    elif isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                if "role" in msg and "content" in msg:
                    upgraded_messages.append(
                        Message(role=msg["role"], content=msg["content"])
                    )
                else:
                    raise ValueError(
                        f"Invalid message format: {msg}. Each message dict must contain 'role' and 'content' keys."
                    )

            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                upgraded_messages.append(Message(role=role, content=content))

            elif (
                isinstance(msg, Message)
                or isinstance(msg, SystemMessage)
                or isinstance(msg, HumanMessage)
                or isinstance(msg, AIMessage)
            ):
                upgraded_messages.append(msg)

            elif isinstance(msg, Content):
                # If a Content object is provided without a role, assume it's a user message
                upgraded_messages.append(Message(role="user", content=msg))

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                upgraded_messages.append(Message(role="user", content=msg))

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    return upgraded_messages
