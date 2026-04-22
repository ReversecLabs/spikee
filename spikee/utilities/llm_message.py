from typing import Dict, List, Any, Union


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self.metadata = {}

    @property
    def contents(self):
        """For compatibility with list representation of contents"""
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


def format_messages(
    messages: Union[str, List[Union[Message, dict, tuple, str]]],
) -> List[Dict[str, str]]:
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
                formatted_messages.append(msg.to_dict())

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                formatted_messages.append({"role": "user", "content": msg})

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    return formatted_messages


def upgrade_messages(
    messages: Union[str, List[Union[Message, dict, tuple, str]]],
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

            elif isinstance(msg, str):
                # Assume it's a user message if only a string is provided
                upgraded_messages.append(Message(role="user", content=msg))

            else:
                raise ValueError(f"Unsupported message format type: {type(msg)}.")

    else:
        raise ValueError(f"Unsupported messages format type: {type(messages)}.")

    return upgraded_messages
