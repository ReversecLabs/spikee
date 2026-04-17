import inspect
from typing import Dict, Union, List, Tuple, Callable, Any
import typing

from spikee.utilities.enums import ModuleTag


# region Content Hinting
class ParentContent:
    def __init__(self, content):
        self.content = content


class Audio(ParentContent):
    pass


class Image(ParentContent):

    def base64_inline(self) -> str:
        """Return the image content as a Base64-encoded string suitable for inline embedding."""
        return f"data:image/png;base64,{self.content}"


Content = Union[str, Audio, Image]


def content_factory(content, content_type: str = "text") -> Content:
    """Factory function to create Content objects based on content type."""

    match content_type.lower():
        case "text":
            return str(content)
        case "audio":
            return Audio(content)
        case "image":
            return Image(content)
        case _:
            raise ValueError(f"Unsupported content type: {content_type}")


def get_content(content: Content) -> str:
    """Extract the raw content from a Content object."""
    if isinstance(content, (Audio, Image)):
        return content.content
    elif isinstance(content, str):
        return content
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")


def get_content_type(content: Content) -> str:
    """Determine the content type of the given content."""

    match content:
        case str():
            return "text"
        case Audio():
            return "audio"
        case Image():
            return "image"
        case _:
            raise ValueError(f"Unsupported content type: {type(content)}")


def validate_content_signature(content: Content, function: Callable, parameter: str) -> bool:
    """Validate that the content matches the expected type based on the function's type annotations.

    For backward compatibility with legacy judges/modules, if the parameter exists but has no
    type hints, validation is permissive (returns True).
    """
    # Use inspect.signature to check parameter existence (works with or without type hints)
    sig = inspect.signature(function)
    if parameter not in sig.parameters:
        raise ValueError(f"Parameter '{parameter}' not found in function signature.")

    # Check if parameter has type annotation
    param = sig.parameters[parameter]
    return validate_content_annotation(content, param.annotation)


def validate_content_annotation(content: Content, annotation) -> bool:
    """Validate that the content matches the expected type based on the annotation."""

    if annotation is inspect.Parameter.empty:
        annotation = str  # Default to str if no annotation

    # Handle Union types by extracting member types
    args = typing.get_args(annotation)
    if args:
        return isinstance(content, args)

    # Handle simple type annotations (non-Union)
    try:
        return isinstance(content, annotation)
    except TypeError:
        return False


# endregion


ModuleDescriptionHint = Tuple[List[ModuleTag], str]
ModuleOptionsHint = Tuple[List[str], bool]

TargetResponseHint = Union[Content, bool, Tuple[Union[Content, bool], Any]]
AttackResponseHint = Tuple[int, bool, Union[Content, Dict[str, Any]], Content]


def process_target_content(response: TargetResponseHint) -> str:
    """Process the content through the target module and return the response as a string."""
    if isinstance(response, tuple):
        if len(response) == 2:
            response, _ = response

        else:
            raise ValueError(f"Invalid tuple return from target's process_input. Expected (Content/bool, meta), got {len(response)} elements.")

    if isinstance(response, Content):
        return get_content(response)

    else:
        raise ValueError(f"Unexpected return type from target's process_input: {type(response)}. Expected Content.")
