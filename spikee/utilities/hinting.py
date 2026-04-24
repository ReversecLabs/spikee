import binascii
import inspect
from typing import Dict, Optional, Union, List, Tuple, Callable, Any
import typing
import base64
import io
import warnings

from spikee.utilities.enums import ModuleTag


# region Content Hinting
class ParentContent:
    def __init__(self, content):
        self.content = content


class Audio(ParentContent):
    """Stored audio content as a Base64-encoded string. The format can be optionally specified for better handling downstream."""

    def __init__(self, content: str, audio_format: Optional[str] = None):
        if not isinstance(content, str):
            raise ValueError(f"Audio content must be a base64-encoded string, got {type(content)}")

        super().__init__(content)

        self.format = audio_format

    def detect_audio_format(self) -> Optional[str]:
        """Detect the audio format from the base64-encoded content using magic bytes.

        Returns a lowercase format string (e.g. 'mp3', 'wav', 'flac') or 'pcm' if the format cannot be determined.
        """
        try:
            # Decode only the first 16 bytes — enough for all magic byte checks
            header = base64.b64decode(self.content[:24])[:16]
        except (ValueError, binascii.Error):
            return None

        # FLAC
        if header[:4] == b'fLaC':
            return 'flac'

        # WAV / AIFF (RIFF container — check sub-type)
        if header[:4] == b'RIFF':
            if header[8:12] == b'WAVE':
                return 'wav'
            if header[8:12] == b'AIFF':
                return 'aiff'

        # AIFF (big-endian FORM container)
        if header[:4] == b'FORM' and header[8:12] in (b'AIFF', b'AIFC'):
            return 'aiff'

        # OGG container (Vorbis, Opus, FLAC-in-OGG, Speex …)
        if header[:4] == b'OggS':
            return 'ogg'

        # MP3 — ID3 tag or raw sync-word variants
        if header[:3] == b'ID3':
            return 'mp3'
        if header[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
            return 'mp3'

        # AAC — ADTS sync word (0xFFF1 = MPEG-4 AAC, 0xFFF9 = MPEG-2 AAC)
        if header[:2] in (b'\xff\xf1', b'\xff\xf9'):
            return 'aac'

        # MP4 / M4A / M4B — 'ftyp' box at byte 4
        if header[4:8] == b'ftyp':
            return 'm4a'

        # WebM / Matroska — EBML magic
        if header[:4] == b'\x1a\x45\xdf\xa3':
            return 'webm'

        # AMR narrowband / wideband
        if header[:6] == b'#!AMR\n':
            return 'amr'
        if header[:9] == b'#!AMR-WB\n':
            return 'amr'

        # AU / Sun audio (.au / .snd)
        if header[:4] == b'.snd':
            return 'au'

        # CAF (Apple Core Audio)
        if header[:4] == b'caff':
            return 'caf'

        # No magic bytes matched — assume raw PCM
        return 'pcm'

    def convert_audio_format(
        self, target_format: str, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2
    ) -> Optional["Audio"]:
        """Convert the audio content to a different format using pydub + static-ffmpeg.

        Mutates self in-place and returns self, or None if the source format cannot be determined.
        For raw PCM sources (no header), sample_rate/channels/sample_width describe the input.

        Requires: ``pip install pydub audioop-lts static-ffmpeg``
        """

        source_format = self.format or self.detect_audio_format()
        if source_format is None:
            return None

        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                from pydub import AudioSegment
        except ImportError as exc:
            raise ImportError(
                "convert_audio_format() requires `pip install pydub audioop-lts static-ffmpeg`."
            ) from exc

        audio_bytes = base64.b64decode(self.content)

        if source_format == "pcm":
            segment = AudioSegment.from_raw(
                io.BytesIO(audio_bytes),
                sample_width=sample_width,
                frame_rate=sample_rate,
                channels=channels,
            )
        else:
            segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)

        output = io.BytesIO()
        segment.export(output, format=target_format)
        converted_b64 = base64.b64encode(output.getvalue()).decode("utf-8")

        self.content = converted_b64
        self.format = target_format
        return self

    def get_raw_audio(self) -> bytes:
        """Get the raw audio bytes by decoding the base64 content."""
        return base64.b64decode(self.content)

    def set_raw_audio(self, audio_bytes: bytes, audio_format: Optional[str] = None):
        """Set the audio content from raw audio bytes, encoding it as base64."""
        self.content = base64.b64encode(audio_bytes).decode("utf-8")
        if audio_format:
            self.format = audio_format


class Image(ParentContent):
    def __init__(self, content: str):
        if not isinstance(content, str):
            raise ValueError(f"Image content must be a base64-encoded string, got {type(content)}")

        super().__init__(content)

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
