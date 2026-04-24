"""
OpenAI Text-to-Speech provider module for Spikee.

Additional Args:
- `voice`:
    - gpt-4o-mini-tts: alloy (default), ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar
    - tts-1 and tts-1-hd: alloy, ash, coral, echo, fable, onyx, nova, sage, and shimmer.
    
- `response_format`: mp3 (default), opus, aac, flac, wav, pcm.
- `speed`: 1.0
"""
import base64
import os

from spikee.templates.streaming_provider import StreamingProvider
from spikee.utilities.hinting import ModuleDescriptionHint, Content, Audio, get_content
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, single_message, AIMessage, HumanMessage
from typing import Callable, Union, Dict, Tuple, Sequence, Set


class OpenAITTSProvider(StreamingProvider):
    """OpenAI Text-to-Speech provider"""

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini-tts"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gpt-4o-mini-tts": "gpt-4o-mini-tts",
            "tts-1-hd": "tts-1-hd",
            "tts-1": "tts-1",
        }

    @property
    def audio_formats(self) -> Set[str]:
        return {"mp3", "opus", "aac", "flac", "wav", "pcm"}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.model = model
        self.voice = additional_kwargs.get("voice", "alloy")
        self.response_format = additional_kwargs.get("response_format", "pcm")
        self.speed = float(additional_kwargs.get("speed", 1.0))

        if self.response_format not in self.audio_formats:
            raise ValueError(f"Invalid response_format '{self.response_format}'. Supported formats: {self.audio_formats}")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'openai_tts' is missing required packages. "
                "Please run `pip install spikee[openai]` to install them."
            )

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_TTS], "TTS Provider for OpenAI text-to-speech models."

    def _validate_messages(self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]) -> Tuple[str, str]:
        """Validate and extract instruction and text from messages."""
        msg, instruction = single_message(messages)

        if msg.content_type != "text":
            raise ValueError("OpenAI TTS Provider requires text content as input.")

        if instruction is None:
            instruction = "Speak in a cheerful and positive tone."
        else:
            instruction = get_content(instruction.content)

        return instruction, get_content(msg.content)

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke OpenAI TTS with the provided text. Returns audio bytes in metadata."""

        instruction, text = self._validate_messages(messages)

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            instructions=instruction,
            response_format=self.response_format,
            speed=self.speed,
        )

        audio_bytes = response.content
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return AIMessage(
            content=Audio(base64_audio, audio_format=self.response_format),
            original_response=response,
            response_format=self.response_format,
        )

    def invoke_streaming(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]], callback: Callable
    ):
        instruction, text = self._validate_messages(messages)

        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            instructions=instruction,
            response_format=self.response_format,
            speed=self.speed,
        ) as response:
            for audio_bytes in response.iter_bytes():
                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                callback(base64_audio)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am Spikee."
    provider = OpenAITTSProvider()
    provider.setup(model="gpt-4o-mini-tts", voice="alloy", response_format="pcm")
    response = provider.invoke([HumanMessage(content=text)])
    raw = response.content.get_raw_audio()
    with open("audio_file.pcm", "wb") as f:
        f.write(raw)
    print("Written to audio_file.pcm")
