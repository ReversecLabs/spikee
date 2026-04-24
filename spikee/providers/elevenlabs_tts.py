"""
ElevenLabs Text-to-Speech provider module for Spikee.

Additional Args:
- `voice_id`: ElevenLabs voice ID (default: "JBFqnCBsd6RMkjVDRZzb" = "George")
  Browse available voices at: https://elevenlabs.io/voice-library
- `output_format`: mp3_44100_128 (default), mp3_22050_32, pcm_16000, pcm_22050, pcm_44100, ulaw_8000
"""
import base64
import os
from typing import Callable, Set, Union, Dict, Sequence


from spikee.templates.streaming_provider import StreamingProvider
from spikee.utilities.hinting import ModuleDescriptionHint, Content, Audio, get_content
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, single_message, AIMessage, HumanMessage


class ElevenLabsTTSProvider(StreamingProvider):
    """ElevenLabs Text-to-Speech provider"""

    @property
    def default_model(self) -> str:
        return "eleven_flash_v2_5"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "eleven_flash_v2_5": "eleven_flash_v2_5",
            "eleven_turbo_v2_5": "eleven_turbo_v2_5",
            "eleven_multilingual_v2": "eleven_multilingual_v2",
            "eleven_monolingual_v1": "eleven_monolingual_v1",
        }

    @property
    def audio_formats(self) -> Set[str]:
        return {"mp3_44100_128", "mp3_22050_32", "pcm_16000", "pcm_22050", "pcm_44100", "ulaw_8000"}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.model = model
        self.voice_id = additional_kwargs.get("voice_id", "JBFqnCBsd6RMkjVDRZzb")
        self.output_format = additional_kwargs.get("output_format", "pcm_16000")

        if self.output_format not in self.audio_formats:
            raise ValueError(f"Invalid output_format '{self.output_format}'. Supported formats: {self.audio_formats}")

        try:
            from elevenlabs import ElevenLabs
            self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'elevenlabs_tts' is missing required packages. "
                "Please run `pip install elevenlabs` to install them."
            )

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_TTS], "TTS Provider for ElevenLabs text-to-speech models."

    def _validate_messages(self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]) -> str:
        """Extract text from messages."""
        msg, _ = single_message(messages)

        if msg.content_type != "text":
            raise ValueError("ElevenLabs TTS Provider requires text content as input.")

        return get_content(msg.content)

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke ElevenLabs TTS with the provided text. Returns base64-encoded audio."""

        text = self._validate_messages(messages)

        response = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model,
            output_format=self.output_format,
        )

        audio_bytes = b"".join(response)
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return AIMessage(
            content=Audio(base64_audio, audio_format=None),
            response_format=self.output_format,
        )

    def invoke_streaming(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]], callback: Callable
    ) -> None:
        """Invoke ElevenLabs TTS with streaming, calling callback for each audio chunk."""

        text = self._validate_messages(messages)

        response = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model,
            output_format=self.output_format,
        )

        for audio_bytes in response:
            base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            callback(base64_audio)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am Spikee."
    provider = ElevenLabsTTSProvider()
    provider.setup(model="eleven_flash_v2_5", voice_id="JBFqnCBsd6RMkjVDRZzb", output_format="pcm_16000")
    response = provider.invoke([HumanMessage(content=text)])
    raw = response.content.get_raw_audio()
    with open("audio_file.pcm", "wb") as f:
        f.write(raw)
    print("Written to audio_file.pcm")
