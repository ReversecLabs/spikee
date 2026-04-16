"""
ElevenLabs Text-to-Speech provider module for Spikee.

Additional Args:
- `voice_id`: ElevenLabs voice ID (default: "JBFqnCBsd6RMkjVDRZzb" = "George")
  Browse available voices at: https://elevenlabs.io/voice-library
- `output_format`: mp3_44100_128 (default), mp3_22050_32, pcm_16000, pcm_22050, pcm_44100, ulaw_8000
"""
import base64
import os

from spikee.templates.streaming_provider import StreamingProvider
from spikee.utilities.hinting import ModuleDescriptionHint
from spikee.utilities.content import Content, Audio
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, single_message, AIMessage, HumanMessage
from typing import Callable, Union, Dict, Sequence


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

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.model = model
        self.voice_id = additional_kwargs.get("voice_id", "JBFqnCBsd6RMkjVDRZzb")
        self.output_format = additional_kwargs.get("output_format", "pcm_22050")

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

        return msg.content

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
            content=Audio(base64_audio),
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
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("Playback requires 'numpy' and 'sounddevice' packages. Please install them.")
        exit(1)

    # ElevenLabs PCM format: pcm_22050
    SAMPLE_RATE = 22050
    CHANNELS = 1
    BYTES_PER_SAMPLE = 2  # 16-bit = 2 bytes

    chunk_count = 0
    stream = None
    buffer = b""  # Buffer for incomplete frames

    def play_audio(base64_audio):
        global chunk_count, stream, buffer
        chunk_count += 1

        audio_bytes = base64.b64decode(base64_audio)
        print(f"[Chunk {chunk_count}] Received {len(audio_bytes)} bytes")

        # Append to buffer
        buffer += audio_bytes

        # Extract complete samples (16-bit = 2 bytes per sample)
        complete_bytes = (len(buffer) // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE
        if complete_bytes == 0:
            print(f"  Buffering {len(buffer)} incomplete bytes, waiting for more...")
            return

        # Split buffered data
        to_play = buffer[:complete_bytes]
        buffer = buffer[complete_bytes:]  # Keep incomplete samples for next chunk

        # Convert raw PCM bytes to numpy array
        audio_data = np.frombuffer(to_play, dtype=np.int16)

        # Normalize to [-1, 1] for playback
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Create stream on first chunk
        if stream is None:
            stream = sd.OutputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype=np.float32)
            stream.start()
            print(f"Starting playback ({SAMPLE_RATE}Hz, mono, 16-bit PCM)")

        # Play chunk
        try:
            stream.write(audio_float)
            print(f"  Playing {len(audio_float)} samples ({complete_bytes} bytes)")
        except Exception as e:
            print(f"Error playing audio: {e}")

    # Test with streaming
    provider = ElevenLabsTTSProvider()
    provider.setup(model="eleven_flash_v2_5", voice_id="JBFqnCBsd6RMkjVDRZzb", output_format="pcm_22050")

    if False:
        # Non-streaming test
        messages = [
            HumanMessage(content="Hello, how are you today?"),
        ]
        response = provider.invoke(messages)
        # print("Base64 Audio Content:", response.content)

        audio_bytes = base64.b64decode(response.content)
        try:
            import io
            import soundfile as sf

            data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            sd.play(data, sample_rate)
            sd.wait()
            print("Playback complete.")
        except Exception as e:
            print(f"Playback error: {e}")

    else:
        # Streaming test with PCM format
        messages = [
            HumanMessage(content="This is a streaming test with ElevenLabs. The audio will play as it is received."),
        ]
        provider.invoke_streaming(messages, callback=play_audio)

        # Flush any remaining buffered bytes
        if buffer:
            print(f"\nFlushing {len(buffer)} final bytes from buffer...")
            audio_data = np.frombuffer(buffer, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            if stream is not None:
                stream.write(audio_float)

        # Close stream
        if stream is not None:
            stream.stop()
            stream.close()
        print("Playback complete.")
