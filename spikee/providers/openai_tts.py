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
from spikee.utilities.hinting import ModuleDescriptionHint
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, upgrade_messages, AIMessage, HumanMessage, SystemMessage
from typing import Callable, Union, Dict, List, Tuple


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

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.model = model
        self.voice = additional_kwargs.get("voice", "alloy")
        self.response_format = additional_kwargs.get("response_format", "mp3")
        self.speed = float(additional_kwargs.get("speed", 1.0))

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

    def _validate_messages(self, messages: Union[str, List[Union[Message, dict, tuple, str]]]) -> Tuple[str, str]:
        """Validate and extract instruction and text from messages."""
        messages = upgrade_messages(messages)

        if len(messages) > 2:
            raise ValueError("OpenAI TTS Provider only supports an instruction an user prompt input.")

        else:
            instruction = None
            text = None

            for msg in messages:
                if isinstance(msg, SystemMessage):
                    instruction = msg.content
                elif isinstance(msg, HumanMessage):
                    text = msg.content

            if instruction is None:
                instruction = "Speak in a cheerful and positive tone."

            if text is None:
                raise ValueError("OpenAI TTS Provider requires a user prompt input.")

        return instruction, text

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
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
            content=base64_audio,
            original_response=response,
            response_format=self.response_format,
        )

    def invoke_streaming(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]], callback: Callable
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
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        print("Playback requires 'numpy' and 'sounddevice' packages. Please install them.")
        exit(1)

    # PCM format from OpenAI: 24kHz, mono, 16-bit little-endian
    SAMPLE_RATE = 24000
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
            print("Starting playback (24kHz, mono, 16-bit PCM)")

        # Play chunk
        try:
            stream.write(audio_float)
            print(f"  Playing {len(audio_float)} samples ({complete_bytes} bytes)")
        except Exception as e:
            print(f"Error playing audio: {e}")

    provider = OpenAITTSProvider()
    provider.setup(model="gpt-4o-mini-tts", voice="alloy", response_format="pcm", speed=1.0)

    if False:
        messages = [
            HumanMessage(content="Hello, how are you today?"),
        ]
        response = provider.invoke(messages)
        # print("Base64 Audio Content:", response.content)

    else:
        messages = [
            HumanMessage(content="This is a streaming response test. The audio will play as it is received."),
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
