"""
ElevenLabs Text-to-Speech provider module for Spikee.

Additional Args:
- `voice_id`: ElevenLabs voice ID (default: "JBFqnCBsd6RMkjVDRZzb" = "George")
  Browse available voices at: https://elevenlabs.io/voice-library
- `output_format`: mp3_44100_128 (default), mp3_22050_32, pcm_16000, pcm_22050, pcm_44100, ulaw_8000
"""
import base64
import os

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, upgrade_messages, AIMessage, HumanMessage, SystemMessage
from typing import List, Tuple, Dict, Union


class ElevenLabsTTSProvider(Provider):
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
        self.output_format = additional_kwargs.get("output_format", "mp3_44100_128")

        try:
            from elevenlabs import ElevenLabs
            self.client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'elevenlabs_tts' is missing required packages. "
                "Please run `pip install elevenlabs` to install them."
            )

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.AUDIO, ModuleTag.LLM], "TTS Provider for ElevenLabs text-to-speech models."

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke ElevenLabs TTS with the provided text. Returns base64-encoded audio."""

        messages = upgrade_messages(messages)
        
        if len(messages) > 1 and not isinstance(messages[0], HumanMessage):
            raise ValueError("ElevenLabs TTS Provider only supports a single user message as input.")
        
        else:
            text = None
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    text = msg.content
                    break
                
        response = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model,
            output_format=self.output_format,
        )

        audio_bytes = b"".join(response)
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return AIMessage(
            content=base64_audio,
            response_format=self.output_format,
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    provider = ElevenLabsTTSProvider()
    provider.setup(model="eleven_flash_v2_5")
    messages = [
        HumanMessage(content="Hello, how are you today?"),
    ]
    response = provider.invoke(messages)
    print("Base64 Audio Content:", response.content)
    
    audio_bytes = base64.b64decode(response.content)

    try:
        import io
        import soundfile as sf
        import sounddevice as sd

        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        sd.play(data, sample_rate)
        sd.wait()
    except ImportError:
        print("Audio playback requires 'soundfile' and 'sounddevice' packages. Please install them to enable audio playback.")
