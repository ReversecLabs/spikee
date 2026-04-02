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

from spikee.templates.provider import Provider
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, upgrade_messages, AIMessage, HumanMessage, SystemMessage, format_messages
from typing import List, Tuple, Dict, Union, Any


class OpenAITTSProvider(Provider):
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
        self.speed = additional_kwargs.get("speed", 1.0)

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'openai_tts' is missing required packages. "
                "Please run `pip install spikee[openai]` to install them."
            )

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM], "TTS Provider for OpenAI text-to-speech models."

    def invoke(
        self, messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke OpenAI TTS with the provided text. Returns audio bytes in metadata."""
        
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

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    provider = OpenAITTSProvider()
    provider.setup(model="gpt-4o-mini-tts", voice="alloy", response_format="mp3", speed=1.0)
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
        print("Audio playback requires 'soundfile' and 'sounddevice' packages. Please install them to play the audio response.")
