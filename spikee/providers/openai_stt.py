"""
OpenAI Speech-to-Text provider module for Spikee.

Additional Args:

"""
import base64
from io import BytesIO
import os
from typing import Union, Dict, Sequence


from spikee.templates.provider import Provider
from spikee.utilities.hinting import ModuleDescriptionHint, Content, get_content, Audio
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, single_message, AIMessage, HumanMessage


class OpenAISTTProvider(Provider):
    """OpenAI Speech-to-Text provider"""

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini-transcribe"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gpt-4o-mini-transcribe": "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe": "gpt-4o-transcribe",
            "gpt-4o-transcribe-diarize": "gpt-4o-transcribe-diarize",
            "whisper-1": "whisper-1",
        }

    @property
    def audio_formats(self) -> set:
        return {"mp3", "mp4", "mpeg", "mpga", "wav", "m4a", "webm"}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.model = model

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'openai_stt' is missing required packages. "
                "Please run `pip install spikee[openai]` to install them."
            )

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_STT], "STT Provider for OpenAI speech-to-text models."

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke OpenAI STT with the provided audio. Returns transcribed text in metadata."""

        msg, _ = single_message(messages)

        content = msg.content

        if not isinstance(content, Audio):
            raise ValueError("OpenAI STT Provider requires a user message containing audio content.")

        audio_bytes = content.get_raw_audio()
        audio_format = content.format

        if audio_format not in self.audio_formats:
            content.convert_audio_format(target_format="mp3")
            audio_bytes = content.get_raw_audio()
            audio_format = "mp3"

        audio_buffer = BytesIO(audio_bytes)
        audio_buffer.name = f"input_audio.{audio_format}"

        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_buffer,
            response_format="text",
        )

        transcribed_text = response.rstrip()

        return AIMessage(
            content=transcribed_text
        )


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    pcm_path = sys.argv[1] if len(sys.argv) > 1 else "audio_file.pcm"
    with open(pcm_path, "rb") as f:
        raw = f.read()
    audio = Audio(base64.b64encode(raw).decode(), audio_format="pcm")
    provider = OpenAISTTProvider()
    provider.setup(model="gpt-4o-mini-transcribe")
    response = provider.invoke([HumanMessage(content=audio)])
    print(response.content)
