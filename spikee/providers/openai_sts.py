"""
OpenAI Speech-to-Speech provider module for Spikee.

Uses the OpenAI Realtime API to process audio input and return audio output.

Additional Args:
- `voice`: alloy (default), ash, ballad, coral, echo, sage, shimmer, verse
"""
import asyncio
import base64
import os
from typing import Union, Dict, Sequence, Optional, Set

from spikee.templates.provider import Provider
from spikee.utilities.hinting import ModuleDescriptionHint, Content, Audio, get_content
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, single_message, AIMessage, HumanMessage


class OpenAISTSProvider(Provider):
    """OpenAI Speech-to-Speech provider using the Realtime API."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.client = None
        self.voice = "alloy"

    @property
    def default_model(self) -> str:
        return "gpt-4o-realtime-preview"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "gpt-4o-realtime-preview": "gpt-4o-realtime-preview",
            "gpt-4o-mini-realtime-preview": "gpt-4o-mini-realtime-preview",
        }

    @property
    def audio_formats(self) -> Set[str]:
        return {"pcm"}

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

        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError as exc:
            raise ImportError(
                "[Import Error] Provider Module 'openai_sts' is missing required packages. "
                "Please run `pip install spikee[openai] openai[realtime]` to install them."
            ) from exc

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_STS], "STS Provider for OpenAI speech-to-speech models via the Realtime API."

    async def _invoke_async(self, audio_b64: str, instructions: Optional[str] = None) -> str:
        """Async call to the OpenAI Realtime API for speech-to-speech conversion."""
        session_config = {
            "modalities": ["audio"],
            "voice": self.voice,
        }
        if instructions:
            session_config["instructions"] = instructions

        audio_chunks = []
        async with self.client.beta.realtime.connect(model=self.model) as connection:
            await connection.session.update(session=session_config)
            await connection.input_audio_buffer.append(audio=audio_b64)
            await connection.input_audio_buffer.commit()
            await connection.response.create()

            async for event in connection:
                if event.type == "response.audio.delta":
                    audio_chunks.append(base64.b64decode(event.delta))
                elif event.type == "response.done":
                    break

        combined_audio = b"".join(audio_chunks)
        return base64.b64encode(combined_audio).decode("utf-8")

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke OpenAI STS via the Realtime API. Takes audio input, returns audio output."""
        msg, system_msg = single_message(messages, system_prompt=True)

        content = msg.content

        if not isinstance(content, Audio):
            raise ValueError("OpenAI STS Provider requires audio content as input.")

        if system_msg is not None and not isinstance(system_msg.content, str):
            raise ValueError("OpenAI STS Provider requires system instructions to be a text string.")

        audio_b64 = get_content(content)
        audio_format = content.format

        if audio_format not in self.audio_formats:
            content.convert_audio_format("pcm")
            audio_b64 = get_content(content)
            audio_format = "pcm"

        instructions = get_content(system_msg.content) if system_msg else None

        result_b64 = asyncio.run(self._invoke_async(audio_b64, instructions))

        return AIMessage(content=Audio(result_b64, audio_format="pcm"))


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    pcm_path = sys.argv[1] if len(sys.argv) > 1 else "audio_file.pcm"
    with open(pcm_path, "rb") as f:
        raw = f.read()
    audio = Audio(base64.b64encode(raw).decode(), audio_format="pcm")
    provider = OpenAISTSProvider()
    provider.setup(model="gpt-4o-realtime-preview", voice="alloy")
    response = provider.invoke([HumanMessage(content=audio)])
    out_raw = response.content.get_raw_audio()
    with open("audio_file_out.pcm", "wb") as f:
        f.write(out_raw)
    print("Written to audio_file_out.pcm")
