"""
AWS Transcribe Speech-to-Text provider module for Spikee.

Additional Args:
- `language_code`: BCP-47 language code (default: en-GB). E.g. fr-FR, de-DE.
- `sample_rate_hz`: Audio sample rate in Hz (default: 16000). Used for raw PCM and FLAC.
                    Automatically detected from WAV headers.

Supported audio formats:
  - pcm  — raw signed 16-bit little-endian PCM (pass-through)
  - flac — passed directly to Transcribe (natively supported)
  - wav, mp3, ogg — decoded to PCM via pydub + static-ffmpeg

Authentication via environment variables:
  - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  - AWS_PROFILE, AWS_DEFAULT_REGION
"""
import asyncio
import base64
import os
from typing import Optional, Set, Union, List, Dict, Sequence

from spikee.templates.provider import Provider
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, Content, Audio
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import AIMessage, HumanMessage, Message, single_message


class AWSTranscribeSTTProvider(Provider):
    """AWS Transcribe Speech-to-Text provider (streaming API)"""

    def __init__(self):
        super().__init__()
        self.region: Optional[str] = None
        self.language_code: Optional[str] = None
        self.sample_rate_hz: int = 16000
        self._credentials: dict = {}

    @property
    def default_model(self) -> str:
        return "transcribe"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "transcribe": "transcribe",
        }

    @property
    def audio_formats(self) -> Set[str]:
        return {"pcm", "flac", "wav", "mp3", "ogg"}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.language_code = additional_kwargs.get("language_code", "en-GB")
        self.sample_rate_hz = int(additional_kwargs.get("sample_rate_hz", 16000))

        try:
            import boto3
            import amazon_transcribe  # noqa: F401 - imported to validate package availability
        except ImportError as exc:
            raise ImportError(
                "[Import Error] Provider Module 'aws_transcribe_stt' is missing required packages. "
                "Please run `pip install boto3 amazon_transcribe` to install them."
            ) from exc

        self.region = os.getenv("AWS_DEFAULT_REGION", None)

        if self.region is None:
            raise ValueError(
                "AWS_DEFAULT_REGION environment variable must be set for AWS Transcribe STT Provider."
            )

        if os.getenv("AWS_PROFILE"):
            session = boto3.Session(profile_name=os.getenv("AWS_PROFILE"))
            frozen = session.get_credentials().get_frozen_credentials()

            # Inject as env vars so awscrt picks them up via the default chain
            os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token

        elif not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            raise ValueError(
                "AWS Transcribe STT Provider requires AWS credentials. "
                "Please set either AWS_PROFILE or AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables."
            )

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_STT], "STT Provider for AWS Transcribe speech-to-text."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [
            "language_code=en-GB,sample_rate_hz=16000",
        ], False

    async def _transcribe_async(
        self, audio_data: bytes, media_encoding: str, sample_rate: int
    ) -> str:
        try:
            from amazon_transcribe.client import TranscribeStreamingClient
            from amazon_transcribe.handlers import TranscriptResultStreamHandler
            from amazon_transcribe.model import TranscriptEvent
        except ImportError as exc:
            raise ImportError(
                "[Import Error] Provider Module 'aws_transcribe_stt' is missing required packages. "
                "Please run `pip install amazon-transcribe` to install them."
            ) from exc

        client = TranscribeStreamingClient(region=self.region)

        stream = await client.start_stream_transcription(
            language_code=self.language_code,
            media_sample_rate_hz=sample_rate,
            media_encoding=media_encoding,
        )

        transcript_parts: List[str] = []

        class _EventHandler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, transcript_event: TranscriptEvent):
                for result in transcript_event.transcript.results:
                    if not result.is_partial:
                        for alt in result.alternatives:
                            transcript_parts.append(alt.transcript)

        async def _write_chunks():
            chunk_size = 16 * 1024  # 16 KB
            offset = 0
            while offset < len(audio_data):
                chunk = audio_data[offset: offset + chunk_size]
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
                offset += chunk_size
            await stream.input_stream.end_stream()

        handler = _EventHandler(stream.output_stream)
        await asyncio.gather(_write_chunks(), handler.handle_events())

        return " ".join(transcript_parts).strip()

    def invoke(
        self, messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke AWS Transcribe streaming STT with base64-encoded audio. Returns transcribed text."""

        msg, _ = single_message(messages)

        content = msg.content

        if not isinstance(content, Audio):
            raise ValueError(
                "AWS Transcribe STT Provider requires a user message containing base64-encoded audio."
            )

        audio_bytes = content.get_raw_audio()
        audio_format = content.format

        if audio_format not in self.audio_formats:
            content.convert_audio_format(target_format="pcm")
            audio_bytes = content.get_raw_audio()
            audio_format = "pcm"

        transcribed_text = asyncio.run(
            self._transcribe_async(audio_bytes, audio_format, self.sample_rate_hz)
        )

        return AIMessage(content=transcribed_text)


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    pcm_path = sys.argv[1] if len(sys.argv) > 1 else "audio_file.pcm"
    with open(pcm_path, "rb") as f:
        raw = f.read()
    audio = Audio(base64.b64encode(raw).decode(), audio_format="pcm")
    provider = AWSTranscribeSTTProvider()
    provider.setup(model="transcribe", sample_rate_hz=24000)
    response = provider.invoke([HumanMessage(content=audio)])
    print(response.content)
