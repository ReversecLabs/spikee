"""
AWS Polly Text-to-Speech provider module for Spikee.

Additional Args:
- `voice_id`: Polly VoiceId (default: "Joanna" — neural, en-US)
  See: https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
- `output_format`: mp3 (default), ogg_vorbis, pcm16

Engines (set via model):
- neural (default): Neural TTS — natural, high-quality voices
- generative: Generative TTS — most expressive
- long-form: Optimised for long documents
- standard: Classic concatenative synthesis

Allows for AWS Key-based or profile-based authentication via environment variables:
 - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
 - AWS_PROFILE, AWS_DEFAULT_REGION
"""
import base64
import os

from spikee.templates.provider import Provider
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, Content, Audio
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, single_message, AIMessage, HumanMessage
from typing import Set, Union, Dict, Sequence


class AWSPollyTTSProvider(Provider):
    """AWS Polly Text-to-Speech provider"""

    def __init__(self):
        super().__init__()
        self.engine = None
        self.voice_id = None
        self.output_format = "pcm"
        self.client = None

    @property
    def default_model(self) -> str:
        return "neural"

    @property
    def models(self) -> Dict[str, str]:
        return {
            "neural": "neural",         # Neural TTS — natural, high-quality voices (default)
            "generative": "generative",  # Generative TTS — most expressive
            "long-form": "long-form",   # Optimised for longer content
            "standard": "standard",     # Classic concatenative synthesis
        }

    @property
    def audio_formats(self) -> Set[str]:
        return {"mp3", "ogg_vorbis", "pcm"}

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.engine = model
        self.voice_id = additional_kwargs.get("voice_id", "Joanna")
        self.output_format = additional_kwargs.get("output_format", "pcm")

        if self.output_format not in self.audio_formats:
            raise ValueError(f"Invalid output_format '{self.output_format}'. Supported formats: {self.audio_formats}")

        try:
            import boto3
            if not os.getenv("AWS_DEFAULT_REGION"):
                raise ValueError("AWS_DEFAULT_REGION environment variable must be set for AWS Polly TTS Provider.")

            if os.getenv("AWS_PROFILE"):  # AWS Profile-based authentication
                session = boto3.Session(profile_name=os.getenv("AWS_PROFILE"))
                self.client = session.client(
                    "polly",
                    region_name=os.getenv("AWS_DEFAULT_REGION"),
                )

            elif os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):  # AWS Key-based authentication
                self.client = boto3.client(
                    "polly",
                    region_name=os.getenv("AWS_DEFAULT_REGION"),
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                )

            else:
                raise ValueError(
                    "AWS Polly TTS Provider requires AWS credentials. Please set either AWS_PROFILE or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
                )

        except ImportError:
            raise ImportError(
                "[Import Error] Provider Module 'aws_polly_tts' is missing required packages. "
                "Please run `pip install boto3` to install them."
            )

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.AUDIO, ModuleTag.LLM_TTS], "TTS Provider for AWS Polly text-to-speech."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return [
            "voice_id=Joanna,output_format=pcm",
        ], False

    def invoke(
        self, input_messages: Union[str, Sequence[Union[Message, dict, tuple, str, Content]]]
    ) -> AIMessage:
        """Invoke AWS Polly TTS with the provided text. Returns base64-encoded audio."""

        msg, _ = single_message(input_messages)

        if msg.content_type != "text":
            raise ValueError("AWS Polly TTS Provider requires text content as input.")

        response = self.client.synthesize_speech(
            Engine=self.engine,
            VoiceId=self.voice_id,
            OutputFormat=self.output_format,
            Text=msg.content,
            TextType="text",
        )

        audio_bytes = response["AudioStream"].read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return AIMessage(
            content=Audio(base64_audio, audio_format=self.output_format),
            response_format=self.output_format,
        )


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am Spikee."
    provider = AWSPollyTTSProvider()
    provider.setup(model="neural", voice_id="Joanna", output_format="pcm")
    response = provider.invoke([HumanMessage(content=text)])
    raw = response.content.get_raw_audio()
    with open("audio_file.pcm", "wb") as f:
        f.write(raw)
    print("Written to audio_file.pcm")
