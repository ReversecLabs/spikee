"""
AWS Polly Text-to-Speech provider module for Spikee.

Additional Args:
- `voice_id`: Polly VoiceId (default: "Joanna" — neural, en-US)
  See: https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
- `output_format`: mp3 (default), ogg_vorbis, pcm

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
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm_message import Message, upgrade_messages, AIMessage, HumanMessage
from typing import List, Tuple, Dict, Union


class AWSPollyTTSProvider(Provider):
    """AWS Polly Text-to-Speech provider"""

    def __init__(self):
        super().__init__()
        self.engine = None
        self.voice_id = None
        self.output_format = None
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

    def setup(
        self,
        model: str,
        max_tokens: Union[int, None] = None,
        temperature: Union[float, None] = None,
        **additional_kwargs,
    ) -> None:
        self.engine = model
        self.voice_id = additional_kwargs.get("voice_id", "Joanna")
        self.output_format = additional_kwargs.get("output_format", "mp3")

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

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.AUDIO, ModuleTag.LLM_TTS], "TTS Provider for AWS Polly text-to-speech."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        return [
            "voice_id=Joanna,output_format=mp3",
        ], False

    def invoke(
        self, input_messages: Union[str, List[Union[Message, dict, tuple, str]]]
    ) -> AIMessage:
        """Invoke AWS Polly TTS with the provided text. Returns base64-encoded audio."""

        upgraded_messages: List[Message] = upgrade_messages(input_messages)

        if len(upgraded_messages) > 1 and not isinstance(upgraded_messages[0], HumanMessage):
            raise ValueError("AWS Polly TTS Provider only supports a single user message input.")

        else:
            text = None
            for msg in upgraded_messages:
                if isinstance(msg, HumanMessage):
                    text = msg.content
                    break

        response = self.client.synthesize_speech(
            Engine=self.engine,
            VoiceId=self.voice_id,
            OutputFormat=self.output_format,
            Text=text,
            TextType="text",
        )

        audio_bytes = response["AudioStream"].read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return AIMessage(
            content=base64_audio,
            response_format=self.output_format,
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    provider = AWSPollyTTSProvider()
    provider.setup(model="standard", voice_id="Joanna", output_format="mp3")
    messages = [
        HumanMessage(content="Hello, this is a test of the AWS Polly text-to-speech provider."),
    ]

    print("Invoking AWS Polly TTS Provider...")
    response = provider.invoke(input_messages=messages)
    # print("Base64 Audio Content:", response.content)

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
