"""
TTS (Text-to-Speech) plugin for Spikee.

Converts the input text into base64-encoded audio using any configured TTS provider.

Usage:
  spikee generate --plugins tts --plugin-options "tts:model=openai_tts/gpt-4o-mini-tts"
  spikee generate --plugins tts --plugin-options "tts:model=openai_tts/gpt-4o-mini-tts,voice=alloy"
  spikee generate --plugins tts --plugin-options "tts:model=elevenlabs_tts/eleven_flash_v2_5,voice_id=JBFqnCBsd6RMkjVDRZzb"

Options:
  model   - TTS provider/model string passed to get_llm() (required)
            Default: openai_tts/gpt-4o-mini-tts

Additional options are forwarded to the provider's setup() as keyword arguments:
  openai_tts:      voice, response_format, speed
  elevenlabs_tts:  voice_id, output_format
"""

from typing import List, Tuple

from spikee.templates.plugin import Plugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm import get_llm
from spikee.utilities.llm_message import HumanMessage
from spikee.utilities.modules import parse_options


class TTSPlugin(Plugin):
    DEFAULT_MODEL = "openai_tts/gpt-4o-mini-tts"

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.AUDIO, ModuleTag.LLM_TTS], "Converts text to base64-encoded audio using a TTS provider."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        return [
            "model=openai_tts/gpt-4o-mini-tts",
            "model=elevenlabs_tts/eleven_flash_v2_5",
        ], True

    def transform(
        self,
        text: str,
        exclude_patterns: List[str] = [],
        plugin_option: str = "",
    ) -> str:
        opts = parse_options(plugin_option)
        llm_model = opts.get("model", self.DEFAULT_MODEL)
        setup_kwargs = {k: v for k, v in opts.items() if k != "model"}
        
        llm = get_llm(llm_model, max_tokens=None, temperature=None, **setup_kwargs)
        
        llm_description = llm.get_description()[0]
        if ModuleTag.LLM_TTS not in llm_description:
            raise ValueError(f"Selected model '{llm_model}' is not a valid TTS provider.")
        
        return llm.invoke([HumanMessage(content=text)]).content

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    plugin = TTSPlugin()
    response = plugin.transform("Hello, how are you today?", plugin_option="model=openai_tts/gpt-4o-mini-tts,voice=alloy,response_format=mp3,speed=1.0")
    #print("Base64 Audio Content:", response)
    
    import base64
    audio_bytes = base64.b64decode(response)
    
    try:
        import io
        import soundfile as sf
        import sounddevice as sd

        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        sd.play(data, sample_rate)
        sd.wait()
    except ImportError:
        print("Audio playback requires 'soundfile' and 'sounddevice' packages. Please install them to enable audio playback.")
