"""
OPUS-MT Universal Translation Plugin

Translates input text to any language pair using local Helsinki-NLP OPUS-MT models.

Usage:
    spikee generate --plugins opus_translator
    spikee generate --plugins opus_translator --plugin-options "opus_translator:source=en,targets=zh"
    spikee generate --plugins opus_translator --plugin-options "opus_translator:source=en,targets=es+fr+de" # Multiple targets
    spikee generate --plugins opus_translator --plugin-options "opus_translator:targets=en:fr|fr:de|de:es" # Language chains
    
    
Options:
    source: Source language code (default: "en")
    targets: Target language(s) in formats:
        - Single: "zh"
        - Multiple: "es+fr+de"
        - Chains: "en:fr|fr:de|de:es"
    quality: Translation quality (1=fast/greedy, >1=beam search for better quality)
    device: Device to use ("cuda" or "cpu", default auto-detect)
    cache_dir: Directory to cache models (optional)

Reference:
    https://huggingface.co/Helsinki-NLP

Requirements:
    pip install transformers torch sentencepiece

    To use GPU acceleration, ensure you have the appropriate CUDA toolkit and PyTorch version installed.
    'nvidia-smi' can be used to verify GPU availability and CUDA installation.

    To specify torch version with CUDA support, use:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<version>
"""

from typing import List, Tuple, Union, Optional

from transformers import MarianMTModel, MarianTokenizer
import torch

from spikee.templates.plugin import Plugin
from spikee.utilities.enums import ModuleTag
from spikee.utilities.modules import parse_options


class OpusTranslator(Plugin):
    # Common language codes (not exhaustive—OPUS-MT supports 500+)
    SUPPORTED_LANGUAGES = {
        "ta": "Tamil", "te": "Telugu", "my": "Burmese", "am": "Amharic", "sw": "Swahili",
        "ny": "Chichewa", "gn": "Guarani", "pap": "Papiamento", "eo": "Esperanto",
        "mk": "Macedonian", "sq": "Albanian", "mt": "Maltese", "ka": "Georgian",
        "hy": "Armenian", "tl": "Tagalog", "fi": "Finnish", "hu": "Hungarian",
        "ro": "Romanian", "cs": "Czech", "sk": "Slovak", "uk": "Ukrainian",
        "bg": "Bulgarian", "el": "Greek", "th": "Thai", "vi": "Vietnamese",
        "id": "Indonesian", "fil": "Filipino", "bn": "Bengali", "hi": "Hindi",
        "gu": "Gujarati", "pa": "Punjabi", "kn": "Kannada", "ml": "Malayalam",
        "or": "Oriya", "si": "Sinhala", "ne": "Nepali", "ur": "Urdu",
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "pl": "Polish",
        "tr": "Turkish", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
        "ko": "Korean", "ru": "Russian",
    }

    def __init__(self):
        super().__init__()
        self.model_cache = {}  # Cache format: {"src-tgt": (tokenizer, model)}

        # Detect GPU availability
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[OpusTranslator] Using device: {self.device}")
        except ImportError:
            self.device = "cpu"

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.ML], "Translates text to any language(s) using local OPUS-MT models. (Requires dependencies [see docs]: transformers, torch, sentencepiece) and 'HF_TOKEN')"

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        """Return supported options; Tuple[options (default is first), llm_required]"""
        return [
            "source=en",
            "targets=zh",
            "targets=es+fr+de",
            "targets=en:fr|fr:de|de:es",
            "quality=4",
            "device=cuda",
            "cache_dir=<path>"
        ], False

    def _load_translator(self, src_lang: str, tgt_lang: str, cache_dir: Optional[str] = None, device: Optional[str] = None):
        """Load and cache translator model. Reuses cached models on subsequent calls.

        Args:
            device: Device to move model to ('cuda', 'cpu'). Defaults to auto-detected.
        """
        cache_key = f"{src_lang}-{tgt_lang}"

        # Return cached model if available
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        target_device = device or self.device

        try:
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            # Move model to device (GPU or CPU)
            model = model.to(target_device)
            # Store in cache for reuse
            self.model_cache[cache_key] = (tokenizer, model, target_device)
            return tokenizer, model, target_device
        except Exception as e:
            raise RuntimeError(
                f"[OpusTranslator] Failed to load model '{model_name}': {str(e)}"
            )

    def _translate(self, text: str, src_lang: str, tgt_lang: str, cache_dir: Optional[str] = None, num_beams: int = 1, device: Optional[str] = None) -> str:
        """Translate text from src_lang to tgt_lang with optional beam search.

        Args:
            num_beams: Number of beams for beam search. 1 = greedy, >1 = beam search for better quality.
            device: Device to use ('cuda' or 'cpu'). Defaults to auto-detected.
        """
        try:
            tokenizer, model, target_device = self._load_translator(src_lang, tgt_lang, cache_dir, device)
        except RuntimeError as e:
            raise e

        try:
            inputs = tokenizer(text, return_tensors="pt").to(target_device)
            outputs = model.generate(**inputs, max_length=512, num_beams=num_beams)
            translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translated
        except Exception as e:
            raise RuntimeError(
                f"[OpusTranslator] Translation failed for {src_lang}→{tgt_lang}: {str(e)}"
            )

    def transform(
        self,
        text: str,
        exclude_patterns: List[str] = [],
        plugin_option: str = ""
    ) -> Union[str, List[str]]:
        """
        Translates input text to target language(s).

        Supports:
        - Simple translation: source → target
        - Multi-target: source → [target1, target2, target3]
        - Language chains: source → intermediate → ... → final

        Args:
            text (str): Input text to translate.
            exclude_patterns (List[str], optional): Ignored.
            plugin_option (str, optional): Options like "source=en,targets=ta+te+my" or "targets=en:fr|fr:de".

        Returns:
            str or List[str]: Translated version(s).
        """
        opts = parse_options(plugin_option)
        source_lang = opts.get("source", "en")
        targets_str = opts.get("targets", "zh")
        cache_dir = opts.get("cache_dir", None)
        num_beams = int(opts.get("quality", "1"))
        device = opts.get("device", None)  # None = auto-detect

        # Parse target specs (supports pipes for language chains, + for multiple targets)
        if "|" in targets_str:
            target_specs = [t.strip() for t in targets_str.split("|")]
        else:
            target_specs = [t.strip() for t in targets_str.split("+")]

        translations = []

        for target_spec in target_specs:
            try:
                result = text

                # Handle language chains (e.g., "en:fr" or "en:fr:de:es")
                if ":" in target_spec:
                    chain = [lang.strip() for lang in target_spec.split(":")]
                    for i in range(len(chain) - 1):
                        src = chain[i]
                        tgt = chain[i + 1]
                        result = self._translate(result, src, tgt, cache_dir, num_beams, device)
                else:
                    # Simple translation
                    result = self._translate(text, source_lang, target_spec, cache_dir, num_beams, device)

                translations.append(result)
            except RuntimeError:
                continue

        if len(translations) == 1:
            return translations[0]
        return translations if translations else text


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    plugin = OpusTranslator()

    for i in range(25):
        text = plugin.transform(
            "Hi my name is Spikee, I'm a helpful prompt injection assistant.",
            plugin_option="source=en,targets=zh+ny,cache_dir=./opus_cache"
        )
        print(f"{i+1}/100 - {text}")
