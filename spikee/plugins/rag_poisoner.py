"""
RAG Poisoner plugin for spikee.
This plugin is based on the RAG Poisoner attack, and injects fake RAG context that 
appears to be legitimate document snippets supporting the attack objective.

Usage:
  spikee test --plugins rag_poisoner --plugin-options "rag_poisoner:model=openai-gpt-4o,variants=5"
"""

from typing import Dict, List, Tuple, Union

from spikee.templates.plugin import Plugin
from spikee.utilities.enums import PluginType
from spikee.utilities.llm import (
    get_llm,
    get_supported_llm_models,
    get_example_llm_models,
    get_supported_prefixes,
)
from spikee.utilities.modules import parse_options, extract_json_or_fail

from spikee.attacks.rag_poisoner import SPIKEE_RAG_POISONER_PROMPT


class RAGPoisoner(Plugin):
    DEFAULT_MODEL = "openai-gpt-4o"
    VARIANTS = 5

    def get_description(self) -> Tuple[PluginType, str]:
        return PluginType.LLM_BASED, "Generates RAG Poisoner attack prompts using an LLM."

    def get_available_option_values(self) -> List[str]:
        return [f"model={model}" for model in get_example_llm_models()] + [
            f"model={model}" for model in get_supported_llm_models()
        ]

    def get_available_prefixes(self) -> Tuple[bool, List[str]]:
        """Return supported prefixes."""
        return False, get_supported_prefixes()

    def _generate_rag_attack(
        self, llm, objective: str, previous_attempts: List[Dict]
    ) -> str:
        """Generate a RAG poisoning attack using the LLM."""
        # Format previous attempts properly
        prev_attempts_formatted = []
        for i, attempt in enumerate(previous_attempts):
            prompt = attempt.get("attack_prompt", "")
            response = attempt.get("response", "")
            if prompt and response:
                prev_attempts_formatted.append(
                    f"Attempt {i + 1}:\nPrompt: {prompt}\nResponse: {response}"
                )

        prev_attempts_str = "\\n\\n".join(prev_attempts_formatted)

        prompt = SPIKEE_RAG_POISONER_PROMPT.replace("{{objective}}", objective).replace(
            "{{previous_attempts}}",
            prev_attempts_str
            if prev_attempts_formatted
            else "No previous attempts yet.",
        )
        res_text = llm.invoke(prompt).content.strip()

        obj = extract_json_or_fail(res_text)
        attack_prompt = obj.get("attack_prompt", "")
        if not attack_prompt:
            raise RuntimeError("LLM failed to produce an attack prompt")
        return attack_prompt.strip()

    def transform(
        self,
        text: str,
        exclude_patterns: List[str] = None,
        plugin_option: str = None
    ) -> Union[str, List[str]]:
        opts = parse_options(plugin_option)
        llm_model = opts.get("model", self.DEFAULT_MODEL)
        variants = int(opts.get("variants", self.VARIANTS))

        try:
            llm = get_llm(
                llm_model,
                max_tokens=400
            )
        except Exception as e:
            return None

        previous_attempts = []
        attack_prompts = []

        for i in range(1, variants + 1):
            try:
                attack_prompts.append(self._generate_rag_attack(
                    llm,
                    text,
                    previous_attempts
                ))
            except Exception as e:
                print(f"[RAGPoisoner] Error generating prompt {i}: {str(e)}")

        return attack_prompts
