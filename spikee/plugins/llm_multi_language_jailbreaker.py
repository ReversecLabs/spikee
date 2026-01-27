"""
LLM Multi Language Jailbreaker plugin for spikee.
This plugin is based on the Multi Language LLM Jailbreaker attack, and generates jailbreak attempts 
using different languages, focusing on low-resource languages that may bypass safety filters in 
the target system.

Usage:
  spikee test --plugins llm_multi_language_jailbreaker --plugin-options "llm_multi_language_jailbreaker:model=openai-gpt-4o,variants=5"
"""
from spikee.templates.plugin import Plugin
from typing import Dict, List, Tuple, Union
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm import get_llm
from spikee.attacks.llm_multi_language_jailbreaker import SPIKEE_MULTI_LANGUAGE_JAILBREAKER_PROMPT
from spikee.utilities.modules import parse_options, extract_json_or_fail


class LLMMultiLanguageJailbreaker(Plugin):
    DEFAULT_MODEL = "openai-gpt-4o"
    VARIANTS = 5

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM, ModuleTag.ATTACK_BASED], "Generates jailbreak attack prompts using an LLM and multi language techniques."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        return [], True

    def _generate_multi_language_jailbreak_attack(
        self, llm, objective: str, previous_attempts: List[Dict]
    ) -> str:
        """Generate a multi language jailbreak attack using the LLM."""
        # Format previous attempts for the prompt
        prev_attempts_formatted = []
        for i, attempt in enumerate(previous_attempts):
            prompt = attempt.get("attack_prompt", "")
            response = attempt.get("response", "")
            if prompt and response:
                prev_attempts_formatted.append(
                    f"Attempt {i + 1}:\nPrompt: {prompt}\nResponse: {response}"
                )

        prev_attempts_str = "\\n\\n".join(prev_attempts_formatted)

        prompt = SPIKEE_MULTI_LANGUAGE_JAILBREAKER_PROMPT.replace(
            "{{objective}}", objective
        ).replace(
            "{{previous_attempts}}",
            prev_attempts_str
            if prev_attempts_formatted
            else "No previous attempts yet.",
        )

        # call the model via .invoke
        response = llm.invoke(prompt)
        res_text = response.content.strip()

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
                max_tokens=1000
            )
        except Exception as e:
            raise RuntimeError(f"[LLMMultiLanguageJailbreaker] Error initializing LLM '{llm_model}': {str(e)}")

        previous_attempts = []
        attack_prompts = []

        for i in range(1, variants + 1):
            try:
                attack_prompts.append(self._generate_multi_language_jailbreak_attack(
                    llm,
                    text,
                    previous_attempts
                ))
            except Exception as e:
                print(f"[LLMMultiLanguageJailbreaker] Error generating prompt {i}: {str(e)}")

        return attack_prompts
