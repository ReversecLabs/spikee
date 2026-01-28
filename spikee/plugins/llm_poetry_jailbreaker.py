"""
LLM Poetry Jailbreaker plugin for spikee.
This plugin is based on the LLM Poetry Jailbreaker attack, and generates single-turn jailbreak 
attempts using poetry to bypass the target system's safety mechanisms and get it to respond to 
harmful or restricted questions.

Usage:
  spikee test --plugins llm_poetry_jailbreaker --plugin-options "llm_poetry_jailbreaker:model=openai-gpt-4o,variants=5"
"""

from spikee.templates.plugin import Plugin
from typing import Dict, List, Tuple, Union
from spikee.utilities.enums import ModuleTag
from spikee.utilities.llm import get_llm
from spikee.attacks.llm_poetry_jailbreaker import SPIKEE_LLM_POETRY_JAILBREAKER_PROMPT
from spikee.utilities.modules import parse_options, extract_json_or_fail


class LLMPoetryJailbreaker(Plugin):
    DEFAULT_MODEL = "openai-gpt-4o"
    VARIANTS = 5

    def get_description(self) -> Tuple[List[ModuleTag], str]:
        return [ModuleTag.LLM, ModuleTag.ATTACK_BASED], "Generates jailbreak attack prompts using an LLM and poetry techniques."

    def get_available_option_values(self) -> Tuple[List[str], bool]:
        return [], True

    def get_variants(self, plugin_option: str = None) -> int:
        """Get the number of variants to generate based on plugin options."""
        opts = parse_options(plugin_option)
        return int(opts.get("variants", self.VARIANTS))

    def _generate_jailbreak_attack(
        self, llm, objective: str, previous_attempts: List[Dict]
    ) -> str:
        """Generate a jailbreak attack using the LLM."""
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

        prompt = SPIKEE_LLM_POETRY_JAILBREAKER_PROMPT.replace(
            "{{objective}}", objective
        ).replace(
            "{{previous_attempts}}",
            prev_attempts_str
            if prev_attempts_formatted
            else "No previous attempts yet.",
        )

        # Call the model via .invoke and get content
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
                max_tokens=800
            )
        except Exception as e:
            raise RuntimeError(f"[LLMJailbreaker] Error initializing LLM '{llm_model}': {str(e)}")

        previous_attempts = []
        attack_prompts = []

        for i in range(1, variants + 1):
            try:
                attack_prompts.append(self._generate_jailbreak_attack(
                    llm,
                    text,
                    previous_attempts
                ))
            except Exception as e:
                print(f"[LLMPoetryJailbreaker] Error generating prompt {i}: {str(e)}")

        return attack_prompts
