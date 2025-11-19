"""
sample_attack.py

This is a sample dynamic attack plugin for spikee. Attack plugins have a lot of power—
they can call the target LLM as many times as needed to perform dynamic attacks. However,
they should respect the maximum attempts specified by the user (via --attack-iterations).
Additionally, any throttling or retry logic beyond what the target module already implements
must be handled by the attack plugin itself.

The attack plugin is responsible for calling the provided call_judge() method to determine
whether an attack attempt is successful.

It can choose to operate on the full input text or only on the malicious payload (if the dataset
entry provides a "payload" field). In this example, we simply append a random digit to the payload
(or the full input) to simulate an attack variation.

Usage:
    spikee test --attack sample_attack --attack-iterations 50
    spikee test --attack sample_attack --attack-iterations 25 --attack-options "strategy=aggressive"

Parameters:
    entry: dict
        The dataset entry. It should include at least "text" and may include "payload" and
        "exclude_from_transformations_regex" (a list of regex strings to specify parts of the
        input that should remain unmodified).
    target_module: module
        The target module that provides process_input(input_text, system_message).
    call_judge: function
        A function to be called as call_judge(entry, llm_response) that returns True if the
        attack was successful, and False otherwise.
    max_iterations: int
        The maximum number of attack iterations to perform.
    attempts_bar: (optional) tqdm progress bar for live progress updates.
    attack_option: (optional) configuration option passed via --attack-options.

Returns:
    tuple: (iterations_attempted, success_flag, modified_input, last_response)
           - iterations_attempted: number of iterations performed.
           - success_flag: True if an attack attempt was successful; False otherwise.
           - modified_input: the input text (or payload) used in the final attack attempt.
           - last_response: the LLM's response from the final attempt.

Notes:
  - The plugin must respect the max_iterations value.
  - The plugin is responsible for any additional throttling or retry strategy beyond
    what the target module implements.
  - The plugin should call call_judge() to determine if the LLM response indicates success.
"""

import random
from typing import List

from spikee.templates.attack import Attack


class SampleAttack(Attack):
    @property
    def __name__(self):
        return "sample_attack"

    def get_available_option_values(self) -> List[str]:
        """Return supported attack options; first option is default."""
        return ["strategy=random", "strategy=aggressive", "strategy=stealth"]

    def attack(
        self,
        entry,
        target_module,
        call_judge,
        max_iterations,
        attempts_bar=None,
        bar_lock=None,
        attack_option=None,
    ):
        """
        Executes a dynamic attack on the given entry.

        This sample attack plugin operates on either the malicious payload (if provided) or the full input.
        For each iteration (up to max_iterations), it creates a modified candidate by appending a random
        digit (0-9) to the base text. It then calls target_module.process_input() with the modified text,
        and uses call_judge() to determine if the attack was successful.

        The plugin must ensure it does not exceed max_iterations.

        Args:
            entry (dict): The dataset entry. Expected keys: "text", optionally "payload" and
                        "exclude_from_transformations_regex".
            target_module (module): The target module (must implement process_input(input_text, system_message)).
            call_judge (function): A function that accepts (entry, llm_response) and returns True if the attack
                                is successful.
            max_iterations (int): The maximum number of attack iterations to try.
            attempts_bar (tqdm, optional): A progress bar to update for each iteration.
            attack_option (str, optional): Configuration option like "strategy=aggressive".

        Returns:
            tuple: (iterations_attempted, success_flag, modified_input, last_response)
        """
        # Parse attack option
        strategy = "random"  # default
        if attack_option and attack_option.startswith("strategy="):
            strategy = attack_option.replace("strategy=", "")

        # Retrieve the full input text and, if available, the malicious payload.
        original_text = entry.get("text", "")
        payload = entry.get("payload", "")
        # Use the payload if present; otherwise, operate on the full input.
        base = payload if payload else original_text

        last_response = ""
        candidate_text = original_text  # default fallback

        for i in range(1, max_iterations + 1):
            # Create a candidate variation based on strategy
            if strategy == "aggressive":
                candidate_base = base + f" [ATTACK_{i}]" + str(random.randint(0, 999))
            elif strategy == "stealth":
                candidate_base = base + str(random.randint(0, 9))
            else:  # random strategy
                candidate_base = base + str(random.randint(0, 9))

            # If operating on only the payload, substitute the modified payload back into the full text.
            if payload and payload in original_text:
                candidate_text = original_text.replace(payload, candidate_base)
            else:
                candidate_text = candidate_base

            try:
                response, logprobs = target_module.process_input(
                    candidate_text, entry.get("system_message", None)
                )
                last_response = response
                # Determine if this candidate is successful using the provided call_judge() function.
                if call_judge(entry, response):
                    if attempts_bar:
                        # If we finish early, update the progress bar to remove iterations we did not run
                        remaining = max_iterations - i
                        attempts_bar.total = attempts_bar.total - remaining
                    return i, True, candidate_text, response
            except Exception as e:
                last_response = str(e)
            if attempts_bar:
                with bar_lock:
                    attempts_bar.update(1)
            # Implement throttling: wait briefly before next attempt.
            # time.sleep(0.5)
        return max_iterations, False, candidate_text, last_response
