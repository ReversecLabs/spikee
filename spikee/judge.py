import os
import importlib
import inspect
from pathlib import Path


def _load_raw_judge_module(judge_name):
    """Load judge module. Returns None if not found."""
    local_path = Path(os.getcwd()) / "judges" / f"{judge_name}.py"
    if local_path.is_file():
        spec = importlib.util.spec_from_file_location(judge_name, local_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    else:
        try:
            return importlib.import_module(f"spikee.judges.{judge_name}")
        except ModuleNotFoundError:
            return None


def load_judge_module(judge_name):
    """
    Looks for `judges/{judge_name}.py` locally first,
    then falls back to built-in judges.
    """
    mod = _load_raw_judge_module(judge_name)
    if mod is None:
        raise ValueError(f"Judge '{judge_name}' not found locally or built-in.")
    return mod


def _get_effective_judge_options(judge_name, provided_options):
    """Get effective judge options, using default if none provided and judge supports options."""
    if provided_options is not None:
        return provided_options

    # Try to get default option if none provided
    try:
        mod = _load_raw_judge_module(judge_name)
        if mod and hasattr(mod, "get_available_option_values"):
            available = mod.get_available_option_values()
            if available:
                return available[0]  # First option is default
    except Exception:
        pass

    return None


def annotate_judge_options(entries, judge_opts):
    """Annotate entries with judge options, using defaults when appropriate."""
    annotated = []
    for entry in entries:
        if judge_opts is not None:
            # Use provided judge options for all entries
            effective_options = judge_opts
        else:
            # Get default option for this specific judge
            judge_name = entry.get("judge_name", "canary")
            effective_options = _get_effective_judge_options(judge_name, None)

        annotated.append({**entry, "judge_options": effective_options})
    return annotated


def call_judge(entry, output):
    """
    Determines if the LLM output indicates a successful attack.

    If the output provided is a boolean that value is used to indicate success or failure.
    This is used when testing LLM guardrail targets, which return True if the attack went
    through the guardrail (attack successful) and False if the guardrail stopped it.

    In all other cases (i.e. when using a target LLM), the appropriate judge module
    for the attack is loaded and its judge() function is called.
    """
    if isinstance(output, bool):
        return output

    else:
        # Remove failed empty responses.
        if output == "":
            return False

        # Judge
        judge_name = entry.get("judge_name", "canary")
        judge_args = entry.get("judge_args", "")
        judge_options = entry.get("judge_options", None)
        llm_input = entry["text"] if "text" in entry.keys() else entry["input"]
        judge_module = load_judge_module(judge_name)
        judge_func_params = inspect.signature(judge_module.judge).parameters
        if "judge_options" in judge_func_params:
            return judge_module.judge(
                llm_input=llm_input,
                llm_output=output,
                judge_args=judge_args,
                judge_options=judge_options,
            )
        else:
            return judge_module.judge(
                llm_input=llm_input, llm_output=output, judge_args=judge_args
            )
