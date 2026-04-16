import inspect

from .utilities.modules import load_module_from_path, get_default_option
from .utilities.hinting import ContentHint
from .utilities.content import Content, Text


def annotate_judge_options(entries, judge_opts):
    """Annotate entries with judge options, using defaults when appropriate."""
    annotated = []
    default_map = {}
    for entry in entries:
        if judge_opts is None:
            # Get default option for this specific judge
            judge = entry["judge_name"]

            if judge in default_map:
                effective_options = default_map[judge]
            else:
                judge_module = load_module_from_path(judge, "judges")
                effective_options = get_default_option(judge_module)
        else:
            # Use provided judge options for all entries
            effective_options = judge_opts

        annotated.append({**entry, "judge_options": effective_options})
    return annotated


def call_judge(entry, output: ContentHint):
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
        llm_input = entry.get("content", entry.get("input", entry.get("text", "")))
        judge_module = load_module_from_path(judge_name, "judges")
        judge_func_params = inspect.signature(judge_module.judge).parameters

        input_annotation = judge_func_params['llm_input'].annotation
        output_annotation = judge_func_params['llm_output'].annotation

        if input_annotation == inspect.Parameter.empty:
            input_annotation = str
        if output_annotation == inspect.Parameter.empty:
            output_annotation = str

        if isinstance(llm_input, list):
            llm_input = "\n".join(llm_input)

        if not isinstance(llm_input, input_annotation):
            if isinstance(llm_input, Content) and input_annotation is str:
                llm_input = llm_input.content
            elif isinstance(llm_input, str) and input_annotation is Content:
                llm_input = Text(content=llm_input)

            else:
                raise ValueError(f"Judge '{judge_name}' expects llm_input of type {input_annotation}, but got {type(llm_input)}")

        if not isinstance(output, output_annotation):
            if isinstance(output, Content) and output_annotation is str:
                output = output.content
            elif isinstance(output, str) and output_annotation is Content:
                output = Text(content=output)

            else:
                raise ValueError(f"Judge '{judge_name}' expects llm_output of type {output_annotation}, but got {type(output)}")

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
