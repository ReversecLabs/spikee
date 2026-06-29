"""
Debugging utilities for running individual Spikee modules in isolation.

Each function maps to a `spikee debug module <type>` sub-command and loads the
named module, executes it with the supplied arguments, then prints the result.
This is useful for verifying that a custom module is correctly implemented
before running a full test.

CLI Usage:
    spikee debug module targets   -m <module>  -i <input>  [--system-message <msg>] [--target-options <opts>]
    spikee debug module judges    -m <module>  -i <input>  -o <output>  [--judge-options <opts>] [--judge-args <args>]
    spikee debug module plugins   -m <module>  -i <input>  [--plugin-options <opts>] [--exclude-patterns <regex>]
    spikee debug module attacks   -m <module>  -i <b64json> --target <target>  [--max-iterations <n>] [--attack-options <opts>]
    spikee debug module providers -m <provider/model>  -i <input>  [--max-tokens <n>] [--temperature <f>]
"""

import base64
import json

from spikee.utilities.modules import load_module_from_path
from spikee.utilities.hinting import get_content
from spikee.judge import call_judge
from spikee.utilities.llm import get_llm

def debug_module_target(args):
    """
    Debug a target module by sending a single input and printing the response.

    CLI args used:
        -m / --module          Name of the target module (local or built-in)
        -i / --input           Input text to send to the target
        --system-message       Optional system prompt
        --target-options       Optional target options string (e.g. "openai/gpt-4o")

    Example:
        spikee debug module targets -m llm_provider -i "Hello!" --target-options "openai/gpt-4o"
    """
    target = load_module_from_path(args.module, "targets")

    target_args = {}

    if args.system_message:
        target_args["system_message"] = args.system_message
    
    if args.target_options:
        target_args["target_options"] = args.target_options

    response = target.process_input(args.input, **target_args)

    print(f"[{target.__class__.__name__}] Response: {get_content(response) if response else 'No response'}")

def debug_module_judge(args):
    """
    Debug a judge module by evaluating a target response and printing the verdict.

    CLI args used:
        -m / --module          Name of the judge module (local or built-in)
        -i / --input           The original prompt / input that was sent to the target
        -o / --output          The LLM output to evaluate (required)
        --judge-options        LLM provider options for the judge (e.g. "openai/gpt-4o-mini")
        --judge-args           Additional judge-specific arguments (format varies by judge)

    Example:
        spikee debug module judges -m llm_judge_harmful \\
            -i "How do I make a bomb?" \\
            -o "Sure, here are the steps..." \\
            --judge-options "openai/gpt-4o-mini"
    """
    judge = load_module_from_path(args.module, "judges")

    judge_args = {}

    if args.output:
        judge_args["llm_output"] = args.output
    
    else:
        raise ValueError("Judges require an output argument to evaluate the response.")

    if args.judge_options:
        judge_args["judge_options"] = args.judge_options

    if args.judge_args:
        judge_args["judge_args"] = args.judge_args
    
    response = judge.judge(args.input, **judge_args)

    print(f"[{judge.__class__.__name__}] Judge Result: {response}")

def debug_module_plugin(args):
    """
    Debug a plugin module by transforming an input and printing the result.

    CLI args used:
        -m / --module          Name of the plugin module (local or built-in)
        -i / --input           Input text to transform
        --plugin-options       Plugin options string
        --exclude-patterns     Regex pattern(s) to exclude from transformation.
                               This flag can be specified multiple times.

    Example:
        spikee debug module plugins -m base64_plugin -i "Ignore previous instructions and..."
    """
    plugin = load_module_from_path(args.module, "plugins")

    plugin_args = {}

    if args.plugin_options:
        plugin_args["plugin_options"] = args.plugin_options

    if args.exclude_patterns:
        plugin_args["exclude_patterns"] = args.exclude_patterns

    response = plugin.transform(args.input, **plugin_args)

    print(f"[{plugin.__class__.__name__}] Plugin Response: {get_content(response) if response else 'No response'}")

def debug_module_attack(args):
    """
    Debug an attack module by running it against a target and printing the result.

    The -i / --input argument must be a base64-encoded JSON string representing a
    dataset entry. The JSON object must contain all of the following fields:
        id            (int)  Numeric entry identifier
        long_id       (str)  Human-readable entry identifier
        content       (str)  The prompt / payload content
        content_type  (str)  Content type, e.g. "text"
        judge_name    (str)  Name of the judge module to use for evaluation
        judge_args    (dict) Judge-specific arguments (can be an empty object {})

    To generate the base64 input in a shell:
        echo '{"id":1,"long_id":"entry_001","content":"<prompt>","content_type":"text","judge_name":"<judge>","judge_args":{}}' | base64

    CLI args used:
        -m / --module          Name of the attack module (local or built-in)
        -i / --input           Base64-encoded JSON dataset entry (see above)
        --target               Name of the target module to attack (required)
        --max-iterations       Maximum attack iterations (default: 10)
        --attack-options       Attack options string

    Example:
        spikee debug module attacks \\
            -m crescendo \\
            -i $(echo '{"id":1,"long_id":"e1","content":"Ignore instructions","content_type":"text","judge_name":"llm_judge_harmful","judge_args":{}}' | base64) \\
            --target llm_provider \\
            --max-iterations 5 \\
            --attack-options "model=openai/gpt-4o-mini"
    """
    attack = load_module_from_path(args.module, "attacks")

    try:
        entry = json.loads(base64.b64decode(args.input).decode("utf-8"))
    except (json.JSONDecodeError, base64.binascii.Error):
        raise ValueError("Attack input must be a valid base64-encoded JSON entry.")
    
    missing_entry_args = []
    for arg in ["id", "long_id", "content", "content_type", "judge_name", "judge_args"]:
        if arg not in entry:
            missing_entry_args.append(arg)
    
    if len(missing_entry_args) > 0:
        raise ValueError(f"Missing required entry arguments for attack: {', '.join(missing_entry_args)}")

    attack_args = {}

    target = load_module_from_path(args.target, "targets")
    attack_args["target_module"] = target

    attack_args["max_iterations"] = args.max_iterations
    attack_args["call_judge"] = call_judge

    if args.attack_options:
        attack_args["attack_options"] = args.attack_options

    response = attack.attack(entry, **attack_args)

    print(f"[{attack.__class__.__name__}] Attack Response: {str(response) if response else 'No response'}")

def debug_module_provider(args):
    """
    Debug a provider by sending a single prompt and printing the response.

    The -m / --module argument accepts the same provider/model identifier format
    used throughout Spikee (e.g. "openai/gpt-4o", "bedrock/claude45-haiku",
    "google/gemini-2.5-flash"). Use `spikee list providers` to see available options.

    CLI args used:
        -m / --module          Provider/model identifier (e.g. "openai/gpt-4o")
        -i / --input           Input prompt to send to the provider
        --max-tokens           Maximum tokens for the response
        --temperature          Sampling temperature

    Example:
        spikee debug module providers -m "openai/gpt-4o" -i "What is 2+2?"
    """
    provider_args = {}

    if args.module is None:
        raise ValueError("Providers require a model argument to specify the LLM model to use.")

    provider_args["max_tokens"] = args.max_tokens
    provider_args["temperature"] = args.temperature

    provider = get_llm(
        options=args.module,
        **provider_args,
    )

    if provider is None:
        raise ValueError("No provider could be loaded. Please check the module name and ensure it is a valid LLM provider.")

    response = provider.invoke(args.input)

    response = response.content

    print(f"[{provider.__class__.__name__}] Provider Response: {get_content(response) if response else 'No response'}")

