import base64
import json

from spikee.utilities.modules import load_module_from_path
from spikee.utilities.hinting import get_content
from spikee.judge import call_judge
from spikee.utilities.llm import get_llm

def debug_module_target(args):
    target = load_module_from_path(args.module, "targets")

    target_args = {}

    if args.system_message:
        target_args["system_message"] = args.system_message
    
    if args.target_options:
        target_args["target_options"] = args.target_options

    response = target.process_input(args.input, **target_args)

    print(f"[{target.__class__.__name__}] Response: {get_content(response) if response else 'No response'}")

def debug_module_judge(args):
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
    plugin = load_module_from_path(args.module, "plugins")

    plugin_args = {}

    if args.plugin_options:
        plugin_args["plugin_options"] = args.plugin_options

    if args.exclude_patterns:
        plugin_args["exclude_patterns"] = args.exclude_patterns

    response = plugin.transform(args.input, **plugin_args)

    print(f"[{plugin.__class__.__name__}] Plugin Response: {get_content(response) if response else 'No response'}")

def debug_module_attack(args):
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

