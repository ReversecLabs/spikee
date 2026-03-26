from enum import Enum
import os
import inspect
import json
from re import I
import time
from collections import defaultdict
from typing import Union, List
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm

from .utilities.files import read_jsonl_file, read_toml_file, write_jsonl_file
from .utilities.modules import load_module_from_path
from .utilities.tags import validate_tag


class EntryType(Enum):
    DOCUMENT = "document"
    SUMMARY = "summarization"
    QA = "qna"
    ATTACK = "attack"


class Entry():
    """Class representing a single dataset entry, with methods to convert to output formats."""

    def __init__(
        self,

        entry_type: EntryType,

        entry_id,
        base_id,
        jailbreak_id,
        instruction_id,
        prefix_id,
        suffix_id,

        text,
        entry_text,
        system_message,
        payload,
        lang,

        plugin_suffix,
        plugin_name,

        judge_name,
        judge_args,

        position,
        jailbreak_type,
        instruction_type,
        injection_pattern,
        spotlighting_data_markers,

        exclude_from_transformations_regex=None,

        # Extras
        steering_keywords=None,
    ):
        self.entry_type = entry_type

        self.id = entry_id
        self.base_id = base_id
        self.jailbreak_id = jailbreak_id
        self.instruction_id = instruction_id
        self.prefix_id = prefix_id
        self.suffix_id = suffix_id

        self.text = text
        self.entry_text = entry_text
        self.system_message = system_message
        self.payload = payload
        self.lang = lang if lang else "en"

        self.plugin_suffix = plugin_suffix
        self.plugin_name = plugin_name

        self.judge_name = judge_name
        self.judge_args = judge_args

        self.position = position
        self.jailbreak_type = jailbreak_type
        self.instruction_type = instruction_type
        self.injection_pattern = injection_pattern
        self.spotlighting_data_markers = spotlighting_data_markers

        self.exclude_from_transformations_regex = exclude_from_transformations_regex

        if self.entry_type == EntryType.ATTACK:
            self.long_id = f"{base_id}" + self.plugin_suffix

        else:
            self.long_id = f"{self.entry_type.value}_{base_id}_{jailbreak_id}_{instruction_id}_{position}{plugin_suffix}"

        match self.entry_type:
            case EntryType.SUMMARY:
                self.text = f"Summarize the following document:\n{self.text}"

            case EntryType.QA:
                self.text = f"Given this document:\n{self.text}\nAnswer the following question: {self.entry_text.get('question', '')}"

        # Extras
        self.steering_keywords = steering_keywords

    def to_entry(self):
        """Converts the Entry object to a dictionary format suitable for output."""
        entry = {
            "id": self.id,
            "long_id": self.long_id,
            "text": self.text,
            "judge_name": self.judge_name,
            "judge_args": self.judge_args,
            "injected": "true",
            "task_type": self.entry_type.value,
            "jailbreak_type": self.jailbreak_type,
            "instruction_type": self.instruction_type,
            "document_id": self.base_id,
            "position": self.position,
            "spotlighting_data_markers": self.spotlighting_data_markers,
            "injection_delimiters": self.injection_pattern,
            "lang": self.lang,
            "prefix_id": self.prefix_id,
            "suffix_id": self.suffix_id,
            "system_message": self.system_message,
            "plugin": self.plugin_name,
            "payload": self.payload,
            "exclude_from_transformations_regex": self.exclude_from_transformations_regex,
        }

        if self.prefix_id:
            entry["long_id"] += f"-p{self.prefix_id}"
        if self.suffix_id:
            entry["long_id"] += f"-s{self.suffix_id}"
        if self.system_message:
            entry["long_id"] += "-sys"
        if self.steering_keywords:
            entry["steering_keywords"] = self.steering_keywords

        # Add the ideal answer or summary to the entry based on the entry type
        match self.entry_type:
            case EntryType.SUMMARY:
                entry["ideal_summary"] = self.entry_text.get("ideal_summary", "")

            case EntryType.QA:
                entry["ideal_answer"] = self.entry_text.get("ideal_answer", "")

        return entry

    def to_attack(self):
        """Converts the Entry object to a dictionary format suitable for standalone attacks."""
        attack = {
            "id": self.long_id,
            "long_id": self.long_id,
            "text": self.text,
            "judge_name": self.judge_name,
            "judge_args": self.judge_args,
            "injected": "true",
            "jailbreak_type": self.jailbreak_type,
            "instruction_type": self.instruction_type,
            "task_type": None,
            "document_id": None,
            "position": None,
            "spotlighting_data_markers": None,
            "injection_delimiters": None,
            "lang": self.lang,
            "prefix_id": self.prefix_id,
            "suffix_id": self.suffix_id,
            "payload": self.payload,
            "plugin": self.plugin_name,
            "exclude_from_transformations_regex": self.exclude_from_transformations_regex,
        }

        if self.steering_keywords:
            attack["steering_keywords"] = self.steering_keywords

        return attack


# region resolve file
def resolve_seed_folder(seed_folder_name):
    """
    Return the absolute path to the seed folder, searching local workspace first,
    then built-in package data.
    """
    # local path
    local_path = os.path.join(os.getcwd(), seed_folder_name)
    if os.path.isdir(local_path):
        return local_path

    # built-in path
    builtin_path = os.path.join(os.path.dirname(__file__), "data", seed_folder_name)
    if os.path.isdir(builtin_path):
        return builtin_path

    # Fallback: raise error
    raise FileNotFoundError(
        f"Seed folder '{seed_folder_name}' not found "
        f"in local datasets/ or in built-in spikee/data/"
    )


def resolve_base_inputs_path(seed_folder: str) -> Path:
    base = Path(seed_folder) / "base_user_inputs.jsonl"
    if base.exists():
        return base
    legacy = Path(seed_folder) / "base_documents.jsonl"
    if legacy.exists():
        print(
            "NOTICE: base_user_inputs.jsonl not found. Using legacy base_documents.jsonl."
        )
        return legacy
    raise FileNotFoundError(
        "No base_user_inputs.jsonl or base_documents.jsonl found in seed folder."
    )


def resolve_standalone_inputs_path(seed_folder: str):
    cur = Path(seed_folder) / "standalone_user_inputs.jsonl"
    if cur.exists():
        return cur
    legacy = Path(seed_folder) / "standalone_attacks.jsonl"
    if legacy.exists():
        print(
            "NOTICE: standalone_user_inputs.jsonl not found. Using legacy standalone_attacks.jsonl."
        )
        return legacy
    raise FileNotFoundError(
        "No standalone_user_inputs.jsonl or standalone_attacks.jsonl found in seed folder "
        "(required by --include-standalone-inputs)."
    )


# endregion


# region dataset builders
def insert_jailbreak(document, combined_text, position, injection_pattern, placeholder):
    """
    Inserts the combined_text into the document at the specified position
    using the provided injection_pattern. The pattern must contain the
    placeholder 'INJECTION_PAYLOAD'.
    """
    if "INJECTION_PAYLOAD" not in injection_pattern:
        raise ValueError(
            "Injection pattern must contain 'INJECTION_PAYLOAD' placeholder."
        )
    injected_text = injection_pattern.replace("INJECTION_PAYLOAD", combined_text)

    # if there is an explicit placeholder, replace it with the injected text
    # and ignore any explicit position
    if placeholder:
        return document.replace(placeholder, injected_text)

    if position == "start":
        return f"{injected_text}{document}"
    elif position == "middle":
        mid_point = len(document) // 2
        insert_index = find_nearest_whitespace(document, mid_point)
        return f"{document[:insert_index]}{injected_text}{document[insert_index:]}"
    elif position == "end":
        return f"{document}{injected_text}"
    else:
        raise ValueError(f"Invalid position: {position}")


def find_nearest_whitespace(text, index):
    """
    Finds the nearest whitespace character to the given index in the text.
    Returns the index of that whitespace character (or original index if none found).
    """
    forward_index = text.find(" ", index)
    backward_index = text.rfind(" ", 0, index)

    if forward_index == -1 and backward_index == -1:
        return index
    elif forward_index == -1:
        return backward_index
    elif backward_index == -1:
        return forward_index
    else:
        return (
            forward_index
            if abs(forward_index - index) < abs(index - backward_index)
            else backward_index
        )


def get_system_message(system_message_config, spotlighting_data_marker=None):
    """
    Retrieves the appropriate system message from the system_message_config
    based on a given spotlighting data marker. Falls back to 'default' if no
    exact match is found.
    """
    if system_message_config is None:
        return None

    default = None

    for config in system_message_config["configurations"]:
        if config["spotlighting_data_markers"] == spotlighting_data_marker:
            return config["system_message"]

        if config["spotlighting_data_markers"] == "default":
            default = config["system_message"]

    return default

# endregion


# region plugins
def load_plugins(plugin_names):
    """
    For each plugin name, try:
      1) <cwd>/plugins/<name>.py
      2) built-in package plugin (spikee.plugins.<name>)
    If found, dynamically import and return it.
    """
    plugins = []

    for name in plugin_names:
        name = parse_plugin_piping(name)  # Handle plugin piping syntax

        if isinstance(name, str):
            try:
                plugins.append((name, load_module_from_path(name, "plugins")))
            except (ImportError) as e:
                print(e)
                exit(1)

        else:  # If it's a plugin pipe, load each sub-plugin and store as a list
            plugin_pipe = []
            for sub_name in name:
                try:
                    plugin_pipe.append((sub_name, load_module_from_path(sub_name, "plugins")))
                except (ImportError) as e:
                    print(e)
                    exit(1)

            plugins.append(('~'.join(name), plugin_pipe))

    return plugins


def parse_plugin_piping(plugin: str) -> Union[str, List[str], None]:
    """
    Parses a plugin piping string like "plugin1|plugin2|plugin3" into a list of plugin modules.
    Each plugin is loaded using the load_plugins function.
    """
    if not plugin:
        return None

    if "|" in plugin:
        plugin_names = [name.strip() for name in plugin.split("|")]
        return plugin_names

    else:
        return plugin.strip()


def parse_plugin_options(plugin_options_str):
    """
    Parse plugin options string like "plugin1:option1;plugin2:option2"
    Returns dict mapping plugin name to option string.
    """
    if not plugin_options_str:
        return {}

    options_map = {}
    pairs = plugin_options_str.split(";")
    for pair in pairs:
        if ":" in pair:
            plugin_name, option = pair.split(":", 1)
            options_map[plugin_name.strip()] = option.strip()
    return options_map


def get_plugin_variants(plugin_module, plugin_option):
    """
    Obtains the number of variants a plugin will produce based on its options.
    """

    if hasattr(plugin_module, "get_variants"):
        return plugin_module.get_variants(plugin_option)

    else:
        return 1


def apply_plugin(
    plugin_name, plugin_module, text, exclude_patterns=None, plugin_option_map=None
):
    """
    Applies a plugin module's transform function to the given text if available.
    """

    plugins = []

    if "~" in plugin_name:  # Handle plugin piping syntax
        plugins.extend(plugin_module)
    else:
        plugins.append((plugin_name, plugin_module))

    text = [text]

    for name, module in plugins:
        new_text = []
        if hasattr(module, "transform"):
            # Check if the plugin's transform function accepts plugin_option parameter

            sig = inspect.signature(module.transform)
            params = sig.parameters

            for t in text:

                if "plugin_option" in params:
                    res = module.transform(t, exclude_patterns, plugin_option_map.get(name) if plugin_option_map else None)
                else:
                    # Older plugin without plugin_option support
                    res = module.transform(t, exclude_patterns)

                if isinstance(res, str):
                    new_text.append(res)
                elif isinstance(res, list):
                    new_text.extend(res)

            text = new_text

        else:
            print(f"Plugin '{plugin_name}' does not have a 'transform' function.")

    return text


def parse_exclude_patterns(jailbreak, instruction):
    """
    Parses the 'exclude_from_transformations_regex' field from both jailbreak and instruction,
    combining them into a single list of patterns to exclude.
    """
    exclude_patterns = set()

    for item in [jailbreak, instruction]:
        if "exclude_from_transformations_regex" in item:
            value = item["exclude_from_transformations_regex"]
            if isinstance(value, list):
                exclude_patterns.update(value)
            else:
                exclude_patterns.add(value)

    return list(exclude_patterns) if exclude_patterns else None
# endregion


def process_standalone_attacks(
    standalone_attacks,
    dataset,
    entry_id,
    adv_prefixes=None,
    adv_suffixes=None,
    plugins=None,
    plugin_options_map=None,
    plugin_only=False,
):
    """
    Processes standalone attacks and appends them to the dataset.
    If plugins are provided, applies them to each standalone attack.
    Returns the updated dataset and the next entry_id.
    """

    if plugin_only:
        plugins = [] + plugins if plugins else []  # Only include plugin variations, no base attack

    else:
        plugins = [(None, None)] + plugins if plugins else [(None, None)]  # [(plugin name, plugin module)] with a dummy entry for no plugin

    prefixes = adv_prefixes
    suffixes = adv_suffixes

    # Obtain plugin options and calculate total variants
    plugin_variants = {}
    if plugins:
        for plugin_name, plugin_module in plugins:
            if plugin_name is None:
                plugin_variants[plugin_name] = 1

            elif "~" in plugin_name:  # Plugin Pipe
                sub_plugins = plugin_name.split("~")
                total_variants = 1
                for sub_plugin in sub_plugins:
                    sub_plugin_option = (
                        plugin_options_map.get(sub_plugin)
                        if plugin_options_map
                        else None
                    )
                    variants = get_plugin_variants(plugin_module[1], sub_plugin_option)
                    total_variants *= variants
                plugin_variants[plugin_name] = total_variants

            else:  # Standalone Plugin
                plugin_option = (
                    plugin_options_map.get(plugin_name)
                    if plugin_options_map
                    else None
                )
                plugin_variants[plugin_name] = get_plugin_variants(plugin_module, plugin_option)

    # Calculate total entries for progress bar
    total_entries = len(standalone_attacks) * (sum(plugin_variants.values() or [1]) + (1 if not plugin_only else 0))
    bar_standalone = tqdm(
        total=total_entries,
        desc="Standalone Attacks",
        initial=1
    )

    for attack in standalone_attacks:
        # If no judge_name, fallback
        if "judge_name" not in attack:
            attack["judge_name"] = "canary"
        if "judge_args" not in attack:
            attack["judge_args"] = attack.get("canary", "")

        # Get the base attack text and exclude patterns
        attack_text = attack["text"]
        exclude_patterns = attack.get("exclude_from_transformations_regex", None)

        # Get permutations for prefixes and suffixes
        combined_texts = []  # Stored all permutations of prefixes/suffixes and plugin outputs for an attack entry.
        fix_permutations = [(prefix, suffix) for prefix in prefixes for suffix in suffixes]

        # Apply plugins to the base attack text
        for plugin_name, plugin_module in plugins:
            plugin_texts = apply_plugin(
                plugin_name,
                plugin_module,
                attack_text,
                exclude_patterns,
                plugin_options_map
            ) if plugin_name else attack_text

            # Ensure plugin_texts is a list of variations. If it's a single string, convert it to a list with one element.
            if not isinstance(plugin_texts, list):
                plugin_texts = [plugin_texts]

            # Combine each plugin variation with each prefix/suffix permutation and add to combined_texts
            for plugin_index, plugin_text in enumerate(plugin_texts, start=1):
                for prefix, suffix in fix_permutations:
                    combined_texts.append({
                        "text": (prefix.get("prefix", "") + " " if prefix else "") + plugin_text + (" " + suffix.get("suffix", "") if suffix else ""),
                        "prefix_id": prefix.get("id", None) if prefix else None,
                        "suffix_id": suffix.get("id", None) if suffix else None,
                        "plugin_name": plugin_name,
                        "plugin_suffix": f"_{plugin_name}-{plugin_index}" if plugin_name else ""
                    })

        for combined_text in combined_texts:
            entry = Entry(
                entry_type=EntryType.ATTACK,
                entry_id=entry_id,
                base_id=attack["id"],
                jailbreak_id=None,
                instruction_id=None,
                prefix_id=combined_text.get("prefix_id", None),
                suffix_id=combined_text.get("suffix_id", None),
                text=combined_text["text"],
                entry_text={},
                system_message=None,
                payload=combined_text["text"],
                lang=attack.get("lang", "en"),

                plugin_suffix=combined_text.get("plugin_suffix", ""),
                plugin_name=combined_text.get("plugin_name", None),

                judge_name=attack["judge_name"],
                judge_args=attack["judge_args"],

                position=None,
                jailbreak_type=None,
                instruction_type=None,
                injection_pattern=None,
                spotlighting_data_markers=None,

                exclude_from_transformations_regex=exclude_patterns,

                steering_keywords=attack.get("steering_keywords", None),
            ).to_attack()

            dataset.append(entry)
            entry_id += 1
            bar_standalone.update(1)

    bar_standalone.close()
    return dataset, entry_id


def generate_variations(
    base_docs,
    jailbreaks,
    instructions,
    positions,
    injection_delimiters,
    spotlighting_data_markers_list,
    plugins,
    adv_prefixes=None,
    adv_suffixes=None,
    output_format="full-prompt",
    match_languages=False,
    system_message_config=None,
    plugin_options_map=None,
    plugin_only=False,
):
    """
    Generates dataset variations from the given base documents, jailbreaks,
    instructions, injection positions, delimiters, data markers, and plugins.
    Returns the dataset and the last used entry_id.
    """
    dataset = []
    entry_id = 1

    if plugin_only:
        plugins = [] + plugins if plugins else []  # Only include plugin variations, no base attack

    else:
        plugins = [(None, None)] + plugins if plugins else [(None, None)]  # [(plugin name, plugin module)] with a dummy entry for no plugin

    prefixes = adv_prefixes
    suffixes = adv_suffixes

    # Define output format specific entry types
    match output_format:
        case "full-prompt":
            output_format = [EntryType.SUMMARY, EntryType.QA]

        case "user-input":
            output_format = [EntryType.DOCUMENT]

        case _:
            output_format = ["burp"]

    # Obtain plugin options and calculate total variants for progress bar
    plugin_variants = {}
    if plugins:
        for plugin_name, plugin_module in plugins:
            if plugin_name is None:
                plugin_variants[plugin_name] = 1

            elif "~" in plugin_name:  # Plugin Pipe
                sub_plugins = plugin_name.split("~")
                total_variants = 1
                for sub_plugin in sub_plugins:
                    sub_plugin_option = (
                        plugin_options_map.get(sub_plugin)
                        if plugin_options_map
                        else None
                    )
                    variants = get_plugin_variants(plugin_module[1], sub_plugin_option)
                    total_variants *= variants
                plugin_variants[plugin_name] = total_variants

            else:  # Standalone Plugin
                plugin_option = (
                    plugin_options_map.get(plugin_name)
                    if plugin_options_map
                    else None
                )
                plugin_variants[plugin_name] = get_plugin_variants(plugin_module, plugin_option)

    # Calculate total entries for progress bar
    total_entries = (
        len(base_docs)
        * len(jailbreaks)
        * len(instructions)
        * len(positions)
        * len(injection_delimiters)
        * len(spotlighting_data_markers_list)
        * len(suffixes)
        * sum(plugin_variants.values() or [1]) + 1 if not plugin_only else 0
    )
    bar_variations = tqdm(
        total=total_entries, desc="Variations", initial=0
    )

    for base_doc in base_docs:
        base_id = base_doc["id"]
        document = base_doc["document"]
        placeholder = base_doc.get("placeholder", "")

        # Define entry type specific text
        question = base_doc.get("question", "")
        ideal_answer = base_doc.get("ideal_answer", "")
        ideal_summary = base_doc.get("ideal_summary", "")
        entry_text = {}
        if question != "":
            entry_text["question"] = question

        if ideal_answer != "":
            entry_text["ideal_answer"] = ideal_answer

        if ideal_summary != "":
            entry_text["ideal_summary"] = ideal_summary

        # If the current document has a placeholder attribute, it means the user
        # want the payload to be inserted into a fixed location, so we override
        # the inject positions for this document
        insert_positions = ["fixed"] if placeholder else positions

        for jailbreak in jailbreaks:
            jailbreak_id = jailbreak["id"]
            jailbreak_text = jailbreak["text"]
            jailbreak_type = jailbreak.get("jailbreak_type", "")
            jailbreak_lang = jailbreak.get("lang", "en")

            for instruction in instructions:
                instruction_id = instruction["id"]
                instruction_text = instruction["instruction"]
                instruction_type = instruction.get("instruction_type", "")
                instruction_lang = instruction.get("lang", "en")
                instruction_steering_keywords = instruction.get("steering_keywords", None)

                judge_name = instruction.get("judge_name", "canary")
                judge_args = instruction.get(
                    "judge_args", instruction.get("canary", "")
                )

                # If match_languages is enabled, skip if jailbreak and instruction languages do not match
                if match_languages and jailbreak_lang != instruction_lang:
                    total_entries -= (
                        len(positions)
                        * len(injection_delimiters)
                        * len(spotlighting_data_markers_list)
                        * len(suffixes)
                        * sum(plugin_variants.values() or [1])
                    )
                    bar_variations.total = total_entries
                    bar_variations.refresh()
                    continue

                # Combines jailbreak and instruction texts
                # Instruction is placed into jailbreak at <INSTRUCTION> placeholder
                combined_base = jailbreak_text.replace("<INSTRUCTION>", instruction_text)
                lang = instruction_lang

                # Create plugin / transformation regex exclusion lists
                local_exclude = parse_exclude_patterns(jailbreak, instruction)

                # Apply all plugin, prefix and suffix combinations for the combined_base text (jailbreak + instruction)
                # Applied with --plugins, --include-prefixes and --include-suffixes, by default None
                combined_texts = []
                fix_permutations = [(prefix, suffix) for prefix in prefixes for suffix in suffixes]

                for plugin_name, plugin_module in plugins:
                    plugin_texts = apply_plugin(
                        plugin_name,
                        plugin_module,
                        combined_base,
                        local_exclude,
                        plugin_options_map
                    ) if plugin_name else combined_base

                    # Ensure plugin_texts is a list of variations. If it's a single string, convert it to a list with one element.
                    if not isinstance(plugin_texts, list):
                        plugin_texts = [plugin_texts]

                    for plugin_index, plugin_text in enumerate(plugin_texts, start=1):
                        for prefix, suffix in fix_permutations:
                            prefix_lang = prefix.get("lang", None) if prefix else None
                            suffix_lang = suffix.get("lang", None) if suffix else None

                            if match_languages and ((prefix_lang and prefix_lang != lang) or (suffix_lang and suffix_lang != lang)):
                                continue

                            combined_texts.append({
                                "text": (prefix.get("prefix", "") + " " if prefix else "") + plugin_text + (" " + suffix.get("suffix", "") if suffix else ""),
                                "prefix_id": prefix.get("id", None) if prefix else None,
                                "suffix_id": suffix.get("id", None) if suffix else None,
                                "plugin_name": plugin_name,
                                "plugin_suffix": f"_{plugin_name}-{plugin_index}" if plugin_name else ""
                            })

                for combined_text in combined_texts:
                    # Applied with --positions, by default ['end']
                    for position in insert_positions:

                        # Applied with --injection-delimiters, by default "\nINJECTION_PAYLOAD\n"
                        for injection_pattern in injection_delimiters:
                            # Combine document with suffix_combined_text (jailbreak + instruction + suffix)
                            # suffix_combined_text is inserted into the injection_pattern at 'INJECTION_PAYLOAD' placeholder
                            # Document Placeholder: Injection is placed into document placeholder
                            # Otherwise: Injection is placed into document at position (start, middle, end)
                            injected_doc = insert_jailbreak(
                                document,
                                combined_text["text"],
                                position,
                                injection_pattern,
                                placeholder,
                            )

                            for entry_type in output_format:
                                if entry_type == "burp":
                                    burp_payload_encoded = json.dumps(injected_doc)[1:-1]
                                    dataset.append(burp_payload_encoded)

                                else:
                                    # Applied with --spotlighting-data-markers, by default "\nDOCUMENT\n"
                                    for spotlighting_data_marker in spotlighting_data_markers_list:

                                        # Applied with --include-system-message flag - gets system message associated with spotlighting data marker
                                        system_message = get_system_message(
                                            system_message_config, spotlighting_data_marker
                                        )

                                        # Combines injected document with spotlighting data marker, for full-prompt entries
                                        wrapped_document = (
                                            injected_doc
                                            if spotlighting_data_marker == "none"
                                            else spotlighting_data_marker.replace(
                                                "DOCUMENT", injected_doc
                                            )
                                        )

                                        entry = Entry(
                                            entry_type=entry_type,
                                            entry_id=entry_id,
                                            base_id=base_id,
                                            jailbreak_id=jailbreak_id,
                                            instruction_id=instruction_id,
                                            prefix_id=combined_text.get("prefix_id", None),
                                            suffix_id=combined_text.get("suffix_id", None),

                                            text=injected_doc if entry_type == EntryType.DOCUMENT else wrapped_document,
                                            entry_text=entry_text,
                                            system_message=system_message,
                                            payload=combined_text.get("text", None),
                                            lang=lang,

                                            plugin_suffix=combined_text.get("plugin_suffix", ""),
                                            plugin_name=combined_text.get("plugin_name", None),

                                            judge_args=judge_args,
                                            judge_name=judge_name,

                                            position=position,
                                            jailbreak_type=jailbreak_type,
                                            instruction_type=instruction_type,
                                            injection_pattern=injection_pattern,
                                            spotlighting_data_markers=spotlighting_data_marker,

                                            exclude_from_transformations_regex=local_exclude,
                                        ).to_entry()
                                        dataset.append(entry)
                                        entry_id += 1
                                        bar_variations.update(1)

    return dataset, entry_id


def generate_dataset(args):
    """
    Main entry point for generating the dataset. Loads files, filters content,
    generates variations, writes results to disk, and prints stats.
    """
    seed_folder = resolve_seed_folder(args.seed_folder)
    output_format = args.format
    include_system_message = args.include_system_message
    plugin_options_map = parse_plugin_options(args.plugin_options)

    # Tags
    tag = args.tag
    if tag:
        is_valid_tag, tag_error = validate_tag(tag)
        if not is_valid_tag:
            print(f"Error: Invalid tag: {tag_error}")
            return

    # Pattern Lists
    injection_delimiters_input = args.injection_delimiters
    spotlighting_data_markers_input = args.spotlighting_data_markers
    languages_input = args.languages
    match_languages = args.match_languages
    instruction_filter_input = args.instruction_filter
    jailbreak_filter_input = args.jailbreak_filter
    include_fixes = [fix.strip() for fix in args.include_fixes.split(",")] if args.include_fixes else []

    # legacy cli arg
    if args.include_suffixes and "adv_suffixes" not in include_fixes:
        include_fixes.append("adv_suffixes")

    injection_delimiters = [
        delim.encode("utf-8").decode("unicode_escape")
        for delim in injection_delimiters_input.split(",")
    ]
    spotlighting_data_markers_list = [
        marker.encode("utf-8").decode("unicode_escape")
        for marker in spotlighting_data_markers_input.split(",")
    ]

    if languages_input:
        languages = [lang.strip() for lang in languages_input.split(",")]
    else:
        languages = None

    if instruction_filter_input:
        instruction_filters = [i.strip() for i in instruction_filter_input.split(",")]
    else:
        instruction_filters = None

    if jailbreak_filter_input:
        jailbreak_filters = [j.strip() for j in jailbreak_filter_input.split(",")]
    else:
        jailbreak_filters = None

    # Get Base Inputs
    base_file = resolve_base_inputs_path(seed_folder)
    base_documents_file = str(base_file)

    # Get Additional Files
    jailbreaks_file = os.path.join(seed_folder, "jailbreaks.jsonl")
    instructions_file = os.path.join(seed_folder, "instructions.jsonl")
    adv_prefixes_file = os.path.join(seed_folder, "adv_prefixes.jsonl")
    adv_suffixes_file = os.path.join(seed_folder, "adv_suffixes.jsonl")
    system_messages = os.path.join(seed_folder, "system_messages.toml")

    # Validate Files
    required_files = [base_documents_file, jailbreaks_file, instructions_file]

    for fix in include_fixes:
        if fix == "adv_prefixes":
            required_files.append(adv_prefixes_file)

        elif fix == "adv_suffixes":
            required_files.append(adv_suffixes_file)

        elif fix.startswith("prefixes="):
            _, prefix_file_name = fix.split("=", 1)
            required_files.append(os.path.abspath(prefix_file_name))

        elif fix.startswith("suffixes="):
            _, suffix_file_name = fix.split("=", 1)
            required_files.append(os.path.abspath(suffix_file_name))

    for file_path in required_files:
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            return

    # Read/Ingest Files
    base_docs = read_jsonl_file(base_documents_file)
    jailbreaks = read_jsonl_file(jailbreaks_file)
    instructions = read_jsonl_file(instructions_file)

    # Ingest prefixes and suffixes
    adv_prefixes = []
    adv_suffixes = []

    prefix_none_flag = True
    suffix_none_flag = True

    custom_fix = 1
    for fix in include_fixes:
        if fix == "adv_prefixes":
            adv_prefixes += read_jsonl_file(adv_prefixes_file)

        elif fix == "adv_suffixes":
            adv_suffixes += read_jsonl_file(adv_suffixes_file)

        elif fix.startswith("prefixes="):
            _, prefix_file_name = fix.split("=", 1)
            adv_prefixes += read_jsonl_file(os.path.abspath(prefix_file_name))

        elif fix.startswith("suffixes="):
            _, suffix_file_name = fix.split("=", 1)
            adv_suffixes += read_jsonl_file(os.path.abspath(suffix_file_name))

        elif fix.startswith("prefix="):
            _, prefix = fix.split("=", 1)
            adv_prefixes.append({
                "id": f"custom-{custom_fix}",
                "prefix": prefix
            })
            custom_fix += 1

        elif fix.startswith("suffix="):
            _, suffix = fix.split("=", 1)
            adv_suffixes.append({
                "id": f"custom-{custom_fix}",
                "suffix": suffix
            })
            custom_fix += 1

        elif fix == "none_prefix":
            prefix_none_flag = False

        elif fix == "none_suffix":
            suffix_none_flag = False

    if prefix_none_flag:
        adv_prefixes = [None] + adv_prefixes

    if suffix_none_flag:
        adv_suffixes = [None] + adv_suffixes

    # Process Jailbreaks
    processed_jailbreaks = []
    for jb in jailbreaks:
        jb["lang"] = jb.get("lang", "en")
        jb_type = jb.get("jailbreak_type", "")

        # Jailbreak does not match user-defined language or jb filter, skip.
        if (languages and jb["lang"] not in languages) or (
            jailbreak_filters and jb_type not in jailbreak_filters
        ):
            continue

        processed_jailbreaks.append(jb)
    jailbreaks = processed_jailbreaks

    # Process Instructions
    processed_instructions = []
    for instr in instructions:
        instr["lang"] = instr.get("lang", "en")
        instr_type = instr.get("instruction_type", "")

        # If no judge_name, fallback to 'canary'
        if "judge_name" not in instr:
            instr["judge_name"] = "canary"

        # If no judge_args, fallback to any 'canary' string or empty (compatibility with v0.1)
        if "judge_args" not in instr:
            instr["judge_args"] = instr.get("canary", "")

        # Instruction does not match user-defined language or instruction filter, skip.
        if (
            (languages and instr["lang"] not in languages)
            or instruction_filters
            and instr_type not in instruction_filters
        ):
            continue

        processed_instructions.append(instr)
    instructions = processed_instructions

    # Load user-defined plugins
    plugins = load_plugins(args.plugins)

    # Load system message config
    system_message_config = (
        read_toml_file(system_messages) if include_system_message else None
    )

    try:
        # Generate Dataset
        dataset, entry_id = generate_variations(
            base_docs,
            jailbreaks,
            instructions,
            args.positions,
            injection_delimiters,
            spotlighting_data_markers_list,
            plugins,
            adv_prefixes=adv_prefixes,
            adv_suffixes=adv_suffixes,
            output_format=output_format,
            match_languages=match_languages,
            system_message_config=system_message_config,
            plugin_options_map=plugin_options_map,
            plugin_only=args.plugin_only,
        )

        # Generate Standalone Attacks
        if getattr(args, "include_standalone_inputs", False):
            standalone_file = resolve_standalone_inputs_path(seed_folder)
            standalone_inputs = read_jsonl_file(str(standalone_file))
            dataset, entry_id = process_standalone_attacks(
                standalone_inputs,
                dataset,
                entry_id,
                adv_prefixes=adv_prefixes,
                adv_suffixes=adv_suffixes,
                plugins=plugins if args.plugins else None,
                plugin_options_map=plugin_options_map,
                plugin_only=args.plugin_only,
            )
    except ImportError as e:
        print(f"Missing dependency: {e}")
        exit(1)

    timestamp = int(time.time())
    seed_folder_name = os.path.basename(os.path.normpath(seed_folder))
    output_file_name = f"{seed_folder_name.replace('seeds-', '')}-{output_format}"

    if include_system_message:
        output_file_name += "-sys"

    if tag:
        output_file_name += f"-{tag}"

    output_file_path = os.path.join("datasets", output_file_name)

    if output_format == "burp":
        output_file_path += f"-dataset-{timestamp}.txt"
        os.makedirs("datasets", exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            for payload in dataset:
                f.write(payload + "\n")
    else:
        output_file_path += f"-dataset-{timestamp}.jsonl"
        os.makedirs("datasets", exist_ok=True)
        write_jsonl_file(output_file_path, dataset)

    print(f"Dataset generated and saved to {output_file_path}")

    stats = {
        "total_entries": len(dataset),
        "by_jailbreak_type": defaultdict(int),
        "by_instruction_type": defaultdict(int),
        "by_lang": defaultdict(int),
        "by_task_type": defaultdict(int),
        "by_suffix_id": defaultdict(int),
        "by_plugin_id": defaultdict(int),
        "by_prefix_id": defaultdict(int),
    }

    for entry in dataset:
        if isinstance(entry, str):
            continue
        jb_type = entry.get("jailbreak_type") or "None"
        instr_type = entry.get("instruction_type") or "None"
        lang = entry.get("lang", "en")
        task_type = entry.get("task_type") or "None"
        prefix_id = entry.get("prefix_id") or "None"
        suffix_id = entry.get("suffix_id") or "None"
        plugin_id = entry.get("plugin") or "None"

        stats["by_jailbreak_type"][jb_type] += 1
        stats["by_instruction_type"][instr_type] += 1
        stats["by_lang"][lang] += 1
        stats["by_task_type"][task_type] += 1
        stats["by_prefix_id"][prefix_id] += 1
        stats["by_suffix_id"][suffix_id] += 1
        stats["by_plugin_id"][plugin_id] += 1

    print("\n=== Dataset Statistics ===")
    print(f"Total Entries: {stats['total_entries']}")

    def print_stats(title, data):
        print(f"\nBreakdown by {title}:")
        table = []
        for key, count in data.items():
            display_key = key if key else "None"
            table.append([display_key, count])
        print(tabulate(table, headers=[title, "Count"]))

    print_stats("Jailbreak Type", stats["by_jailbreak_type"])
    print_stats("Instruction Type", stats["by_instruction_type"])
    print_stats("Language", stats["by_lang"])
    print_stats("Task Type", stats["by_task_type"])
    print_stats("Prefix ID", stats["by_prefix_id"])
    print_stats("Suffix ID", stats["by_suffix_id"])
    print_stats("Plugin ID", stats["by_plugin_id"])


def generate_plugin(args):
    text = args.input_string
    exclude_patterns = args.exclude_patterns
    iterations = int(args.iterations)

    plugin_options_map = parse_plugin_options(args.plugin_options)
    plugins = load_plugins(args.plugins)

    if len(plugins) == 0:
        print("No valid plugins found. Please check the plugin names and ensure they are in the correct directory.")
        return

    else:
        print(f"Applying plugins: {[name for name, _ in plugins]} to input string: '{text}' with exclude patterns: {exclude_patterns}")

    counter = 0
    print("----------------------------------------------")
    for plugin_name, module in plugins:

        for _ in range(iterations):
            output = apply_plugin(plugin_name, module, text, exclude_patterns, plugin_options_map)

            if len(output) == 1:
                print(f"{plugin_name}|{counter}: {output[0]}")

            else:
                print(f"{plugin_name}|{counter}:")
                for variant in output:
                    print(f"  - {variant}")

            counter += 1

            print("----------------------------------------------")
