import os
import inspect
import json
import time
from collections import defaultdict
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm

from .utilities.files import read_jsonl_file, read_toml_file, write_jsonl_file
from .utilities.modules import load_module_from_path
from .utilities.tags import validate_tag


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


def _create_summary_entry(
    entry_id,
    base_id,
    jailbreak_id,
    instruction_id,
    position,
    plugin_suffix,
    wrapped_document,
    judge_name,
    judge_args,
    ideal_summary,
    jailbreak_type,
    instruction_type,
    injection_pattern,
    spotlighting_data_marker,
    lang,
    suffix_id,
    system_message,
    plugin_name,
    payload,
    exclude_regex_list=None,
    steering_keywords=None,
):
    summary_entry = {
        "id": entry_id,
        "long_id": f"summary_{base_id}_{jailbreak_id}_{instruction_id}_{position}{plugin_suffix}",
        "text": f"Summarize the following document:\n{wrapped_document}",
        "judge_name": judge_name,
        "judge_args": judge_args,
        "injected": "true",
        "ideal_answer": ideal_summary,
        "task_type": "summarization",
        "jailbreak_type": jailbreak_type,
        "instruction_type": instruction_type,
        "document_id": base_id,
        "position": position,
        "spotlighting_data_markers": spotlighting_data_marker,
        "injection_delimiters": injection_pattern,
        "lang": lang,
        "suffix_id": suffix_id,
        "system_message": system_message,
        "plugin": plugin_name,
        "payload": payload,
        "exclude_from_transformations_regex": exclude_regex_list,
    }

    if suffix_id:
        summary_entry["long_id"] += f"-{suffix_id}"
    if system_message:
        summary_entry["long_id"] += "-sys"

    if steering_keywords is not None:
        summary_entry["steering_keywords"] = steering_keywords

    return summary_entry


def _create_qa_entry(
    entry_id,
    base_id,
    jailbreak_id,
    instruction_id,
    position,
    plugin_suffix,
    wrapped_document,
    judge_name,
    judge_args,
    question,
    ideal_answer,
    jailbreak_type,
    instruction_type,
    injection_pattern,
    spotlighting_data_marker,
    lang,
    suffix_id,
    system_message,
    plugin_name,
    payload,
    exclude_regex_list=None,
    steering_keywords=None,
):
    qa_entry = {
        "id": entry_id,
        "long_id": f"qa_{base_id}_{jailbreak_id}_{instruction_id}_{position}{plugin_suffix}",
        "text": f"Given this document:\n{wrapped_document}\nAnswer the following question: {question}",
        "judge_name": judge_name,
        "judge_args": judge_args,
        "injected": "true",
        "ideal_answer": ideal_answer,
        "task_type": "qna",
        "jailbreak_type": jailbreak_type,
        "instruction_type": instruction_type,
        "document_id": base_id,
        "position": position,
        "spotlighting_data_markers": spotlighting_data_marker,
        "injection_delimiters": injection_pattern,
        "lang": lang,
        "suffix_id": suffix_id,
        "system_message": system_message,
        "plugin": plugin_name,
        "payload": payload,
        "exclude_from_transformations_regex": exclude_regex_list,
    }
    if suffix_id:
        qa_entry["long_id"] += f"-{suffix_id}"
    if system_message:
        qa_entry["long_id"] += "-sys"

    if steering_keywords is not None:
        qa_entry["steering_keywords"] = steering_keywords

    return qa_entry


def _create_document_entry(
    entry_id,
    base_id,
    jailbreak_id,
    instruction_id,
    position,
    plugin_suffix,
    modified_document,
    judge_name,
    judge_args,
    jailbreak_type,
    instruction_type,
    injection_pattern,
    spotlighting_data_markers,
    lang,
    suffix_id,
    system_message,
    plugin_name,
    output_format,
    payload,
    exclude_regex_list=None,
    steering_keywords=None,
):
    doc_entry = {
        "id": entry_id,
        "long_id": f"{output_format}_{base_id}_{jailbreak_id}_{instruction_id}_{position}{plugin_suffix}",
        "text": modified_document,
        "judge_name": judge_name,
        "judge_args": judge_args,
        "injected": "true",
        "jailbreak_type": jailbreak_type,
        "instruction_type": instruction_type,
        "task_type": None,
        "document_id": base_id,
        "position": position,
        "spotlighting_data_markers": spotlighting_data_markers,
        "injection_delimiters": injection_pattern,
        "lang": lang,
        "suffix_id": suffix_id,
        "system_message": system_message,
        "plugin": plugin_name,
        "payload": payload,
        "exclude_from_transformations_regex": exclude_regex_list,
    }
    if suffix_id:
        doc_entry["long_id"] += f"-{suffix_id}"
    if system_message:
        doc_entry["long_id"] += "-sys"

    if steering_keywords is not None:
        doc_entry["steering_keywords"] = steering_keywords

    return doc_entry


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
        try:
            plugins.append((name, load_module_from_path(name, "plugins")))
        except (ImportError, ValueError) as e:
            print(
                f"Warning: Plugin '{name}' not found locally or in built-in plugins.: {e}"
            )
    return plugins


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
    plugin_name, plugin_module, text, exclude_patterns=None, plugin_option=None
):
    """
    Applies a plugin module's transform function to the given text if available.
    """

    if hasattr(plugin_module, "transform"):
        # Check if the plugin's transform function accepts plugin_option parameter

        sig = inspect.signature(plugin_module.transform)
        params = sig.parameters

        if "plugin_option" in params:
            return plugin_module.transform(text, exclude_patterns, plugin_option)
        else:
            # Older plugin without plugin_option support
            return plugin_module.transform(text, exclude_patterns)

    print(f"Plugin '{plugin_name}' does not have a 'transform' function.")
    return text


# endregion


def process_standalone_attacks(
    standalone_attacks,
    dataset,
    entry_id,
    plugins=None,
    output_format="full-prompt",
    plugin_options_map=None,
    plugin_only=False,
):
    """
    Processes standalone attacks and appends them to the dataset.
    If plugins are provided, applies them to each standalone attack.
    Returns the updated dataset and the next entry_id.
    """
    plugin_variants = {}
    if plugins:
        for plugin_name, plugin_module in plugins:
            plugin_option = (
                plugin_options_map.get(plugin_name)
                if plugin_options_map
                else None
            )
            plugin_variants[plugin_name] = get_plugin_variants(plugin_module, plugin_option)

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

        # Process the base attack without plugins first
        if not plugin_only:
            entry = {
                "id": entry_id,
                "long_id": attack["id"],
                "text": attack_text,
                "judge_name": attack["judge_name"],
                "judge_args": attack["judge_args"],
                "injected": "true",
                "jailbreak_type": attack.get("jailbreak_type", ""),
                "instruction_type": attack.get("instruction_type", ""),
                "task_type": None,
                "document_id": None,
                "position": None,
                "spotlighting_data_markers": None,
                "injection_delimiters": None,
                "lang": attack.get("lang", "en"),
                "suffix_id": None,
                "payload": attack_text,
                "plugin": None,
                "exclude_from_transformations_regex": exclude_patterns,
            }

            if "steering_keywords" in attack:
                entry["steering_keywords"] = attack["steering_keywords"]

            dataset.append(entry)
            entry_id += 1
            bar_standalone.update(1)

        # Apply plugins if provided
        if plugins:
            for plugin_name, plugin_module in plugins:
                try:
                    # Get option for this specific plugin
                    plugin_option = (
                        plugin_options_map.get(plugin_name)
                        if plugin_options_map
                        else None
                    )

                    # Convert exclude_patterns to a list if it's a string or None
                    exclude_list = []
                    if exclude_patterns:
                        if isinstance(exclude_patterns, list):
                            exclude_list = exclude_patterns
                        else:
                            exclude_list = [exclude_patterns]

                    # Get the transformed text(s) from the plugin
                    plugin_result = apply_plugin(
                        plugin_name,
                        plugin_module,
                        attack_text,
                        exclude_list,
                        plugin_option,
                    )

                    # Ensure the result is a list of variations
                    if not isinstance(plugin_result, list):
                        plugin_result = [plugin_result]

                    # Add each plugin variation as a separate entry
                    for variant_index, plugin_variant in enumerate(
                        plugin_result, start=1
                    ):
                        plugin_entry = {
                            "id": entry_id,
                            "long_id": f"{attack['id']}_{plugin_name}-{variant_index}",
                            "text": plugin_variant,
                            "judge_name": attack["judge_name"],
                            "judge_args": attack["judge_args"],
                            "injected": "true",
                            "jailbreak_type": attack.get("jailbreak_type", ""),
                            "instruction_type": attack.get("instruction_type", ""),
                            "task_type": None,
                            "document_id": None,
                            "position": None,
                            "spotlighting_data_markers": None,
                            "injection_delimiters": None,
                            "lang": attack.get("lang", "en"),
                            "suffix_id": None,
                            "payload": plugin_variant,
                            "plugin": plugin_name,
                            "exclude_from_transformations_regex": exclude_patterns,
                        }

                        if "steering_keywords" in attack:
                            plugin_entry["steering_keywords"] = attack["steering_keywords"]

                        dataset.append(plugin_entry)
                        entry_id += 1

                    if len(plugin_result) < plugin_variants[plugin_name]:
                        total_entries -= plugin_variants[plugin_name] - len(plugin_result)
                        bar_standalone.total = total_entries
                        bar_standalone.refresh()

                    bar_standalone.update(len(plugin_result))
                except Exception as e:
                    print(
                        f"Warning: Plugin '{plugin_name}' failed for standalone attack '{attack['id']}': {e}"
                    )
                    continue

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

    suffixes = [None] + adv_suffixes if adv_suffixes else [None]

    plugin_variants = {}
    if plugins:
        for plugin_name, plugin_module in plugins:
            plugin_option = (
                plugin_options_map.get(plugin_name)
                if plugin_options_map
                else None
            )
            plugin_variants[plugin_name] = get_plugin_variants(plugin_module, plugin_option)

    total_entries = (
        len(base_docs)
        * len(jailbreaks)
        * len(instructions)
        * len(positions)
        * len(injection_delimiters)
        * len(spotlighting_data_markers_list)
        * len(suffixes)
        * sum(plugin_variants.values() or [1]) + (1 if not plugin_only else 0)
    )
    bar_variations = tqdm(
        total=total_entries, desc="Variations", initial=0
    )

    for base_doc in base_docs:
        base_id = base_doc["id"]
        document = base_doc["document"]
        question = base_doc.get("question", "")
        placeholder = base_doc.get("placeholder", "")
        ideal_answer = base_doc.get("ideal_answer", "")
        ideal_summary = base_doc.get("ideal_summary", "")

        # If the current document has a placeholder attribute, it means the user
        # want the payload to be insernted into a fixed location, so we override
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

                # Creates combined jailbreak and instruction texts
                # Instruction is placed into jailbreak at <INSTRUCTION> placeholder
                if "<INSTRUCTION>" in jailbreak_text:
                    combined_text = jailbreak_text.replace(
                        "<INSTRUCTION>", instruction_text
                    )
                    lang = instruction_lang
                else:  # TODO / BUG: If no instruction it will just repeat the variant instruction times.
                    combined_text = jailbreak_text
                    lang = jailbreak_lang

                # Create plugin / transformation regex exclusion lists
                local_exclude = []
                for doc in [jailbreak, instruction]:
                    if "exclude_from_transformations_regex" in doc:
                        value = doc["exclude_from_transformations_regex"]
                        if isinstance(value, list):
                            local_exclude.extend(value)
                        else:
                            local_exclude.append(value)

                local_exclude = list(set(local_exclude))  # Remove duplicates

                # 1) No-plugin entries
                for (
                    suffix
                ) in suffixes:  # Applied with --include-suffixes, by defualt [None]
                    # Combine suffix with combined text (jailbreak + instruction)
                    suffix_combined_text, suffix_id = combined_text, None
                    if suffix:
                        suffix_combined_text += " " + suffix["suffix"]
                        suffix_id = suffix["id"]

                    if not plugin_only:
                        for position in (
                            insert_positions
                        ):  # Applied with --positions, by default ['end']
                            for injection_pattern in injection_delimiters:  # Applied with --injection-delimiters, by default "\nINJECTION_PAYLOAD\n"
                                # Combine document with suffix_combined_text (jailbreak + instruction + suffix)
                                # suffix_combined_text is inserted into the injection_pattern at 'INJECTION_PAYLOAD' placeholder
                                # Document Placeholder: Injection is placed into document placeholder
                                # Otherwise: Injection is placed into document at position (start, middle, end)
                                injected_doc = insert_jailbreak(
                                    document,
                                    suffix_combined_text,
                                    position,
                                    injection_pattern,
                                    placeholder,
                                )

                                if output_format == "burp":
                                    burp_payload_encoded = json.dumps(injected_doc)[1:-1]
                                    dataset.append(burp_payload_encoded)
                                else:
                                    for spotlighting_data_marker in spotlighting_data_markers_list:  # Applied with --spotlighting-data-markers, by default "\nDOCUMENT\n"
                                        # Applied with --include-system-message flag - gets system message associated with spotlighting data marker
                                        system_message = get_system_message(
                                            system_message_config, spotlighting_data_marker
                                        )

                                        if output_format == "full-prompt":
                                            # Combines injected document with spotlighting data marker
                                            wrapped_document = (
                                                injected_doc
                                                if spotlighting_data_marker == "none"
                                                else spotlighting_data_marker.replace(
                                                    "DOCUMENT", injected_doc
                                                )
                                            )

                                            # Create summary entry - "Summarize the following document: {wrapped_document}"
                                            summary_entry = _create_summary_entry(
                                                entry_id,
                                                base_id,
                                                jailbreak_id,
                                                instruction_id,
                                                position,
                                                "",
                                                wrapped_document,
                                                judge_name,
                                                judge_args,
                                                ideal_summary,
                                                jailbreak_type,
                                                instruction_type,
                                                injection_pattern,
                                                spotlighting_data_marker,
                                                lang,
                                                suffix_id,
                                                system_message,
                                                None,
                                                suffix_combined_text,
                                                local_exclude,
                                                instruction_steering_keywords,
                                            )
                                            dataset.append(summary_entry)
                                            entry_id += 1

                                            # Create Q&A entry - "Given this document: {wrapped_document} Answer the following question: {question}"
                                            qa_entry = _create_qa_entry(
                                                entry_id,
                                                base_id,
                                                jailbreak_id,
                                                instruction_id,
                                                position,
                                                "",
                                                wrapped_document,
                                                judge_name,
                                                judge_args,
                                                question,
                                                ideal_answer,
                                                jailbreak_type,
                                                instruction_type,
                                                injection_pattern,
                                                spotlighting_data_marker,
                                                lang,
                                                suffix_id,
                                                system_message,
                                                None,
                                                suffix_combined_text,
                                                local_exclude,
                                                instruction_steering_keywords,
                                            )
                                            dataset.append(qa_entry)
                                            entry_id += 1

                                        elif output_format == "user-input":
                                            # Create document entry
                                            doc_entry = _create_document_entry(
                                                entry_id,
                                                base_id,
                                                jailbreak_id,
                                                instruction_id,
                                                position,
                                                "",
                                                injected_doc,
                                                judge_name,
                                                judge_args,
                                                jailbreak_type,
                                                instruction_type,
                                                injection_pattern,
                                                spotlighting_data_marker,
                                                lang,
                                                suffix_id,
                                                system_message,
                                                None,
                                                output_format,
                                                suffix_combined_text,
                                                local_exclude,
                                                instruction_steering_keywords,
                                            )
                                            dataset.append(doc_entry)
                                            entry_id += 1

                                    bar_variations.update(len(spotlighting_data_markers_list))
                    # 2) Plugin entries
                    for plugin_name, plugin_module in plugins:
                        plugin_option = (
                            plugin_options_map.get(plugin_name)
                            if plugin_options_map
                            else None
                        )

                        for suffix in suffixes:
                            # Get the transformed text(s) from the plugin.
                            plugin_result = apply_plugin(
                                plugin_name,
                                plugin_module,
                                combined_text,
                                local_exclude,
                                plugin_option,
                            )

                            # Ensure the plugin result is a list of variations.
                            if not isinstance(plugin_result, list):
                                plugin_result = [plugin_result]
                            suffix_id = None

                            # Combine suffix with plugin_results (jailbreak + instruction | plugin transformation)
                            if suffix:
                                plugin_result = [
                                    variation + " " + suffix["suffix"]
                                    for variation in plugin_result
                                ]
                                suffix_id = suffix["id"]

                            # Iterate over each variation (with index)
                            for variant_index, plugin_variant in enumerate(
                                plugin_result, start=1
                            ):
                                # Create a plugin suffix that includes the plugin name and variant index.
                                plugin_suffix = f"_{plugin_name}-{variant_index}"

                                for position in (
                                    insert_positions
                                ):  # Applied with --positions, by default ['end']
                                    for injection_pattern in injection_delimiters:  # Applied with --injection-delimiters, by default "\nINJECTION_PAYLOAD\n"
                                        # Combine document with plugin_variant ((jailbreak + instruction | plugin transformation) + suffix)
                                        injected_doc = insert_jailbreak(
                                            document,
                                            plugin_variant,
                                            position,
                                            injection_pattern,
                                            placeholder,
                                        )

                                        if output_format == "burp":
                                            burp_payload_encoded = json.dumps(
                                                injected_doc
                                            )[1:-1]
                                            dataset.append(burp_payload_encoded)
                                        else:
                                            for (
                                                spotlighting_data_marker
                                            ) in spotlighting_data_markers_list:
                                                # Applied with --include-system-message flag - gets system message associated with spotlighting data marker
                                                system_message = get_system_message(
                                                    system_message_config,
                                                    spotlighting_data_marker,
                                                )

                                                if output_format == "full-prompt":
                                                    # Combines injected document with spotlighting data marker
                                                    wrapped_document = (
                                                        injected_doc
                                                        if spotlighting_data_marker
                                                        == "none"
                                                        else spotlighting_data_marker.replace(
                                                            "DOCUMENT", injected_doc
                                                        )
                                                    )
                                                    # Create summary entry - "Summarize the following document: {wrapped_document}"
                                                    summary_entry = (
                                                        _create_summary_entry(
                                                            entry_id,
                                                            base_id,
                                                            jailbreak_id,
                                                            instruction_id,
                                                            position,
                                                            plugin_suffix,
                                                            wrapped_document,
                                                            judge_name,
                                                            judge_args,
                                                            ideal_summary,
                                                            jailbreak_type,
                                                            instruction_type,
                                                            injection_pattern,
                                                            spotlighting_data_marker,
                                                            lang,
                                                            suffix_id,
                                                            system_message,
                                                            plugin_name,
                                                            plugin_variant,
                                                            local_exclude,
                                                            instruction_steering_keywords,
                                                        )
                                                    )
                                                    dataset.append(summary_entry)
                                                    entry_id += 1

                                                    # Create Q&A entry - "Given this document: {wrapped_document} Answer the following question: {question}"
                                                    qa_entry = _create_qa_entry(
                                                        entry_id,
                                                        base_id,
                                                        jailbreak_id,
                                                        instruction_id,
                                                        position,
                                                        plugin_suffix,
                                                        wrapped_document,
                                                        judge_name,
                                                        judge_args,
                                                        question,
                                                        ideal_answer,
                                                        jailbreak_type,
                                                        instruction_type,
                                                        injection_pattern,
                                                        spotlighting_data_marker,
                                                        lang,
                                                        suffix_id,
                                                        system_message,
                                                        plugin_name,
                                                        plugin_variant,
                                                        local_exclude,
                                                        instruction_steering_keywords,
                                                    )
                                                    dataset.append(qa_entry)
                                                    entry_id += 1

                                                elif output_format == "user-input":
                                                    # Create document entry
                                                    doc_entry = _create_document_entry(
                                                        entry_id,
                                                        base_id,
                                                        jailbreak_id,
                                                        instruction_id,
                                                        position,
                                                        plugin_suffix,
                                                        injected_doc,
                                                        judge_name,
                                                        judge_args,
                                                        jailbreak_type,
                                                        instruction_type,
                                                        injection_pattern,
                                                        spotlighting_data_marker,
                                                        lang,
                                                        suffix_id,
                                                        system_message,
                                                        plugin_name,
                                                        output_format,
                                                        plugin_variant,
                                                        local_exclude,
                                                        instruction_steering_keywords,
                                                    )
                                                    dataset.append(doc_entry)
                                                    entry_id += 1
                            if len(plugin_result) < plugin_variants[plugin_name]:
                                total_entries -= (plugin_variants[plugin_name] - len(plugin_result)) * len(insert_positions) * len(injection_delimiters) * len(spotlighting_data_markers_list)
                                bar_variations.total = total_entries
                                bar_variations.refresh()

                            bar_variations.update(len(plugin_result) * len(insert_positions) * len(injection_delimiters) * len(spotlighting_data_markers_list))
    bar_variations.close()
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
    include_suffixes = args.include_suffixes

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
    adv_suffixes_file = os.path.join(seed_folder, "adv_suffixes.jsonl")
    system_messages = os.path.join(seed_folder, "system_messages.toml")

    # Validate Files
    required_files = [base_documents_file, jailbreaks_file, instructions_file]
    if include_suffixes:
        required_files.append(adv_suffixes_file)

    for file_path in required_files:
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            return

    # Read/Ingest Files
    base_docs = read_jsonl_file(base_documents_file)
    jailbreaks = read_jsonl_file(jailbreaks_file)
    instructions = read_jsonl_file(instructions_file)
    adv_suffixes = read_jsonl_file(adv_suffixes_file) if include_suffixes else None

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

    # Generate Dataset
    dataset, entry_id = generate_variations(
        base_docs,
        jailbreaks,
        instructions,
        args.positions,
        injection_delimiters,
        spotlighting_data_markers_list,
        plugins,
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
            plugins=plugins if args.plugins else None,
            output_format=output_format,
            plugin_options_map=plugin_options_map,
            plugin_only=args.plugin_only,
        )

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
    }

    for entry in dataset:
        if isinstance(entry, str):
            continue
        jb_type = entry.get("jailbreak_type") or "None"
        instr_type = entry.get("instruction_type") or "None"
        lang = entry.get("lang", "en")
        task_type = entry.get("task_type") or "None"
        suffix_id = entry.get("suffix_id") or "None"
        plugin_id = entry.get("plugin") or "None"

        stats["by_jailbreak_type"][jb_type] += 1
        stats["by_instruction_type"][instr_type] += 1
        stats["by_lang"][lang] += 1
        stats["by_task_type"][task_type] += 1
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
    print_stats("Suffix ID", stats["by_suffix_id"])
    print_stats("Plugin ID", stats["by_plugin_id"])
