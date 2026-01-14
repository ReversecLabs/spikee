import json
import os
import pandas as pd  # Required for Excel conversion
from collections import defaultdict
from tabulate import tabulate
import html
import traceback
from tqdm import tqdm

from .judge import annotate_judge_options, call_judge
from .utilities.files import (
    read_jsonl_file,
    write_jsonl_file,
    process_jsonl_input_files,
    extract_prefix_from_file_name,
    extract_directory_from_file_path,
    build_resource_name,
    prepare_output_file,
)
from .utilities.tags import validate_and_get_tag


def encode_special_characters(value):
    """Encodes special characters like newlines as '\\n' for Excel export."""
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return value  # If not a string, return as-is


def preprocess_results(results):
    """Preprocess results to encode special characters in specific fields."""
    for result in results:
        # Encode special characters in these fields if they exist
        if "injection_delimiters" in result:
            result["injection_delimiters"] = encode_special_characters(
                result["injection_delimiters"]
            )
        if "spotlighting_data_markers" in result:
            result["spotlighting_data_markers"] = encode_special_characters(
                result["spotlighting_data_markers"]
            )
    return results


def group_entries_with_attacks(results):
    """
    Group original entries with their corresponding dynamic attack entries.

    Returns:
        dict: A mapping from original IDs to a list of entries (original + attacks)
        dict: A mapping from attack IDs to their original ID
    """
    groups = defaultdict(list)
    attack_to_original = {}

    # First pass - identify all entries
    for entry in results:
        entry_id = entry["id"]
        source_file = entry.get(
            "source_file", None
        )  # Used for combined result analysis

        # Check if this is an attack entry
        if isinstance(entry_id, str) and "-attack" in entry_id:
            # Extract the original ID from the attack ID
            original_id_str = entry_id.split("-attack")[0]
            # Convert to the same type as the original ID (int or str)
            try:
                original_id = int(original_id_str)
            except ValueError:
                original_id = original_id_str

            if source_file is not None:  # For consistent key lookup
                str_original_id = str(original_id) + "-" + source_file
                str_entry_id = str(entry_id) + "-" + source_file
            else:
                str_original_id = str(original_id)
                str_entry_id = str(entry_id)

            groups[str_original_id].append(entry)
            attack_to_original[str_entry_id] = str_original_id
        else:
            # This is an original entry - use string representation for consistent keys
            if source_file is not None:
                str_entry_id = str(entry_id) + "-" + source_file
            else:
                str_entry_id = str(entry_id)
            groups[str_entry_id].append(entry)

    return groups, attack_to_original


def escape_special_chars(text):
    """Escapes special characters for console output."""
    if text is None:
        return "None"
    return repr(text)


# region analyze
def analyze_results(args):
    result_files = process_jsonl_input_files(
        args.result_file,
        args.result_folder,
        file_type=["results", "rejudge", "extract"],
    )
    output_format = args.output_format

    # False positive status and multi-file check
    fp_check_file = (
        args.false_positive_checks if hasattr(args, "false_positive_checks") else None
    )

    if len(result_files) > 1 and fp_check_file:
        print(
            f"[Error] false positive checks cannot be used when analyzing multiple results. Currently selected {len(result_files)} results."
        )
        exit(1)

    def process_results(args, results, result_file):
        # --- 1. SETUP ---
        # Group original entries with their dynamic attack entries
        entry_groups, attack_to_original = group_entries_with_attacks(results)

        # Count unique entries (considering original + attack as one logical entry)
        total_entries = len(entry_groups)

        # Count success at the group level (success if any entry in the group succeeded)
        successful_groups = 0
        initially_successful_groups = 0  # Succeeded without dynamic attack
        attack_only_successful_groups = 0  # Succeeded only with dynamic attack
        failed_groups = 0
        guardrail_groups = 0
        error_groups = 0
        total_attempts = 0

        # Track which attack types were successful
        attack_types = defaultdict(
            lambda: {"total": 0, "successes": 0, "attempts": 0, "guardrail": 0}
        )

        # Track which unique features (like jailbreak_type) are associated with each group
        group_features = {}

        # Track Triggered Guardrail Categories
        guardrail_categories = {}

        # Check if any dynamic attacks were used
        has_dynamic_attacks = False

        # --- 2. PROCESS GROUPS ---
        for original_id, entries in entry_groups.items():
            # Count total attempts across all entries in the group
            group_attempts = sum(entry.get("attempts", 1) for entry in entries)
            total_attempts += group_attempts

            # Check if any entry in this group is a dynamic attack
            attack_entries = [
                e
                for e in entries
                if isinstance(e["id"], str) and "-attack" in str(e["id"])
            ]
            if attack_entries:
                has_dynamic_attacks = True

            # Check initial success (original entry without attack)
            initial_entries = [
                e
                for e in entries
                if not (isinstance(e["id"], str) and "-attack" in str(e["id"]))
            ]
            initial_success = any(e.get("success", False) for e in initial_entries)

            # Check attack success (entries with -attack)
            attack_success = any(e.get("success", False) for e in attack_entries)

            # Overall success
            group_success = initial_success or attack_success

            # check if all entries had guardrails triggered
            group_has_guardrail = all(
                entry.get("guardrail", False) for entry in entries
            )

            # Check if all entries had errors
            group_has_error = all(
                entry.get("error") not in [None, "No response received"]
                for entry in entries
            )

            # Track attack types
            for attack_entry in attack_entries:
                attack_name = attack_entry.get("attack_name", "None")
                if attack_name != "None":
                    # Clean up the attack name by removing 'spikee.' prefix
                    clean_attack_name = attack_name.replace(
                        "spikee.attacks.", ""
                    ).replace("spikee.", "")
                    attack_types[clean_attack_name]["total"] += 1
                    attack_types[clean_attack_name]["attempts"] += attack_entry.get(
                        "attempts", 1
                    )
                    if attack_entry.get("success", False):
                        attack_types[clean_attack_name]["successes"] += 1

                    elif attack_entry.get("guardrail", False):
                        attack_types[clean_attack_name]["guardrail"] += 1

            # Increment appropriate counters
            if group_success:
                successful_groups += 1
                if initial_success:
                    initially_successful_groups += 1
                elif attack_success:
                    attack_only_successful_groups += 1
            elif group_has_error:
                error_groups += 1
                if group_has_guardrail:
                    guardrail_groups += 1
            else:
                failed_groups += 1

            # Increment guardrail categories
            for entry in entries:
                if entry.get("guardrail", False):
                    categories = entry.get("guardrail_categories", {})
                    for category, triggered in categories.items():
                        if triggered:
                            if category not in guardrail_categories:
                                guardrail_categories[category] = 0
                            guardrail_categories[category] += 1

            # Store the original entry's features to use in breakdowns
            # (We use the original entry for consistency)
            original_entry = next(
                (
                    e
                    for e in entries
                    if not isinstance(e["id"], str) or "-attack" not in str(e["id"])
                ),
                entries[0],
            )
            group_features[original_id] = original_entry

        # --- 3. CALCULATE STATISTICS ---
        # Calculate attack success rates
        attack_success_rate = (
            (successful_groups / total_entries) * 100 if total_entries else 0
        )
        initial_success_rate = (
            (initially_successful_groups / total_entries) * 100 if total_entries else 0
        )
        attack_improvement = (
            (attack_only_successful_groups / total_entries) * 100
            if total_entries
            else 0
        )

        # --- 4. OUTPUT GENERAL OVERVIEW ---
        print(f"\n=== Analysis of Results File: {result_file} ===")
        print("=== General Statistics ===")
        print(f"Total Unique Entries: {total_entries}")

        if has_dynamic_attacks:
            print(
                f"Successful Attacks (Total): {successful_groups} [{(successful_groups / total_entries) * 100:.2f}%]"
            )
            print(
                f"  - Initially Successful: {initially_successful_groups} [{(initially_successful_groups / total_entries) * 100:.2f}%]"
            )
            print(
                f"  - Only Successful with Dynamic Attack: {attack_only_successful_groups} [{(attack_only_successful_groups / total_entries) * 100:.2f}%]"
            )
            print(
                f"Failed Attacks: {failed_groups} [{(failed_groups / total_entries) * 100:.2f}%]"
            )
            print(
                f"Errors: {error_groups} [{(error_groups / total_entries) * 100:.2f}%]"
            )
            if guardrail_groups > 0:
                print(
                    f"Guardrail Triggers: {guardrail_groups} [{(guardrail_groups / total_entries) * 100:.2f}%]"
                )
            print(f"Total Attempts: {total_attempts}")
            print(f"Attack Success Rate (Overall): {attack_success_rate:.2f}%")
            print(
                f"Attack Success Rate (Without Dynamic Attack): {initial_success_rate:.2f}%"
            )
            print(
                f"Attack Success Rate (Improvement from Dynamic Attack): {attack_improvement:.2f}%\n"
            )
        else:
            print(
                f"Successful Attacks: {successful_groups} [{(successful_groups / total_entries) * 100:.2f}%]"
            )
            print(
                f"Failed Attacks: {failed_groups} [{(failed_groups / total_entries) * 100:.2f}%]"
            )
            print(
                f"Errors: {error_groups} [{(error_groups / total_entries) * 100:.2f}%]"
            )
            if guardrail_groups > 0:
                print(
                    f"Guardrail Triggers: {guardrail_groups} [{(guardrail_groups / total_entries) * 100:.2f}%]"
                )
            print(f"Total Attempts: {total_attempts}")
            print(f"Attack Success Rate: {attack_success_rate:.2f}%\n")

        # Print attack type statistics if any dynamic attacks were used
        if attack_types:
            print("=== Dynamic Attack Statistics ===")
            table = []
            for attack_name, stats in attack_types.items():
                success_rate = (
                    (stats["successes"] / stats["total"]) * 100 if stats["total"] else 0
                )
                table.append(
                    [
                        attack_name,
                        stats["total"],
                        stats["successes"],
                        stats["attempts"],
                        stats["guardrail"],
                        f"{success_rate:.2f}%",
                    ]
                )

            # Sort by success rate
            table.sort(key=lambda x: float(x[5].strip("%")), reverse=True)
            headers = [
                "Attack Type",
                "Total",
                "Successes",
                "Attempts",
                "Guardrail",
                "Success Rate",
            ]
            print(tabulate(table, headers=headers))
            print()

        if not args.overview:
            # --- 5. FALSE POSITIVE ANALYSIS ---
            # False positive analysis
            fp_data = None
            if fp_check_file:
                if not os.path.exists(fp_check_file):
                    print(
                        f"\nWARNING: False positive check file '{fp_check_file}' not found or not accessible."
                    )
                else:
                    fp_data = calc_and_print_false_positives(
                        fp_check_file, successful_groups, failed_groups
                    )

            # --- 6. DETAILED BREAKDOWNS ---
            breakdown_fields, breakdowns, combination_counts = calc_breakdowns(
                group_features, entry_groups
            )

            # Print breakdowns
            for field in breakdown_fields:
                data = breakdowns[field]
                if data:
                    print_breakdown(field, data, has_dynamic_attacks)

            # --- 7. GUARDRAIL CATEGORIES ---
            if len(guardrail_categories) > 0:
                print("=== Guardrail Category Breakdown ===")
                table = []
                for category, count in guardrail_categories.items():
                    table.append(
                        [category, count, f"{(count / total_entries) * 100:.2f}%"]
                    )
                # Sort by count descending
                table.sort(key=lambda x: x[1], reverse=True)
                headers = ["Guardrail Category", "Trigger Count", "Trigger Rate"]
                print(tabulate(table, headers=headers))
                print()

            # --- 8. COMBINATION ANALYSIS ---
            top_10, bottom_10, combination_stats_sorted = calc_combinations(
                combination_counts
            )

            # Print top 10 and bottom 10 combinations
            print_combination_stats(
                "Top 10 Most Successful Combinations", top_10, has_dynamic_attacks
            )
            print_combination_stats(
                "Top 10 Least Successful Combinations", bottom_10, has_dynamic_attacks
            )
        else:
            fp_data = None
            combination_stats_sorted = []

        # Optionally, generate HTML report
        if output_format == "html":
            attack_statistics = None
            if has_dynamic_attacks:
                attack_statistics = {
                    "has_dynamic_attacks": True,
                    "initially_successful": initially_successful_groups,
                    "attack_only_successful": attack_only_successful_groups,
                    "initial_success_rate": initial_success_rate,
                    "attack_improvement": attack_improvement,
                    "attack_types": attack_types,
                }

            generate_html_report(
                result_file,
                results,
                total_entries,
                successful_groups,
                failed_groups,
                error_groups,
                total_attempts,
                attack_success_rate,
                breakdowns,
                combination_stats_sorted,
                fp_data,
                attack_statistics,
            )

    if args.combine:
        print("[Overview] Printing statistics for the following result(s): ")
        combined_results = []
        for result_file in result_files:
            print(" - {}".format(result_file))
            # Load the results data
            results = read_jsonl_file(result_file)

            for result in results:
                result["source_file"] = result_file  # Track source file for each entry

            combined_results.extend(results)

        # process and analyze combined result file
        process_results(args, combined_results, "Combined")
    else:
        for result_file in result_files:
            # Load the results data
            results = read_jsonl_file(result_file)

            # process and analyze single result file
            process_results(args, results, result_file)


def calc_and_print_false_positives(fp_check_file, successful_groups, failed_groups):
    print("\n=== False Positive Analysis ===")
    print(f"False Positive Check File: {fp_check_file}")

    try:
        # Load the false positive checks data
        fp_results = read_jsonl_file(fp_check_file)
        print(f"Loaded {len(fp_results)} entries from false positive check file")

        # Group the false positive entries with their dynamic attack entries (if any)
        fp_groups, _ = group_entries_with_attacks(fp_results)
        print(f"Grouped into {len(fp_groups)} unique entries")

        # Count success/failure in false positive checks
        fp_success = 0  # True Negatives (benign prompts correctly allowed)
        fp_failure = 0  # False Positives (benign prompts incorrectly blocked)

        for fp_id, entries in fp_groups.items():
            # If any entry in the group was successful, count it as a success
            if any(entry.get("success", False) for entry in entries):
                fp_success += 1
            else:
                fp_failure += 1

        # Calculate confusion matrix values
        false_negatives = (
            successful_groups  # Attacks that went through (should have been blocked)
        )
        true_positives = failed_groups  # Attacks that were blocked (correctly)
        true_negatives = fp_success  # Benign prompts that went through (correctly)
        false_positives = fp_failure  # Benign prompts that were blocked (incorrectly)

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (
            (true_positives + true_negatives)
            / (true_positives + true_negatives + false_positives + false_negatives)
            if (true_positives + true_negatives + false_positives + false_negatives) > 0
            else 0
        )

        # Explicitly print the confusion matrix to CLI
        print("\n=== Confusion Matrix ===")
        print(f"True Positives (attacks correctly blocked): {true_positives}")
        print(f"False Negatives (attacks incorrectly allowed): {false_negatives}")
        print(f"True Negatives (benign prompts correctly allowed): {true_negatives}")
        print(
            f"False Positives (benign prompts incorrectly blocked): {false_positives}"
        )

        # Explicitly print the metrics to CLI
        print("\n=== Performance Metrics ===")
        print(
            f"Precision: {precision:.4f} - Of all blocked prompts, what fraction were actual attacks"
        )
        print(
            f"Recall: {recall:.4f} - Of all actual attacks, what fraction were blocked"
        )
        print(f"F1 Score: {f1_score:.4f} - Harmonic mean of precision and recall")
        print(f"Accuracy: {accuracy:.4f} - Overall accuracy across all prompts\n")

        # Store metrics for HTML report
        return {
            "file": fp_check_file,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
        }
    except Exception as e:
        print(f"Error processing false positive check file: {e}")


def calc_breakdowns(group_features, entry_groups):
    # Initialize counters for breakdowns
    breakdown_fields = [
        "jailbreak_type",
        "instruction_type",
        "task_type",
        "position",
        "spotlighting_data_markers",
        "injection_delimiters",
        "lang",
        "suffix_id",
        "plugin",
    ]

    breakdowns = {
        field: defaultdict(
            lambda: {
                "total": 0,
                "successes": 0,
                "initial_successes": 0,
                "attack_only_successes": 0,
                "attempts": 0,
            }
        )
        for field in breakdown_fields
    }

    # Initialize combination counters
    combination_counts = defaultdict(
        lambda: {
            "total": 0,
            "successes": 0,
            "initial_successes": 0,
            "attack_only_successes": 0,
            "attempts": 0,
        }
    )

    # Process groups for breakdowns
    for original_id, entry in group_features.items():
        # Get all entries in this group
        entries = entry_groups[original_id]

        # Check success types
        initial_entries = [
            e
            for e in entries
            if not (isinstance(e["id"], str) and "-attack" in str(e["id"]))
        ]
        attack_entries = [
            e for e in entries if isinstance(e["id"], str) and "-attack" in str(e["id"])
        ]

        initial_success = any(e.get("success", False) for e in initial_entries)
        attack_success = any(e.get("success", False) for e in attack_entries)
        overall_success = initial_success or attack_success
        attack_only_success = not initial_success and attack_success

        # Total attempts for this group
        group_attempts = sum(e.get("attempts", 1) for e in entries)

        # Prepare fields, replacing missing or empty values with 'None'
        jailbreak_type = entry.get("jailbreak_type") or "None"
        instruction_type = entry.get("instruction_type") or "None"
        lang = entry.get("lang") or "None"
        suffix_id = entry.get("suffix_id") or "None"
        plugin = entry.get("plugin") or "None"

        # Update combination counts
        combo_key = (jailbreak_type, instruction_type, lang, suffix_id, plugin)
        combination_counts[combo_key]["total"] += 1
        combination_counts[combo_key]["attempts"] += group_attempts
        if overall_success:
            combination_counts[combo_key]["successes"] += 1
        if initial_success:
            combination_counts[combo_key]["initial_successes"] += 1
        if attack_only_success:
            combination_counts[combo_key]["attack_only_successes"] += 1

        # Update breakdowns
        for field in breakdown_fields:
            value = entry.get(field, "None") or "None"
            breakdowns[field][value]["total"] += 1
            breakdowns[field][value]["attempts"] += group_attempts
            if overall_success:
                breakdowns[field][value]["successes"] += 1
            if initial_success:
                breakdowns[field][value]["initial_successes"] += 1
            if attack_only_success:
                breakdowns[field][value]["attack_only_successes"] += 1

    return breakdown_fields, breakdowns, combination_counts


def print_breakdown(field_name, data, has_dynamic_attacks):
    print(f"=== Breakdown by {field_name.replace('_', ' ').title()} ===")
    table = []

    if has_dynamic_attacks:
        # Full version with attack statistics
        for value, stats in data.items():
            total = stats["total"]
            successes = stats["successes"]
            initial_successes = stats["initial_successes"]
            attack_only_successes = stats["attack_only_successes"]
            attempts = stats["attempts"]

            success_rate = (successes / total) * 100 if total else 0
            initial_success_rate = (initial_successes / total) * 100 if total else 0
            attack_improvement = (attack_only_successes / total) * 100 if total else 0

            escaped_value = escape_special_chars(value)
            table.append(
                [
                    escaped_value,
                    total,
                    successes,
                    initial_successes,
                    attack_only_successes,
                    attempts,
                    f"{success_rate:.2f}%",
                    f"{initial_success_rate:.2f}%",
                    f"{attack_improvement:.2f}%",
                ]
            )

        # Sort the table by overall success rate descending
        table.sort(key=lambda x: float(x[6].strip("%")), reverse=True)
        headers = [
            field_name.title(),
            "Total",
            "All Successes",
            "Initial Successes",
            "Attack-Only Successes",
            "Attempts",
            "Success Rate",
            "Initial Success Rate",
            "Attack Improvement",
        ]
    else:
        # Simplified version without attack statistics
        for value, stats in data.items():
            total = stats["total"]
            successes = stats["successes"]
            attempts = stats["attempts"]

            success_rate = (successes / total) * 100 if total else 0

            escaped_value = escape_special_chars(value)
            table.append(
                [escaped_value, total, successes, attempts, f"{success_rate:.2f}%"]
            )

        # Sort the table by success rate descending
        table.sort(key=lambda x: float(x[4].strip("%")), reverse=True)
        headers = [
            field_name.title(),
            "Total",
            "Successes",
            "Attempts",
            "Success Rate",
        ]

    print(tabulate(table, headers=headers), "\n")


def calc_combinations(combination_counts):
    # Analyze combinations
    # Calculate success rates for each combination
    combination_stats = []
    for combo, stats in combination_counts.items():
        total = stats["total"]
        successes = stats["successes"]
        initial_successes = stats["initial_successes"]
        attack_only_successes = stats["attack_only_successes"]
        attempts = stats["attempts"]

        success_rate = (successes / total) * 100 if total else 0
        initial_success_rate = (initial_successes / total) * 100 if total else 0
        attack_improvement = (attack_only_successes / total) * 100 if total else 0

        combination_stats.append(
            {
                "jailbreak_type": combo[0],
                "instruction_type": combo[1],
                "lang": combo[2],
                "suffix_id": combo[3],
                "plugin": combo[4],
                "total": total,
                "successes": successes,
                "initial_successes": initial_successes,
                "attack_only_successes": attack_only_successes,
                "attempts": attempts,
                "success_rate": success_rate,
                "initial_success_rate": initial_success_rate,
                "attack_improvement": attack_improvement,
            }
        )

    # Sort combinations by success rate
    combination_stats_sorted = sorted(
        combination_stats, key=lambda x: x["success_rate"], reverse=True
    )

    # Get top 10 most successful combinations
    top_10 = combination_stats_sorted[:10]

    # Get bottom 10 least successful combinations (excluding combinations with zero total)
    bottom_10 = [combo for combo in combination_stats_sorted if combo["total"] > 0][
        -10:
    ]

    return top_10, bottom_10, combination_stats_sorted


def print_combination_stats(title, combo_list, has_dynamic_attacks):
    print(f"\n=== {title} ===")
    table = []

    if has_dynamic_attacks:
        # Full version with attack statistics
        for combo in combo_list:
            jailbreak_type = escape_special_chars(combo["jailbreak_type"])
            instruction_type = escape_special_chars(combo["instruction_type"])
            lang = escape_special_chars(combo["lang"])
            suffix_id = escape_special_chars(combo["suffix_id"])
            plugin = escape_special_chars(combo["plugin"])

            total = combo["total"]
            successes = combo["successes"]
            initial_successes = combo["initial_successes"]
            attack_only_successes = combo["attack_only_successes"]
            attempts = combo["attempts"]

            success_rate = f"{combo['success_rate']:.2f}%"
            initial_success_rate = f"{combo['initial_success_rate']:.2f}%"
            attack_improvement = f"{combo['attack_improvement']:.2f}%"

            table.append(
                [
                    jailbreak_type,
                    instruction_type,
                    lang,
                    suffix_id,
                    plugin,
                    total,
                    successes,
                    initial_successes,
                    attack_only_successes,
                    attempts,
                    success_rate,
                    initial_success_rate,
                    attack_improvement,
                ]
            )

        headers = [
            "Jailbreak Type",
            "Instruction Type",
            "Language",
            "Suffix ID",
            "Plugin",
            "Total",
            "All Successes",
            "Initial Successes",
            "Attack-Only Successes",
            "Attempts",
            "Success Rate",
            "Initial Rate",
            "Attack Improvement",
        ]
    else:
        # Simplified version without attack statistics
        for combo in combo_list:
            jailbreak_type = escape_special_chars(combo["jailbreak_type"])
            instruction_type = escape_special_chars(combo["instruction_type"])
            lang = escape_special_chars(combo["lang"])
            suffix_id = escape_special_chars(combo["suffix_id"])
            plugin = escape_special_chars(combo["plugin"])

            total = combo["total"]
            successes = combo["successes"]
            attempts = combo["attempts"]

            success_rate = f"{combo['success_rate']:.2f}%"

            table.append(
                [
                    jailbreak_type,
                    instruction_type,
                    lang,
                    suffix_id,
                    plugin,
                    total,
                    successes,
                    attempts,
                    success_rate,
                ]
            )

        headers = [
            "Jailbreak Type",
            "Instruction Type",
            "Language",
            "Suffix ID",
            "Plugin",
            "Total",
            "Successes",
            "Attempts",
            "Success Rate",
        ]

    print(tabulate(table, headers=headers), "\n")


# endregion


def rejudge_results(args):
    result_files = process_jsonl_input_files(
        args.result_file, args.result_folder, file_type="results"
    )

    print("[Overview] The following file(s) will be re-judged: ")
    print("\n - " + "\n - ".join(result_files))

    for result_file in result_files:
        print(f" \n\n[Start] Currently Re-judging: {result_file.split(os.sep)[-1]}")

        # Obtain file names
        file_dir = extract_directory_from_file_path(result_file)
        prefix, resource_name = extract_prefix_from_file_name(result_file)

        # Obtain results to re-judge and annotate judge options
        results = read_jsonl_file(result_file)

        judge_options = args.judge_options
        results = annotate_judge_options(results, judge_options)

        # Resume handling (per tester.py behavior)
        output_file = None
        mode = None
        completed_ids = set()
        success_count = 0

        if args.resume:
            # Attempt to obtain file name
            resume_name = build_resource_name("rejudge" + resource_name)
            file_index = os.listdir(file_dir)
            newest = 0

            # Obtain newest valid rejudge file, or fallback to new rejudge file.
            for file in file_index:
                if str(file).startswith(resume_name):
                    try:
                        age = int(file.removeprefix(resume_name).removesuffix(".jsonl"))

                        if age > newest:
                            newest = age
                            output_file = file

                    except Exception:
                        continue

            # Resume file exists
            if output_file is not None:
                output_file = os.path.join(file_dir, output_file)

                existing = read_jsonl_file(output_file)
                completed_ids = {r["id"] for r in existing}
                success_count = sum(1 for r in existing if r.get("success"))
                mode = "a"

                print(
                    f"[Resume] Found {len(completed_ids)} completed entries in {'temp'}."
                )
            else:
                print(
                    "[Resume] Existing rejudge results file not found, generating new results."
                )

        if output_file is None:
            output_file = prepare_output_file(
                file_dir, "rejudge", resource_name, None, None
            )
            mode = "w"

        # Stream write, allows CTRL+C leaves a partial file
        with open(output_file, mode, encoding="utf-8") as out_f:
            try:
                with tqdm(
                    total=len(results),
                    desc="Rejudged: ",
                    position=1,
                    initial=len(completed_ids),
                ) as pbar:
                    # Shows current successes in the loading bar
                    pbar.set_postfix(success=success_count)

                    # Process results
                    for entry in results:
                        # Skip already completed
                        if entry["id"] in completed_ids:
                            continue

                        try:
                            entry["success"] = call_judge(entry, entry["response"])

                        except Exception as e:
                            error_message = str(e)
                            entry["success"] = False
                            print("[Error] {}: {}".format(entry["id"], error_message))
                            traceback.print_exc()

                        # Update progress bar
                        if entry.get("success", False):
                            success_count += 1

                        json.dump(entry, out_f, ensure_ascii=False)
                        out_f.write("\n")
                        out_f.flush()

                        pbar.update(1)
                        pbar.set_postfix(success=success_count)

            except KeyboardInterrupt:
                print(
                    f"\n[Interrupt] CTRL+C pressed. Partial results saved to {output_file}"
                )


def extract_results(args):
    result_files = process_jsonl_input_files(
        args.result_file, args.result_folder, file_type="results"
    )

    # Category validation
    category = args.category or "success"
    if category not in [
        "success",
        "failure",
        "error",
        "guardrail",
        "no-guardrail",
        "custom",
    ]:
        print(
            f"[Error] Invalid category '{category}' specified for extraction. Must be one of: success, failure, error, guardrail, no-guardrail, custom."
        )
        exit(1)

    # Custom Category
    custom_query = []
    if args.category == "custom":
        if args.custom_search is None:
            print("[Error] Custom search requires the --custom_value to be specified.")
            exit(1)

        for query in args.custom_search:
            query = query.split(":", 1)
            query.reverse()
            custom_query.append(query)

        def search(query, text):
            invert = query.startswith("!")
            if invert:
                query = query[1:]

            result = query in str(text)
            return not result if invert else result

    # Print overview
    print("[Overview] Results will be extracted from the following file(s): ")

    matching_results = []
    id_count = 0
    total_count = 0
    for result_file in result_files:
        print(f" - {result_file}")

        result_name = result_file.split(os.sep)[-1].removesuffix(".jsonl")

        # Load the results data
        results = read_jsonl_file(result_file)

        for result in results:
            total_count += 1
            result["source_file"] = result_file  # Track source file for each entry
            match category:
                case "success":
                    if result.get("success", False):
                        id_count += 1
                        result["id"] = id_count
                        result["long_id"] = (
                            f"{result['long_id']}_extracted_{result_name}"
                        )

                        matching_results.append(result)

                case "failure":
                    if not result.get("success", False):
                        id_count += 1
                        result["id"] = id_count
                        result["long_id"] = (
                            f"{result['long_id']}_extracted_{result_name}"
                        )

                        matching_results.append(result)

                case "error":
                    if result.get("error", None) not in [None, "No response received"]:
                        id_count += 1
                        result["id"] = id_count
                        result["long_id"] = (
                            f"{result['long_id']}_extracted_{result_name}"
                        )

                        matching_results.append(result)

                case "guardrail":
                    if result.get("guardrail_triggered", False):
                        id_count += 1
                        result["id"] = id_count
                        result["long_id"] = (
                            f"{result['long_id']}_extracted_{result_name}"
                        )

                        matching_results.append(result)

                case "no-guardrail":
                    if not result.get("guardrail_triggered", False):
                        id_count += 1
                        result["id"] = id_count
                        result["long_id"] = (
                            f"{result['long_id']}_extracted_{result_name}"
                        )

                        matching_results.append(result)

                case "custom":
                    query_match = True
                    for query in custom_query:
                        if len(query) > 1:
                            field = result.get(query[1], None)
                            if field is None:
                                trimmed_query = query[0].lstrip("!")
                                invert = query[0].startswith("!")
                                if trimmed_query == "None" or trimmed_query == "null":
                                    query_match = not invert
                                else:
                                    query_match = invert

                            elif not search(query[0], field):
                                query_match = False

                        elif not search(query[0], result):
                            query_match = False

                    if query_match:
                        id_count += 1
                        result["id"] = id_count
                        result["long_id"] = (
                            f"{result['long_id']}_extracted_{result_name}"
                        )

                        matching_results.append(result)

    # Output File
    tag = validate_and_get_tag(args.tag)
    output_file = prepare_output_file("results", "extract", category, None, tag)

    # Output matching results
    write_jsonl_file(output_file, matching_results)
    print(
        f"[Overview] Extracted {id_count} / {total_count} results to {output_file}. Extraction Rate: {round(id_count / total_count if total_count > 0 else 0, 2)}"
    )


def dataset_comparison(args):
    # Get and sort result file data
    dataset = sorted(read_jsonl_file(args.dataset), key=lambda r: r.get("long_id", ""))

    result_files = process_jsonl_input_files(
        args.result_file, args.result_folder, file_type="results"
    )
    results = {}
    for result_file in result_files:
        file_results = {
            r.get("long_id", "").removesuffix("-ERROR"): r
            for r in read_jsonl_file(result_file)
        }
        results[result_file] = file_results

    print(
        "[Overview] Comparing dataset entries against results from the following file(s): "
    )
    print(" - " + "\n - ".join(result_files))

    # Dataset validation
    if args.skip_validation:
        print("[Warning] Skipping dataset validation as per user request.")
    else:
        errors = {}
        for entry in dataset:
            entry_long_id = entry["long_id"]

            for result_file, file_results in results.items():
                if entry_long_id not in file_results:
                    if result_file not in errors:
                        errors[result_file] = 0
                    errors[result_file] += 1

        if len(errors) > 0:
            print(
                "[Error] Dataset validation failed. The following discrepancies were found:"
            )
            for result_file, error_count in errors.items():
                print(f" - {result_file}: {error_count} missing entries")

    # Dataset Comparison
    success_rates = {}
    successful_entries = []
    for entry in dataset:
        entry_long_id = entry["long_id"]
        success_rates[entry_long_id] = 0

        # Obtain successes across result files
        for result_file, file_results in results.items():
            if entry_long_id in file_results and file_results[entry_long_id].get(
                "success", False
            ):
                success_rates[entry_long_id] += 1

        # Calculate success rate
        success_rates[entry_long_id] = success_rates[entry_long_id] / len(result_files)

        # Check against success criteria
        match args.success_definition:
            case "gt":
                if success_rates[entry_long_id] > args.success_threshold:
                    entry["success_rate"] = success_rates[entry_long_id]
                    successful_entries.append(entry)

            case "lt":
                if success_rates[entry_long_id] < args.success_threshold:
                    entry["success_rate"] = success_rates[entry_long_id]
                    successful_entries.append(entry)

    print(
        f"[Overview] Dataset comparison completed. {len(successful_entries)} entries matched the success criteria."
    )
    if len(successful_entries) == 0:
        print(
            "[Overview] No entries matched the success criteria. No output file will be generated."
        )
        return

    # Sort entries by success rate descending
    successful_entries.sort(key=lambda e: e["success_rate"], reverse=True)

    if args.number > 0:
        print(
            f"[Overview] Limiting output to top {args.number} entries based on success rate."
        )
        successful_entries = successful_entries[: args.number]

    tag = validate_and_get_tag(args.tag)
    output_file = prepare_output_file("datasets", "comparison", None, args.dataset, tag)

    write_jsonl_file(output_file, successful_entries)
    print(f"[Overview] Comparison dataset saved to {output_file}.")


def convert_results_to_excel(args):
    result_file = args.result_file

    # Read results
    results = read_jsonl_file(result_file)

    # Preprocess results to encode special characters
    results = preprocess_results(results)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Output Excel file
    output_file = os.path.splitext(result_file)[0] + ".xlsx"
    df.to_excel(output_file, index=False)

    print(f"Results successfully converted to Excel: {output_file}")


def generate_html_report(
    result_file,
    results,
    total_entries,
    total_successes,
    total_failures,
    total_errors,
    total_attempts,
    attack_success_rate,
    breakdowns,
    combination_stats_sorted,
    fp_data=None,
    attack_statistics=None,
):
    import os
    from jinja2 import Template

    # Check if attack statistics are available
    has_dynamic_attacks = attack_statistics and attack_statistics.get(
        "has_dynamic_attacks", False
    )

    # Prepare data for the template
    template_data = {
        "result_file": result_file,
        "total_entries": total_entries,
        "total_successes": total_successes,
        "total_failures": total_failures,
        "total_errors": total_errors,
        "total_attempts": total_attempts,
        "attack_success_rate": f"{attack_success_rate:.2f}%",
        "breakdowns": {},
        "top_combinations": combination_stats_sorted[:10],
        "bottom_combinations": [
            combo for combo in combination_stats_sorted if combo["total"] > 0
        ][-10:],
        "fp_data": fp_data,
        "attack_statistics": attack_statistics,
        "has_dynamic_attacks": has_dynamic_attacks,
    }

    # Prepare breakdown data
    for field, data in breakdowns.items():
        breakdown_list = []
        for value, stats in data.items():
            total = stats["total"]
            successes = stats["successes"]
            attempts = stats["attempts"]
            success_rate = (successes / total) * 100 if total else 0

            item = {
                "value": html.escape(str(value)) if value else "None",
                "total": total,
                "successes": successes,
                "attempts": attempts,
                "success_rate": f"{success_rate:.2f}%",
            }

            # Add attack-specific stats if available
            if has_dynamic_attacks:
                initial_successes = stats["initial_successes"]
                attack_only_successes = stats["attack_only_successes"]
                initial_success_rate = (initial_successes / total) * 100 if total else 0
                attack_improvement = (
                    (attack_only_successes / total) * 100 if total else 0
                )

                item.update(
                    {
                        "initial_successes": initial_successes,
                        "attack_only_successes": attack_only_successes,
                        "initial_success_rate": f"{initial_success_rate:.2f}%",
                        "attack_improvement": f"{attack_improvement:.2f}%",
                    }
                )

            # Replace newlines and tabs with visible representations
            item["value"] = item["value"].replace("\n", "\\n").replace("\t", "\\t")
            breakdown_list.append(item)

        # Sort by success rate descending
        breakdown_list.sort(
            key=lambda x: float(x["success_rate"].strip("%")), reverse=True
        )
        template_data["breakdowns"][field] = breakdown_list

    # Load HTML template
    html_template = """
    <html>
    <head>
        <title>Results Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            pre { margin: 0; }
            .metrics { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .metric-good { color: green; }
            .metric-bad { color: red; }
            .attack-stats { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Results Analysis Report</h1>
        <p><strong>Result File:</strong> {{ result_file }}</p>
        
        <h2>General Statistics</h2>
        <ul>
            <li>Total Unique Entries: {{ total_entries }}</li>
            {% if has_dynamic_attacks %}
                <li>Successful Attacks (Total): {{ total_successes }}</li>
                <li>&nbsp;&nbsp;- Initially Successful: {{ attack_statistics.initially_successful }}</li>
                <li>&nbsp;&nbsp;- Only Successful with Dynamic Attack: {{ attack_statistics.attack_only_successful }}</li>
                <li>Failed Attacks: {{ total_failures }}</li>
                <li>Errors: {{ total_errors }}</li>
                <li>Total Attempts: {{ total_attempts }}</li>
                <li>Attack Success Rate (Overall): {{ attack_success_rate }}</li>
                <li>Attack Success Rate (Without Dynamic Attack): {{ "%.2f%%" | format(attack_statistics.initial_success_rate) }}</li>
                <li>Attack Success Rate (Improvement from Dynamic Attack): {{ "%.2f%%" | format(attack_statistics.attack_improvement) }}</li>
            {% else %}
                <li>Successful Attacks: {{ total_successes }}</li>
                <li>Failed Attacks: {{ total_failures }}</li>
                <li>Errors: {{ total_errors }}</li>
                <li>Total Attempts: {{ total_attempts }}</li>
                <li>Attack Success Rate: {{ attack_success_rate }}</li>
            {% endif %}
        </ul>
        
        {% if has_dynamic_attacks and attack_statistics.attack_types %}
        <div class="attack-stats">
            <h3>Dynamic Attack Statistics</h3>
            <table>
                <tr>
                    <th>Attack Type</th>
                    <th>Total</th>
                    <th>Successes</th>
                    <th>Attempts</th>
                    <th>Success Rate</th>
                </tr>
                {% for attack_name, stats in attack_statistics.attack_types.items() %}
                <tr>
                    <td>{{ attack_name }}</td>
                    <td>{{ stats.total }}</td>
                    <td>{{ stats.successes }}</td>
                    <td>{{ stats.attempts }}</td>
                    <td>{{ "%.2f%%" | format((stats.successes / stats.total * 100) if stats.total else 0) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if fp_data %}
        <h2>False Positive Analysis</h2>
        <p><strong>False Positive Check File:</strong> {{ fp_data.file }}</p>
        
        <div class="metrics">
            <h3>Confusion Matrix</h3>
            <ul>
                <li><strong>True Positives</strong> (attacks correctly blocked): {{ fp_data.true_positives }}</li>
                <li><strong>False Negatives</strong> (attacks incorrectly allowed): {{ fp_data.false_negatives }}</li>
                <li><strong>True Negatives</strong> (benign prompts correctly allowed): {{ fp_data.true_negatives }}</li>
                <li><strong>False Positives</strong> (benign prompts incorrectly blocked): {{ fp_data.false_positives }}</li>
            </ul>
            
            <h3>Performance Metrics</h3>
            <ul>
                <li><strong>Precision:</strong> {{ "%.4f"|format(fp_data.precision) }} - Of all blocked prompts, what fraction were actual attacks</li>
                <li><strong>Recall:</strong> {{ "%.4f"|format(fp_data.recall) }} - Of all actual attacks, what fraction were blocked</li>
                <li><strong>F1 Score:</strong> {{ "%.4f"|format(fp_data.f1_score) }} - Harmonic mean of precision and recall</li>
                <li><strong>Accuracy:</strong> {{ "%.4f"|format(fp_data.accuracy) }} - Overall accuracy across all prompts</li>
            </ul>
        </div>
        {% endif %}
        
        {% for field, breakdown in breakdowns.items() %}
            <h2>Breakdown by {{ field.replace('_', ' ').title() }}</h2>
            <table>
                <tr>
                    <th>{{ field.title() }}</th>
                    <th>Total</th>
                    <th>Successes</th>
                    {% if has_dynamic_attacks %}
                    <th>Initial Successes</th>
                    <th>Attack-Only Successes</th>
                    {% endif %}
                    <th>Attempts</th>
                    <th>Success Rate</th>
                    {% if has_dynamic_attacks %}
                    <th>Initial Success Rate</th>
                    <th>Attack Improvement</th>
                    {% endif %}
                </tr>
                {% for item in breakdown %}
                <tr>
                    <td><pre>{{ item.value }}</pre></td>
                    <td>{{ item.total }}</td>
                    <td>{{ item.successes }}</td>
                    {% if has_dynamic_attacks %}
                    <td>{{ item.initial_successes }}</td>
                    <td>{{ item.attack_only_successes }}</td>
                    {% endif %}
                    <td>{{ item.attempts }}</td>
                    <td>{{ item.success_rate }}</td>
                    {% if has_dynamic_attacks %}
                    <td>{{ item.initial_success_rate }}</td>
                    <td>{{ item.attack_improvement }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </table>
        {% endfor %}
        
        <h2>Top 10 Most Successful Combinations</h2>
        <table>
            <tr>
                <th>Jailbreak Type</th>
                <th>Instruction Type</th>
                <th>Language</th>
                <th>Suffix ID</th>
                <th>Plugin</th>
                <th>Total</th>
                <th>Successes</th>
                {% if has_dynamic_attacks %}
                <th>Initial Successes</th>
                <th>Attack-Only Successes</th>
                {% endif %}
                <th>Attempts</th>
                <th>Success Rate</th>
                {% if has_dynamic_attacks %}
                <th>Initial Rate</th>
                <th>Attack Improvement</th>
                {% endif %}
            </tr>
            {% for combo in top_combinations %}
            <tr>
                <td>{{ combo.jailbreak_type }}</td>
                <td>{{ combo.instruction_type }}</td>
                <td>{{ combo.lang }}</td>
                <td>{{ combo.suffix_id }}</td>
                <td>{{ combo.plugin }}</td>
                <td>{{ combo.total }}</td>
                <td>{{ combo.successes }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ combo.initial_successes }}</td>
                <td>{{ combo.attack_only_successes }}</td>
                {% endif %}
                <td>{{ combo.attempts }}</td>
                <td>{{ "%.2f%%" % combo.success_rate }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ "%.2f%%" % combo.initial_success_rate }}</td>
                <td>{{ "%.2f%%" % combo.attack_improvement }}</td>
                {% endif %}
            </tr>
            {% endfor %}
        </table>
        
        <h2>Top 10 Least Successful Combinations</h2>
        <table>
            <tr>
                <th>Jailbreak Type</th>
                <th>Instruction Type</th>
                <th>Language</th>
                <th>Suffix ID</th>
                <th>Plugin</th>
                <th>Total</th>
                <th>Successes</th>
                {% if has_dynamic_attacks %}
                <th>Initial Successes</th>
                <th>Attack-Only Successes</th>
                {% endif %}
                <th>Attempts</th>
                <th>Success Rate</th>
                {% if has_dynamic_attacks %}
                <th>Initial Rate</th>
                <th>Attack Improvement</th>
                {% endif %}
            </tr>
            {% for combo in bottom_combinations %}
            <tr>
                <td>{{ combo.jailbreak_type }}</td>
                <td>{{ combo.instruction_type }}</td>
                <td>{{ combo.lang }}</td>
                <td>{{ combo.suffix_id }}</td>
                <td>{{ combo.plugin }}</td>
                <td>{{ combo.total }}</td>
                <td>{{ combo.successes }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ combo.initial_successes }}</td>
                <td>{{ combo.attack_only_successes }}</td>
                {% endif %}
                <td>{{ combo.attempts }}</td>
                <td>{{ "%.2f%%" % combo.success_rate }}</td>
                {% if has_dynamic_attacks %}
                <td>{{ "%.2f%%" % combo.initial_success_rate }}</td>
                <td>{{ "%.2f%%" % combo.attack_improvement }}</td>
                {% endif %}
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """

    # Render template
    template = Template(html_template)
    html_content = template.render(template_data)

    # Write to HTML file
    output_file = os.path.splitext(result_file)[0] + "_analysis.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_file}")
