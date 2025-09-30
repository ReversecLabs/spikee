import json
import os
import pandas as pd  # Required for Excel conversion
from collections import defaultdict
from tabulate import tabulate
import html
import traceback
import time
from tqdm import tqdm

from .judge import annotate_judge_options, call_judge


def read_jsonl_file(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl_file(output_file, data):
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


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

        # Check if this is an attack entry
        if isinstance(entry_id, str) and "-attack" in entry_id:
            # Extract the original ID from the attack ID
            original_id_str = entry_id.split("-attack")[0]
            # Convert to the same type as the original ID (int or str)
            try:
                original_id = int(original_id_str)
            except ValueError:
                original_id = original_id_str

            str_original_id = str(original_id)  # For consistent key lookup
            groups[str_original_id].append(entry)
            attack_to_original[entry_id] = str_original_id
        else:
            # This is an original entry - use string representation for consistent keys
            str_entry_id = str(entry_id)
            groups[str_entry_id].append(entry)

    return groups, attack_to_original


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


def escape_special_chars(text):
    """Escapes special characters for console output."""
    if text is None:
        return "None"
    return repr(text)


def analyze_results(args):
    result_file = args.result_file
    output_format = args.output_format

    fp_check_file = (
        args.false_positive_checks if hasattr(args, "false_positive_checks") else None
    )

    # Load the results data
    results = read_jsonl_file(result_file)

    # Group original entries with their dynamic attack entries
    entry_groups, attack_to_original = group_entries_with_attacks(results)

    # Count unique entries (considering original + attack as one logical entry)
    total_entries = len(entry_groups)

    # Count success at the group level (success if any entry in the group succeeded)
    successful_groups = 0
    initially_successful_groups = 0  # Succeeded without dynamic attack
    attack_only_successful_groups = 0  # Succeeded only with dynamic attack
    failed_groups = 0
    error_groups = 0
    total_attempts = 0

    # Track which attack types were successful
    attack_types = defaultdict(lambda: {"total": 0, "successes": 0, "attempts": 0})

    # Track which unique features (like jailbreak_type) are associated with each group
    group_features = {}

    # Check if any dynamic attacks were used
    has_dynamic_attacks = False

    for original_id, entries in entry_groups.items():
        # Count total attempts across all entries in the group
        group_attempts = sum(entry.get("attempts", 1) for entry in entries)
        total_attempts += group_attempts

        # Check if any entry in this group is a dynamic attack
        attack_entries = [
            e for e in entries if isinstance(e["id"], str) and "-attack" in str(e["id"])
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
                clean_attack_name = attack_name.replace("spikee.attacks.", "").replace(
                    "spikee.", ""
                )
                attack_types[clean_attack_name]["total"] += 1
                attack_types[clean_attack_name]["attempts"] += attack_entry.get(
                    "attempts", 1
                )
                if attack_entry.get("success", False):
                    attack_types[clean_attack_name]["successes"] += 1

        # Increment appropriate counters
        if group_success:
            successful_groups += 1
            if initial_success:
                initially_successful_groups += 1
            elif attack_success:
                attack_only_successful_groups += 1
        elif group_has_error:
            error_groups += 1
        else:
            failed_groups += 1

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

    # Calculate attack success rates
    attack_success_rate = (
        (successful_groups / total_entries) * 100 if total_entries else 0
    )
    initial_success_rate = (
        (initially_successful_groups / total_entries) * 100 if total_entries else 0
    )
    attack_improvement = (
        (attack_only_successful_groups / total_entries) * 100 if total_entries else 0
    )

    # Print general statistics
    print("\n=== General Statistics ===")
    print(f"Total Unique Entries: {total_entries}")

    if has_dynamic_attacks:
        print(f"Successful Attacks (Total): {successful_groups}")
        print(f"  - Initially Successful: {initially_successful_groups}")
        print(
            f"  - Only Successful with Dynamic Attack: {attack_only_successful_groups}"
        )
        print(f"Failed Attacks: {failed_groups}")
        print(f"Errors: {error_groups}")
        print(f"Total Attempts: {total_attempts}")
        print(f"Attack Success Rate (Overall): {attack_success_rate:.2f}%")
        print(
            f"Attack Success Rate (Without Dynamic Attack): {initial_success_rate:.2f}%"
        )
        print(
            f"Attack Success Rate (Improvement from Dynamic Attack): {attack_improvement:.2f}%\n"
        )
    else:
        print(f"Successful Attacks: {successful_groups}")
        print(f"Failed Attacks: {failed_groups}")
        print(f"Errors: {error_groups}")
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
                    f"{success_rate:.2f}%",
                ]
            )

        # Sort by success rate
        table.sort(key=lambda x: float(x[4].strip("%")), reverse=True)
        headers = ["Attack Type", "Total", "Successes", "Attempts", "Success Rate"]
        print(tabulate(table, headers=headers))
        print()

    # False positive analysis
    fp_data = None
    if fp_check_file:
        if not os.path.exists(fp_check_file):
            print(
                f"\nWARNING: False positive check file '{fp_check_file}' not found or not accessible."
            )
        else:
            print("\n=== False Positive Analysis ===")
            print(f"False Positive Check File: {fp_check_file}")

            try:
                # Load the false positive checks data
                fp_results = read_jsonl_file(fp_check_file)
                print(
                    f"Loaded {len(fp_results)} entries from false positive check file"
                )

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
                false_negatives = successful_groups  # Attacks that went through (should have been blocked)
                true_positives = failed_groups  # Attacks that were blocked (correctly)
                true_negatives = (
                    fp_success  # Benign prompts that went through (correctly)
                )
                false_positives = (
                    fp_failure  # Benign prompts that were blocked (incorrectly)
                )

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
                    / (
                        true_positives
                        + true_negatives
                        + false_positives
                        + false_negatives
                    )
                    if (
                        true_positives
                        + true_negatives
                        + false_positives
                        + false_negatives
                    )
                    > 0
                    else 0
                )

                # Explicitly print the confusion matrix to CLI
                print("\n=== Confusion Matrix ===")
                print(f"True Positives (attacks correctly blocked): {true_positives}")
                print(
                    f"False Negatives (attacks incorrectly allowed): {false_negatives}"
                )
                print(
                    f"True Negatives (benign prompts correctly allowed): {true_negatives}"
                )
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
                print(
                    f"F1 Score: {f1_score:.4f} - Harmonic mean of precision and recall"
                )
                print(
                    f"Accuracy: {accuracy:.4f} - Overall accuracy across all prompts\n"
                )

                # Store metrics for HTML report
                fp_data = {
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

    # Function to print breakdowns
    def print_breakdown(field_name, data):
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
                attack_improvement = (
                    (attack_only_successes / total) * 100 if total else 0
                )

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

        print(tabulate(table, headers=headers))
        print()

    # Print breakdowns
    for field in breakdown_fields:
        data = breakdowns[field]
        if data:
            print_breakdown(field, data)

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

    # Function to print combination stats
    def print_combination_stats(title, combo_list):
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

        print(tabulate(table, headers=headers))
        print()

    # Print top 10 and bottom 10 combinations
    print_combination_stats("Top 10 Most Successful Combinations", top_10)
    print_combination_stats("Top 10 Least Successful Combinations", bottom_10)

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


def rejudge_results(args):
    result_files = args.result_file

    print(f"Re-judging the following file(s): \n - {'\n - '.join(result_files)}")

    for result_file in result_files:
        print(f" \n\nCurrently Re-judging: {result_file.split(os.sep)[-1]}")

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
            file_dir, prefix_name = os.path.split(os.path.abspath(result_file))
            prefix_name = prefix_name.removesuffix(".jsonl") + "-rejudge-"

            file_index = os.listdir(file_dir)
            newest = 0

            # Obtain newest valid rejudge file, or fallback to new rejudge file.
            for file in file_index:
                if str(file).startswith(prefix_name):
                    try:
                        age = int(file.removeprefix(prefix_name).removesuffix(".jsonl"))

                        if age > newest:
                            newest = age
                            output_file = file

                    except Exception:
                        continue

            if output_file is not None:
                output_file = file_dir + os.path.sep + output_file

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
            output_file = (
                result_file.removesuffix(".jsonl")
                + "-rejudge-"
                + str(round(time.time()))
                + ".jsonl"
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
