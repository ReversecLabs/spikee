import json
import os
import pandas as pd  # Required for Excel conversion
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
from .utilities.results import (
    preprocess_results,
    ResultProcessor
)
from .utilities.tags import validate_and_get_tag


def analyze_results(args):
    """Analyze and summarize results from JSONL files."""
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

    print("[Overview] Analyzing the following file(s): ")
    print(" - " + "\n - ".join(result_files))

    if args.combine:
        combined_results = []
        for result_file in result_files:
            # Load the results data
            results = read_jsonl_file(result_file)

            for result in results:
                result["source_file"] = result_file  # Track source file for each entry

            combined_results.extend(results)

        # process and analyze combined result file
        results_processor = ResultProcessor(combined_results, "Combined", fp_check_file)
        if output_format == "console":
            print(results_processor.generate_output(overview=args.overview))

        elif output_format == "html":
            results_processor.generate_html_report()

    else:
        for result_file in result_files:
            # Load the results data
            results = read_jsonl_file(result_file)

            # process and analyze single result file
            results_processor = ResultProcessor(results, result_file, fp_check_file)
            if output_format == "console":
                print(results_processor.generate_output(overview=args.overview))

            elif output_format == "html":
                results_processor.generate_html_report()


def rejudge_results(args):
    """Re-judge results from JSONL files."""
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
    """Extract specific results from JSONL files based on criteria."""
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

            result = query in text
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
    """Compare dataset entries against multiple result JSONL files."""
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
    """Convert results from a JSONL file to an Excel file."""
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
