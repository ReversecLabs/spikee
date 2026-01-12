import os
import re
import time
import random
import threading
import inspect
import traceback
import sys
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
from InquirerPy import inquirer

from .judge import annotate_judge_options, call_judge
from .utilities.enums import Turn
from .utilities.files import (
    read_jsonl_file,
    write_jsonl_file,
    append_jsonl_entry,
    process_jsonl_input_files,
    extract_resource_name,
    build_resource_name,
    prepare_output_file,
    does_resource_name_match,
)
from .utilities.modules import load_module_from_path, get_default_option
from .utilities.tags import validate_and_get_tag


class GuardrailTrigger(Exception):
    """Exception raised when a guardrail is triggered."""

    def __init__(self, message, categories={}):
        super().__init__(message)
        self.categories = categories


class RetryableError(Exception):
    """Exception raised for errors that are retryable, such as 429 errors."""

    def __init__(self, message, retry_period=None):
        super().__init__(message)
        self.retry_period = retry_period


class MultiTurnSkip(Exception):
    """Exception raised to skip multi-turn entries being processed as single-turn."""
    pass


class AdvancedTargetWrapper:
    """
    A wrapper for a target module's process_input method that incorporates both:
      - A retry strategy for handling 429 errors (max_retries) with throttling.
      - Transparent forwarding of only the kwargs the wrapped target supports.

    This is designed to be passed to the attack() function so that each call to process_input()
    will try up to max_retries times before failing.

    Parameters:
      target_module: The original target module that provides process_input(input_text, system_message[, logprobs]).
      target_options: Target options, typically a string representing the name of the llm to call
      max_retries (int): Maximum number of retries per attempt (e.g. on 429 errors).
      throttle (float): Number of seconds to wait after a successful call.

    KWARGS (if supported):
      target_options (str): User requested options to pass to the target.
      logprobs (bool): Whether to request logprobs from the target.
      input_id (str): Input identifier of current entry.
      output_file (str): Output results file path.

      spikee_session_id (str): Spikee session identifier - Multi-Turn.
      backtrack (bool): Whether to backtrack the last turn - Multi-Turn.
    """

    def __init__(self, target_module, target_options=None, max_retries=3, throttle=0):
        self.target_module = target_module
        self.target_options = target_options
        self.max_retries = max_retries
        self.throttle = throttle

        if not hasattr(self.target_module, "config"):
            self.config = {
                "single-turn": True,
                "multi-turn": False,
                "backtrack": False,
            }
            self.target_module.config = self.config
        else:
            self.config = self.target_module.config

        sig = inspect.signature(self.target_module.process_input)
        params = sig.parameters
        # Detect support for optional features using available parameters and target config
        self.supports_options = "target_options" in params
        self.supports_logprobs = "logprobs" in params
        self.supports_input_id = "input_id" in params
        self.supports_output_file = "output_file" in params

        self.supports_spikee_session_id = "spikee_session_id" in params
        self.supports_backtrack = "backtrack" in params

    @classmethod
    def create_target_wrapper(cls, target_name, target_options, max_retries, throttle):
        """Static method to create an AdvancedTargetWrapper for a given target name."""
        target_mod = load_module_from_path(target_name, "targets")

        # Wrap the target module with AdvancedTargetWrapper
        return cls(
            target_mod,
            max_retries=max_retries,
            throttle=throttle,
            target_options=target_options,
        )

    def get_target(self):
        """Returns the underlying target module."""
        return self.target_module

    def process_input(
        self,
        input_text,
        system_message=None,
        logprobs=False,
        input_id=None,
        output_file=None,
        spikee_session_id=None,
        backtrack=False,
    ):
        last_error = None
        retries = 0

        while retries < self.max_retries:
            try:
                # Build only the kwargs the underlying target supports.
                # Older targets without these parameters will simply be called without them.
                kwargs = {}
                if self.supports_options and self.target_options is not None:
                    kwargs["target_options"] = self.target_options
                if self.supports_logprobs:
                    kwargs["logprobs"] = logprobs
                if self.supports_input_id:
                    kwargs["input_id"] = input_id
                if self.supports_output_file:
                    kwargs["output_file"] = output_file
                if self.supports_spikee_session_id:
                    kwargs["spikee_session_id"] = spikee_session_id
                if self.supports_backtrack:
                    kwargs["backtrack"] = backtrack

                # Correct Multi-Turn -> Single-Turn handling
                if isinstance(input_text, list):
                    # input_text = "\n".join(input_text)
                    raise MultiTurnSkip("Multi-Turn Skip - Process via Multi-Turn capable attack.")

                    # Delegate to the wrapped process_input
                if kwargs:
                    result = self.target_module.process_input(
                        input_text, system_message, **kwargs
                    )
                else:
                    result = self.target_module.process_input(
                        input_text, system_message
                    )

                # Unpack (response, meta) if tuple returned
                if isinstance(result, tuple) and len(result) == 2:
                    response, meta = result
                else:
                    response, meta = result, None
                if self.throttle > 0:
                    time.sleep(self.throttle)

                return response, meta

            except RetryableError as e:
                last_error = e
                if retries < self.max_retries - 1:
                    wait_time = (
                        e.retry_period
                        if e.retry_period is not None
                        else random.randint(30, 120)
                    )
                    time.sleep(wait_time)
                    retries += 1
                else:
                    break

            except MultiTurnSkip as e:
                last_error = e
                break

            except Exception as e:
                last_error = e
                if "429" in str(e) and retries < self.max_retries - 1:
                    wait_time = random.randint(30, 120)
                    time.sleep(wait_time)
                    retries += 1
                else:
                    break

        # All retries exhausted
        raise last_error


# region resource_utilities
def _build_target_name(target, target_options):
    """
    Builds a target's name, returning "target-target_options".
    If no target_options provided, attempts to get default option from target module.
    """

    regex_pattern = '(^[<>:"/\|?*]+)|([<>:"/\|?*]+$)|([<>:"/\|?*]+)'  # Matches Invalid Windows Characters,

    def replacer(match):
        if match.group(1) or match.group(2):  # If at start/end of string, just remove
            return ""
        else:  # If in middle of string, replace with '~'
            return "~"

    # If no target options provided, try to get default module option
    if target_options is None:
        try:
            mod = load_module_from_path(target, "targets")
            target_options = get_default_option(mod)
        except Exception:
            pass

    if target_options is None:
        return target

    target_options = re.sub(
        regex_pattern, replacer, target_options
    )  # Remove Invalid Windows Characters
    return f"{target}-{target_options}"


def _load_results_file(resume_file, attack_module, attack_iters):
    completed_ids, results, already_done = set(), [], 0

    # Load Resume File, if selected.
    if resume_file and os.path.exists(resume_file):
        results = read_jsonl_file(resume_file)
        completed_ids = {r["id"] for r in results}
        print(
            f"[Resume] Found {len(completed_ids)} completed entries in {resume_file}."
        )

        # Identify attack results
        no_attack = sum(1 for r in results if r.get("attack_name") == "None")
        with_attack = len(results) - no_attack
        already_done = no_attack + with_attack * attack_iters
    return completed_ids, results, already_done


# endregion


# region entry_processing
def _apply_sampling(dataset, sample_percent, sample_seed):
    """Apply random sampling to the dataset based on sample_percent and sample_seed."""
    if sample_seed == "random":  # apply random seed
        seed = random.randint(0, 2**32 - 1)
        print(f"[Info] Using random seed for sampling: {seed}")

    else:  # apply user-defined seed
        seed = int(sample_seed)
        print(f"[Info] Using seed for sampling: {seed}")

    # Obtain random sample
    random.seed(seed)
    size = round(len(dataset) * sample_percent)
    print(
        f"[Info] Sampled {size} entries from {len(dataset)} total entries ({sample_percent:.1%})"
    )
    return random.sample(dataset, size)


def _calculate_total_attempts(
    n_entries, attempts, attack_iters, already_done, has_attack
):
    per_item = attempts + (attack_iters if has_attack else 0)
    return n_entries * per_item + already_done


# endregion


# region resume_handling
def _determine_resume_file(args, dataset, is_tty: bool) -> str | None:
    """
    Determine resume behaviour depending on tty status and resume flags.
    Returns a result file path, or 'None' to create a new results file.
    """
    # Use explicit --resume-file
    if getattr(args, "resume_file", None):
        return args.resume_file

    # --no-auto-resume flag - create new results file
    if getattr(args, "no_auto_resume", False):
        return None

    # Identify previous results files
    target_name_full = _build_target_name(args.target, args.target_options)
    candidates = _find_resume_candidates("results", target_name_full, dataset, args.tag)

    if not candidates:
        return None

    # --auto-resume flag, silently pick latest results file
    if getattr(args, "auto_resume", False):
        print(f"[Auto-Resume] Using latest: {candidates[0].name}")
        return str(candidates[0])

    # ---- TTY behavior: user select prompt ----
    if is_tty:
        picked = _select_resume_file_interactive(candidates, preselect_index=0)
        return str(picked) if picked else None

    return None


def _find_resume_candidates(
    results_dir: str | Path, target_name_full: str, dataset_path: str, tag: str | None
) -> list[Path]:
    """Identify potential resume candidates within the results_dir using the same resource name"""
    # Load results directory
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    # Get resource name
    resource_name = build_resource_name(
        "results", target_name_full, extract_resource_name(dataset_path), tag
    )

    # Only accept exact matches for the requested tag (or lack of tag).
    # No fallback to untagged files when a tag is specified.
    candidates = [
        p
        for p in results_dir.glob(f"{resource_name}_*.jsonl")
        if does_resource_name_match(p, resource_name)
    ]

    return sorted(
        candidates,
        key=_parse_timestamp_from_filename,
        reverse=True,
    )


def _select_resume_file_interactive(
    cands: list[Path], preselect_index: int = 0
) -> Path | None:
    """Interactive Results Prompt"""
    items = ["Start fresh (do not resume)"] + [_format_candidate_line(p) for p in cands]

    result = inquirer.select(
        message="Resume from which results file? (Enter = Start fresh)",
        choices=items,
        default=items[0],  # default to Start fresh
        pointer="➤ ",
    ).execute()

    if result == items[0]:  # "Start fresh" selected
        return None

    # Find which candidate was selected
    idx = items.index(result) - 1
    return cands[idx]


def _format_candidate_line(p: Path) -> str:
    ts = _parse_timestamp_from_filename(p)
    dt = datetime.fromtimestamp(ts)
    age_sec = max(0, int((datetime.now() - dt).total_seconds()))
    # compact age display
    if age_sec < 90:
        age = f"{age_sec}s"
    elif age_sec < 90 * 60:
        age = f"{age_sec // 60}m"
    elif age_sec < 48 * 3600:
        age = f"{age_sec // 3600}h"
    else:
        age = f"{age_sec // 86400}d"
    return f"[{dt.strftime('%Y-%m-%d %H:%M')}] {p.name}  (age {age})"


def _parse_timestamp_from_filename(p: Path) -> int:
    # Expect ..._<ts>.jsonl at the end; fall back to mtime if parse fails
    name = p.name
    try:
        ts_str = name.rsplit("_", 1)[-1].removesuffix(".jsonl")
        return int(ts_str)
    except Exception:
        return int(p.stat().st_mtime)


# endregion


def _do_single_request(
    entry,
    input_text,
    target_module,
    output_file,
    num_attempt,
    attempts_bar,
    global_lock,
):
    """
    Executes one request against the target by calling its process_input() method.
    The target_module is assumed to be an instance of AdvancedTargetWrapper that
    already implements retries and throttling.

    Parameters:
      entry (dict): The dataset entry.
      input_text (str): The prompt text.
      target_module: The wrapped target module.
      num_attempt: The current attempt number.
      attempts_bar (tqdm): Progress bar to update.
      global_lock (threading.Lock): Lock for safely updating the progress bar.

    Returns:
      tuple: (result_dict, success)
    """
    # Extract metadata from the entry.
    entry_id = entry.get("id", None)
    task_type = entry.get("task_type", None)
    jailbreak_type = entry.get("jailbreak_type", None)
    instruction_type = entry.get("instruction_type", None)
    document_id = entry.get("document_id", None)
    position = entry.get("position", None)
    spotlighting_data_markers = entry.get("spotlighting_data_markers", None)
    injection_delimiters = entry.get("injection_delimiters", None)
    suffix_id = entry.get("suffix_id", None)
    lang = entry.get("lang", "en")
    system_message = entry.get("system_message", None)
    plugin = entry.get("plugin", None)

    # Guardrail Specific Errors
    guardrail = False
    guardrail_categories = {}

    try:
        start_time = time.time()
        response, _ = target_module.process_input(
            input_text, system_message, False, entry_id, output_file
        )
        end_time = time.time()
        response_time = end_time - start_time
        success = call_judge(entry, response)
        response_str = response if isinstance(response, str) else ""
        error_message = None

    except GuardrailTrigger as gt:
        error_message = str(gt)
        response_str = ""
        response_time = time.time() - start_time
        success = False
        guardrail = True
        if hasattr(gt, "categories"):
            guardrail_categories = gt.categories
        print("[Guardrail Triggered] {}: {}".format(entry["id"], error_message))

    except MultiTurnSkip as ms:
        error_message = str(ms)
        response_str = ""
        response_time = None
        success = False

    except Exception as e:
        error_message = str(e)
        response_str = ""
        response_time = None
        success = False
        print("[Error] {}: {}".format(entry["id"], error_message))
        traceback.print_exc()

    with global_lock:
        attempts_bar.update(1)

    result_dict = {
        "id": entry["id"],
        "long_id": entry["long_id"],
        "input": input_text,
        "response": response_str,
        "response_time": response_time,
        "success": success,
        "judge_name": entry["judge_name"],
        "judge_args": entry["judge_args"],
        "judge_options": entry["judge_options"],
        "attempts": num_attempt,
        "task_type": task_type,
        "jailbreak_type": jailbreak_type,
        "instruction_type": instruction_type,
        "document_id": document_id,
        "position": position,
        "spotlighting_data_markers": spotlighting_data_markers,
        "injection_delimiters": injection_delimiters,
        "suffix_id": suffix_id,
        "lang": lang,
        "system_message": system_message,
        "plugin": plugin,
        "attack_name": "None",
        "error": error_message,
    }

    # Add guardrail info if triggered
    if guardrail:
        result_dict["guardrail"] = True
        result_dict["guardrail_categories"] = guardrail_categories

    return result_dict, success


def process_entry(
    entry,
    target_module,
    attempts=1,
    attack_name="",
    attack_module=None,
    attack_iterations=0,
    attack_options=None,
    output_file=None,
    attempts_bar=None,
    global_lock=None,
):
    """
    Processes one dataset entry.

    First, it performs a single standard attempt by calling _do_single_request().
    The final standard attempt result is recorded (with "attack_name": "None").
    If this attempt is unsuccessful and an attack module is provided,
    it then calls the attack() method and records its result as a separate entry.

    The target_module passed here is assumed to be wrapped (AdvancedTargetWrapper)
    and therefore already handles retries and multiple attempts.

    Returns:
      List[dict]: A list containing one or two result entries.
    """
    original_input = entry["text"]
    std_result = None
    std_success = False

    # Attempt Logic
    for attempt_num in range(1, attempts + 1):
        std_result, success_now = _do_single_request(
            entry,
            original_input,
            target_module,
            output_file,
            attempt_num,
            attempts_bar,
            global_lock,
        )
        if success_now:
            std_success = True
            break

    results_list = [std_result]

    if std_success and attack_module:
        # Remove all the attempts that we are not going to do any longer as we are skipping the dynamic attacks
        with global_lock:
            attempts_bar.total = attempts_bar.total - attack_iterations

    # If the standard attempt fail and an attack module is provided, run the dynamic attack.
    if (not std_success) and attack_module:
        try:
            start_time = time.time()
            effective_attack_options = (
                attack_options if attack_options else get_default_option(attack_module)
            )

            # Check if attack function accepts attack_options parameter
            sig = inspect.signature(attack_module.attack)
            params = sig.parameters

            if "attack_option" in params:
                attack_attempts, attack_success, attack_input, attack_response = (
                    attack_module.attack(
                        entry,
                        target_module,
                        call_judge,
                        attack_iterations,
                        attempts_bar,
                        global_lock,
                        attack_options,
                    )
                )
            else:
                # Backward compatibility for attacks without attack_option support
                attack_attempts, attack_success, attack_input, attack_response = (
                    attack_module.attack(
                        entry,
                        target_module,
                        call_judge,
                        attack_iterations,
                        attempts_bar,
                        global_lock,
                    )
                )

            end_time = time.time()
            response_time = end_time - start_time

            attack_result = {
                "id": f"{entry['id']}-attack",
                "long_id": entry["long_id"] + "-" + attack_name,
                "input": attack_input,
                "response": attack_response,
                "response_time": response_time,
                "success": attack_success,
                "judge_name": entry["judge_name"],
                "judge_args": entry["judge_args"],
                "judge_options": entry["judge_options"],
                "attempts": attack_attempts,
                "task_type": entry.get("task_type", None),
                "jailbreak_type": entry.get("jailbreak_type", None),
                "instruction_type": entry.get("instruction_type", None),
                "document_id": entry.get("document_id", None),
                "position": entry.get("position", None),
                "spotlighting_data_markers": entry.get(
                    "spotlighting_data_markers", None
                ),
                "injection_delimiters": entry.get("injection_delimiters", None),
                "suffix_id": entry.get("suffix_id", None),
                "lang": entry.get("lang", "en"),
                "system_message": entry.get("system_message", None),
                "plugin": entry.get("plugin", None),
                "error": None,
                "attack_name": attack_name,
                "attack_options": effective_attack_options,
            }
            results_list.append(attack_result)
        except Exception as e:
            traceback.print_exc()
            error_result = {
                "id": f"{entry['id']}-attack",
                "long_id": entry["long_id"] + "-" + attack_name + "-ERROR",
                "input": original_input,
                "response": "",
                "success": False,
                "judge_name": entry["judge_name"],
                "judge_args": entry["judge_args"],
                "judge_options": entry["judge_options"],
                "attempts": 1,
                "task_type": entry.get("task_type", None),
                "jailbreak_type": entry.get("jailbreak_type", None),
                "instruction_type": entry.get("instruction_type", None),
                "document_id": entry.get("document_id", None),
                "position": entry.get("position", None),
                "spotlighting_data_markers": entry.get(
                    "spotlighting_data_markers", None
                ),
                "injection_delimiters": entry.get("injection_delimiters", None),
                "suffix_id": entry.get("suffix_id", None),
                "lang": entry.get("lang", "en"),
                "system_message": entry.get("system_message", None),
                "plugin": entry.get("plugin", None),
                "error": str(e),
                "attack_name": attack_name,
                "attack_options": effective_attack_options,
            }
            results_list.append(error_result)

    return results_list


def _run_threaded(
    entries,
    target_module,
    attempts,
    attack_name,
    attack_module,
    attack_iters,
    attack_options,
    num_threads,
    total_attempts,
    initial_attempts,
    output_file,
    total_dataset_size,
    initial_processed,
    initial_success,
):
    lock = threading.Lock()
    bar_all = tqdm(
        total=total_attempts, desc="All attempts", position=1, initial=initial_attempts
    )
    bar_entries = tqdm(
        total=total_dataset_size,
        desc="Processing entries",
        position=0,
        initial=initial_processed,
    )
    bar_entries.set_postfix(success=initial_success)
    executor = ThreadPoolExecutor(max_workers=num_threads)
    futures = {
        executor.submit(
            process_entry,
            entry,
            target_module,
            attempts,
            attack_name,
            attack_module,
            attack_iters,
            attack_options,
            output_file,
            bar_all,
            lock,
        ): entry
        for entry in entries
    }
    success = initial_success
    try:
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                res = fut.result()
                if isinstance(res, list):
                    for r in res:
                        success += int(r.get("success", False))
                        append_jsonl_entry(output_file, r, lock)
                else:
                    success += int(res.get("success", False))
                    append_jsonl_entry(output_file, res, lock)
                bar_entries.update(1)
                bar_entries.set_postfix(success=success)
            except Exception as e:
                print(f"[Error] Entry ID {entry['id']}: {e}")
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n[Interrupt] CTRL+C pressed. Cancelling...")
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        bar_all.close()
        bar_entries.close()


def test_dataset(args):
    """
    Orchestrate testing of a dataset against a target.
    """
    # 1. Validate and process args, load modules and datasets
    tag = validate_and_get_tag(args.tag)

    # Load Attack module if specified
    attack_name = args.attack if args.attack else ""
    attack_module = (
        load_module_from_path(args.attack, "attacks") if args.attack else None
    )

    # Load Target module, with AdvancedTargetWrapper
    target_module = AdvancedTargetWrapper.create_target_wrapper(
        args.target,
        args.target_options,
        args.max_retries,
        args.throttle,
    )

    # Validate multi-turn capability
    if attack_module and hasattr(attack_module, "turn_type") and attack_module.turn_type == Turn.MULTI:
        # Validate target supports multi-turn
        if target_module.config.get("multi-turn", False):
            manager = multiprocessing.Manager()
            target_data = manager.dict()
            target_module.get_target().add_managed_dicts(target_data)

            print(f"[Info] Performing a multi-turn attack using '{attack_name}' on target '{args.target}'.")

        else:
            print(
                f"[Error] The selected attack '{attack_name}' requires multi-turn support, but the target '{args.target}' does not support multi-turn testing."
            )
            exit(1)

    else:
        # Validate target supports single-turn
        if not target_module.config.get("single-turn", False):
            print(
                f"[Error] The selected target '{args.target}' does not support single-turn testing."
            )
            exit(1)

        print(
            f"[Info] Performing a single-turn attack using '{attack_name}' on target '{args.target}'." if attack_module else f"[Info] Performing single-turn testing on target '{args.target}'."
        )

    # Obtain datasets and ensure resume-file is only used with single dataset
    datasets = process_jsonl_input_files(args.dataset, args.dataset_folder)

    # Prevent single dataset flags from being used with multiple datasets
    if len(datasets) > 1 and args.resume_file is not None:
        print(
            f"[Error] --resume-file cannot be used when testing multiple datasets. Currently selected {len(datasets)} datasets."
        )
        exit(1)

    # Print overview of datasets
    print("[Overview] Testing the following dataset(s): ")
    print("\n - " + "\n - ".join(datasets))

    # Print information about alternative resume flags
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    if not args.auto_resume and not args.no_auto_resume and tty:
        print(
            "\n[Info] Spikee supports the following resume flags, instead of the interactive prompt:\n  --auto-resume ~ silently pick the latest matching results file.\n  --no-auto-resume ~ create a new results file."
        )

    # 2. Prep datasets
    for dataset in datasets:
        print(
            f" \n[Start] Initiating testing of '{dataset.split(os.sep)[-1]}' against target '{args.target}'"
        )

        dataset_json = read_jsonl_file(dataset)
        dataset_json = (
            _apply_sampling(dataset_json, args.sample, args.sample_seed)
            if args.sample
            else dataset_json
        )

        # Determine resume action / file
        current_resume_file = args.resume_file
        picked = _determine_resume_file(args, dataset, tty)
        if picked:
            current_resume_file = picked

        # Load resume data if any has been selected
        completed_ids, results, already_done = _load_results_file(
            current_resume_file, attack_module, args.attack_iterations
        )

        # Identify unprocessed entries
        to_process = [
            entry for entry in dataset_json if entry["id"] not in completed_ids
        ]
        to_process = annotate_judge_options(to_process, args.judge_options)

        # Print if results completed, and skip
        if len(to_process) == 0:
            print(
                f"[Done] All entries have already been processed for dataset '{dataset}'. Skipping, please use `--no-auto-resume` to re-test."
            )
            continue

        # Create new results file and for resume, write existing results
        target_name_full = _build_target_name(args.target, args.target_options)
        output_file = prepare_output_file(
            "results",
            "results",
            target_name_full,
            dataset,
            tag,
        )
        write_jsonl_file(output_file, results)

        # 3. Run tests
        total_attempts = _calculate_total_attempts(
            len(to_process),
            args.attempts,
            args.attack_iterations,
            already_done,
            bool(attack_module),
        )
        print(f"[Info] Testing {len(to_process)} new entries (threads={args.threads}).")
        print(f"[Info] Output will be saved to: {output_file}")

        success_count = sum(1 for r in results if r.get("success"))
        _run_threaded(
            to_process,
            target_module,
            args.attempts,
            attack_name,
            attack_module,
            args.attack_iterations,
            args.attack_options,
            args.threads,
            total_attempts,
            already_done,
            output_file,
            len(dataset_json),
            len(completed_ids),
            success_count,
        )

        print(f"[Done] Testing finished. Results saved to {output_file}")
    print(f"[Overview] Tested {len(datasets)} dataset(s).")
