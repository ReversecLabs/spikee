import os
import re
import json
import time
import importlib
import random
import threading
import inspect
import traceback
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

from .judge import annotate_judge_options, call_judge


class AdvancedTargetWrapper:
    """
    A wrapper for a target module's process_input method that incorporates both:
      - A loop for a given number of independent attempts (num_attempts), and
      - A retry strategy for handling 429 errors (max_retries) with throttling.

    This is designed to be passed to the attack() function so that each call to process_input()
    will try up to num_attempts times (each with up to max_retries on quota errors) before failing.

    Parameters:
      target_module: The original target module that provides process_input(input_text, system_message[, logprobs]).
      target_options: Target options, typically a string representing the name of the llm to call
      num_attempts (int): Number of independent attempts to call process_input per invocation.
      max_retries (int): Maximum number of retries per attempt (e.g. on 429 errors).
      throttle (float): Number of seconds to wait after a successful call.
    """

    def __init__(self, target_module, target_options=None, max_retries=3, throttle=0):
        self.target_module = target_module
        self.target_options = target_options
        self.max_retries = max_retries
        self.throttle = throttle

        sig = inspect.signature(self.target_module.process_input)
        params = sig.parameters
        # detect optional parameters that were only added in newer Spikee versions
        self.supports_options = "target_options" in params
        self.supports_logprobs = "logprobs" in params

    def process_input(self, input_text, system_message=None, logprobs=False):
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

                # Delegate to the wrapped process_input
                if kwargs:
                    result = self.target_module.process_input(
                        input_text, system_message, **kwargs
                    )
                else:
                    result = self.target_module.process_input(
                        input_text, system_message
                    )

                # Unpack (response, logprobs) if tuple returned
                if isinstance(result, tuple) and len(result) == 2:
                    response, lp = result
                else:
                    response, lp = result, None

                if self.throttle > 0:
                    time.sleep(self.throttle)

                return response, lp

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


def validate_tag(tag):
    """
    Validates that a tag is safe to use in a filename.

    Args:
        tag (str): The tag to validate

    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if tag is valid, False otherwise
            - error_message (str): Reason for validation failure or None if valid
    """
    if tag is None:
        return True, None

    # Check for empty string after stripping whitespace
    if len(tag.strip()) == 0:
        return False, "Tag cannot be empty or whitespace only"

    # Check length (reasonable max length for filename component)
    MAX_LENGTH = 50
    if len(tag) > MAX_LENGTH:
        return False, f"Tag exceeds maximum length of {MAX_LENGTH} characters"

    # Check for valid characters - alphanumeric, dash and underscore only
    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
    if not pattern.match(tag):
        return (
            False,
            "Tag can only contain letters, numbers, dash (-) and underscore (_)",
        )

    return True, None


def extract_dataset_name(file_name):
    file_name = os.path.basename(file_name)
    file_name = re.sub(r"^\d+-", "", file_name)
    file_name = re.sub(r".jsonl$", "", file_name)
    if file_name.startswith("seeds-"):
        file_name = file_name[len("seeds-") :]
    return file_name


def _results_prefix(target_name_full: str, dataset_path: str, tag: str | None) -> str:
    base = extract_dataset_name(dataset_path)
    parts = [f"results_{target_name_full}", base]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def _parse_timestamp_from_filename(p: Path) -> int:
    # Expect ..._<ts>.jsonl at the end; fall back to mtime if parse fails
    name = p.name
    try:
        ts_str = name.rsplit("_", 1)[-1].removesuffix(".jsonl")
        return int(ts_str)
    except Exception:
        return int(p.stat().st_mtime)


def _find_resume_candidates(
    results_dir: str | Path, target_name_full: str, dataset_path: str, tag: str | None
) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    # Prefer exact tag if provided, else list all
    prefix = _results_prefix(target_name_full, dataset_path, tag)
    primary = sorted(
        results_dir.glob(f"{prefix}_*.jsonl"),
        key=_parse_timestamp_from_filename,
        reverse=True,
    )

    if tag:
        # Also include related files without the tag if none found with tag
        fallback_prefix = _results_prefix(target_name_full, dataset_path, None)
        fallback = sorted(
            results_dir.glob(f"{fallback_prefix}_*.jsonl"),
            key=_parse_timestamp_from_filename,
            reverse=True,
        )
        # De-duplicate while preserving order
        seen = set(p.resolve() for p in primary)
        for f in fallback:
            r = f.resolve()
            if r not in seen:
                primary.append(f)
                seen.add(r)

    return primary


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


def _select_resume_file_interactive(
    cands: list[Path], preselect_index: int = 0
) -> Path | None:
    # Try simple-term-menu if available
    try:
        from simple_term_menu import TerminalMenu  # type: ignore

        items = [_format_candidate_line(p) for p in cands]
        menu = TerminalMenu(
            items,
            title="Resume from which results file? (Enter to confirm)",
            menu_cursor="➤ ",
            menu_cursor_style=("bold",),
            cycle_cursor=True,
            clear_screen=True,
            preview_command=None,
            preview_size=0,
            cursor_index=preselect_index,
        )
        idx = menu.show()
        if idx is None:
            return None
        return cands[idx]
    except Exception:
        # Fallback: compact pager showing 3 at a time
        i = 0
        page = 3
        default = 0  # newest
        while True:
            end = min(i + page, len(cands))
            print("\nResume candidates (newest first):")
            for k, p in enumerate(cands[i:end], start=1):
                print(f"{k}) {_format_candidate_line(p)}")
            if end < len(cands):
                print("[M] More   [L] Latest   [N] None   [#] Choose", end=" ")
            else:
                print("[L] Latest   [N] None   [#] Choose", end=" ")
            choice = input().strip().lower()
            if choice == "":
                return cands[default]
            if choice == "l":
                return cands[default]
            if choice == "n":
                return None
            if choice == "m" and end < len(cands):
                i += page
                continue
            if choice.isdigit():
                idx = int(choice) - 1 + i
                if 0 <= idx < end:
                    return cands[idx]
            print("Invalid choice.")


def _maybe_pick_resume_file(args, is_tty: bool) -> str | None:
    # Respect explicit --resume-file
    if getattr(args, "resume_file", None):
        return args.resume_file

    # If TTY and user explicitly disabled auto-resume, do nothing
    if is_tty and getattr(args, "no_auto_resume", False):
        return None

    # Build scope and find candidates
    target_name_full = _build_target_name(args.target, args.target_options)
    cands = _find_resume_candidates("results", target_name_full, args.dataset, args.tag)

    if not cands:
        return None

    # ---- TTY behavior: prompt by default unless --no-auto-resume was set above ----
    if is_tty:
        if len(cands) == 1:
            print(f"[Auto-Resume] Found: {_format_candidate_line(cands[0])}")
            resp = input("Resume this file? [Y/n] ").strip().lower()
            if resp in ("", "y", "yes"):
                return str(cands[0])
            return None

        picked = _select_resume_file_interactive(cands, preselect_index=0)
        return str(picked) if picked else None

    # ---- Non-TTY behavior: do nothing unless --auto-resume ----
    if getattr(args, "auto_resume", False):
        # silently pick latest
        print(f"[Auto-Resume] Using latest: {cands[0].name}")
        return str(cands[0])

    return None


def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl_file(output_file, data):
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def append_jsonl_entry(output_file, entry, file_lock):
    """Appends a single entry to a JSONL file in a thread-safe manner."""
    with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def check_canary_in_response(response, canary):
    if not canary:
        return False
    return canary in response


def load_module_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(f"Could not load module {module_name} from {path}")


def _load_raw_target_module(target_name):
    """Load target module without wrapping. Returns None if not found."""
    local_path = os.path.join(os.getcwd(), "targets", f"{target_name}.py")
    if os.path.isfile(local_path):
        return load_module_from_path(local_path, target_name)
    else:
        try:
            return importlib.import_module(f"spikee.targets.{target_name}")
        except ModuleNotFoundError:
            return None


def load_target_module(target_name, target_options, max_retries, throttle):
    target_mod = _load_raw_target_module(target_name)
    if target_mod is None:
        raise ValueError(
            f"Target '{target_name}' not found locally or in spikee.targets/"
        )

    # Wrap the target module with AdvancedTargetWrapper
    return AdvancedTargetWrapper(
        target_mod,
        max_retries=max_retries,
        throttle=throttle,
        target_options=target_options,
    )


def load_attack_by_name(attack_name):
    """
    Loads an attack module from a new "attacks" folder or from built-in package data.
    """
    local_attack_path = Path(os.getcwd()) / "attacks" / f"{attack_name}.py"
    if local_attack_path.is_file():
        spec = importlib.util.spec_from_file_location(attack_name, local_attack_path)
        attack_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(attack_module)
        return attack_module
    try:
        return importlib.import_module(f"spikee.attacks.{attack_name}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Attack '{attack_name}' not found locally or in spikee.attacks"
        )


def _get_effective_attack_options(attack_module, provided_options):
    """Get effective attack options, using default if none provided and attack supports options."""
    if provided_options is not None:
        return provided_options

    # Try to get default option if none provided
    if attack_module and hasattr(attack_module, "get_available_option_values"):
        try:
            available = attack_module.get_available_option_values()
            if available:
                return available[0]  # First option is default
        except Exception:
            pass

    return None


def _do_single_request(
    entry, input_text, target_module, num_attempt, attempts_bar, global_lock
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

    try:
        start_time = time.time()
        response, _ = target_module.process_input(input_text, system_message)
        end_time = time.time()
        response_time = end_time - start_time
        success = call_judge(entry, response)
        response_str = response if isinstance(response, str) else ""
        error_message = None
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
    return result_dict, success


def process_entry(
    entry,
    target_module,
    attempts=1,
    attack_module=None,
    attack_iterations=0,
    attack_options=None,
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

    for attempt_num in range(1, attempts + 1):
        std_result, success_now = _do_single_request(
            entry, original_input, target_module, attempt_num, attempts_bar, global_lock
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

            effective_attack_options = _get_effective_attack_options(
                attack_module, attack_options
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
                "long_id": entry["long_id"] + "-" + attack_module.__name__,
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
                "attack_name": attack_module.__name__,
                "attack_options": effective_attack_options,
            }
            results_list.append(attack_result)
        except Exception as e:
            error_result = {
                "id": f"{entry['id']}-attack",
                "long_id": entry["long_id"] + "-" + attack_module.__name__ + "-ERROR",
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
                "attack_name": attack_module.__name__,
                "attack_options": effective_attack_options,
            }
            results_list.append(error_result)

    return results_list


def _validate_and_get_tag(tag):
    if not tag:
        return None
    valid, err = validate_tag(tag)
    if not valid:
        print(f"Error: Invalid tag: {err}")
        exit(1)
    return tag


def _load_attack(attack_name):
    if not attack_name:
        return None
    return load_attack_by_name(attack_name)


def _apply_sampling(dataset, pct, seed_arg):
    if pct is None:
        return dataset
    if seed_arg == "random":
        seed = random.randint(0, 2**32 - 1)
        print(f"[Info] Using random seed for sampling: {seed}")
    else:
        seed = int(seed_arg)
        print(f"[Info] Using seed for sampling: {seed}")
    random.seed(seed)
    size = round(len(dataset) * pct)
    print(
        f"[Info] Sampled {size} entries from {len(dataset)} total entries ({pct:.1%})"
    )
    return random.sample(dataset, size)


def _load_resume(resume_file, attack_module, attack_iters):
    completed, results = set(), []
    already_done = 0
    if resume_file and os.path.exists(resume_file):
        existing = read_jsonl_file(resume_file)
        completed = {r["id"] for r in existing}
        results = existing
        print(f"[Resume] Found {len(completed)} completed entries in {resume_file}.")
        no_attack = sum(1 for r in existing if r.get("attack_name") == "None")
        with_attack = len(existing) - no_attack
        already_done = no_attack + with_attack * attack_iters
    return completed, results, already_done


def _filter_entries(dataset, completed_ids):
    return [e for e in dataset if e["id"] not in completed_ids]


def _build_target_name(name, opts):
    """Build target name, using default option if opts is None and target supports options."""

    regex_pattern = '(^[<>:"/\|?*]+)|([<>:"/\|?*]+$)|([<>:"/\|?*]+)'  # Matches Invalid Windows Characters,

    def replacer(match):
        if match.group(1) or match.group(2):  # If at start/end of string, just remove
            return ""
        else:  # If in middle of string, replace with '~'
            return "~"

    if opts is not None:
        opts = re.sub(
            regex_pattern, replacer, opts
        )  # Remove Invalid Windows Characters
        return f"{name}-{opts}"

    # Try to get default option if none provided
    try:
        mod = _load_raw_target_module(name)
        if mod and hasattr(mod, "get_available_option_values"):
            available = mod.get_available_option_values()
            if available:
                opts = re.sub(
                    regex_pattern, replacer, available[0]
                )  # Remove Invalid Windows Characters
                return f"{name}-{opts}"
    except Exception:
        pass

    return name


def _prepare_output_file(results_dir, target_name_full, dataset_path, tag):
    ts = int(time.time())
    base = extract_dataset_name(dataset_path)
    parts = [f"results_{target_name_full}", base]
    if tag:
        parts.append(tag)
    parts.append(str(ts))
    filename = "_".join(parts) + ".jsonl"
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)


def _write_initial_results(path, results):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def _calculate_total_attempts(
    n_entries, attempts, attack_iters, already_done, has_attack
):
    per_item = attempts + (attack_iters if has_attack else 0)
    return n_entries * per_item + already_done


def _print_info(n_new, threads, output_file):
    print(f"[Info] Testing {n_new} new entries (threads={threads}).")
    print(f"[Info] Output will be saved to: {output_file}")


def _run_threaded(
    entries,
    target_module,
    attempts,
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
            attack_module,
            attack_iters,
            attack_options,
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
    # 1. Validate inputs and prepare
    tag = _validate_and_get_tag(args.tag)

    # Auto-resume (decide resume file before loading anything)
    tty = sys.stdin.isatty() and sys.stdout.isatty()
    picked = _maybe_pick_resume_file(args, tty)
    if picked:
        args.resume_file = picked

    attack_module = _load_attack(args.attack)
    target_module = load_target_module(
        args.target,
        args.target_options,
        args.max_retries,
        args.throttle,
    )
    dataset = read_jsonl_file(args.dataset)
    dataset = _apply_sampling(dataset, args.sample, args.sample_seed)

    completed_ids, results, already_done = _load_resume(
        args.resume_file, attack_module, args.attack_iterations
    )

    to_process = _filter_entries(dataset, completed_ids)
    to_process = annotate_judge_options(to_process, args.judge_options)

    target_name_full = _build_target_name(args.target, args.target_options)

    output_file = _prepare_output_file(
        "results",
        target_name_full,
        args.dataset,
        tag,
    )
    _write_initial_results(output_file, results)

    # 2. Run tests
    total_attempts = _calculate_total_attempts(
        len(to_process),
        args.attempts,
        args.attack_iterations,
        already_done,
        bool(attack_module),
    )
    _print_info(len(to_process), args.threads, output_file)

    success_count = sum(1 for r in results if r.get("success"))
    _run_threaded(
        to_process,
        target_module,
        args.attempts,
        attack_module,
        args.attack_iterations,
        args.attack_options,
        args.threads,
        total_attempts,
        already_done,
        output_file,
        len(dataset),
        len(completed_ids),
        success_count,
    )

    print(f"[Done] Testing finished. Results saved to {output_file}")
