import asyncio
import inspect
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from InquirerPy import inquirer
from tqdm import tqdm

from spikee.judge import annotate_judge_options, call_judge
from spikee.templates.attack import Attack
from spikee.templates.target import Target
from spikee.utilities.attack import (
    AttackProgress,
    accepts_attack_history,
    invoke_attack,
)
from spikee.utilities.enums import Turn
from spikee.utilities.files import (
    append_jsonl_entry,
    build_resource_name,
    does_resource_name_match,
    extract_resource_name,
    prepare_output_file,
    process_jsonl_input_files,
    read_jsonl_file,
    write_jsonl_file,
)
from spikee.utilities.hinting import (
    AttackAttempt,
    Content,
    TargetResponseHint,
    content_factory,
    get_content,
    get_content_type,
    validate_content_signature,
)
from spikee.utilities.modules import get_default_option, load_module_from_path
from spikee.utilities.results import attack_parent_id, group_entries_with_attacks
from spikee.utilities.tags import validate_and_get_tag


class GuardrailTrigger(Exception):
    """Exception raised when a guardrail is triggered."""

    def __init__(self, message, categories: dict | None = None):
        super().__init__(message)
        self.categories = {} if categories is None else categories


class RetryableError(Exception):
    """Exception raised for errors that are retryable, such as 429 errors."""

    def __init__(self, message, retry_period=None):
        super().__init__(message)
        self.retry_period = retry_period


class MultiTurnSkip(Exception):
    """Exception raised to skip multi-turn entries being processed as single-turn."""


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

    def __init__(
        self, target_module: Target, target_options="", max_retries=3, throttle=0
    ):
        self.target_module: Target = target_module
        self.target_options: str = target_options
        self.max_retries: int = max_retries
        self.throttle: float = throttle

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
        input_text: Content,
        system_message: Content | None = None,
        logprobs=False,
        input_id=None,
        output_file=None,
        spikee_session_id=None,
        backtrack=False,
    ) -> TargetResponseHint:
        last_error: Exception | None = None
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

                if not validate_content_signature(
                    input_text, self.target_module.process_input, "input_text"
                ):
                    raise TypeError(
                        "Input content does not match the expected type for the target's process_input function."
                    )

                if system_message and not validate_content_signature(
                    system_message, self.target_module.process_input, "system_message"
                ):
                    raise ValueError(
                        "System message content does not match the expected type for the target's process_input function."
                    )

                if kwargs:
                    response: TargetResponseHint = self.target_module.process_input(
                        input_text=input_text, system_message=system_message, **kwargs
                    )
                else:
                    response: TargetResponseHint = self.target_module.process_input(
                        input_text=input_text, system_message=system_message
                    )

                # Unpack (response, meta) if tuple returned
                result: Content | bool
                meta: Any = None
                if isinstance(response, tuple):
                    if len(response) == 2:
                        response, meta = response

                    else:
                        raise ValueError(
                            f"Invalid tuple return from target's process_input. Expected (Content/bool, meta), got {len(response)} elements."
                        )

                if isinstance(response, (Content, bool)):
                    result = response

                else:
                    raise TypeError(
                        "Invalid response type from target's process_input. Expected Content, bool.",
                        str(type(response)),
                    )

                if self.throttle > 0:
                    time.sleep(self.throttle)

                return result, meta

            except GuardrailTrigger as gt:
                last_error = gt
                retries += 1

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

            except Exception as e:  # noqa: BLE001
                last_error = e
                if "429" in str(e) and retries < self.max_retries - 1:
                    wait_time = random.randint(30, 120)
                    time.sleep(wait_time)
                    retries += 1
                else:
                    break

        # All retries exhausted

        if last_error is None:
            last_error = Exception("Unknown error in AdvancedTargetWrapper.")
        raise last_error


# region resource_utilities
def _build_target_name(target, target_options):
    """
    Builds a target's name, returning "target-target_options".
    If no target_options provided, attempts to get default option from target module.
    """

    # Matches Invalid Windows Characters
    regex_pattern = r'(^[<>:"/\|?*]+)|([<>:"/\|?*]+$)|([<>:"/\|?*]+)'

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
        except Exception:  # noqa: BLE001
            pass

    if target_options is None:
        return target

    target_options = re.sub(
        regex_pattern, replacer, target_options
    )  # Remove Invalid Windows Characters
    return f"{target}-{target_options}"


def _load_results_file(resume_file, attack_module, attack_iters):
    completed_ids, results, already_done, entries_done = set(), [], 0, 0

    # Load Resume File, if selected.
    if resume_file and os.path.exists(resume_file):
        results = read_jsonl_file(resume_file)
        groups, _ = group_entries_with_attacks(results)
        complete_groups = [
            rows
            for rows in groups.values()
            if not any("entry_complete" in r for r in rows)
            or any(r.get("entry_complete") for r in rows)
        ]
        results = [r for rows in complete_groups for r in rows]
        completed_ids = {attack_parent_id(rows[0]) for rows in complete_groups}
        already_done = sum(r.get("attempts", 1) for r in results)
        entries_done = len(complete_groups)

        print(f"[Resume] Found {entries_done} completed entries in {resume_file}.")
    return completed_ids, results, already_done, entries_done


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
    n_entries, attempts, attack_iters, attack_only, already_done, has_attack
):
    per_item = attempts * (
        (attack_iters if has_attack else 0) + (0 if attack_only else 1)
    )
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
    dt = datetime.fromtimestamp(ts, UTC)
    age_sec = max(0, int((datetime.now(UTC) - dt).total_seconds()))
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
    except Exception:  # noqa: BLE001
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

    # Meta
    meta = None

    try:
        start_time = time.time()
        response, meta = target_module.process_input(
            input_text, system_message, False, entry_id, output_file
        )
        # Don't convert Content types to string - preserve wrapper objects for judge
        # Only convert to str if it's not bool and not a Content type
        if not isinstance(response, bool) and not isinstance(response, Content):
            response = str(response)

        end_time = time.time()
        response_time = end_time - start_time
        success = call_judge(entry, response)
        response_str = response
        error_message = None

    except GuardrailTrigger as gt:
        error_message = str(gt)
        response_str = ""
        response_time = time.time() - start_time
        success = False
        guardrail = True
        if hasattr(gt, "categories"):
            guardrail_categories = gt.categories
        # print("[Guardrail Triggered] {}: {}".format(entry["id"], error_message))

    except MultiTurnSkip as ms:
        error_message = str(ms)
        response_str = ""
        response_time = None
        success = False

    except ImportError:
        raise

    except Exception as e:  # noqa: BLE001
        error_message = str(e)
        response_str = ""
        response_time = None
        success = False
        print("[Error] {}: {}".format(entry["id"], error_message))
        traceback.print_exc()

    with global_lock:
        attempts_bar.update(1)

    # Handle boolean responses (from guardrail targets)
    if isinstance(response_str, bool):
        response_content = str(response_str)
        response_content_type = "text"
    else:
        response_content = get_content(response_str)
        response_content_type = get_content_type(response_str)

    result_dict = {
        "id": entry["id"],
        "long_id": entry["long_id"],
        "input": get_content(input_text),
        "input_type": get_content_type(input_text),
        "response": response_content,
        "response_type": response_content_type,
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
        "system_message": get_content(system_message) if system_message else None,
        "plugin": plugin,
        "attack_name": "None",
        "error": error_message,
    }

    # Add guardrail info if triggered
    if guardrail:
        result_dict["guardrail"] = True
        result_dict["guardrail_categories"] = guardrail_categories

    if meta:
        result_dict["meta"] = meta

    return result_dict, success


def _validate_attack_attempt(item):
    if not isinstance(item, AttackAttempt):
        raise TypeError("Attack history must contain AttackAttempt records")
    if type(item.attempts) is not int or item.attempts < 0:
        raise ValueError("Attack attempts must be a non-negative integer")
    if item.success is not None and type(item.success) is not bool:
        raise ValueError("Attack success must be True, False, or None (unjudged)")
    return item


def _attack_result(entry, item, attack_name, options):
    payload = item.input
    details = payload if isinstance(payload, dict) else {}
    payload = details.get("input", str(payload)) if details else payload
    input_type = get_content_type(payload) if isinstance(payload, Content) else "text"
    response_type = (
        get_content_type(item.response)
        if isinstance(item.response, Content)
        else "text"
    )
    row = {
        key: entry.get(key)
        for key in (
            "judge_name",
            "judge_args",
            "judge_options",
            "task_type",
            "jailbreak_type",
            "instruction_type",
            "document_id",
            "position",
            "spotlighting_data_markers",
            "injection_delimiters",
            "suffix_id",
            "system_message",
            "plugin",
        )
    }
    row.update(
        id=f"{entry['id']}-attack",
        long_id=entry["long_id"] + "-" + attack_name + ("-ERROR" if item.error else ""),
        input=get_content(payload) if isinstance(payload, Content) else str(payload),
        input_type=input_type,
        response=get_content(item.response)
        if isinstance(item.response, Content)
        else str(item.response),
        response_type=response_type,
        response_time=item.response_time,
        success=item.success,
        attempts=item.attempts,
        lang=entry.get("lang", "en"),
        error=item.error,
        attack_name=attack_name,
        attack_options=options,
    )
    if "conversation" in details:
        row["conversation"] = details["conversation"]
    if "objective" in details:
        row["objective"] = details["objective"]
    if item.guardrail:
        row.update(guardrail=True, guardrail_categories=item.guardrail_categories or {})
    return row


def process_entry(
    entry,
    target_module,
    attempts=1,
    attack_name="",
    attack_module: Attack | None = None,
    attack_iterations=0,
    attack_options=None,
    attack_only=False,
    output_file=None,
    attempts_bar=None,
    global_lock=None,
    attack_return_all_attempts=False,
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
    # Create Content object from entry (new format) or fall back to plain text (legacy)
    content_type = entry.get("content_type", "text")
    content = entry.get("content", entry.get("text"))
    entry["text"] = (
        content  # For backward compatibility with attacks that expect 'text' field
    )
    original_input = content_factory(content, content_type)

    std_result = None
    std_success = False

    # Attempt Logic
    if not attack_only:
        request_attempts = 0
        for attempt_num in range(1, attempts + 1):
            request_attempts += 1
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

        if std_success:
            # Remove all the attempts that we are not going to do any longer as we are skipping the dynamic attacks
            with global_lock:
                attempts_bar.total = (
                    attempts_bar.total
                    - (attempts - request_attempts)
                    - (attack_iterations * attempts if attack_module else 0)
                )
                attempts_bar.refresh()

    else:
        std_success = False
        results_list = []

    if (not std_success) and attack_module:
        effective_options = attack_options or get_default_option(attack_module)
        request_attempts = 0
        recorded = []
        has_history = False
        representative = None
        start_time = time.monotonic()
        for invocation in range(1, attempts + 1):
            invocation_failed = False
            progress = (
                AttackProgress(attempts_bar, attack_iterations)
                if attempts_bar
                else None
            )
            try:
                returned = invoke_attack(
                    attack_module.attack,
                    entry,
                    target_module,
                    call_judge,
                    attack_iterations,
                    progress,
                    global_lock,
                    effective_options,
                    attack_return_all_attempts,
                )
                if isinstance(returned, list):
                    if not returned:
                        raise ValueError("Attack returned an empty attempt list")
                    current = [_validate_attack_attempt(item) for item in returned]
                else:
                    count, success, payload, response = returned
                    current = [
                        _validate_attack_attempt(
                            AttackAttempt(payload, response, success, attempts=count)
                        )
                    ]
                used = sum(item.attempts for item in current)
                attack_success = any(item.success for item in current)
                representative = next(
                    (item for item in current if item.success), current[-1]
                )
                request_attempts += used
                if attack_return_all_attempts:
                    expanded = isinstance(returned, list)
                    has_history |= expanded
                    recorded.extend((invocation, item, expanded) for item in current)
            except Exception as exc:  # noqa: BLE001
                invocation_failed = True
                representative = AttackAttempt(
                    original_input, "", False, attempts=0, error=str(exc)
                )
                used = progress.n if progress else 0
                representative.attempts = used
                request_attempts += used
                attack_success = False
                if attack_return_all_attempts:
                    recorded.append((invocation, representative, False))
            if progress:
                with global_lock:
                    # Reconcile attacks that omit the successful call's update.
                    attempts_bar.update(used - progress.n)
            if attack_success or invocation_failed:
                break

        if attempts_bar:
            with global_lock:
                attempts_bar.total -= attempts * attack_iterations - request_attempts
                attempts_bar.refresh()

        if has_history:
            for number, (invocation, item, expanded) in enumerate(recorded, 1):
                row = _attack_result(entry, item, attack_name, effective_options)
                row.update(
                    id=f"{entry['id']}-attack-{number}",
                    long_id=f"{entry['long_id']}-{attack_name}-attempt-{number}",
                    attack_parent_id=entry["id"],
                    attack_parent_long_id=entry["long_id"],
                    attack_attempt=number,
                    attack_invocation=invocation,
                    attack_result_format="attempt" if expanded else "representative",
                )
                results_list.append(row)
        else:
            representative.attempts = request_attempts
            representative.response_time = time.monotonic() - start_time
            row = _attack_result(entry, representative, attack_name, effective_options)
            if attack_return_all_attempts:
                row.update(
                    attack_result_format="representative",
                    attack_parent_id=entry["id"],
                    attack_parent_long_id=entry["long_id"],
                )
            results_list.append(row)

    if attack_return_all_attempts and results_list:
        # A final marker lets resume distinguish a complete entry from a partial
        # JSONL group. Partial groups are rerun from the original dataset entry.
        for row in results_list:
            row["entry_complete"] = False
        results_list[-1]["entry_complete"] = True

    return results_list


def _run_threaded(
    entries,
    target_module,
    attempts,
    attack_name,
    attack_module,
    attack_iters,
    attack_options,
    attack_only,
    num_threads,
    total_attempts,
    initial_attempts,
    output_file,
    total_dataset_size,
    initial_processed,
    initial_success,
    initial_guardrail,
    attack_return_all_attempts=False,
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
    if initial_guardrail > 0:
        bar_entries.set_postfix(success=initial_success, guardrails=initial_guardrail)
    else:
        bar_entries.set_postfix(success=initial_success)

    _thread_local = threading.local()

    def _thread_init():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.asyncio_loop = loop

    def _thread_cleanup():
        loop = getattr(_thread_local, "asyncio_loop", None)
        if loop is not None:
            loop.close()
            asyncio.set_event_loop(None)

    executor = ThreadPoolExecutor(max_workers=num_threads, initializer=_thread_init)
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
            attack_only,
            output_file,
            bar_all,
            lock,
            attack_return_all_attempts,
        ): entry
        for entry in entries
    }
    success = initial_success
    guardrail = initial_guardrail
    try:
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                res = fut.result()
                rows = res if isinstance(res, list) else [res]
                success += int(any(r.get("success") for r in rows))
                guardrail += int(all(r.get("guardrail") for r in rows))
                for row in rows:
                    append_jsonl_entry(output_file, row, lock)
                bar_entries.update(1)

                if guardrail > 0:
                    bar_entries.set_postfix(success=success, guardrails=guardrail)
                else:
                    bar_entries.set_postfix(success=success)

            except ImportError as ie:
                bar_all.close()
                bar_entries.close()
                print(f"Import Error Entry ID {entry['id']}: {ie}")
                print(
                    "Exiting early due to ImportError in attack module. Please import required dependencies and re-run."
                )
                sys.exit(1)

            except Exception as e:  # noqa: BLE001
                print(f"[Error] Entry ID {entry['id']}: {e}")
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n[Interrupt] CTRL+C pressed. Cancelling...")
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        bar_all.close()
        for thread in executor._threads:
            _thread_cleanup()
        bar_entries.close()


def test_dataset(args):
    """
    Orchestrate testing of a dataset against a target.
    """
    # 1. Validate and process args, load modules and datasets
    tag = validate_and_get_tag(args.tag)

    # Load Attack module if specified
    attack_name = args.attack if args.attack else ""
    try:
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
    except ImportError as e:
        print(e)
        sys.exit(1)

    # Validate multi-turn capability
    if (
        getattr(args, "attack_return_all_attempts", False)
        and attack_module
        and not accepts_attack_history(attack_module.attack)
    ):
        print(
            f"[Warning] Attack '{attack_name}' does not support returning all attempts; "
            "retaining its representative result."
        )

    if (
        attack_module
        and hasattr(attack_module, "turn_type")
        and attack_module.turn_type == Turn.MULTI
    ):
        # Validate target supports multi-turn
        if target_module.config.get("multi-turn", False):
            print(
                f"[Info] Performing a multi-turn attack using '{attack_name}' on target '{args.target}'."
            )

        else:
            print(
                f"[Error] The selected attack '{attack_name}' requires multi-turn support, but the target '{args.target}' does not support multi-turn testing."
            )
            sys.exit(1)

    else:
        # Validate target supports single-turn
        if not target_module.config.get("single-turn", False):
            print(
                f"[Error] The selected target '{args.target}' does not support single-turn testing."
            )
            sys.exit(1)

        print(
            f"[Info] Performing a single-turn attack using '{attack_name}' on target '{args.target}'."
            if attack_module
            else f"[Info] Performing single-turn testing on target '{args.target}'."
        )

    if hasattr(target_module.get_target(), "add_managed_dicts"):
        manager = multiprocessing.Manager()
        target_data = manager.dict()
        target_module.get_target().add_managed_dicts(target_data)

    # Obtain datasets and ensure resume-file is only used with single dataset
    datasets = process_jsonl_input_files(args.dataset, args.dataset_folder)

    # Prevent single dataset flags from being used with multiple datasets
    if len(datasets) > 1 and args.resume_file is not None:
        print(
            f"[Error] --resume-file cannot be used when testing multiple datasets. Currently selected {len(datasets)} datasets."
        )
        sys.exit(1)

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
        completed_ids, results, already_done, entries_done = _load_results_file(
            current_resume_file, attack_module, args.attack_iterations
        )

        # Identify unprocessed entries
        completed_id_strings = {str(i) for i in completed_ids}
        to_process = [
            entry
            for entry in dataset_json
            if str(entry["id"]) not in completed_id_strings
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
            args.attack_only,
            already_done,
            bool(attack_module),
        )
        print(f"[Info] Testing {len(to_process)} new entries (threads={args.threads}).")
        print(f"[Info] Output will be saved to: {output_file}")

        groups, _ = group_entries_with_attacks(results)
        success_count = sum(
            any(r.get("success") for r in rows) for rows in groups.values()
        )
        guardrail_count = sum(
            all(r.get("guardrail") for r in rows) for rows in groups.values()
        )

        _run_threaded(
            to_process,
            target_module,
            args.attempts,
            attack_name,
            attack_module,
            args.attack_iterations,
            args.attack_options,
            args.attack_only,
            args.threads,
            total_attempts,
            already_done,
            output_file,
            len(dataset_json),
            entries_done,
            success_count,
            guardrail_count,
            getattr(args, "attack_return_all_attempts", False),
        )

        print(f"[Done] Testing finished. Results saved to {output_file}")
    print(f"[Overview] Tested {len(datasets)} dataset(s).")
