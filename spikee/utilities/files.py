import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# ==== File I/O Operations for JSONL files ====
def read_jsonl_file(file_path):
    """Reads a JSONL file and returns a list of JSON objects."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on line {i}: {e}")
                print(f"Faulty line {i}: {line}")
                raise

    return data


def read_toml_file(file_path):
    """
    Reads a TOML file and returns its parsed content.
    """
    with open(file_path, "rb") as f:
        return tomllib.load(f)


def write_jsonl_file(output_file, data):
    """Writes a list of JSON objects to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def append_jsonl_entry(output_file, entry, file_lock):
    """Appends a single entry to a JSONL file in a thread-safe manner."""
    with file_lock, open(output_file, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")


# ==== Processing JSONL Input Files ====
def process_jsonl_input_files(file_paths, folder_paths, file_type=None):
    """
    Spikee allows for input files to be specified either directly or via folders.
    This function processes both and returns a combined list of JSONL file paths.
    """
    result_files = file_paths or []

    for results_folder in folder_paths or []:
        folder_files = list_jsonl_files(results_folder, startswith=file_type)
        result_files.extend(folder_files)

    if result_files == []:
        print("[Error] No JSONL files included for analysis.")
        sys.exit(1)

    return result_files


def list_jsonl_files(folder_path, startswith=None):
    """Lists all JSONL results files in a given folder. (Must start with 'results' and end with '.jsonl')"""
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if (startswith is None or any(f.startswith(prefix) for prefix in startswith))
        and f.endswith(".jsonl")
        and os.path.isfile(os.path.join(folder_path, f))
    ]


# ==== Name Operations ====
def extract_resource_name(file_name: str):
    """
    Takes a file path/name and extracts the resource name

    Example:
    datasets\\cybersec-2026-01-user-input-dataset-1762359770.jsonl => cybersec-2026-01-user-input-dataset-1762359770

    """
    file_name = os.path.basename(file_name)
    file_name = re.sub(r"^\d+-", "", file_name)
    file_name = re.sub(r".jsonl$", "", file_name)
    file_name = file_name.removeprefix("seeds-")
    return file_name


def extract_prefix_from_file_name(file_name: str):
    """
    Takes a file path/name and extracts the prefix before the first underscore.

    Example:
    /results/results_cybersec-2026-01-user-input-dataset_1762359770.jsonl => (results, cybersec-2026-01-user-input-dataset_1762359770)
    """
    file_name = os.path.basename(file_name)
    match = re.match(r"([^_]+)_(.+)", file_name)
    if match:
        return match.group(1), match.group(2)
    return None


def extract_directory_from_file_path(file_path: str):
    """
    Takes a file path and extracts the directory.

    Example:
    /results/results_cybersec-2026-01-user-input-dataset_1762359770.jsonl => /results
    """
    return os.path.dirname(file_path)


def compact_filename_part(value: str, max_length: int = 64) -> str:
    """Keep short safe names; append a hash whenever cleaning or shortening loses text."""
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    readable = re.sub(r"-+", "-", readable).strip("-._")
    if readable == value and len(readable) <= max_length:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    readable = readable[: max_length - 13].rstrip("-._")
    return f"{readable}-{digest}" if readable else digest


def build_resource_name(*args) -> str:
    parts = [compact_filename_part(arg) for arg in args if arg is not None]
    # Leave room for the timestamp, extension, and derived output suffixes.
    return compact_filename_part("_".join(parts), max_length=160)


def build_file_name(prefix, *args) -> str:
    ts = int(time.time())
    return f"{build_resource_name(prefix, *args)}_{ts}.jsonl"


def does_resource_name_match(path: Path, resource_name: str) -> bool:
    """
    Accept files named like:
      <resource_name>_<ts>.jsonl

    Reject anything that has extra segments before the timestamp.
    """

    name = path.name
    if name.startswith(resource_name + "_") and name.endswith(".jsonl"):
        remainder = name[
            len(resource_name) + 1 : -len(".jsonl")
        ]  # The remaining text should only be a timestamp
        return remainder.isdigit()
    else:
        return False


def prepare_output_file(directory, file_type, name_full, dataset_path, tag):
    filename = build_file_name(
        file_type,
        name_full,
        None if dataset_path is None else extract_resource_name(dataset_path),
        tag,
    )

    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)
