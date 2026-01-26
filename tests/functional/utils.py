from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
import os
import re


SEED_FOLDER = "datasets/seeds-functional-basic"


def _dataset_files(datasets_dir: Path) -> set[Path]:
    return set(datasets_dir.glob("*-dataset-*.jsonl"))


def run_generate_command(
    run_spikee,
    workspace: Path,
    extra_args: list[str] | None = None,
    match_languages: bool = True,
    seed_folder: str | None = None,
):
    datasets_dir = workspace / "datasets"
    before = _dataset_files(datasets_dir)
    folder = seed_folder or SEED_FOLDER
    args = ["generate", "--seed-folder", folder]
    if not match_languages:
        args.extend(["--match-languages", "false"])
    if extra_args:
        args.extend(extra_args)
    result = run_spikee(args, cwd=workspace)
    after = _dataset_files(datasets_dir)
    new_files = after - before
    assert len(new_files) == 1, f"Expected one new dataset, found {len(new_files)}"
    dataset_path = new_files.pop()
    return dataset_path, result


def run_test_command(run_spikee, workspace: Path, extra_args: list[str]):
    args = ["test", *extra_args]
    return run_spikee(args, cwd=workspace)


def run_results_command(
    run_spikee, workspace: Path, subcommand: str, extra_args: list[str]
):
    args = ["results", subcommand, *extra_args]
    return run_spikee(args, cwd=workspace)


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")


def filter_entries(entries: list[dict], **criteria) -> list[dict]:
    return [
        entry
        for entry in entries
        if all(entry.get(key) == value for key, value in criteria.items())
    ]


def base_long_id(long_id: str, plugin_name: str) -> str:
    marker = f"_{plugin_name}-"
    assert marker in long_id, f"Expected '{marker}' in long_id '{long_id}'"
    return long_id.rsplit(marker, 1)[0]


def split_base_and_plugin_entries(entries: list[dict], plugin_name: str):
    base_entries = [entry for entry in entries if entry.get("plugin") in (None, "None")]
    plugin_entries = [entry for entry in entries if entry.get("plugin") == plugin_name]
    base_by_long_id = {entry["long_id"]: entry for entry in base_entries}
    return base_entries, plugin_entries, base_by_long_id


def load_plugin_module(project_root: Path, module_path: str):
    module_rel = Path(*module_path.split("."))
    module_file = (project_root / module_rel).with_suffix(".py")
    assert module_file.exists(), f"Plugin module file not found: {module_file}"
    project_root_resolved = str(project_root.resolve())
    inserted = False
    if project_root_resolved not in sys.path:
        sys.path.insert(0, project_root_resolved)
        inserted = True
    try:
        spec = importlib.util.spec_from_file_location(module_path, module_file)
        assert spec and spec.loader, f"Unable to load spec for {module_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and sys.path[0] == project_root_resolved:
            sys.path.pop(0)


def extract_results_path(stdout: str, workspace: Path) -> Path:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if "Results saved to" in line or "Output will be saved to" in line:
            _, path_str = line.split("to", 1)
            path_str = path_str.strip()
            candidates = [
                workspace / path_str,
                Path(path_str),
            ]
            for candidate in candidates:
                candidate = candidate.expanduser().resolve()
                if candidate.exists():
                    return candidate
    raise AssertionError(f"Could not determine results path from output:\n{stdout}")


def extract_resource_name(file_name: str):
    """
    Takes a file path/name and extracts the resource name

    Example:
    datasets\\cybersec-2025-04-user-input-dataset-1762359770.jsonl => cybersec-2025-04-user-input-dataset-1762359770

    """
    file_name = os.path.basename(file_name)
    file_name = re.sub(r"^\d+-", "", file_name)
    file_name = re.sub(r".jsonl$", "", file_name)
    if file_name.startswith("seeds-"):
        file_name = file_name[len("seeds-"):]
    return file_name


def build_resource_name(*args) -> str:
    parts = [arg for arg in args if arg is not None]
    return "_".join(parts)


def extract_prefix_from_file_name(file_name: str):
    """
    Takes a file path/name and extracts the prefix before the first underscore.

    Example:
    /results/results_cybersec-2025-04-user-input-dataset_1762359770.jsonl => (results, cybersec-2025-04-user-input-dataset_1762359770)
    """
    file_name = os.path.basename(file_name)
    match = re.match(r"([^_]+)_(.+)", file_name)
    if match:
        return match.group(1), match.group(2)
    return None
