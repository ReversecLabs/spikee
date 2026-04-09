from pathlib import Path
from typing import List
import os
from unittest import result

def get_datasets(workspace_dir: Path) -> set[Path]:
    datasets_dir = workspace_dir / "datasets"
    if not datasets_dir.exists():
        return set()
    
    return set(p for p in datasets_dir.iterdir() if p.is_file())

def get_results(workspace_dir: Path) -> set[Path]:
    results_dir = workspace_dir / "results"
    if not results_dir.exists():
        return set()
    
    return set(p for p in results_dir.iterdir() if p.is_file())

def judge_dataset_filename(judge_variant: str) -> str:
    return (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )

def create_judge_results(
    run_spikee, workspace_dir: Path, target_name: str, judge_variant: str
):
    dataset_path = workspace_dir / "datasets" / judge_dataset_filename(judge_variant)
    assert dataset_path.exists()

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
            target=target_name,
            datasets=[dataset_path],
            additional_args=[
                "--judge-options",
                f"{judge_variant}:mode=fail",
            ],
    )
    return results_file[0] if isinstance(results_file, list) else results_file

def spikee_list(
        run_spikee, 
        workspace_dir, 
        module: str
    ) -> list[str]:
    """Helper function to run `spikee list <entity>` and return the output lines as a list."""
    result = run_spikee(["list", module], cwd=workspace_dir)
    return result.stdout.strip().splitlines()

def spikee_generate_cli(
        run_spikee, 
        workspace_dir,
        seed_folder: str = "datasets/seeds-functional-basic",
        additional_args: list[str] = [],
    ):
    """Helper function to run `spikee generate`"""

    init_datasets = get_datasets(workspace_dir)
    run_spikee(["generate", "--seed-folder", seed_folder, *additional_args], cwd=workspace_dir)
    new_datasets = get_datasets(workspace_dir) - init_datasets

    assert len(new_datasets) == 1, f"Expected exactly one new dataset to be generated, but found {len(new_datasets)}. New datasets: {new_datasets}"
    return new_datasets.pop()

def spikee_test_cli(
        run_spikee, 
        workspace_dir,
        target: str = "mock_target",
        datasets: List[Path] = [],
        additional_args: list[str] = [],
    ):
    """Helper function to run `spikee test`"""

    if datasets == []:
        dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
        datasets = [dataset_path]

    for dataset in datasets:
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset}")
        
        elif dataset.is_file():
            additional_args = [ "--dataset", str(dataset), *additional_args]
        
        elif dataset.is_dir():
            additional_args = [ "--dataset-folder", str(dataset), *additional_args]

    init_results = get_results(workspace_dir)
    result = run_spikee(["test", "--target", target, *additional_args], cwd=workspace_dir)
    new_results = get_results(workspace_dir) - init_results

    assert len(new_results) > 0, f"Expected at least one new results file to be generated, but found {len(new_results)}. New results: {new_results}"
    return list(new_results), result

def spikee_analyze_cli(
        run_spikee, 
        workspace_dir,
        result_files: List[Path] = [],
        additional_args: list[str] = [],
    ):
    """Helper function to run `spikee analyze`"""

    if result_files == []:
        raise ValueError("At least one result file must be provided for analysis.")
    
    for result_file in result_files:
        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")
        
        elif result_file.is_file():
            additional_args = [ "--result-file", str(result_file), *additional_args]
        
        elif result_file.is_dir():
            additional_args = [ "--result-folder", str(result_file), *additional_args]
    
    analyze_result = run_spikee(["results", "analyze", *additional_args], cwd=workspace_dir)

    return analyze_result.stdout

def spikee_extract_cli(
        run_spikee,
        workspace_dir,
        result_files: List[Path] = [],
        category: str = "success",
        custom_search: List[str] = [],
    ):
    """Helper function to run `spikee results extract`.

    Returns (list[Path] of new extract files, CompletedProcess result).
    """

    if result_files == []:
        raise ValueError("At least one result file must be provided for extraction.")

    additional_args: List[str] = []

    for result_file in result_files:
        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")

        elif result_file.is_file():
            additional_args.extend(["--result-file", str(result_file)])

        elif result_file.is_dir():
            additional_args.extend(["--result-folder", str(result_file)])

    command = ["results", "extract", "--category", category, *additional_args]

    for search in custom_search:
        command.extend(["--custom-search", search])

    init_results = get_results(workspace_dir)
    result = run_spikee(command, cwd=workspace_dir)
    new_results = get_results(workspace_dir) - init_results

    return list(new_results), result
    