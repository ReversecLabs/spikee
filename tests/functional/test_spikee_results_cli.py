from __future__ import annotations

import time
from pathlib import Path

import pytest

from .utils import (
    extract_results_path,
    read_jsonl,
    run_results_command,
    run_test_command,
    extract_prefix_from_file_name,
)


def _judge_dataset_filename(judge_variant: str) -> str:
    return (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )


def _create_judge_results(
    run_spikee, workspace_dir: Path, target_name: str, judge_variant: str
):
    dataset_path = workspace_dir / "datasets" / _judge_dataset_filename(judge_variant)
    assert dataset_path.exists()

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_path),
            "--target",
            target_name,
            "--judge-options",
            f"{judge_variant}:mode=fail",
        ],
    )
    return extract_results_path(result.stdout, workspace_dir)


@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_results_analyze(run_spikee, workspace_dir, target_name, judge_variant):
    results_file = _create_judge_results(
        run_spikee, workspace_dir, target_name, judge_variant
    )

    analyze = run_results_command(
        run_spikee,
        workspace_dir,
        "analyze",
        ["--result-file", str(results_file)],
    )

    output = analyze.stdout
    assert "General Statistics" in output
    assert "Total Unique Entries" in output
    assert "Successful Attacks: 0" in output
    assert "Attack Success Rate" in output


@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_results_rejudge_with_options(
    run_spikee, workspace_dir, target_name, judge_variant
):
    results_file = _create_judge_results(
        run_spikee, workspace_dir, target_name, judge_variant
    )

    rejudge = run_results_command(
        run_spikee,
        workspace_dir,
        "rejudge",
        [
            "--result-file",
            str(results_file),
            "--judge-options",
            f"{judge_variant}:mode=success",
        ],
    )
    assert "Currently Re-judging" in rejudge.stdout

    _, resource_name = extract_prefix_from_file_name(results_file.name)
    expected_prefix = f"rejudge_{resource_name}_"

    timeout = time.time() + 5
    new_file = None
    while time.time() < timeout:
        candidates = list(
            path
            for path in results_file.parent.glob("rejudge_*.jsonl")
            if path.name.startswith(expected_prefix)
        )
        if candidates:
            new_file = max(candidates, key=lambda p: p.stat().st_mtime)
            break
        time.sleep(0.2)

    assert new_file and new_file.exists(), "Rejudge output file not created"

    rejudged_results = read_jsonl(new_file)
    assert rejudged_results
    assert all(entry["success"] for entry in rejudged_results)

    analyze = run_results_command(
        run_spikee,
        workspace_dir,
        "analyze",
        ["--result-file", str(new_file)],
    )
    output = analyze.stdout
    assert "Successful Attacks: 2" in output
