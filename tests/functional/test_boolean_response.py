from __future__ import annotations

import pytest

from .utils import (
    extract_results_path,
    read_jsonl,
    run_generate_command,
    run_test_command,
)


def test_boolean_response_handling(run_spikee, workspace_dir):
    # 1. Generate dataset
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    # 2. Run spikee test with mock_boolean target
    # This target returns True (bool).
    # If bug exists: tester converts True -> "True". call_judge("True") runs canary judge.
    # Canary judge checks key. "True" does not contain key. result=False.
    # If bug fixed: tester keeps True. call_judge(True) returns True. result=True.
    result = run_test_command(
        run_spikee,
        workspace_dir,
        ["--dataset", str(dataset_rel), "--target", "mock_boolean"],
    )

    # 3. Verify results
    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results, "No results recorded by spikee test"
    
    # We expect success=True if the boolean logic is working correctly
    success_count = sum(1 for entry in results if entry["success"])
    failure_count = sum(1 for entry in results if not entry["success"])
    
    print(f"Successes: {success_count}, Failures: {failure_count}")
    
    # Assert all succeeded
    assert all(entry["success"] for entry in results), \
        f"Expected all entries to succeed with boolean True, but found {failure_count} failures."
