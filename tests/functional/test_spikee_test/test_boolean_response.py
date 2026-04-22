from __future__ import annotations

from spikee.utilities.files import read_jsonl_file
from ..utils import spikee_test_cli


def test_boolean_response_handling(run_spikee, workspace_dir):
    # Run spikee test with mock_boolean target
    # This target returns True (bool).
    # If bug exists: tester converts True -> "True". call_judge("True") runs canary judge.
    # Canary judge checks key. "True" does not contain key. result=False.
    # If bug fixed: tester keeps True. call_judge(True) returns True. result=True.
    results_path, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="mock_boolean"
    )

    # 3. Verify results
    results = read_jsonl_file(results_path[0])
    assert results, "No results recorded by spikee test"
    
    # We expect success=True if the boolean logic is working correctly
    success_count = sum(1 for entry in results if entry["success"])
    failure_count = sum(1 for entry in results if not entry["success"])

    print(f"Successes: {success_count}, Failures: {failure_count}")

    # Assert all succeeded
    assert all(entry["success"] for entry in results), (
        f"Expected all entries to succeed with boolean True, but found {failure_count} failures."
    )
