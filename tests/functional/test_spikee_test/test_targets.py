import pytest

from spikee.utilities.files import read_jsonl_file
from ..utils import spikee_test_cli, spikee_generate_cli


@pytest.mark.parametrize(
    "target_name,expected_success",
    [
        ("always_refuse", False),
        ("always_refuse_legacy", False),
        ("always_success", True),
        ("always_success_legacy", True),
        ("always_guardrail", False),  # This target raises a GuardrailTrigger, which should be treated as a failure with the canary response
    ],
)
def test_spikee_test_targets(run_spikee, workspace_dir, target_name, expected_success):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_files, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
    )

    results = read_jsonl_file(results_files[0])

    assert len(results) > 0, "No results recorded by spikee test"
    assert len(results) == len(entries), f"Expected {len(entries)} results, got {len(results)}"
    assert all(entry["success"] == expected_success for entry in results)

    if target_name == "always_guardrail":
        # For the always_guardrail target, we expect all entries to have success=False and the canary response indicating the guardrail was triggered
        assert all("guardrail" in r and r['guardrail'] for r in results), "Expected all entries to have guardrail=True for the always_guardrail target {}".format(results)
