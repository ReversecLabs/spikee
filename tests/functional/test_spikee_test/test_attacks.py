import pytest

from spikee.utilities.files import read_jsonl_file
from ..utils import spikee_generate_cli, spikee_test_cli

def _attack_base_name(entry):
    attack_name = entry.get("attack_name")
    if not attack_name:
        return None
    return attack_name.split(".")[-1]

@pytest.mark.parametrize("target_name", ["always_refuse", "always_refuse_legacy"])
@pytest.mark.parametrize("attack_name", ["mock_attack", "mock_attack_legacy"])
def test_spikee_test_runs_attack_when_base_fails(run_spikee, workspace_dir, target_name, attack_name):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            attack_name,
            "--attack-iterations",
            "5",
        ],
    )

    results = read_jsonl_file(results_file[0])
    assert len(results) == (len(entries) * 2), f"Expected {len(entries) * 2} results entries, got {len(results)}"

    base_results = [entry for entry in results if entry.get("attack_name") == "None"]
    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]

    assert len(base_results) == len(attack_results), f"Expected same number of base and attack results, got {len(base_results)} base and {len(attack_results)} attack results"
    for attack_entry in attack_results:
        attempts = attack_entry["attempts"]
        assert attempts == 5, f"Expected 5 attempts, got {attempts}"
        assert not attack_entry["success"], "Expected attack to fail, but it succeeded"

@pytest.mark.parametrize("target_name", ["always_refuse", "always_refuse_legacy"])
@pytest.mark.parametrize("attack_name", ["mock_attack"])
def test_spikee_test_runs_attack_only(run_spikee, workspace_dir, target_name, attack_name):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            attack_name,
            "--attack-iterations",
            "5",
            "--attack-only",
        ],
    )

    results = read_jsonl_file(results_file[0])
    assert len(results) == len(entries), f"Expected {len(entries)} results entries, got {len(results)}"

    base_results = [entry for entry in results if entry.get("attack_name") == "None"]
    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]

    assert len(base_results) == 0, f"Expected no base results since --attack-only is set, but found {len(base_results)} base results"
    assert len(attack_results) == len(entries), f"Expected one attack result per entry, got {len(attack_results)}"

@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("attack_name", ["mock_attack", "mock_attack_legacy"])
def test_spikee_test_skips_attack_when_base_succeeds(run_spikee, workspace_dir, target_name, attack_name):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            attack_name,
            "--attack-iterations",
            "5",
        ],
    )

    results = read_jsonl_file(results_file[0])
    assert len(results) == len(entries), f"Expected {len(entries)} results entries (no attacks), got {len(results)}"

    base_results = [entry for entry in results if entry.get("attack_name") == "None"]
    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]

    assert len(base_results) == len(entries), f"Expected all entries to be base results, got {len(base_results)}"
    assert len(attack_results) == 0, f"Expected no attack results since base succeeded, but found {len(attack_results)} attack results"
    assert all(entry["success"] for entry in base_results)
    assert not attack_results

@pytest.mark.parametrize(
    "attack_name",
    [
        "anti_spotlighting",
        "best_of_n",
        "prompt_decomposition",
    ],
)
def test_spikee_test_builtin_attacks(run_spikee, workspace_dir, attack_name):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_files, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="always_refuse",
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            attack_name,
            "--attack-iterations",
            "4",
        ],
    )

    results = read_jsonl_file(results_files[0])
    assert len(results) == (len(entries) * 2), f"Expected {len(entries) * 2} results entries, got {len(results)}"

    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]
    assert len(attack_results) == len(entries), f"Expected one attack result per entry, got {len(attack_results)}"

    for entry in attack_results:
        assert 1 <= entry["attempts"] <= 4
        assert not entry["success"]
