from __future__ import annotations

import pytest

from .utils import extract_results_path, read_jsonl, run_generate_command, run_test_command


def _attack_base_name(entry):
    attack_name = entry.get("attack_name")
    if not attack_name:
        return None
    return attack_name.split(".")[-1]


@pytest.mark.parametrize(
    "target_name,expected_success",
    [
        ("always_refuse", False),
        ("always_refuse_legacy", False),
        ("always_success", True),
        ("always_success_legacy", True),
    ],
)
def test_spikee_test_targets(run_spikee, workspace_dir, target_name, expected_success):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        ["--dataset", str(dataset_rel), "--target", target_name],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results, "No results recorded by spikee test"
    assert all(entry["success"] == expected_success for entry in results)


@pytest.mark.parametrize("target_name", ["always_refuse", "always_refuse_legacy"])
@pytest.mark.parametrize("attack_name", ["mock_attack", "mock_attack_legacy"])
def test_spikee_test_runs_attack_when_base_fails(
    run_spikee, workspace_dir, target_name, attack_name
):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            target_name,
            "--attack",
            attack_name,
            "--attack-iterations",
            "5",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results

    base_results = [entry for entry in results if entry.get("attack_name") == "None"]
    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]

    assert len(base_results) == len(attack_results)
    for attack_entry in attack_results:
        attempts = attack_entry["attempts"]
        assert attempts == 5
        assert not attack_entry["success"]


@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("attack_name", ["mock_attack", "mock_attack_legacy"])
def test_spikee_test_skips_attack_when_base_succeeds(
    run_spikee, workspace_dir, target_name, attack_name
):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            target_name,
            "--attack",
            attack_name,
            "--attack-iterations",
            "5",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results

    base_results = [entry for entry in results if entry.get("attack_name") == "None"]
    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]

    assert base_results
    assert all(entry["success"] for entry in base_results)
    assert not attack_results


@pytest.mark.parametrize("target_name", ["partial_success", "partial_success_legacy"])
@pytest.mark.parametrize("attack_name", ["mock_attack", "mock_attack_legacy"])
def test_spikee_test_partial_success_target(
    run_spikee, workspace_dir, target_name, attack_name
):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            target_name,
            "--attack",
            attack_name,
            "--attack-iterations",
            "5",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results

    base_results = [entry for entry in results if entry.get("attack_name") == "None"]
    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]

    assert len(base_results) > 0
    successes = [entry for entry in base_results if entry["success"]]
    failures = [entry for entry in base_results if not entry["success"]]
    assert successes
    assert failures
    assert attack_results
    assert len(attack_results) == len(failures)
    assert all(entry["attempts"] == 5 for entry in attack_results)
    assert all(not entry["success"] for entry in attack_results)


@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_test_custom_judge_default_mode(
    run_spikee, workspace_dir, target_name, judge_variant
):
    dataset_filename = (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )
    dataset_path = workspace_dir / "datasets" / dataset_filename
    assert dataset_path.exists()

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_path),
            "--target",
            target_name,
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results
    assert all(not entry["success"] for entry in results)


@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_test_custom_judge_with_options(
    run_spikee, workspace_dir, target_name, judge_variant
):
    dataset_filename = (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )
    dataset_path = workspace_dir / "datasets" / dataset_filename
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
            f"{judge_variant}:mode=success",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results
    assert all(entry["success"] for entry in results)


@pytest.mark.parametrize(
    "attack_name",
    [
        "anti_spotlighting",
        "best_of_n",
        "prompt_decomposition",
    ],
)
def test_spikee_test_builtin_attacks(run_spikee, workspace_dir, attack_name):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            "always_refuse",
            "--attack",
            attack_name,
            "--attack-iterations",
            "4",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    assert results

    attack_results = [
        entry for entry in results if _attack_base_name(entry) == attack_name
    ]
    assert attack_results

    for entry in attack_results:
        assert 1 <= entry["attempts"] <= 4
        assert not entry["success"]
