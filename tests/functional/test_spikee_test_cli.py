from __future__ import annotations

import time

import pytest

from .utils import (
    extract_results_path,
    read_jsonl,
    run_generate_command,
    run_test_command,
    write_jsonl,
    build_resource_name,
    extract_resource_name,
)


def _attack_base_name(entry):
    attack_name = entry.get("attack_name")
    if not attack_name:
        return None
    return attack_name.split(".")[-1]


def _result_stub(entry: dict) -> dict:
    return {
        "id": entry["id"],
        "long_id": entry["long_id"],
        "attack_name": "None",
        "success": False,
    }


def _target_name_full(target_name: str) -> str:
    # Tests rely on built-in mock targets without extra options.
    return target_name


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


def test_spikee_test_resume_file_merges_existing_results(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)
    dataset_entries = read_jsonl(dataset_path)

    resume_count = max(1, len(dataset_entries) // 2)
    resume_entries = [_result_stub(entry) for entry in dataset_entries[:resume_count]]
    resume_file = workspace_dir / "resume_partial.jsonl"
    write_jsonl(resume_file, resume_entries)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            "always_refuse",
            "--resume-file",
            str(resume_file),
        ],
    )

    output_file = extract_results_path(result.stdout, workspace_dir)
    final_results = read_jsonl(output_file)

    assert len(final_results) == len(dataset_entries)
    final_ids = {entry["id"] for entry in final_results}
    expected_ids = {entry["id"] for entry in dataset_entries}
    assert final_ids == expected_ids

    resume_map = {entry["id"]: entry for entry in resume_entries}
    final_resume_map = {
        entry["id"]: entry for entry in final_results if entry["id"] in resume_map
    }
    assert final_resume_map == resume_map


def test_spikee_test_auto_resume_picks_latest_candidate(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)
    dataset_entries = read_jsonl(dataset_path)
    assert len(dataset_entries) >= 3

    target_name = "always_refuse"
    target_name_full = _target_name_full(target_name)
    resource_name = build_resource_name(
        "results",
        target_name_full,
        extract_resource_name(str(dataset_path)),
    )

    results_dir = workspace_dir / "results"
    results_dir.mkdir(exist_ok=True)
    ts_base = int(time.time())
    older_path = results_dir / f"{resource_name}_{ts_base - 10}.jsonl"
    newer_path = results_dir / f"{resource_name}_{ts_base - 1}.jsonl"

    older_entries = [_result_stub(dataset_entries[0])]
    newer_entries = [
        _result_stub(dataset_entries[1]),
        _result_stub(dataset_entries[2]),
    ]

    write_jsonl(older_path, older_entries)
    write_jsonl(newer_path, newer_entries)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            target_name,
            "--auto-resume",
        ],
    )

    assert f"[Auto-Resume] Using latest: {newer_path.name}" in result.stdout

    output_file = extract_results_path(result.stdout, workspace_dir)
    final_results = read_jsonl(output_file)
    final_ids = {entry["id"] for entry in final_results}
    expected_ids = {entry["id"] for entry in dataset_entries}
    assert final_ids == expected_ids

    newer_map = {entry["id"]: entry for entry in newer_entries}
    final_resume_map = {
        entry["id"]: entry for entry in final_results if entry["id"] in newer_map
    }
    assert final_resume_map == newer_map

    older_id = older_entries[0]["id"]
    older_entry_after_resume = next(
        entry for entry in final_results if entry["id"] == older_id
    )
    assert older_entry_after_resume != older_entries[0]


def test_spikee_test_resume_file_skips_completed_dataset(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir)
    dataset_entries = read_jsonl(dataset_path)

    resume_entries = [_result_stub(entry) for entry in dataset_entries]
    resume_file = workspace_dir / "resume_complete.jsonl"
    write_jsonl(resume_file, resume_entries)

    results_dir = workspace_dir / "results"
    results_dir.mkdir(exist_ok=True)
    before_files = set(results_dir.glob("*.jsonl"))

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset",
            str(dataset_rel),
            "--target",
            "always_refuse",
            "--resume-file",
            str(resume_file),
        ],
    )

    assert "[Done] All entries have already been processed" in result.stdout
    after_files = set(results_dir.glob("*.jsonl"))
    assert after_files == before_files


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
