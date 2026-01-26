
import os
import pytest
from .utils import (
    extract_results_path,
    read_jsonl,
    run_generate_command,
    run_test_command,
    write_jsonl,
)

TARGET_NAME = "mock_multiturn"
ATTACK_NAME = "mock_crescendo"
JUDGE_NAME = "mock_keyword"


def _patch_dataset_judge(dataset_path):
    entries = read_jsonl(dataset_path)
    for e in entries:
        e["judge_name"] = JUDGE_NAME
    write_jsonl(dataset_path, entries)


def test_multiturn_success(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    _patch_dataset_judge(dataset_path)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset", str(dataset_rel),
            "--target", TARGET_NAME,
            "--attack", ATTACK_NAME,
            "--attack-iterations", "5",
            "--attack-options", "scenario=success",
            "--judge", JUDGE_NAME,
            "--judge-options", f"{JUDGE_NAME}:keyword=Here is the flag",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    attack_entry = next((e for e in results if e.get("attack_name", "").endswith(ATTACK_NAME)), None)

    try:
        assert attack_entry
        assert attack_entry["success"]
        assert attack_entry["attempts"] == 3  # Q1, Q2, WIN
    except AssertionError:
        print("\n=== STDOUT ===\n", result.stdout)
        print("\n=== STDERR ===\n", result.stderr)
        if os.path.exists(results_file):
            print(f"\n=== RESULTS FILE ({results_file}) ===\n")
            with open(results_file, 'r') as f:
                print(f.read())
        raise


def test_multiturn_refusal_backtrack(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    _patch_dataset_judge(dataset_path)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset", str(dataset_rel),
            "--target", TARGET_NAME,
            "--attack", ATTACK_NAME,
            "--attack-iterations", "10",
            "--attack-options", "scenario=refusal",
            "--judge", JUDGE_NAME,
            "--judge-options", f"{JUDGE_NAME}:keyword=Here is the flag",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    attack_entry = next((e for e in results if e.get("attack_name", "").endswith(ATTACK_NAME)), None)

    try:
        assert attack_entry
        assert attack_entry["success"]
        # Q1 (fail), REFUSE (backtrack, count=1 consumed), Q2 (fail), WIN (success)
        # Total calls: 4
        assert attack_entry["attempts"] == 4
    except AssertionError:
        print("\n=== STDOUT ===\n", result.stdout)
        print("\n=== STDERR ===\n", result.stderr)
        if os.path.exists(results_file):
            print(f"\n=== RESULTS FILE ({results_file}) ===\n")
            with open(results_file, 'r') as f:
                print(f.read())
        raise


def test_multiturn_max_turns_failure(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    _patch_dataset_judge(dataset_path)
    dataset_rel = dataset_path.relative_to(workspace_dir)

    result = run_test_command(
        run_spikee,
        workspace_dir,
        [
            "--dataset", str(dataset_rel),
            "--target", TARGET_NAME,
            "--attack", ATTACK_NAME,
            "--attack-iterations", "4",  # Limit calls to 4
            "--attack-options", "scenario=failure",  # Requires 6
            "--judge", JUDGE_NAME,
            "--judge-options", f"{JUDGE_NAME}:keyword=Here is the flag",
        ],
    )

    results_file = extract_results_path(result.stdout, workspace_dir)
    results = read_jsonl(results_file)
    attack_entry = next((e for e in results if e.get("attack_name", "").endswith(ATTACK_NAME)), None)

    try:
        assert attack_entry
        assert not attack_entry["success"]
        assert attack_entry["attempts"] == 4
    except AssertionError:
        print("\n=== STDOUT ===\n", result.stdout)
        print("\n=== STDERR ===\n", result.stderr)
        raise
