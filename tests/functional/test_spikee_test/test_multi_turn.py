import os

from spikee.utilities.files import read_jsonl_file, write_jsonl_file
from ..utils import spikee_generate_cli, spikee_test_cli

TARGET_NAME = "mock_multiturn"
ATTACK_NAME = "mock_crescendo"
JUDGE_NAME = "mock_keyword"


def _patch_dataset_judge(dataset_path):
    entries = read_jsonl_file(dataset_path)
    for e in entries:
        e["judge_name"] = JUDGE_NAME
    write_jsonl_file(dataset_path, entries)


def test_multiturn_success(run_spikee, workspace_dir):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    _patch_dataset_judge(dataset_path)

    results_file, result = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=TARGET_NAME,
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            ATTACK_NAME,
            "--attack-iterations",
            "5",
            "--attack-options",
            "scenario=success",
            "--judge",
            JUDGE_NAME,
            "--judge-options",
            f"{JUDGE_NAME}:keyword=Here is the flag",
        ],
    )

    results = read_jsonl_file(results_file[0])
    attack_entry = next(
        (e for e in results if e.get("attack_name", "").endswith(ATTACK_NAME)), None
    )

    try:
        assert attack_entry
        assert attack_entry["success"]
        assert attack_entry["attempts"] == 3  # Q1, Q2, WIN
    except AssertionError:
        print("\n=== STDOUT ===\n", result.stdout)
        print("\n=== STDERR ===\n", result.stderr)
        if os.path.exists(results_file[0]):
            print(f"\n=== RESULTS FILE ({results_file[0]}) ===\n")
            with open(results_file[0], "r") as f:
                print(f.read())
        raise


def test_multiturn_refusal_backtrack(run_spikee, workspace_dir):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    _patch_dataset_judge(dataset_path)

    results_file, result = spikee_test_cli(

        run_spikee,
        workspace_dir,
        target=TARGET_NAME,
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            ATTACK_NAME,
            "--attack-iterations",
            "10",
            "--attack-options",
            "scenario=refusal",
            "--judge",
            JUDGE_NAME,
            "--judge-options",
            f"{JUDGE_NAME}:keyword=Here is the flag",
        ],
    )

    results = read_jsonl_file(results_file[0])
    attack_entry = next(
        (e for e in results if e.get("attack_name", "").endswith(ATTACK_NAME)), None
    )

    try:
        assert attack_entry
        assert attack_entry["success"]
        # Q1 (fail), REFUSE (backtrack, count=1 consumed), Q2 (fail), WIN (success)
        # Total calls: 4
        assert attack_entry["attempts"] == 4
    except AssertionError:
        print("\n=== STDOUT ===\n", result.stdout)
        print("\n=== STDERR ===\n", result.stderr)
        if os.path.exists(results_file[0]):
            print(f"\n=== RESULTS FILE ({results_file[0]}) ===\n")
            with open(results_file[0], "r") as f:
                print(f.read())
        raise


def test_multiturn_max_turns_failure(run_spikee, workspace_dir):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    _patch_dataset_judge(dataset_path)

    results_file, result = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=TARGET_NAME,
        datasets=[dataset_path],
        additional_args=[
            "--attack",
            ATTACK_NAME,
            "--attack-iterations",
            "4",  # Limit calls to 4
            "--attack-options",
            "scenario=failure",  # Requires 6
            "--judge",
            JUDGE_NAME,
            "--judge-options",
            f"{JUDGE_NAME}:keyword=Here is the flag",
        ],
    )
    results = read_jsonl_file(results_file[0])
    attack_entry = next(
        (e for e in results if e.get("attack_name", "").endswith(ATTACK_NAME)), None
    )

    try:
        assert attack_entry
        assert not attack_entry["success"]
        assert attack_entry["attempts"] == 4
    except AssertionError:
        print("\n=== STDOUT ===\n", result.stdout)
        print("\n=== STDERR ===\n", result.stderr)
        if os.path.exists(results_file[0]):
            print(f"\n=== RESULTS FILE ({results_file[0]}) ===\n")
            with open(results_file[0], "r") as f:
                print(f.read())
        raise
