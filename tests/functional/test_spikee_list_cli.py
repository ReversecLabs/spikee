from __future__ import annotations

import os
from pathlib import Path

import pytest

from .utils import read_jsonl, run_generate_command, run_test_command


def _run_list(run_spikee, workspace: Path, entity: str):
    result = run_spikee(["list", entity], cwd=workspace)
    return result.stdout.strip().splitlines()


def _assert_contains(lines: list[str], expected_items: set[str]):
    missing = {item for item in expected_items if all(item not in line for line in lines)}
    assert not missing, f"Missing expected entries: {sorted(missing)}"


def test_spikee_list_seeds(run_spikee, workspace_dir):
    output_lines = _run_list(run_spikee, workspace_dir, "seeds")
    expected = {"seeds-functional-basic"}
    _assert_contains(output_lines, expected)


def test_spikee_list_datasets(run_spikee, workspace_dir):
    dataset_path, _ = run_generate_command(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir).as_posix()

    output_lines = _run_list(run_spikee, workspace_dir, "datasets")
    expected = {Path(dataset_rel).name}
    _assert_contains(output_lines, expected)


def test_spikee_list_judges(run_spikee, workspace_dir):
    output_lines = _run_list(run_spikee, workspace_dir, "judges")
    expected_builtins = {"canary", "regex"}
    expected_local = {"test_judge", "test_judge_legacy"}
    _assert_contains(output_lines, expected_builtins | expected_local)


def test_spikee_list_targets(run_spikee, workspace_dir):
    output_lines = _run_list(run_spikee, workspace_dir, "targets")
    expected_local = {
        "always_refuse",
        "always_refuse_legacy",
        "always_success",
        "always_success_legacy",
        "partial_success",
        "partial_success_legacy",
    }
    expected_builtin = {"sample_target", "llm_mailbox"}
    _assert_contains(output_lines, expected_local | expected_builtin)


def test_spikee_list_attacks(run_spikee, workspace_dir):
    output_lines = _run_list(run_spikee, workspace_dir, "attacks")
    expected_local = {"mock_attack", "mock_attack_legacy"}
    expected_builtin = {"anti_spotlighting", "best_of_n", "prompt_decomposition", "random_suffix_search"}
    _assert_contains(output_lines, expected_local | expected_builtin)
