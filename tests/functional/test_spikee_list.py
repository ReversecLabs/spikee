from pathlib import Path

from .utils import spikee_list, spikee_generate_cli

def _assert_contains(lines: list[str], expected_items: set[str]):
    missing = {
        item for item in expected_items if all(item not in line for line in lines)
    }
    assert not missing, f"Missing expected entries: {sorted(missing)}"

def test_list_seeds(run_spikee, workspace_dir):
    """Test that `spikee list seeds` shows the expected seed folders."""

    output_lines = spikee_list(run_spikee, workspace_dir, "seeds")
    expected = {"seeds-functional-basic", "seeds-functional-placeholder"}
    _assert_contains(output_lines, expected)

def test_list_datasets(run_spikee, workspace_dir):
    """Test that `spikee list datasets` shows newly generated datasets."""

    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    dataset_rel = dataset_path.relative_to(workspace_dir).as_posix()

    output_lines = spikee_list(run_spikee, workspace_dir, "datasets")
    expected = {Path(dataset_rel).name}
    _assert_contains(output_lines, expected)

def test_list_targets(run_spikee, workspace_dir):
    """Test that `spikee list targets` shows both built-in and local targets."""

    output_lines = spikee_list(run_spikee, workspace_dir, "targets")
    expected_local = {
        "always_refuse",
        "always_success",
        "partial_success",
    }
    expected_builtin = {"llm_provider"}
    _assert_contains(output_lines, expected_local | expected_builtin)

def test_list_plugins(run_spikee, workspace_dir):
    """Test that `spikee list plugins` shows both built-in and local plugins."""

    output_lines = spikee_list(run_spikee, workspace_dir, "plugins")
    expected_local = {"test_repeat", "test_upper"}
    expected_builtin = {"1337", "base64", "hex"}
    _assert_contains(output_lines, expected_local | expected_builtin)

def test_list_attacks(run_spikee, workspace_dir):
    """Test that `spikee list attacks` shows both built-in and local attacks."""

    output_lines = spikee_list(run_spikee, workspace_dir, "attacks")
    expected_local = {"mock_attack", "goat"}
    expected_builtin = {"best_of_n", "anti_spotlighting", "crescendo"}
    _assert_contains(output_lines, expected_local | expected_builtin)

def test_list_judges(run_spikee, workspace_dir):
    """Test that `spikee list judges` shows both built-in and local judges."""

    output_lines = spikee_list(run_spikee, workspace_dir, "judges")
    expected_local = {"test_judge", "llm_judge_harmful"}
    expected_builtin = {"canary", "regex"}
    _assert_contains(output_lines, expected_local | expected_builtin)

def test_list_providers(run_spikee, workspace_dir):
    """Test that `spikee list providers` shows both built-in and local providers."""

    output_lines = spikee_list(run_spikee, workspace_dir, "providers")
    expected_builtin = {"bedrock", "openai", "deepseek", "google"}
    _assert_contains(output_lines, expected_builtin)