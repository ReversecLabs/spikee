def test_init(workspace_dir):
    """Test that 'spikee init' creates the expected directory structure."""

    expected_dirs = {
        "targets",
        "plugins",
        "judges",
        "attacks",
        "datasets",
        "providers",
    }
    actual_dirs = {p.name for p in workspace_dir.iterdir() if p.is_dir()}
    missing_dirs = expected_dirs - actual_dirs
    assert not missing_dirs, f"Missing expected directories: {missing_dirs}"

def test_init_builtin(workspace_dir_builtin):
    """Test that 'spikee init --include-builtin' creates the expected directory structure with built-in modules."""

    expected_builtin ={
        "targets": "llm_provider.py",
        "plugins": "1337.py",
        "judges": "canary.py",
        "attacks": "best_of_n.py"
    }

    for folder, module in expected_builtin.items():
        module_path = workspace_dir_builtin / folder / module
        assert module_path.exists(), f"Expected built-in module '{module}' not found in '{folder}'"

def test_init_viewer(workspace_dir_viewer):
    """Test that 'spikee init --include-viewer' creates the expected directory structure with viewer."""
    expected_dirs = [
        "viewer",
        "viewer/static",
        "viewer/templates",
    ]

    for dir in expected_dirs:
        dir_path = workspace_dir_viewer / dir
        assert dir_path.exists() and dir_path.is_dir(), f"Expected viewer directory '{dir}' not found"