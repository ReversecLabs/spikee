import os
import shutil
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

import pytest

# Control whether to use an isolated venv for testing.
# Can be overridden via environment variable: SPIKEE_TESTS_USE_ISOLATED_VENV=false
# - True: Create and install spikee in an isolated venv (recommended, clean isolation)
# - False: Use the current Python environment (faster for local development)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))  # Load .env from cwd
USE_ISOLATED_VENV = os.getenv("SPIKEE_TESTS_USE_ISOLATED_VENV", "true").lower() == "true"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Pytest hook that runs once per test session before any tests execute.
    
    Conditionally creates an isolated virtual environment based on USE_ISOLATED_VENV,
    or reuses the current Python environment for faster local testing.
    This ensures tests can run in isolation or against local development.
    """
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))  # Load .env from cwd

    project_root = Path(__file__).resolve().parents[2]

    if USE_ISOLATED_VENV:
        # Create isolated venv for clean testing
        tmp_factory = session.config._tmp_path_factory
        venv_dir = tmp_factory.mktemp("spikee-venv")

        # Print setup message to terminal (suspend output capture so user sees it)
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        capture = session.config.pluginmanager.get_plugin("capturemanager")
        message = f"[functional-tests] Installing spikee into isolated venv at {venv_dir}"
        if capture:
            capture.suspend_global_capture(in_=True)
        try:
            if terminal:
                terminal.write_line(message)
            else:
                print(message, flush=True)
        finally:
            if capture:
                capture.resume_global_capture()

        # Create an isolated Python virtual environment with system packages available
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
        )

        # Install spikee in editable mode from project root (with all optional dependencies)
        pip_executable = (
            venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
        )
        subprocess.run(
            [str(pip_executable), "install", ".[all]"],
            cwd=project_root,
            check=True,
            env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        )

        # Store venv path on session config so fixtures can access it
        session.config.spikee_venv = venv_dir

    else:
        # Use current Python executable (existing venv or system Python)
        terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        capture = session.config.pluginmanager.get_plugin("capturemanager")
        message = "[functional-tests] Using existing Python environment (not isolated)"
        if capture:
            capture.suspend_global_capture(in_=True)
        try:
            if terminal:
                terminal.write_line(message)
            else:
                print(message, flush=True)
        finally:
            if capture:
                capture.resume_global_capture()

        # Store current Python path (use sys.executable wrapped in Path)
        session.config.spikee_venv = Path(sys.executable).parents[1]

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def spikee_venv(request: pytest.FixtureRequest) -> Path:
    """Returns the path to the isolated virtual environment created during pytest_sessionstart.
    
    The venv was populated by pytest_sessionstart with spikee installed.
    """
    return request.config.spikee_venv

@pytest.fixture
def run_spikee(spikee_venv: Path):
    """Factory fixture that returns a function to run spikee commands via subprocess.
    
    Locates the spikee executable from either:
    - The isolated venv (if USE_ISOLATED_VENV=true)
    - The current Python environment (if USE_ISOLATED_VENV=false)
    """
    spikee_executable = (
        spikee_venv / ("Scripts" if sys.platform == "win32" else "bin") / "spikee"
    )

    def _run(args, cwd: Path):
        """Execute spikee command."""
        try:
            env = os.environ.copy()
            result = subprocess.run(
                [str(spikee_executable), *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            return result
        except subprocess.CalledProcessError as e:
            _print_error(' '.join(args), e.stderr or e.stdout)
            raise

    return _run

def _print_error(command: str, output: str) -> None:
    """Print a readable error message from a failed subprocess call.
    
    Extracts the key error message and displays it clearly without overly
    specific hints. Works for any spikee command error.
    """
    print("\n" + "="*80)
    print(f"ERROR: Command failed: spikee {command}")
    print("="*80)
    
    # Try to extract the most relevant error line
    if output:
        lines = output.strip().split("\n")
        # Filter out empty lines
        relevant_lines = [line.strip() for line in lines if line.strip()]
        
        if relevant_lines:
            # Print all non-empty output (usually contains the error)
            print("\n" + "\n".join(relevant_lines))
    else:
        print("\n(No error output captured)")
    
    print("="*80 + "\n")

def workspace_init(tmp_path, project_root: Path, run_spikee, additional_args):
    # Create a temporary workspace directory
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Initialize spikee structure (creates targets/, plugins/, datasets/, etc.)
    run_spikee(["init", *additional_args], cwd=workspace)

    # Copy fixture modules from the test fixtures folder into the workspace
    # This lets tests use mock targets, plugins, judges, attacks, and pre-built datasets
    fixtures_workspace = (
        project_root / "tests" / "functional" / "workspace"
    )
    if fixtures_workspace.exists():
        for item in fixtures_workspace.iterdir():
            target = workspace / item.name
            if item.is_dir():
                # Copy directory (e.g., targets/, plugins/) with all contents
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                # Copy individual file
                shutil.copy2(item, target)

    return workspace

@pytest.fixture
def workspace_dir(tmp_path, project_root: Path, run_spikee):
    """Returns an isolated test workspace with initialized spikee structure and fixtures.
    """
    return workspace_init(tmp_path, project_root, run_spikee, [])

@pytest.fixture
def workspace_dir_builtin(tmp_path, project_root: Path, run_spikee):
    """Returns an isolated test workspace with initialized spikee structure and built-in modules.
    """
    return workspace_init(tmp_path, project_root, run_spikee, ["--include-builtin", "all"])

@pytest.fixture
def workspace_dir_viewer(tmp_path, project_root: Path, run_spikee):
    """Returns an isolated test workspace with initialized spikee structure and viewer.
    """
    return workspace_init(tmp_path, project_root, run_spikee, ["--include-viewer"])