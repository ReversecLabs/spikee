import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def pytest_sessionstart(session: pytest.Session) -> None:
    project_root = Path(__file__).resolve().parents[2]
    tmp_factory = session.config._tmp_path_factory  # type: ignore[attr-defined]
    venv_dir = tmp_factory.mktemp("spikee-venv")

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

    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        check=True,
    )
    pip_executable = (
        venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "pip"
    )
    subprocess.run(
        [str(pip_executable), "install", "."],
        cwd=project_root,
        check=True,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )

    session.config._spikee_venv = venv_dir  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def spikee_venv(request: pytest.FixtureRequest) -> Path:
    return request.config._spikee_venv  # type: ignore[attr-defined]


@pytest.fixture
def run_spikee(spikee_venv: Path, project_root: Path):
    spikee_executable = (
        spikee_venv / ("Scripts" if sys.platform == "win32" else "bin") / "spikee"
    )

    def _run(args, cwd: Path):
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
            print("STDOUT:")
            print(e.stdout)
            print("STDERR:")
            print(e.stderr)
            raise

    return _run


@pytest.fixture
def workspace_dir(tmp_path, project_root: Path, run_spikee):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_spikee(["init"], cwd=workspace)

    fixtures_workspace = (
        project_root / "tests" / "functional" / "fixtures" / "workspace"
    )
    if fixtures_workspace.exists():
        for item in fixtures_workspace.iterdir():
            target = workspace / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    return workspace
