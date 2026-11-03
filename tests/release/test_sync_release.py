"""Exercise real Git histories, including rebased release PRs and retries."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/sync_release.py"


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def versions(repo, value):
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "fixture"\nversion = "{value}"\n'
    )
    (repo / "spikee/__init__.py").write_text(f'__version__ = "{value}"\n')
    (repo / "README.md").write_text(f"# Fixture\n_Version: {value}_\n\nShared text\n")


def commit(repo, message):
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Release test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / "spikee").mkdir()
    versions(tmp_path, "0.9.2")
    commit(tmp_path, "previous release")
    git(tmp_path, "switch", "-c", "develop")
    versions(tmp_path, "0.9.3-dev")
    (tmp_path / "feature.py").write_text("# new feature\n")
    commit(tmp_path, "development")
    return tmp_path


def release(repo, rebased=True):
    if rebased:
        # Replayed release content has no ancestry link to develop, exactly
        # the condition produced by GitHub's rebase/squash merge methods.
        git(repo, "switch", "main")
        (repo / "feature.py").write_text("# new feature\n")
    else:
        git(repo, "switch", "-c", "release", "develop")
    versions(repo, "0.9.3")
    (repo / "CHANGELOG.md").write_text("Released 0.9.3\n")
    sha = commit(repo, "release")
    git(repo, "switch", "develop")
    return sha


def sync(repo, sha, success=True):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), sha],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("rebased", [False, True])
def test_release_merge_and_retry(history, rebased):
    sha = release(history, rebased)
    sync(history, sha)
    for file in ("pyproject.toml", "README.md", "spikee/__init__.py"):
        assert "0.9.4-dev" in (history / file).read_text()
    git(history, "merge-base", "--is-ancestor", sha, "HEAD")
    assert (history / "CHANGELOG.md").exists()
    assert (history / "feature.py").exists()
    head = git(history, "rev-parse", "HEAD")
    sync(history, sha)
    assert git(history, "rev-parse", "HEAD") == head
    assert git(history, "status", "--porcelain") == ""


def test_preserves_develop_edits_in_version_files(history):
    sha = release(history)
    for file in ("README.md", "spikee/__init__.py", "pyproject.toml"):
        with (history / file).open("a") as output:
            output.write("\n# New development content\n")
    commit(history, "concurrent development")
    sync(history, sha)
    for file in ("README.md", "spikee/__init__.py", "pyproject.toml"):
        assert "New development content" in (history / file).read_text()


@pytest.mark.parametrize("file", ["README.md", "feature.py"])
def test_real_conflicts_abort_without_losing_work(history, file):
    sha = release(history)
    git(history, "switch", "--detach", sha)
    with (history / file).open("a") as output:
        output.write("Release edit\n")
    sha = commit(history, "release edit")
    git(history, "switch", "develop")
    with (history / file).open("a") as output:
        output.write("Develop edit\n")
    head = commit(history, "conflicting develop edit")
    result = sync(history, sha, success=False)
    assert "Non-version merge conflict" in result.stderr
    assert git(history, "rev-parse", "HEAD") == head
    assert git(history, "status", "--porcelain") == ""
    assert "Develop edit" in (history / file).read_text()


def test_does_not_downgrade_newer_development_version(history):
    sha = release(history)
    versions(history, "0.10.0-dev")
    commit(history, "next minor development")
    sync(history, sha)
    assert 'version = "0.10.0-dev"' in (history / "pyproject.toml").read_text()


def test_rejects_unpublished_release(history):
    head = git(history, "rev-parse", "HEAD")
    assert "published release version" in sync(history, head, success=False).stderr
    assert git(history, "rev-parse", "HEAD") == head


def test_rejects_dirty_worktree(history):
    sha = release(history)
    (history / "uncommitted.txt").write_text("keep me")
    assert "clean working tree" in sync(history, sha, success=False).stderr
    assert (history / "uncommitted.txt").read_text() == "keep me"
