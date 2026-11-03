"""Merge a published release into develop, resolving version-only conflicts."""

import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

VERSION_FILES = ("pyproject.toml", "spikee/__init__.py", "README.md")
PATTERNS = {
    "pyproject.toml": r'(?ms)^(\[project\]\s*\n.*?^version\s*=\s*")[^"]+(".*?$)',
    "spikee/__init__.py": r'(?m)^(__version__\s*=\s*")[^"]+("\s*)$',
    "README.md": r"(_Version:\s*)\d+\.\d+\.\d+(?:-dev)?(\s*_)",
}


def git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def version_tuple(value):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-dev)?", value)
    if not match:
        raise ValueError(f"Unsupported version: {value}")
    return tuple(map(int, match.groups()))


def set_version(path, content, version):
    updated, count = re.subn(
        PATTERNS[path], lambda m: m[1] + version + m[2], content, count=1
    )
    if count != 1:
        raise ValueError(f"Cannot locate version in {path}")
    return updated


def resolve_version_conflict(path, version):
    # Normalize only the version in all three stages. A real content conflict
    # must still fail; never choose an entire file from one side of the merge.
    with tempfile.TemporaryDirectory() as directory:
        files = []
        for stage in (2, 1, 3):  # ours, base, theirs, as required by merge-file
            content = git("show", f":{stage}:{path}").stdout
            file = Path(directory) / str(stage)
            file.write_text(set_version(path, content, version))
            files.append(str(file))
        merged = git("merge-file", "-p", *files, check=False)
        if merged.returncode:
            raise RuntimeError(f"Non-version merge conflict in {path}")
        Path(path).write_text(merged.stdout)
        git("add", "--", path)


def synchronize(release):
    if git("branch", "--show-current").stdout.strip() != "develop":
        raise RuntimeError("Run synchronization on develop")
    if git("status", "--porcelain").stdout:
        raise RuntimeError("Synchronization requires a clean working tree")
    release = git("rev-parse", "--verify", f"{release}^{{commit}}").stdout.strip()
    published = tomllib.loads(git("show", f"{release}:pyproject.toml").stdout)[
        "project"
    ]["version"]
    if published.endswith("-dev"):
        raise ValueError("Expected a published release version")
    major, minor, patch = version_tuple(published)
    current = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    next_version = max((major, minor, patch + 1), version_tuple(current))
    next_dev = ".".join(map(str, next_version)) + "-dev"
    already_merged = (
        git("merge-base", "--is-ancestor", release, "HEAD", check=False).returncode == 0
    )
    try:
        if not already_merged:
            result = git("merge", "--no-ff", "--no-commit", release, check=False)
            conflicts = git(
                "diff", "--name-only", "--diff-filter=U"
            ).stdout.splitlines()
            if result.returncode and not conflicts:
                raise RuntimeError(result.stderr)
            for path in conflicts:
                if path not in VERSION_FILES:
                    raise RuntimeError(f"Non-version merge conflict in {path}")
                resolve_version_conflict(path, next_dev)
        for path in VERSION_FILES:
            file = Path(path)
            file.write_text(set_version(path, file.read_text(), next_dev))
        git("add", "--", *VERSION_FILES)
        if (
            not already_merged
            or git("diff", "--cached", "--quiet", check=False).returncode
        ):
            git(
                "commit",
                "-m",
                f"chore: synchronize release {published} and set {next_dev}",
            )
            print(f"Synchronized release {published}; develop is {next_dev}")
        else:
            print(f"Release {published} already synchronized; develop is {next_dev}")
    except Exception:
        if git("rev-parse", "--verify", "MERGE_HEAD", check=False).returncode == 0:
            git("merge", "--abort")
        raise


if __name__ == "__main__":
    synchronize(sys.argv[1])
