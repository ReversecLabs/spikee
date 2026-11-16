"""Windows extended-length path handling for the results viewer (issue #141)."""

import os
from pathlib import Path

import pytest

from spikee.utilities import files as files_utils
from spikee.utilities.files import to_extended_path, windows_extended_path_str
from spikee.viewer.blueprints import results as viewer_results


@pytest.mark.parametrize(
    "path, expected",
    [
        ("C:\\a\\b.jsonl", "\\\\?\\C:\\a\\b.jsonl"),
        ("\\\\server\\share\\b.jsonl", "\\\\?\\UNC\\server\\share\\b.jsonl"),
        ("\\\\?\\C:\\a\\b.jsonl", "\\\\?\\C:\\a\\b.jsonl"),
        ("\\\\?\\UNC\\server\\share\\b.jsonl", "\\\\?\\UNC\\server\\share\\b.jsonl"),
        ("\\\\.\\C:\\a\\b.jsonl", "\\\\.\\C:\\a\\b.jsonl"),
    ],
)
def test_windows_extended_path_str(path, expected):
    assert windows_extended_path_str(path) == expected


def test_to_extended_path_is_noop_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(files_utils.os, "name", "posix")
    assert to_extended_path(tmp_path / "results") == tmp_path / "results"


def test_to_extended_path_prefixes_on_windows(monkeypatch):
    monkeypatch.setattr(files_utils.os, "name", "nt")
    monkeypatch.setattr(
        files_utils.os.path, "abspath", lambda p: "C:\\work\\results"
    )
    assert str(to_extended_path("results")) == "\\\\?\\C:\\work\\results"


def test_scan_finds_files_when_legacy_path_checks_fail(tmp_path, monkeypatch):
    """Behavioural regression test for #141.

    Simulates the Windows MAX_PATH limit: Path.is_file() is False for any path
    that is not reached through the extended-length alias. The scan must reach
    the files through to_extended_path(), otherwise it finds nothing.
    """
    results = tmp_path / "results"
    results.mkdir()
    (results / "results_demo_1.jsonl").write_text("{}\n")
    (results / "notes.txt").write_text("x")
    sub = results / "run1"
    sub.mkdir()
    (sub / "rejudge_demo_2.jsonl").write_text("{}\n")

    alias = tmp_path / "EXTENDED_ALIAS"
    alias.symlink_to(results, target_is_directory=True)

    real_is_file = Path.is_file

    def legacy_is_file(self, *args, **kwargs):
        if "EXTENDED_ALIAS" not in str(self):
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "is_file", legacy_is_file)
    # raising=False so the test runs (and fails on behaviour) without the fix
    monkeypatch.setattr(
        viewer_results, "to_extended_path", lambda p: alias, raising=False
    )
    monkeypatch.setattr(viewer_results, "loaded_files", {})

    viewer_results.scan_result_files()

    assert sorted(viewer_results.loaded_files) == [
        "results_demo_1",
        "run1/rejudge_demo_2",
    ]


def test_scan_passes_results_dir_to_extended_helper(tmp_path, monkeypatch):
    seen = []

    def spy(path):
        seen.append(Path(path))
        return Path(path)

    (tmp_path / "results").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(viewer_results, "to_extended_path", spy)
    monkeypatch.setattr(viewer_results, "loaded_files", {})
    viewer_results.scan_result_files()
    assert seen == [Path(os.getcwd()) / "results"]
