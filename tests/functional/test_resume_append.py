"""Resume keeps one results file across interruptions and preserves saved rows."""

import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from spikee import tester
from spikee.utilities.files import append_jsonl_entry, read_jsonl_file, write_jsonl_file


def test_repeated_resume_appends_then_leaves_completed_file_untouched(
    run_spikee, workspace_dir
):
    dataset = workspace_dir / "datasets" / "resume.jsonl"
    entries = [
        {
            "id": i,
            "long_id": f"entry-{i}",
            "content": "Hello",
            "judge_name": "canary",
            "judge_args": "EN-CHECK",
        }
        for i in range(8)
    ]
    write_jsonl_file(dataset, entries)
    resume_file = workspace_dir / "saved-results.jsonl"
    # External/renamed results, non-default spacing, UTF-8, no final newline.
    original = b'{"id":0,"response":"caf\xc3\xa9","success":false}'
    resume_file.write_bytes(original)
    args = [
        "test",
        "--target",
        "always_success",
        "--dataset",
        str(dataset),
        "--resume-file",
        "saved-results.jsonl",
        "--threads",
        "4",
    ]
    for sample in ("0.5", "1"):
        before = resume_file.read_bytes()
        result = run_spikee([*args, "--sample", sample], cwd=workspace_dir)
        assert "Results saved to saved-results.jsonl" in result.stdout
        assert resume_file.read_bytes().startswith(before)
        assert not list((workspace_dir / "results").glob("*.jsonl"))
        rows = read_jsonl_file(resume_file)
        assert len({r["id"] for r in rows}) == len(rows)
    assert len(rows) == len(entries)
    assert rows[0]["response"] == "café"
    before = resume_file.read_bytes()
    modified = resume_file.stat().st_mtime_ns
    result = run_spikee(args, cwd=workspace_dir)
    assert "All entries have already been processed" in result.stdout
    assert resume_file.read_bytes() == before
    assert resume_file.stat().st_mtime_ns == modified
    assert not list((workspace_dir / "results").glob("*.jsonl"))


@pytest.mark.parametrize("exists", [False, True])
def test_bad_resume_file_fails_without_creating_or_changing_results(
    run_spikee, workspace_dir, exists
):
    dataset = workspace_dir / "datasets" / "resume.jsonl"
    write_jsonl_file(dataset, [{"id": 1, "content": "Hello"}])
    resume_file = workspace_dir / "saved-results.jsonl"
    malformed = b'{"id":0}\n{"id":'
    if exists:
        resume_file.write_bytes(malformed)
    with pytest.raises(subprocess.CalledProcessError):
        run_spikee(
            [
                "test",
                "--target",
                "always_success",
                "--dataset",
                str(dataset),
                "--resume-file",
                str(resume_file),
            ],
            cwd=workspace_dir,
        )
    assert resume_file.exists() is exists
    if exists:
        assert resume_file.read_bytes() == malformed
    assert not list((workspace_dir / "results").glob("*.jsonl"))


def test_incomplete_legacy_attack_rows_are_replaced_without_losing_completed_rows(
    tmp_path,
):
    path = tmp_path / "results.jsonl"
    completed = b'{"id":1,"response":"old","success":true}\n'
    incomplete = b'{"id":"2-attack-1","attack_parent_id":2,"entry_complete":false}\n'
    another = b'{"id":3,"response":"other","success":false}\n'
    path.write_bytes(completed + incomplete + another)
    path.chmod(0o640)
    ids, _, _, count = tester._load_results_file(path, None, 1)
    assert count == 2
    tester._prepare_resume_file(path, ids)
    assert path.read_bytes() == completed + another
    assert path.stat().st_mode & 0o777 == 0o640
    append_jsonl_entry(path, {"id": "2-attack", "success": True}, threading.Lock())
    ids, _, _, count = tester._load_results_file(path, None, 1)
    assert {str(i) for i in ids} == {"1", "2", "3"} and count == 3
    assert list(tmp_path.iterdir()) == [path]


def test_cleanup_failure_keeps_original_results(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    original = b'{"id":1}\n{"id":2,"entry_complete":false}\n'
    path.write_bytes(original)

    def fail(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(tester.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failed"):
        tester._prepare_resume_file(path, {1})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_empty_resume_file_can_be_appended(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    tester._prepare_resume_file(path, set())
    append_jsonl_entry(path, {"id": 1}, threading.Lock())
    assert read_jsonl_file(path) == [{"id": 1}]


def test_interactive_selection_uses_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "results" / "results_mock_dataset_1.jsonl"
    path.parent.mkdir()
    path.write_text('{"id":1}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tester, "_select_resume_file_interactive", lambda *a, **k: path)
    args = SimpleNamespace(target="mock", target_options=None, tag=None)
    selected = tester._determine_resume_file(args, "dataset.jsonl", True)
    assert selected == str(path)
    tester._prepare_resume_file(selected, {1})
    append_jsonl_entry(selected, {"id": 2}, threading.Lock())
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == [1, 2]
    assert list(path.parent.iterdir()) == [path]
