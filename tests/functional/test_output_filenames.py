"""Compact output names retain identity and remain discoverable when resuming."""

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from spikee import tester
from spikee.utilities.files import (
    build_file_name,
    compact_filename_part,
    read_jsonl_file,
    write_jsonl_file,
)


def test_short_safe_names_are_unchanged(monkeypatch):
    monkeypatch.setattr("spikee.utilities.files.time.time", lambda: 1788882070)
    assert compact_filename_part("my_dataset-v1.2") == "my_dataset-v1.2"
    assert build_file_name("results", "mock", "dataset", None) == (
        "results_mock_dataset_1788882070.jsonl"
    )


@pytest.mark.parametrize(
    "options",
    [
        "target-url=https://example.test/" + "long-path/" * 100,
        "model=provider/model,temperature=0.5",
        "model=" + "日本語" * 100,
        "../\\:*?<>|\x00",
    ],
)
def test_option_labels_are_short_safe_and_stable(options):
    name = tester._build_target_name("mock", options)
    label = name.removeprefix("mock-")
    assert len(label) <= 37
    assert re.fullmatch(r"[A-Za-z0-9_.-]+-[0-9a-f]{12}|[0-9a-f]{12}", label)
    assert tester._build_target_name("mock", options) == name


def test_hash_distinguishes_truncated_and_cleaned_options():
    options = ["x/y", "x:y", "x-y", "x" * 100 + "a", "x" * 100 + "b"]
    names = {tester._build_target_name("mock", value) for value in options}
    assert len(names) == len(options)


def test_default_options_use_the_same_label(monkeypatch):
    options = "model=provider/model"
    monkeypatch.setattr(tester, "load_module_from_path", lambda *args: object())
    monkeypatch.setattr(tester, "get_default_option", lambda module: options)
    assert tester._build_target_name("mock", None) == tester._build_target_name(
        "mock", options
    )
    monkeypatch.setattr(tester, "get_default_option", lambda module: None)
    assert tester._build_target_name("mock", None) == "mock"


def test_whole_filename_is_bounded_and_keeps_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr("spikee.utilities.files.time.time", lambda: 1788882070)
    parts = ("target-" * 100, "dataset-" * 100, "tag-" * 100)
    filename = build_file_name("results", *parts)
    assert len(filename.encode("utf-8")) <= 180
    assert filename.startswith("results_")
    assert filename.endswith("_1788882070.jsonl")
    assert filename != build_file_name("results", *parts[:-1], parts[-1] + "other")
    (tmp_path / filename).touch()
    assert tester._find_resume_candidates(
        tmp_path, parts[0], parts[1] + ".jsonl", parts[2]
    ) == [tmp_path / filename]


@pytest.mark.parametrize("tag", [None, "tag_with_underscores"])
def test_resume_finds_old_and_new_names_without_matching_other_runs(
    tmp_path, monkeypatch, tag
):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    options = "target-url=https://example.test/run[1]"
    target = tester._build_target_name("mock", options)
    legacy_target = tester._build_target_name("mock", options, legacy=True)
    legacy_stem = "results_mock-target-url=https~example.test~run[1]_dataset"
    if tag:
        legacy_stem += f"_{tag}"
    old = results / f"{legacy_stem}_100.jsonl"
    old.touch()
    monkeypatch.setattr("spikee.utilities.files.time.time", lambda: 200)
    new = results / build_file_name("results", target, "dataset", tag)
    new.touch()
    (results / build_file_name("results", target, "dataset", "other-tag")).touch()
    (results / build_file_name("results", target, "dataset", tag, "extra")).touch()
    other = tester._build_target_name("mock", options + "other")
    (results / build_file_name("results", other, "dataset", tag)).touch()
    # A directory that looks like a result must not be offered for resume.
    (results / new.name.replace("_200.jsonl", "_300.jsonl")).mkdir()
    assert tester._find_resume_candidates(
        results, target, "dataset.jsonl", tag, legacy_target_name_full=legacy_target
    ) == [new, old]
    args = SimpleNamespace(
        target="mock", target_options=options, tag=tag, auto_resume=True
    )
    assert Path(
        tester._determine_resume_file(args, "dataset.jsonl", False)
    ) == new.relative_to(tmp_path)
    new.unlink()
    assert Path(
        tester._determine_resume_file(args, "dataset.jsonl", False)
    ) == old.relative_to(tmp_path)


@pytest.mark.parametrize("output_format", ["user-input", "full-prompt", "burp"])
def test_generate_with_long_seed_name(run_spikee, workspace_dir, output_format):
    seed = workspace_dir / "datasets" / ("seeds-" + "long-name=" * 15)
    shutil.copytree(workspace_dir / "datasets/seeds-functional-basic", seed)
    result = run_spikee(
        [
            "generate",
            "--seed-folder",
            str(seed),
            "--format",
            output_format,
            "--tag",
            "t" * 50,
        ],
        cwd=workspace_dir,
    )
    match = re.search(r"Dataset generated and saved to (.+)", result.stdout)
    assert match, result.stdout
    path = workspace_dir / match.group(1)
    assert path.is_file()
    assert len(path.name.encode("utf-8")) <= 180
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", path.name)
    assert f"-{output_format}-" in path.name
    assert path.suffix == (".txt" if output_format == "burp" else ".jsonl")


def test_cli_long_options_are_only_shortened_in_filename(run_spikee, workspace_dir):
    (workspace_dir / "targets/echo_options.py").write_text(
        "from spikee.templates.target import Target\n"
        "class EchoOptions(Target):\n"
        "    def get_available_option_values(self):\n"
        "        return [], False\n"
        "    def process_input(self, input_text, system_message=None, target_options=None):\n"
        "        return target_options\n"
    )
    dataset = workspace_dir / "datasets" / ("long-dataset-" * 15 + ".jsonl")
    write_jsonl_file(
        dataset,
        [
            {
                "id": 1,
                "long_id": "one",
                "content": "hello",
                "judge_name": "canary",
                "judge_args": "https",
            }
        ],
    )
    options = "target-url=https://example.test/" + "long-path/" * 100
    args = [
        "test",
        "--target",
        "echo_options",
        "--target-options",
        options,
        "--dataset",
        str(dataset),
    ]
    result = run_spikee([*args, "--no-auto-resume"], cwd=workspace_dir)
    match = re.search(r"Results saved to (.+\.jsonl)", result.stdout)
    assert match, result.stdout
    path = workspace_dir / match.group(1)
    assert len(path.name.encode("utf-8")) <= 180
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", path.name)
    assert read_jsonl_file(path)[0]["response"] == options
    resumed = run_spikee([*args, "--auto-resume"], cwd=workspace_dir)
    assert "[Auto-Resume]" in resumed.stdout
    assert "All entries have already been processed" in resumed.stdout
