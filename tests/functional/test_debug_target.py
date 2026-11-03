"""Exercise the debug CLI with the target return types supported by the tester."""

import subprocess

import pytest


def target_workspace(tmp_path, return_expression):
    targets = tmp_path / "targets"
    targets.mkdir()
    (targets / "debug_fixture.py").write_text(
        "def process_input(input_text, system_message=None, target_options=None):\n"
        "    assert input_text == 'hello'\n"
        "    assert system_message == 'system'\n"
        "    assert target_options == 'option'\n"
        f"    return {return_expression}\n"
    )
    return tmp_path


COMMAND = [
    "debug",
    "module",
    "targets",
    "-m",
    "debug_fixture",
    "-i",
    "hello",
    "--system-message",
    "system",
    "--target-options",
    "option",
]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('("reply", {"tokens": 3})', "reply"),
        ('("reply", None)', "reply"),
        ('"reply"', "reply"),
        ('(True, {"reason": "allowed"})', "True"),
        ('(False, {"reason": "blocked"})', "False"),
        ("True", "True"),
        ("False", "False"),
        ('("", {})', "No response"),
        ('""', "No response"),
        ("None", "No response"),
    ],
)
def test_debug_target_supported_responses(run_spikee, tmp_path, response, expected):
    result = run_spikee(COMMAND, cwd=target_workspace(tmp_path, response))
    assert f"Response: {expected}" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("response", ['("reply",)', '("reply", {}, "extra")'])
def test_debug_target_rejects_malformed_tuples(run_spikee, tmp_path, response):
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_spikee(COMMAND, cwd=target_workspace(tmp_path, response))
    output = (error.value.stdout or "") + (error.value.stderr or "")
    assert "Expected (Content/bool, meta)" in output
