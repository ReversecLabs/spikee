import json
import os

import pytest

from tests.inference.models import (
    AZURE,
    BEDROCK,
    DEEPSEEK,
    GOOGLE,
    GROQ,
    OPENAI,
    OPENROUTER,
    PROVIDER_CASES,
    TOGETHER,
)
from tests.inference_assertions import assert_attack_inference

# Require RUN_INFERENCE_TESTS environment variable to run these tests
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INFERENCE_TESTS") != "1",
    reason="Run manually with RUN_INFERENCE_TESTS=1",
)

TARGET_MODEL_COMBINATIONS = [
    ("llm_provider", model) for model, _, _ in PROVIDER_CASES
] + [("llm_provider", "openai/invalid-model-name-123")]

JUDGE_MODEL_COMBINATIONS = [
    ("llm_judge_harmful", "", OPENAI),
    ("llm_judge_objective", "The capital is France.", GROQ),
    (
        "llm_judge_output_criteria",
        '{"schema": {"type": "string", "maxLength": 10}}',
        OPENROUTER,
    ),
    ("llm_judge_objective", "The capital is France.", BEDROCK),
    ("llm_judge_objective", "The capital is France.", "openai/invalid-model-name-123"),
]


def write_single_entry(workspace_dir):
    """Helper to write the single test entry to the workspace datasets folder."""
    dataset_dir = workspace_dir / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    dataset_file = dataset_dir / "single_entry_inf.jsonl"
    entry = {
        "id": 1,
        "long_id": "inf-test-01",
        "text": "What is the capital of France? Just tell me the city name, nothing else.",
        "context": "",
        "system_message": "",
        "judge_name": "canary",
        "judge_args": "test",
        "tags": ["inference-test"],
    }
    with dataset_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return dataset_file


def write_judge_entry(workspace_dir, judge_name, judge_args, judge_options):
    """Helper to write a test entry for a specific LLM judge."""
    dataset_dir = workspace_dir / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    dataset_file = dataset_dir / "judge_entry_inf.jsonl"
    entry = {
        "id": 1,
        "long_id": "inf-judge-test-01",
        "text": "What is the capital of France?",
        "context": "",
        "system_message": "",
        "judge_name": judge_name,
        "judge_args": judge_args,
        "tags": ["inference-test"],
    }
    with dataset_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return dataset_file


@pytest.mark.parametrize("target_name, model_name", TARGET_MODEL_COMBINATIONS)
def test_inference_targets(run_spikee, workspace_dir, target_name, model_name):
    """
    Test live inference against built-in targets using the single-entry dataset.
    """
    dataset_file = write_single_entry(workspace_dir)
    args = [
        "test",
        "--dataset",
        str(dataset_file.relative_to(workspace_dir)),
        "--target",
        target_name,
        "--target-options",
        model_name,
    ]
    spikee_cmd = "spikee " + " ".join(args)

    import re

    result = run_spikee(args, cwd=workspace_dir)

    assert result.returncode == 0, (
        f"spikee test failed for {target_name} ({model_name}). Stderr: {result.stderr}"
    )

    # Verify the results file to ensure inference actually worked
    match = re.search(r"saved to results/(results_[^\s]+\.jsonl)", result.stdout)
    assert match is not None, (
        "Results file not found in stdout. Execution may have failed."
    )

    result_file = workspace_dir / "results" / match.group(1)
    assert result_file.exists(), f"Result file missing: {result_file}"

    with result_file.open("r", encoding="utf-8") as f:
        data = json.loads(f.readline().strip())

    error_val = data.get("error")

    if "invalid-model-name-123" in model_name:
        error_str = str(error_val) if error_val else str(data.get("response", ""))
        is_invalid = (
            "does not exist" in error_str
            or "NotFoundError" in error_str
            or "Unknown" in error_str
        )
        assert is_invalid, (
            f"Expected a 'Not Found' or 'Unknown' error for invalid model {model_name}, but got: {error_str}. Result data: {data}"
        )
        return

    # If the LLM call errored out (e.g., missing API key), the response will typically be missing or flag an error
    response = data.get("response", "")
    assert error_val is None, f"Provider returned an error: {error_val}"
    assert response, (
        f"Empty or missing response from {target_name}. LLM call likely failed.\nCommand: {spikee_cmd}\nResult data: {data}"
    )


ATTACK_MODEL_COMBINATIONS = [
    ("llm_jailbreaker", f"model={OPENAI}"),
    ("llm_poetry_jailbreaker", f"model={GOOGLE}"),
    ("llm_poetry_jailbreaker", f"model={DEEPSEEK}"),
    ("llm_multi_language_jailbreaker", f"model={GROQ},max-tokens=480"),
    ("llm_multi_language_jailbreaker", f"model={BEDROCK}"),
    ("rag_poisoner", f"model={DEEPSEEK}"),
    ("rag_poisoner", f"model={TOGETHER}"),
    ("crescendo", f"model={TOGETHER}"),
    ("crescendo", f"model={OPENROUTER}"),
    ("echo_chamber", f"model={OPENROUTER}"),
    ("echo_chamber", f"model={BEDROCK}"),
    ("echo_chamber", f"model={OPENAI}"),
    ("prompt_decomposition", f"mode={GOOGLE}"),
    ("crescendo", "model=openai/invalid-model-name-123"),
]
if AZURE:
    ATTACK_MODEL_COMBINATIONS.extend(
        [
            ("llm_jailbreaker", f"model={AZURE}"),
            ("prompt_decomposition", f"mode={AZURE}"),
        ]
    )


@pytest.mark.parametrize("attack_name, attack_options", ATTACK_MODEL_COMBINATIONS)
def test_inference_attacks(
    run_spikee, workspace_dir, attack_name, attack_options, monkeypatch
):
    """
    Test live inference for LLM-powered attacks using a simple mock target ('always_refuse' loaded from functional workspaces).
    This verifies that the attacks can successfully format and send provider completion requests.
    """
    monkeypatch.setenv("PROVIDER_DEBUG", "1")
    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", "true")
    dataset_file = write_single_entry(workspace_dir)
    target = {
        "crescendo": "mock_multiturn",
        "echo_chamber": "mock_echo_chamber",
    }.get(attack_name, "always_refuse")

    args = [
        "test",
        "--dataset",
        str(dataset_file.relative_to(workspace_dir)),
        "--target",
        target,
        "--attack",
        attack_name,
        "--attack-options",
        attack_options,
        "--attack-iterations",
        "2",  # Just run 2 iterations to test connectivity
    ]

    import re

    result = run_spikee(args, cwd=workspace_dir)

    assert result.returncode == 0, (
        f"spikee test failed for attack {attack_name} ({attack_options}). Stderr: {result.stderr}"
    )

    # Verify the results file to ensure attack LLM inference actually worked
    match = re.search(r"saved to results/(results_[^\s]+\.jsonl)", result.stdout)
    assert match is not None, (
        "Results file not found in stdout. Execution may have failed."
    )

    result_file = workspace_dir / "results" / match.group(1)
    assert result_file.exists(), f"Result file missing: {result_file}"

    with result_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) >= 2, (
        f"Expected at least standard and attack entries, got {len(lines)} lines."
    )

    # The last entry should represent the attack attempt if it reached that stage and errored,
    # or the final iteration. We verify the attack successfully produced an event.
    attack_data = json.loads(lines[-1].strip())

    # Attacks may report failures in error, response, or attempt_history.
    error_val = attack_data.get("error")

    if "invalid-model-name-123" in attack_options:
        error_str = (
            str(error_val) if error_val else str(attack_data.get("response", ""))
        )
        is_invalid = (
            "does not exist" in error_str
            or "NotFoundError" in error_str
            or "Unknown" in error_str
        )
        assert is_invalid, (
            f"Expected a 'Not Found' or 'Unknown' error for invalid model {attack_options}, but got: {error_str}. Result data: {attack_data}"
        )
        return

    outcome = assert_attack_inference(attack_data, result.stdout)
    if outcome == "target_response":
        response = attack_data["response"]
        if target == "always_refuse":
            assert response == "Sorry, I can't answer that."
        elif target == "mock_echo_chamber":
            assert response == "Paris is the capital of France."
        else:
            assert response.startswith("Response to ") or response in {
                "History cleared",
                "I cannot do that",
                "Here is the flag",
            }, f"Expected a mock target reply, received: {response}"


@pytest.mark.parametrize(
    "judge_name, judge_args, judge_options", JUDGE_MODEL_COMBINATIONS
)
def test_inference_llm_judges(
    run_spikee, workspace_dir, judge_name, judge_args, judge_options
):
    """
    Test live inference for LLM-based judges.
    """
    dataset_file = write_judge_entry(
        workspace_dir, judge_name, judge_args, judge_options
    )

    # "always_refuse" provides clean predictable text back to the judge
    args = [
        "test",
        "--dataset",
        str(dataset_file.relative_to(workspace_dir)),
        "--target",
        "always_refuse",
        "--judge-options",
        judge_options,
    ]
    spikee_cmd = "spikee " + " ".join(args)

    import re

    result = run_spikee(args, cwd=workspace_dir)

    assert result.returncode == 0, (
        f"spikee test failed for judge {judge_name} ({judge_options}). Stderr: {result.stderr}"
    )

    match = re.search(r"saved to results/(results_[^\s]+\.jsonl)", result.stdout)
    assert match is not None, (
        "Results file not found in stdout. Execution may have failed."
    )

    result_file = workspace_dir / "results" / match.group(1)

    with result_file.open("r", encoding="utf-8") as f:
        result_data = json.loads(f.readline().strip())

    error_val = result_data.get("error")

    if "invalid-model-name-123" in judge_options:
        error_str = (
            str(error_val) if error_val else str(result_data.get("response", ""))
        )
        is_invalid = (
            "does not exist" in error_str
            or "NotFoundError" in error_str
            or "Unknown" in error_str
        )
        assert is_invalid, (
            f"Expected a 'Not Found' or 'Unknown' error for invalid model {judge_options}, but got: {error_str}. Result data: {result_data}"
        )
        return

    # For a judge inference test, we just want to ensure the LLM successfully connected
    # and evaluated without crashing or returning an error string in the results.
    assert error_val is None, (
        f"LLM Judge call failed.\nCommand: {spikee_cmd}\nError: {error_val}\nResult data: {result_data}"
    )
