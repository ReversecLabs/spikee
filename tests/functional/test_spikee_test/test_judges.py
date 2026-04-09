import pytest
import os

from spikee.utilities.modules import load_module_from_path
from spikee.utilities.files import read_jsonl_file
from spikee.templates.judge import Judge
from spikee.templates.llm_judge import LLMJudge
from ..utils import spikee_generate_cli, spikee_test_cli

def test_judge_canary():
    judge = load_module_from_path("canary", "judges")

    assert isinstance(judge, Judge), f"Canary judge should be an instance of Judge: {type(judge)}"

    canary = "CANARY_STRING"
    test_cases = {
        "This response contains the CANARY_STRING.": True,
        "This response does not have it.": False,
    }

    for response, expected in test_cases.items():
        result = judge.judge("", response, canary)

        assert result == expected, f"Expected {expected} for response: '{response}', got {result}"

def test_llm_judge_regex():
    judge = load_module_from_path("regex", "judges")

    assert isinstance(judge, Judge), f"Regex judge should be an instance of Judge: {type(judge)}"

    test_cases = {
        "The answer is 42.": True,
        "No numbers here!": False,
        "Multiple numbers: 1, 2, 3.": True,
    }

    regex = r"\d+"  # Matches any string containing a digit

    for response, expected in test_cases.items():
        result = judge.judge("", response, regex)

        assert result == expected, f"Expected {expected} for response: '{response}', got {result}"

@pytest.mark.parametrize(
    "model, required_env, valid",
    [
        ("openai/gpt-4o", ["OPENAI_API_KEY"], True),
        ("openai/nonexistent-model", [], False),
        ("bedrock/", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"], True),
        ("google/", ["GOOGLE_API_KEY"], True),
    ],
)
def test_llm_judge_inference(model, required_env, valid):
    missing_env = [env for env in required_env if not os.environ.get(env)]
    if missing_env:
        pytest.skip(f"Skipping test because required environment variables are missing: {', '.join(missing_env)}")

    judge = load_module_from_path("llm_judge_output_criteria", "judges")

    assert isinstance(judge, LLMJudge), f"LLM Judge should be an instance of LLMJudge: {type(judge)}"

    criteria = "contains the word 'forbidden'"
    test_cases = {
        "This response contains the word forbidden.": True,
        "This response is clean.": False,
    }

    for response, expected in test_cases.items():
        if not valid:
             with pytest.raises(Exception):
                result = judge.judge("", response, criteria, judge_options=model)
        else:
            result = judge.judge("", response, criteria, judge_options=model)
            assert result == expected, f"Expected {expected} for response: '{response}', got {result}"

@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_test_custom_judge_default_mode(run_spikee, workspace_dir, target_name, judge_variant):
    dataset_filename = (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )
    dataset_path = workspace_dir / "datasets" / dataset_filename
    assert dataset_path.exists()

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
    )

    results = read_jsonl_file(results_file[0])
    assert results
    assert all(not entry["success"] for entry in results)

@pytest.mark.parametrize("target_name", ["always_success", "always_success_legacy"])
@pytest.mark.parametrize("judge_variant", ["test_judge", "test_judge_legacy"])
def test_spikee_test_custom_judge_with_options(run_spikee, workspace_dir, target_name, judge_variant):
    dataset_filename = (
        "test_judge_dataset_legacy.jsonl"
        if judge_variant.endswith("_legacy")
        else "test_judge_dataset.jsonl"
    )
    dataset_path = workspace_dir / "datasets" / dataset_filename
    assert dataset_path.exists()

    results_file, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
        additional_args=[
            "--judge-options",
            f"{judge_variant}:mode=success",
        ],
    )

    results = read_jsonl_file(results_file[0])
    assert results
    assert all(entry["success"] for entry in results)