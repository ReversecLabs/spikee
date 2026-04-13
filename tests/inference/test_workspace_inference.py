import os
import pytest

from spikee.utilities.modules import load_module_from_path
from spikee.utilities.files import read_jsonl_file
from spikee.templates.llm_judge import LLMJudge
from .utils import spikee_generate_cli, spikee_test_cli

# Skip the entire test file if RUN_INFERENCE_TESTS is set, to avoid running inference tests in environments where they are not intended
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INFERENCE_TESTS") is not None,
    reason="Skipping inference tests because RUN_INFERENCE_TESTS environment variable is set."
)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live inference plugin test",
)
def test_inference_plugin(run_spikee, workspace_dir):
    """Test test_inference plugin makes a live LLM call during dataset generation.

    Uses --languages en to limit to 2 base combos, reducing API calls.
    With 1 piped plugin and 2 en combos: 2 base + 2 plugin = 4 entries.
    Uses gpt-4o-mini via plugin-options for cheapest available model.
    """
    output_file = spikee_generate_cli(
        run_spikee,
        workspace_dir,
        additional_args=[
            "--plugins", "test_inference",
            "--plugin-options", "test_inference:model=openai/gpt-4o",
            "--plugin-only",
            "--languages", "en",
        ],
    )

    dataset = read_jsonl_file(output_file)

    # 2 en combos × (1 base + 1 plugin) = 4 entries
    assert len(dataset) == 4, f"Expected 4 entries (2 base + 2 plugin), got {len(dataset)}"

    inference_entries = [e for e in dataset if e.get("plugin") == "test_inference"]
    assert len(inference_entries) == 4, \
        f"Expected 2 test_inference plugin entries, got {len(inference_entries)}"

    # Plugin ran and produced non-empty payloads
    for entry in inference_entries:
        assert entry.get("payload"), \
            f"Expected non-empty payload from inference plugin, got: {entry.get('payload')!r}"
        assert "_test_inference-1" in entry["long_id"], \
            f"Expected '_test_inference-1' in long_id, got: {entry['long_id']}"


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


@pytest.mark.parametrize(
    "model, required_env, valid",
    [
        ("openai/gpt-4o", ["OPENAI_API_KEY"], True),
        ("openai/gpt-4o-mini", ["OPENAI_API_KEY"], True),
        ("openai/nonexistent-model", [], False),

        ("azure_openai/gpt-4o", ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY"], True),

        ("bedrock/", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"], True),
        ("bedrock/claude45-haiku", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"], True),
        ("bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"], True),
        ("bedrock/deepseek-v3", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"], True),

        ("deepseek/deepseek-chat", ["DEEPSEEK_API_KEY"], True),

        ("google/", ["GOOGLE_API_KEY"], True),
        ("google/gemini-2.5-flash", ["GOOGLE_API_KEY"], True),
        ("google/gemini-7.0-pro", ["GOOGLE_API_KEY"], False),

        ("groq/", ["GROQ_API_KEY"], True),

        ("llamacpp/", ["LLAMACPP_URL"], True),
        ("ollama/", ["OLLAMA_URL"], True),

        ("openrouter/", ["OPENROUTER_API_KEY"], True),
        ("togetherai/", ["TOGETHER_API_KEY"], True),
    ],
)
def test_spikee_inference_providers(run_spikee, workspace_dir, model, required_env, valid):
    missing_env = [env for env in required_env if not os.environ.get(env)]
    if missing_env:
        pytest.skip(f"Skipping test because required environment variables are missing: {', '.join(missing_env)}")

    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_files, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="llm_provider",
        datasets=[dataset_path],
        additional_args=[
            "--target-options", f"{model}",
        ]
    )

    if not valid:
        if len(results_files) == 0:
            assert True

        else:
            assert len(results_files) == 1, f"Expected 1 results file for invalid provider, got {len(results_files)}"

            results = read_jsonl_file(results_files[0])
            assert all(len(r["error"]) > 0 for r in results), "Expected all entries to fail with invalid provider"

    else:
        assert len(results_files) == 1, f"Expected 1 results file for valid provider, got {len(results_files)}"

        results = read_jsonl_file(results_files[0])

        assert len(results) > 0, "No results recorded by spikee test"
        assert len(results) == len(entries), f"Expected {len(entries)} results, got {len(results)}"
        assert all("response" in r and isinstance(r["response"], str) and len(r["response"]) > 0 for r in results), \
            "Expected all results to have a non-empty 'response' field from the LLM provider"
