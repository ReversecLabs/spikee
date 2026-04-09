import pytest
import os

from spikee.utilities.files import read_jsonl_file
from tests.functional.utils import extract_results_path
from ..utils import spikee_test_cli, spikee_generate_cli

@pytest.mark.parametrize(
    "target_name,expected_success",
    [
        ("always_refuse", False),
        ("always_refuse_legacy", False),
        ("always_success", True),
        ("always_success_legacy", True),
        ("always_guardrail", False),  # This target raises a GuardrailTrigger, which should be treated as a failure with the canary response
    ],
)
def test_spikee_test_targets(run_spikee, workspace_dir, target_name, expected_success):
    dataset_path = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset_path)

    results_files, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target=target_name,
        datasets=[dataset_path],
    )

    results = read_jsonl_file(results_files[0])

    assert len(results) > 0, "No results recorded by spikee test"
    assert len(results) == len(entries), f"Expected {len(entries)} results, got {len(results)}"
    assert all(entry["success"] == expected_success for entry in results)

    if target_name == "always_guardrail":
        # For the always_guardrail target, we expect all entries to have success=False and the canary response indicating the guardrail was triggered
        assert all("guardrail" in r and r['guardrail'] for r in results), "Expected all entries to have guardrail=True for the always_guardrail target {}".format(results)

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