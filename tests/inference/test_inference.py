import os
import json
import pytest
from pathlib import Path

# Require RUN_INFERENCE_TESTS environment variable to run these tests
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INFERENCE_TESTS"),
    reason="Run manually with RUN_INFERENCE_TESTS=1",
)

TARGET_MODEL_COMBINATIONS = [
    # (Target Name, Target Option (Model Name))
    ("openai_api", "gpt-4o-mini"),
    ("openai_api", "gpt-4o"),
    ("azure_api", "gpt-4o-mini"),
    ("google_api", "gemini-2.5-flash"),
    ("openrouter_api", "meta-llama/llama-3.1-8b-instruct"),
    ("groq_api", "llama-3.1-8b-instant"),
    ("deepseek_api", "deepseek-chat"),
    ("togetherai_api", "llama31-8b"),
    ("aws_bedrock_api", "claude45-opus"),
    ("openai_api", "invalid-model-name-123"),
    # Add localized or other targets if needed (like ollama_api / llamacpp)
]

JUDGE_MODEL_COMBINATIONS = [
    # (Judge Name, Judge Args, Judge Options (LLM Model))
    ("llm_judge_harmful", "", "openai-gpt-4o-mini"),
    ("llm_judge_objective", "The capital is France.", "groq-llama-3.1-8b-instant"),
    ("llm_judge_output_criteria", '{"schema": {"type": "string", "maxLength": 10}}', "openrouter-meta-llama/llama-3.1-8b-instruct"),
    ("llm_judge_objective", "The capital is France.", "aws_bedrock-claude35-haiku"),
    ("llm_judge_objective", "The capital is France.", "openai-invalid-model-name-123"),
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
         "tags": ["inference-test"]
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
         "tags": ["inference-test"]
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
    
    import re
    result = run_spikee(args, cwd=workspace_dir)

    assert result.returncode == 0, f"spikee test failed for {target_name} ({model_name}). Stderr: {result.stderr}"
    
    # Verify the results file to ensure inference actually worked
    match = re.search(r"saved to results/(results_[^\s]+\.jsonl)", result.stdout)
    assert match is not None, "Results file not found in stdout. Execution may have failed."
    
    result_file = workspace_dir / "results" / match.group(1)
    assert result_file.exists(), f"Result file missing: {result_file}"
    
    with result_file.open("r", encoding="utf-8") as f:
        data = json.loads(f.readline().strip())
    
    error_val = data.get("error")
    
    if "invalid-model-name-123" in model_name:
        error_str = str(error_val) if error_val else str(data.get("response", ""))
        is_invalid = "does not exist" in error_str or "NotFoundError" in error_str or "Unknown" in error_str
        assert is_invalid, f"Expected a 'Not Found' or 'Unknown' error for invalid model {model_name}, but got: {error_str}. Result data: {data}"
        return

    # If the LLM call errored out (e.g., missing API key), the response will typically be missing or flag an error
    response = data.get("response", "")
    assert response, f"Empty or missing response from {target_name}. LLM call likely failed. Result data: {data}"


ATTACK_MODEL_COMBINATIONS = [
    # (Attack Module, Attack Option (Model))
    ("llm_jailbreaker", "model=openai-gpt-4o-mini"),
    ("llm_jailbreaker", "model=azure-gpt-4o-mini"),

    ("llm_poetry_jailbreaker", "model=google-gemini-2.5-flash"),
    ("llm_poetry_jailbreaker", "model=deepseek-deepseek-chat"),

    ("llm_multi_language_jailbreaker", "model=groq-llama-3.1-8b-instant"),
    ("llm_multi_language_jailbreaker", "model=bedrock-claude35-haiku"),

    ("rag_poisoner", "model=deepseek-deepseek-chat"),
    ("rag_poisoner", "model=together-llama31-8b"),

    ("crescendo", "model=together-llama31-8b"),
    ("crescendo", "model=openrouter-meta-llama/llama-3.1-8b-instruct"),

    ("echo_chamber", "model=openrouter-meta-llama/llama-3.1-8b-instruct"),
    ("echo_chamber", "model=bedrock-claude35-haiku"),
    ("echo_chamber", "model=openai-gpt-4o-mini"),

    ("prompt_decomposition", "modes=azure-gpt-4o-mini;variants=2"),
    ("prompt_decomposition", "modes=google-gemini-2.5-flash;variants=2"),
    
    ("crescendo", "model=openai-invalid-model-name-123"),
]


@pytest.mark.parametrize("attack_name, attack_options", ATTACK_MODEL_COMBINATIONS)
def test_inference_attacks(run_spikee, workspace_dir, attack_name, attack_options):
    """
    Test live inference for LLM-powered attacks using a simple mock target ('always_refuse' loaded from functional workspaces).
    This verifies that the attacks can successfully format and send litellm completion requests.
    """
    dataset_file = write_single_entry(workspace_dir)
    target = "mock_multiturn" if attack_name in ("crescendo", "echo_chamber") else "always_refuse"
    
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
        "2", # Just run 2 iterations to test connectivity
    ]
    
    import re
    result = run_spikee(args, cwd=workspace_dir)

    assert result.returncode == 0, f"spikee test failed for attack {attack_name} ({attack_options}). Stderr: {result.stderr}"
    
    # Verify the results file to ensure attack LLM inference actually worked
    match = re.search(r"saved to results/(results_[^\s]+\.jsonl)", result.stdout)
    assert match is not None, "Results file not found in stdout. Execution may have failed."
    
    result_file = workspace_dir / "results" / match.group(1)
    assert result_file.exists(), f"Result file missing: {result_file}"
    
    with result_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        
    assert len(lines) >= 2, f"Expected at least standard and attack entries, got {len(lines)} lines."
    
    # The last entry should represent the attack attempt if it reached that stage and errored, 
    # or the final iteration. We verify the attack successfully produced an event. 
    attack_data = json.loads(lines[-1].strip())
    
    # If the attack LLM generation fails (e.g missing API keys), the 'error' field will be populated
    # with the caught litellm exception instead of None.
    error_val = attack_data.get("error")
    
    if "invalid-model-name-123" in attack_options:
        error_str = str(error_val) if error_val else str(attack_data.get("response", ""))
        is_invalid = "does not exist" in error_str or "NotFoundError" in error_str or "Unknown" in error_str
        assert is_invalid, f"Expected a 'Not Found' or 'Unknown' error for invalid model {attack_options}, but got: {error_str}. Result data: {attack_data}"
        return
    
    # We consider the test passed if it succeeded (Error is None) OR if the LLM successfully connected
    # but refused to generate the attack prompt. We check for a few common refusal formats.
    is_success = error_val is None
    
    # Common refusal indicators:
    # 1. 'valid JSON object' - The model responded in plain text instead of requested JSON due to a guardrail/refusal.
    # 2. 'refused to answer' - Some APIs (like OpenRouter) explicitly wrapper LLM refusals with this message.
    error_str = str(error_val).lower() if error_val else ""
    is_refusal = "valid json object" in error_str or "refused to answer" in error_str
    
    assert is_success or is_refusal, f"Attack LLM call failed. Error: {error_val}. Result data: {attack_data}"

@pytest.mark.parametrize("judge_name, judge_args, judge_options", JUDGE_MODEL_COMBINATIONS)
def test_inference_llm_judges(run_spikee, workspace_dir, judge_name, judge_args, judge_options):
    """
    Test live inference for LLM-based judges.
    """
    dataset_file = write_judge_entry(workspace_dir, judge_name, judge_args, judge_options)
    
    # "always_refuse" provides clean predictable text back to the judge
    args = [
        "test",
        "--dataset",
        str(dataset_file.relative_to(workspace_dir)),
        "--target",
        "always_refuse",
        "--judge-options",
        judge_options
    ]
    
    import re
    result = run_spikee(args, cwd=workspace_dir)
    
    assert result.returncode == 0, f"spikee test failed for judge {judge_name} ({judge_options}). Stderr: {result.stderr}"
    
    match = re.search(r"saved to results/(results_[^\s]+\.jsonl)", result.stdout)
    assert match is not None, "Results file not found in stdout. Execution may have failed."
    
    result_file = workspace_dir / "results" / match.group(1)
    
    with result_file.open("r", encoding="utf-8") as f:
        result_data = json.loads(f.readline().strip())
        
    error_val = result_data.get("error")
    
    if "invalid-model-name-123" in judge_options:
        error_str = str(error_val) if error_val else str(result_data.get("response", ""))
        is_invalid = "does not exist" in error_str or "NotFoundError" in error_str or "Unknown" in error_str
        assert is_invalid, f"Expected a 'Not Found' or 'Unknown' error for invalid model {judge_options}, but got: {error_str}. Result data: {result_data}"
        return
        
    # For a judge inference test, we just want to ensure the LLM successfully connected 
    # and evaluated without crashing or returning an error string in the results.
    assert error_val is None, f"LLM Judge call failed. Error: {error_val}. Result data: {result_data}"

