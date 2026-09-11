"""Diagnostic history preserves responses without changing attack error handling."""

import importlib
from types import SimpleNamespace

import pytest

ATTACKS = [
    ("anti_spotlighting", "AntiSpotlightingAttack", None),
    ("best_of_n", "BestOfNAttack", None),
    ("random_suffix_search", "RandomSuffixSearch", None),
    ("prompt_decomposition", "PromptDecompositionAttack", None),
    ("llm_jailbreaker", "LLMJailbreaker", "_generate_jailbreak_attack"),
    (
        "llm_multi_language_jailbreaker",
        "LLMMultiLanguageJailbreaker",
        "_generate_multilingual_jailbreak_attack",
    ),
    ("llm_poetry_jailbreaker", "LLMPoetryJailbreaker", "_generate_jailbreak_attack"),
    ("rag_poisoner", "RAGPoisoner", "_generate_rag_attack"),
    ("sample_attack", "SampleAttack", None),
]
LLM_ATTACKS = [attack for attack in ATTACKS if attack[2]]


def run_attack(monkeypatch, spec, enabled, error_stage, error_at):
    name, class_name, generator_method = spec
    package = (
        "spikee.data.workspace.attacks" if name == "sample_attack" else "spikee.attacks"
    )
    module = importlib.import_module(f"{package}.{name}")
    attack = getattr(module, class_name)()
    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(enabled))
    target_inputs = []
    judge_inputs = []
    generated = []

    if generator_method:
        monkeypatch.setattr(module, "get_llm", lambda *args, **kwargs: object())

        def generate(_llm, _objective, previous):
            generated.append(list(previous))
            if error_stage == "generator" and len(generated) == error_at:
                raise RuntimeError("generator failed")
            return f"candidate-{len(generated)}"

        monkeypatch.setattr(attack, generator_method, generate)
    elif name == "anti_spotlighting":
        monkeypatch.setattr(attack, "_generate_variants", lambda text: [text] * 3)
    elif name == "best_of_n":
        monkeypatch.setattr(attack, "_generate_variant", lambda text, _: text)
    elif name == "prompt_decomposition":
        monkeypatch.setattr(attack, "_generate_variants_dumb", lambda text: [text] * 3)
    elif name == "random_suffix_search":
        monkeypatch.setattr(
            module.tiktoken,
            "get_encoding",
            lambda _: SimpleNamespace(n_vocab=128, decode=lambda _: "suffix"),
        )
    elif name == "sample_attack":
        monkeypatch.setattr(module.random, "randint", lambda *_: 1)

    def process_input(candidate, *_args, **_kwargs):
        target_inputs.append(candidate)
        if error_stage == "target" and len(target_inputs) == error_at:
            raise RuntimeError("target failed")
        return f"target-response-{len(target_inputs)}", {}

    def judge(_entry, response):
        judge_inputs.append(response)
        if error_stage == "judge" and len(judge_inputs) == error_at:
            raise RuntimeError("judge failed")
        return False

    result = attack.attack(
        {"id": 1, "text": "objective", "content": "objective"},
        SimpleNamespace(process_input=process_input),
        judge,
        3,
    )
    return result, target_inputs, judge_inputs, generated


@pytest.mark.parametrize("spec", ATTACKS, ids=[item[0] for item in ATTACKS])
@pytest.mark.parametrize("error_stage", ["target", "judge"])
@pytest.mark.parametrize("error_at", [2, 3])
def test_target_and_judge_errors_preserve_available_response(
    monkeypatch, spec, error_stage, error_at
):
    enabled = run_attack(monkeypatch, spec, True, error_stage, error_at)
    disabled = run_attack(monkeypatch, spec, False, error_stage, error_at)
    result, target_inputs, judge_inputs, _generated = enabled
    plain_result = disabled[0]

    assert result[:2] == plain_result[:2] == (3, False)
    assert result[2]["input"] == plain_result[2]["input"]
    assert result[3] == plain_result[3]
    assert enabled[1:] == disabled[1:]
    assert "attempt_history" not in plain_result[2]

    history = result[2]["attempt_history"]
    assert len(history) == len(target_inputs) == 3
    assert [attempt["input"] for attempt in history] == target_inputs
    error = history[error_at - 1]
    assert error["error"] == f"{error_stage} failed"
    assert error["success"] is None
    assert error["response"] == (
        f"target-response-{error_at}" if error_stage == "judge" else ""
    )
    assert len(judge_inputs) == (2 if error_stage == "target" else 3)
    for index, attempt in enumerate(history, 1):
        if index != error_at:
            assert attempt["success"] is False
            assert attempt["response"] == f"target-response-{index}"
    if error_at == 3:
        assert f"{error_stage} failed" in result[3]


@pytest.mark.parametrize("spec", LLM_ATTACKS, ids=[item[0] for item in LLM_ATTACKS])
@pytest.mark.parametrize("error_at", [2, 3])
def test_generator_errors_have_no_stale_input_or_response(monkeypatch, spec, error_at):
    enabled = run_attack(monkeypatch, spec, True, "generator", error_at)
    disabled = run_attack(monkeypatch, spec, False, "generator", error_at)
    result, target_inputs, judge_inputs, generated = enabled
    plain_result = disabled[0]

    assert result[:2] == plain_result[:2] == (3, False)
    assert result[2]["input"] == plain_result[2]["input"]
    assert result[3] == plain_result[3]
    assert enabled[1:] == disabled[1:]
    assert "attempt_history" not in plain_result[2]
    assert len(target_inputs) == len(judge_inputs) == 2
    assert len(generated) == 3

    history = result[2]["attempt_history"]
    assert len(history) == 3
    assert history[error_at - 1] == {
        "input": "",
        "response": "",
        "success": None,
        "error": "generator failed",
    }
    assert [
        attempt["input"] for attempt in history if attempt["input"]
    ] == target_inputs
    assert [
        attempt["response"] for attempt in history if attempt["input"]
    ] == judge_inputs
    if error_at == 3:
        assert result[3] == "Error during attack attempt 3: generator failed"


def test_decomposition_generation_error_has_no_target_response(monkeypatch):
    from spikee.attacks.prompt_decomposition import PromptDecompositionAttack

    attack = PromptDecompositionAttack()
    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", "true")

    def fail_generation(_text):
        raise RuntimeError("generator failed")

    monkeypatch.setattr(attack, "_generate_variants_dumb", fail_generation)
    result = attack.attack({"id": 1, "text": "objective"}, None, None, 3)
    assert result[:2] == (0, False)
    assert result[3] == "generator failed"
    assert result[2]["attempt_history"] == [
        {
            "input": "objective",
            "response": "",
            "success": None,
            "error": "generator failed",
        }
    ]
