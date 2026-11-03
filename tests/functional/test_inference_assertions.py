import pytest

from tests.inference_assertions import assert_attack_inference


def test_generated_attack_reaches_target():
    assert_attack_inference(
        {"input": "question", "response": "target reply", "error": None},
        "[Provider DEBUG] Response: generated question\n",
    )


@pytest.mark.parametrize(
    "error",
    ["HTTP 402", "HTTP 429", "model_not_found", "bound to a different event loop"],
)
def test_error_in_response_cannot_pass(error):
    with pytest.raises(AssertionError, match="Attack inference failed"):
        assert_attack_inference(
            {
                "input": "earlier question",
                "response": f"Error during attack: {error}",
                "error": None,
            }
        )


def test_error_in_history_cannot_pass():
    with pytest.raises(AssertionError):
        assert_attack_inference(
            {
                "input": "question",
                "response": "reply",
                "attempt_history": [{"error": "HTTP 429"}],
            }
        )


def test_invalid_json_is_not_automatically_a_refusal():
    with pytest.raises(AssertionError):
        assert_attack_inference(
            {"response": "Error: invalid response after 2 attempts"}
        )


def test_explicit_model_refusal_is_valid_connectivity_evidence():
    assert_attack_inference(
        {"response": "Error: invalid response after 2 attempts"},
        "[Provider DEBUG] Response: I cannot assist with this request.\n",
    )


def test_refusal_text_in_prompt_does_not_count():
    with pytest.raises(AssertionError):
        assert_attack_inference(
            {"response": "Error: invalid response after 2 attempts"},
            "[Provider DEBUG] Prompt: Example: I cannot assist with this request.\n",
        )


def test_refusal_cannot_hide_a_later_transport_error():
    with pytest.raises(AssertionError):
        assert_attack_inference(
            {"response": "Error: HTTP 429"},
            "[Provider DEBUG] Response: I cannot assist with this request.\n",
        )


def test_echo_chamber_fixture_supplies_valid_seed_sentences():
    from spikee.attacks.echo_chamber import EchoChamber
    from tests.functional.workspace.targets.mock_echo_chamber import (
        MockEchoChamberTarget,
    )

    _, sentences = EchoChamber().get_seed_sentences(
        MockEchoChamberTarget(), None, "test-session", ["France", "capital", "Paris"], 0
    )
    assert len(sentences) == 3
    assert "Paris" in sentences[0]


def test_offline_output_does_not_prove_live_inference():
    with pytest.raises(AssertionError, match="No live model reply"):
        assert_attack_inference({"input": "question", "response": "target reply"})


def test_failed_llm_generation_cannot_pass_using_fallback():
    with pytest.raises(AssertionError, match="fallback"):
        assert_attack_inference(
            {"input": "original question", "response": "target reply"},
            "[Provider DEBUG] Response: invalid JSON\n[Warning] parsing failed\nError generating LLM variants: invalid response",
        )


@pytest.mark.parametrize("option, expected", [("", None), (",max-tokens=480", 480)])
def test_multilingual_attack_passes_output_budget(monkeypatch, option, expected):
    from unittest.mock import Mock

    from spikee.attacks import llm_multi_language_jailbreaker as module

    provider = Mock()
    monkeypatch.setattr(module, "get_llm", provider)
    module.LLMMultiLanguageJailbreaker().attack(
        {"content": "Name the capital of France"},
        Mock(),
        Mock(),
        0,
        attack_option="model=groq/test" + option,
    )
    provider.assert_called_once_with("groq/test", max_tokens=expected)


@pytest.mark.parametrize("budget", ["0", "-1", "invalid"])
def test_multilingual_attack_rejects_invalid_output_budget(budget):
    from unittest.mock import Mock

    from spikee.attacks.llm_multi_language_jailbreaker import (
        LLMMultiLanguageJailbreaker,
    )

    with pytest.raises(ValueError):
        LLMMultiLanguageJailbreaker().attack(
            {"content": "Name the capital of France"},
            Mock(),
            Mock(),
            0,
            attack_option=f"model=groq/test,max-tokens={budget}",
        )
