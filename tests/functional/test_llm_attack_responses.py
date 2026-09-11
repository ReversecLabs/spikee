"""Offline structured attacker response parsing, repair and diagnostic regressions."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spikee.attacks.crescendo import Crescendo
from spikee.attacks.echo_chamber import EchoChamber, RefusalError
from spikee.attacks.llm_jailbreaker import LLMJailbreaker
from spikee.attacks.llm_multi_language_jailbreaker import LLMMultiLanguageJailbreaker
from spikee.attacks.llm_poetry_jailbreaker import LLMPoetryJailbreaker
from spikee.attacks.prompt_decomposition import PromptDecompositionAttack
from spikee.attacks.rag_poisoner import RAGPoisoner
from spikee.data.workspace.attacks import goat
from spikee.templates.standardised_conversation import StandardisedConversation
from spikee.utilities.llm_message import AIMessage, HumanMessage
from spikee.utilities.llm_response import (
    LLMResponseError,
    parse_json_object,
    parse_jsonl_variations,
    query_structured_response,
)


def provider(*responses, metadata=None):
    return SimpleNamespace(
        model="test/model",
        max_tokens=None,
        invoke=Mock(
            side_effect=[
                r
                if isinstance(r, Exception)
                else AIMessage(r, original_response=metadata)
                for r in responses
            ]
        ),
    )


def query(llm, messages=None, **kwargs):
    return query_structured_response(
        llm,
        messages if messages is not None else [HumanMessage("Return an object.")],
        parse_json_object,
        context="test.generate",
        **kwargs,
    )


@pytest.mark.parametrize(
    "wrapped",
    [
        '{"text":"A } brace and a \\"quote\\" and ```json"}',
        '```JSON\n{"text":"A } brace and a \\"quote\\" and ```json"}\n```',
        'Here is the result:\n{"text":"A } brace and a \\"quote\\" and ```json"}\nDone.',
    ],
)
def test_local_parser_preserves_values_inside_wrappers(wrapped):
    assert parse_json_object(wrapped) == {"text": 'A } brace and a "quote" and ```json'}


@pytest.mark.parametrize(
    "response",
    [
        "",
        None,
        "[]",
        "null",
        '"text"',
        '{"text":"complete"',
        '{"text":"unfinished',
        '{"outer":{"text":"nested"}',
        '[{"text":"nested"}',
        '{"text":1}{"text":2}',
        '{"text":1}, {"text":2}',
        '{"text":1}\nOr this:\n{"text":2}',
        '{text:"unquoted"}',
        '{"text":"a"b"}',
    ],
)
def test_local_parser_does_not_invent_content_or_salvage_inner_objects(response):
    with pytest.raises(ValueError):
        parse_json_object(response)


@pytest.mark.parametrize(
    "response,error",
    [
        ("{}", "Missing required JSON fields: text"),
        ('{"text":null}', "nonempty string"),
        ('{"text":12}', "nonempty string"),
        ('{"text":"   "}', "nonempty string"),
    ],
)
def test_schema_errors_are_useful_for_repair(response, error):
    with pytest.raises(ValueError, match=error):
        parse_json_object(response, required_keys=("text",), string_keys=("text",))


def test_repair_contains_failed_reply_and_actual_error_without_mutating_history(capsys):
    bad = '{"text":"A complete value"'
    llm = provider(bad, '{"text":"A complete value"}')
    messages = [HumanMessage("Return JSON with a text field.")]
    assert query(llm, messages) == {"text": "A complete value"}
    first, repaired = [call.args[0] for call in llm.invoke.call_args_list]
    assert first == messages
    assert len(messages) == 1
    assert repaired[0] is messages[0]
    assert repaired[-2].role == "assistant" and repaired[-2].content == bad
    assert "Expecting ',' delimiter" in repaired[-1].content
    assert "retain the intended values" in repaired[-1].content
    assert capsys.readouterr().out == ""


def test_exhaustion_is_bounded_and_history_does_not_grow(capsys):
    llm = provider("bad one", "bad two", "bad three")
    with pytest.raises(LLMResponseError, match="after 3 attempts") as exc:
        query(llm, max_attempts=3)
    assert exc.value.response == "bad three"
    requests = [call.args[0] for call in llm.invoke.call_args_list]
    assert [len(r) for r in requests] == [1, 3, 3]
    assert requests[2][-2].content == "bad two"
    output = capsys.readouterr().out
    assert output.count("invalid LLM response") == 1
    assert '"attempts_remaining": 0' in output


@pytest.mark.parametrize("during_repair", [False, True])
def test_provider_errors_are_not_retried_as_format_errors(during_repair):
    error = TimeoutError("provider timeout")
    responses = ["bad", error] if during_repair else [error]
    llm = provider(*responses)
    with pytest.raises(TimeoutError) as exc:
        query(llm)
    assert exc.value is error
    assert llm.invoke.call_count == len(responses)


@pytest.mark.parametrize("as_object", [False, True])
def test_diagnostics_include_only_selected_metadata(capsys, as_object):
    raw = {
        "model": "returned/model",
        "choices": [{"finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
        "headers": {"authorization": "DO-NOT-LOG"},
    }
    if as_object:
        raw = json.loads(json.dumps(raw), object_hook=lambda x: SimpleNamespace(**x))
    llm = provider("bad", "still bad", metadata=raw)
    with pytest.raises(LLMResponseError):
        query(llm)
    output = capsys.readouterr().out
    details = json.loads(output.split("metadata: ", 1)[1])
    assert details["model"] == "returned/model"
    assert details["finish_reason"] == "length"
    assert details["requested_max_tokens"] is None
    assert [
        details[k]
        for k in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
        )
    ] == [120, 80, 200, 20]
    assert "DO-NOT-LOG" not in output


@pytest.mark.parametrize(
    "raw", [None, {"choices": [], "usage": None}, {"choices": {}, "model": object()}]
)
def test_missing_metadata_does_not_mask_error(capsys, raw):
    llm = provider("bad", metadata=raw)
    llm.max_tokens = 64
    with pytest.raises(LLMResponseError):
        query(llm, max_attempts=1)
    details = json.loads(capsys.readouterr().out.split("metadata: ", 1)[1])
    assert details["finish_reason"] is None
    assert details["completion_tokens"] is None
    assert details["requested_max_tokens"] == 64


SINGLES = [
    LLMJailbreaker,
    LLMMultiLanguageJailbreaker,
    LLMPoetryJailbreaker,
    RAGPoisoner,
]
ATTACKS = [*SINGLES, Crescendo, EchoChamber, goat.GOAT, PromptDecompositionAttack]


def run_generator(kind, llm):
    attack = kind()
    if kind is LLMMultiLanguageJailbreaker:
        return attack._generate_multilingual_jailbreak_attack(llm, "Write a poem.", [])
    if kind is RAGPoisoner:
        return attack._generate_rag_attack(llm, "Write a poem.", [])
    if kind in SINGLES:
        return attack._generate_jailbreak_attack(llm, "Write a poem.", [])
    conversation = StandardisedConversation({"objective": "Write a poem."})
    if kind is Crescendo:
        return attack._generate_question(
            llm, "Write a poem.", conversation, conversation.get_root_id(), 5, 5
        )
    if kind is EchoChamber:
        attack.stage_attempts = 3
        return attack._query_attacker_llm(
            llm, [HumanMessage("Return JSON.")], ["objective"]
        )
    if kind is goat.GOAT:
        return attack._generate_question(
            llm, "Write a poem.", conversation, conversation.get_root_id(), 5
        )
    return attack._query_variations(llm, [HumanMessage("Return JSONL variations.")])


def valid_reply(kind):
    if kind in SINGLES:
        return '{"attack_prompt":"A question"}'
    if kind is Crescendo:
        return '{"generated_question":"A question"}'
    if kind is EchoChamber:
        return '{"objective":"A question"}'
    if kind is goat.GOAT:
        return '{"observation":"", "thought":"", "strategy":"", "next_question":"A question"}'
    return '{"variation":"A question"}'


@pytest.mark.parametrize("kind", ATTACKS)
@pytest.mark.parametrize("bad_first", [False, True])
def test_every_attack_uses_shared_repair_and_valid_replies_are_quiet(
    capsys, monkeypatch, kind, bad_first
):
    monkeypatch.setattr(goat, "APPLICATION_GUARDRAILS", None)
    good = valid_reply(kind)
    llm = provider(*([good[:-1], good] if bad_first else [good]))
    assert run_generator(kind, llm)
    assert llm.invoke.call_count == (2 if bad_first else 1)
    output = capsys.readouterr().out
    assert output == ""


@pytest.mark.parametrize("kind", SINGLES)
def test_single_turn_schema_failure_is_repaired(kind):
    llm = provider('{"attack_prompt":12}', valid_reply(kind))
    assert run_generator(kind, llm) == "A question"
    assert (
        "'attack_prompt' must be a nonempty string"
        in llm.invoke.call_args.args[0][-1].content
    )


def test_crescendo_has_three_calls_total_and_never_sends_invalid_prompt_to_target(
    capsys,
):
    attack = Crescendo()
    conversation = StandardisedConversation({"objective": "Write a poem."})
    llm = provider("bad", "bad", "bad")
    target, judge = Mock(), Mock()
    with pytest.raises(LLMResponseError):
        attack._run_attempt(
            {"content": "Write a poem."},
            target,
            judge,
            llm,
            5,
            5,
            None,
            None,
            "session",
            conversation,
            conversation.get_root_id(),
        )
    assert llm.invoke.call_count == 3
    target.process_input.assert_not_called()
    judge.assert_not_called()


def test_echo_schema_retry_uses_stage_budget_and_final_refusal_fallback():
    attack = EchoChamber()
    attack.stage_attempts = 2
    attack.debug = False
    attack.evaluate_refusal = Mock(return_value=(True, "refused"))
    llm = provider("{}", "{}")
    with pytest.raises(RefusalError):
        attack._query_attacker_llm(
            llm, [HumanMessage("Return objective.")], ["objective"]
        )
    assert llm.invoke.call_count == 2
    assert (
        "Missing required JSON fields: objective"
        in llm.invoke.call_args.args[0][-1].content
    )
    assert attack.evaluate_refusal.call_args.args[-1] == "{}"


def test_echo_refusal_format_failure_remains_bounded():
    attack = EchoChamber()
    attack.stage_attempts = 2
    llm = provider("bad", "bad")
    assert (
        attack._query_attacker_llm(
            llm, [HumanMessage("Check refusal.")], ["answered"], refusal_query=True
        )
        == {}
    )
    assert llm.invoke.call_count == 2


def test_goat_refusal_repair():
    llm = provider('{"answered":false', '{"answered":false,"explanation":"Declined"}')
    assert goat.GOAT().evaluate_refusal(llm, "Question", "Response") == (
        True,
        "Declined",
    )
    assert llm.invoke.call_count == 2


def test_goat_generation_keeps_existing_budget(monkeypatch):
    monkeypatch.setattr(goat, "LLM_AGENT_RETRY", 2)
    monkeypatch.setattr(goat, "APPLICATION_GUARDRAILS", None)
    attack = goat.GOAT()
    attack.evaluate_refusal = Mock(return_value=(False, ""))
    conversation = StandardisedConversation({"objective": "Write a poem."})
    llm = provider("bad", "bad")
    with pytest.raises(RuntimeError, match="required fields"):
        attack._generate_question(
            llm, "Write a poem.", conversation, conversation.get_root_id(), 5
        )
    assert llm.invoke.call_count == 2
    assert attack.evaluate_refusal.call_args.args[-1] == "bad"


def test_jsonl_fences_and_blank_lines():
    assert parse_jsonl_variations(
        '```jsonl\n{"variation":"one"}\n\n{"variation":"two"}\n```'
    ) == ["one", "two"]


def test_jsonl_failed_repair_keeps_valid_original_lines():
    llm = provider(
        '{"variation":"one"}\n{"variation":"two"}\n{"variation":"cut off',
        "invalid again",
    )
    assert PromptDecompositionAttack._query_variations(
        llm, [HumanMessage("Return JSONL.")]
    ) == ["one", "two"]
    repair = llm.invoke.call_args.args[0][-1].content
    assert "Line 3:" in repair and "JSONL" in repair


def test_jsonl_top_up_also_repairs(monkeypatch):
    from spikee.attacks import prompt_decomposition

    llm = provider('{"variation":"one"}', '{"variation":"two"', '{"variation":"two"}')
    monkeypatch.setattr(prompt_decomposition, "get_llm", lambda *a, **k: llm)
    result = PromptDecompositionAttack()._generate_variants_llm(
        "Write a poem.", "test/model", 2
    )
    assert result == ["one", "two"]
    assert llm.invoke.call_count == 3


def test_jsonl_total_failure_keeps_original_prompt_fallback(monkeypatch):
    from spikee.attacks import prompt_decomposition

    llm = provider("bad", "bad", "bad", "bad")
    monkeypatch.setattr(prompt_decomposition, "get_llm", lambda *a, **k: llm)
    assert PromptDecompositionAttack()._generate_variants_llm(
        "Write a poem.", "test/model", 2
    ) == ["Write a poem."]
    assert llm.invoke.call_count == 4


@pytest.mark.parametrize(
    "prefix",
    [
        '<think>Draft: {"text":"discard this"}</think>',
        'Draft: {"text":"discard this"}</think>',
    ],
)
def test_attack_parser_uses_final_object_after_reasoning(prefix):
    assert parse_json_object(prefix + '{"text":"final"}') == {"text": "final"}


@pytest.mark.parametrize("fenced", [False, True])
def test_reasoning_marker_inside_attack_text_is_preserved(fenced):
    obj = {"text": 'Explain </think> and {"example": true}.'}
    raw = json.dumps(obj)
    if fenced:
        raw = "```json\n" + raw + "\n```"
    assert parse_json_object(raw) == obj


def test_jsonl_uses_final_variations_after_reasoning():
    raw = (
        '<think>{"variation":"discard this"}</think>'
        '```jsonl\n{"variation":"first"}\n{"variation":"second"}\n```'
    )
    assert parse_jsonl_variations(raw) == ["first", "second"]


@pytest.mark.parametrize(
    "raw",
    [
        '<think>{"text":"draft"}',
        '<think>{"text":"draft"}</think>',
        '<think>Done.</think>{"text":"unfinished',
    ],
)
def test_reasoning_does_not_make_incomplete_attack_json_valid(raw):
    with pytest.raises(ValueError):
        parse_json_object(raw)


@pytest.mark.parametrize("kind", SINGLES)
def test_single_turn_generation_has_no_hardcoded_token_cap(kind, monkeypatch):
    # A long generated input must reach the target intact, using provider limits.
    prompt = "A harmless line of a poem.\n" * 120
    llm = provider(json.dumps({"attack_prompt": prompt}))
    get_llm = Mock(return_value=llm)
    monkeypatch.setattr(f"{kind.__module__}.get_llm", get_llm)
    target = SimpleNamespace(process_input=Mock(return_value="A harmless reply."))
    judge = Mock(return_value=True)
    entry = {"id": 1, "content": "Write a poem."}
    result = kind().attack(
        entry, target, judge, 1, attack_option="model=custom/test-model"
    )
    get_llm.assert_called_once_with("custom/test-model", max_tokens=None)
    target.process_input.assert_called_once_with(prompt.strip(), None)
    judge.assert_called_once_with(entry, "A harmless reply.")
    assert result[1] is True


@pytest.mark.parametrize("kind", SINGLES)
def test_literal_newline_in_attack_prompt_does_not_need_repair(kind, capsys):
    raw = '{"attack_prompt":"First line.\nSecond line.","rationale":"A poem."}'
    llm = provider(raw)
    assert run_generator(kind, llm) == "First line.\nSecond line."
    llm.invoke.assert_called_once()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "wrapper", ["{}", "```json\n{}\n```", "Here is the object:\n{}"]
)
@pytest.mark.parametrize("control", ["\n", "\t", "\r", "\x00"])
def test_control_characters_inside_json_strings_are_preserved(wrapper, control):
    value = f"First{control}second"
    raw = '{"attack_prompt":"' + value + '"}'
    obj = parse_json_object(wrapper.format(raw), string_keys=("attack_prompt",))
    assert obj["attack_prompt"] == value
    # Saved output remains standard JSON with escaped control characters.
    assert json.loads(json.dumps(obj)) == obj


@pytest.mark.parametrize(
    "raw",
    [
        '{"attack_prompt":"First\nsecond}',
        '{"attack_prompt":"First\nsecond"',
        '{"attack_prompt":"First\nsecond" "rationale":"missing comma"}',
    ],
)
def test_control_character_tolerance_does_not_hide_incomplete_or_invalid_json(raw):
    llm = provider(raw, '{"attack_prompt":"Corrected"}')
    assert query(llm) == {"attack_prompt": "Corrected"}
    assert llm.invoke.call_count == 2
    assert llm.invoke.call_args.args[0][-2].content == raw
