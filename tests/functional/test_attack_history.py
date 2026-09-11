"""Optional history is diagnostic data inside the existing attack result."""

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from typing import ClassVar

import pytest

from spikee import tester
from spikee.attacks.multi_turn import MultiTurnAttack
from spikee.templates.attack import Attack
from spikee.utilities.attack import attack_history_enabled, invoke_attack
from spikee.utilities.files import read_jsonl_file, write_jsonl_file
from spikee.utilities.hinting import Image
from spikee.utilities.results import ResultProcessor


class Bar:
    def __init__(self, total):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n

    def refresh(self):
        pass


class Target:
    config: ClassVar[dict] = {
        "single-turn": True,
        "multi-turn": True,
        "backtrack": True,
    }

    def __init__(self, success_at=None, error_at=None):
        self.calls = []
        self.success_at = success_at
        self.error_at = error_at

    def process_input(self, input_text, system_message=None, *_args, **kwargs):
        self.calls.append((deepcopy(input_text), kwargs))
        if len(self.calls) == self.error_at:
            raise RuntimeError("target failed")
        return ("success" if len(self.calls) == self.success_at else "refused"), {}


@pytest.fixture
def entry():
    return {
        "id": 42,
        "long_id": "dataset-entry",
        "content": "objective",
        "content_type": "text",
        "judge_name": "test",
        "judge_args": {},
        "judge_options": "",
    }


class RecordingAttack:
    def __init__(self, collect=True):
        self.collect = collect

    def attack(
        self, entry, target, judge, iterations, bar=None, lock=None, attack_options=None
    ):
        history = [] if self.collect and attack_history_enabled() else None
        response, payload, success = "", "", False
        for count in range(1, iterations + 1):
            payload = f"candidate-{count}"
            error = None
            try:
                response, _ = target.process_input(payload)
                success = judge(entry, response)
            except RuntimeError as exc:
                response, success, error = "", False, str(exc)
            if history is not None:
                history.append(
                    {
                        "input": payload,
                        "response": response,
                        "success": success,
                        "error": error,
                    }
                )
            if bar:
                bar.update(1)
            if success:
                break
        else:
            count = iterations
        return (
            count,
            success,
            Attack.standardised_input_return(payload, attempt_history=history),
            response,
        )


class LegacyAttack:
    def attack(self, entry, target, judge, iterations, bar=None, lock=None):
        count, success, payload, response = RecordingAttack(False).attack(
            entry, target, judge, iterations, bar, lock
        )
        return count, success, payload["input"], response


def run_attack(
    monkeypatch,
    entry,
    attack=None,
    *,
    trace=True,
    attempts=1,
    iterations=5,
    success_at=None,
    error_at=None,
    attack_only=True,
):
    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(trace))
    monkeypatch.setattr(
        tester, "call_judge", lambda entry, response: response == "success"
    )
    target = Target(success_at, error_at)
    bar = Bar(attempts * (iterations + int(not attack_only)))
    rows = tester.process_entry(
        deepcopy(entry),
        target,
        attempts=attempts,
        attack_module=attack or RecordingAttack(),
        attack_name="mock",
        attack_iterations=iterations,
        attack_options="mode=test",
        attack_only=attack_only,
        attempts_bar=bar,
        global_lock=threading.Lock(),
    )
    return rows, target, bar


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("true", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("no", False),
        (" OFF ", False),
    ],
)
def test_history_environment(monkeypatch, value, expected):
    monkeypatch.delenv("SPIKEE_ATTACK_HISTORY", raising=False)
    if value is not None:
        monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", value)
    assert attack_history_enabled() is expected


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("success_at,expected", [(None, 15), (1, 1), (7, 7)])
def test_nested_history_counts_and_early_stop(
    monkeypatch, entry, trace, success_at, expected
):
    rows, target, bar = run_attack(
        monkeypatch, entry, trace=trace, attempts=3, success_at=success_at
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "42-attack" and row["long_id"] == "dataset-entry-mock"
    assert row["attempts"] == len(target.calls) == bar.n == bar.total == expected
    assert row["success"] == (success_at is not None)
    assert "attack_attempt" not in row and "entry_complete" not in row
    if trace:
        assert len(row["attempt_history"]) == expected
        assert [r["invocation"] for r in row["attempt_history"]] == [
            i // 5 + 1 for i in range(expected)
        ]
        assert row["attempt_history"][-1]["response"] == row["response"]
    else:
        assert "attempt_history" not in row
    p = ResultProcessor(rows, "nested")
    assert p.total_entries == 1 and p.total_attempts == expected


@pytest.mark.parametrize("trace", [False, True])
def test_legacy_modules_remain_callable(monkeypatch, entry, trace):
    rows, target, bar = run_attack(
        monkeypatch, entry, LegacyAttack(), trace=trace, attempts=2
    )
    assert len(rows) == 1 and rows[0]["id"] == "42-attack"
    assert rows[0]["attempts"] == len(target.calls) == bar.total == 10
    assert "attempt_history" not in rows[0]


def test_history_does_not_override_top_level_outcome(monkeypatch, entry):
    class DiagnosticOnly:
        def attack(self, *args):
            return (
                20,
                False,
                {
                    "input": "final",
                    "attempt_history": [
                        {"input": "earlier", "response": "reply", "success": True},
                    ],
                },
                "final response",
            )

    rows, _, _ = run_attack(monkeypatch, entry, DiagnosticOnly())
    p = ResultProcessor(rows, "nested")
    assert rows[0]["success"] is False and rows[0]["attempts"] == 20
    assert p.successful_groups == 0 and p.total_attempts == 20


def test_history_keeps_handled_errors(monkeypatch, entry):
    rows, target, _ = run_attack(monkeypatch, entry, error_at=2, attempts=2)
    assert len(rows) == 1 and rows[0]["attempts"] == len(target.calls) == 10
    assert rows[0]["attempt_history"][1]["error"] == "target failed"


def test_history_survives_later_invocation_error(monkeypatch, entry):
    class Failing(RecordingAttack):
        invocations = 0

        def attack(self, *args):
            self.invocations += 1
            if self.invocations == 2:
                raise RuntimeError("invocation failed")
            return super().attack(*args)

    rows, target, _ = run_attack(monkeypatch, entry, Failing(), attempts=3)
    assert (
        len(target.calls) == len(rows[0]["attempt_history"]) == rows[0]["attempts"] == 5
    )
    assert rows[0]["error"] == "invocation failed"


def test_mixed_optional_history_across_invocations(monkeypatch, entry):
    class Mixed:
        invocations = 0

        def attack(self, *args):
            self.invocations += 1
            payload = {"input": f"input-{self.invocations}"}
            if self.invocations == 2:
                payload["attempt_history"] = [
                    {"input": "middle", "response": "reply", "success": False}
                ]
            return 3, False, payload, "reply"

    rows, _, _ = run_attack(monkeypatch, entry, Mixed(), attempts=3)
    assert rows[0]["attempts"] == 9 and rows[0]["input"] == "input-3"
    assert len(rows[0]["attempt_history"]) == 1
    assert rows[0]["attempt_history"][0]["invocation"] == 2


def test_nested_multimodal_history_serialization(monkeypatch, entry):
    class Images:
        def attack(self, *args):
            return (
                1,
                True,
                {
                    "input": Image("aGVsbG8="),
                    "attempt_history": [
                        {
                            "input": Image("aGVsbG8="),
                            "response": "reply",
                            "success": True,
                        },
                    ],
                },
                "reply",
            )

    rows, _, _ = run_attack(monkeypatch, entry, Images())
    assert rows[0]["input_type"] == "image"
    item = rows[0]["attempt_history"][0]
    assert item["input_type"] == "image" and item["input"] == "aGVsbG8="
    json.dumps(rows)


@pytest.mark.parametrize(
    "history",
    [
        "wrong",
        [1],
        [{"input": "a"}],
        [{"input": "a", "response": "b", "success": "yes"}],
    ],
)
def test_invalid_history_does_not_change_attack_result(
    monkeypatch, entry, history, capsys
):
    class Invalid:
        calls = 0

        def attack(self, *args):
            self.calls += 1
            return 5, True, {"input": "a", "attempt_history": history}, "b"

    attack = Invalid()
    rows, _, bar = run_attack(monkeypatch, entry, attack, attempts=3)
    assert attack.calls == 1
    assert rows[0]["error"] is None and rows[0]["success"] is True
    assert rows[0]["input"] == "a" and rows[0]["response"] == "b"
    assert rows[0]["attempts"] == bar.n == bar.total == 5
    assert "attempt_history" not in rows[0]
    assert "Ignoring invalid attempt_history" in capsys.readouterr().out


def test_history_snapshot_does_not_mutate_attack_owned_data(monkeypatch, entry):
    class Reused:
        def __init__(self):
            self.record = {"input": ["first"], "response": "reply", "success": False}
            self.payload = {"input": "final", "attempt_history": [self.record]}
            self.invocation = 0

        def attack(self, *args):
            self.invocation += 1
            self.record["input"][0] = str(self.invocation)
            return 1, False, self.payload, "reply"

    attack = Reused()
    rows, _, _ = run_attack(monkeypatch, entry, attack, attempts=2)
    records = rows[0]["attempt_history"]
    assert [r["input"] for r in records] == [["1"], ["2"]]
    assert [r["invocation"] for r in records] == [1, 2]
    assert "invocation" not in attack.record
    assert attack.payload["attempt_history"] == [attack.record]


def test_standard_failures_and_nested_attack_count_once(monkeypatch, entry):
    rows, target, _ = run_attack(monkeypatch, entry, attempts=2, attack_only=False)
    assert [r["id"] for r in rows] == [42, "42-attack"]
    assert [r["attempts"] for r in rows] == [2, 10]
    assert len(rows[1]["attempt_history"]) == 10
    p = ResultProcessor(rows, "combined")
    assert p.total_entries == 1 and p.total_attempts == len(target.calls) == 12


def test_options_legacy_signatures(entry):
    def plural(e, t, j, n, b, l, attack_options=None):
        return attack_options

    def singular(e, t, j, n, b, l, attack_option=None):
        return attack_option

    def legacy(e, t, j, n, b, l):
        return "legacy"

    for function, expected in (
        (plural, "chosen"),
        (singular, "chosen"),
        (legacy, "legacy"),
    ):
        assert (
            invoke_attack(function, entry, None, None, 1, None, None, "chosen")
            == expected
        )


def test_history_state_is_local_to_invocation(monkeypatch, entry):
    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", "true")
    attack = RecordingAttack()

    def run(i):
        return attack.attack(
            dict(entry, id=i), Target(success_at=i), lambda e, r: r == "success", 5
        )[2]["attempt_history"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        histories = list(pool.map(run, [1, 2, 3]))
    assert [len(h) for h in histories] == [1, 2, 3]
    assert len({id(item) for h in histories for item in h}) == 6


def test_nested_resume(monkeypatch, entry, tmp_path):
    rows, _, _ = run_attack(monkeypatch, entry, attempts=2)
    path = tmp_path / "results.jsonl"
    write_jsonl_file(path, rows)
    ids, _, count, entries = tester._load_results_file(path, RecordingAttack(), 5)
    assert {str(i) for i in ids} == {"42"} and count == 10 and entries == 1


@pytest.mark.parametrize("complete", [False, True])
def test_resume_older_expanded_history(entry, tmp_path, complete):
    rows = [
        dict(
            entry,
            id=f"42-attack-{i}",
            attack_name="mock",
            attempts=1,
            attack_parent_id=42,
            attack_parent_long_id=entry["long_id"],
            attack_attempt=i,
            attack_result_format="attempt",
            entry_complete=i == 2,
        )
        for i in (1, 2)
    ]
    path = tmp_path / "results.jsonl"
    write_jsonl_file(path, rows if complete else rows[:1])
    ids, retained, count, entries = tester._load_results_file(
        path, RecordingAttack(), 99
    )
    assert {str(i) for i in ids} == ({"42"} if complete else set())
    assert retained == (rows if complete else [])
    assert count == (2 if complete else 0) and entries == (1 if complete else 0)


def test_manual_multiturn_keeps_one_conversation(monkeypatch, entry):
    entry["content"] = ["one", "two", "three"]
    for value in ("true", "false"):
        monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", value)
        result = MultiTurnAttack().attack(entry, Target(), lambda e, r: False, 2)
        assert result[0] == 2 and len(result[2]["conversation"]) == 4
        assert "attempt_history" not in result[2]


def test_standard_success_skips_dynamic_attack(monkeypatch, entry):
    target = tester.AdvancedTargetWrapper(
        SimpleNamespace(process_input=lambda input_text, system_message=None: "success")
    )
    monkeypatch.setattr(tester, "call_judge", lambda entry, response: True)
    bar = Bar(12)
    rows = tester.process_entry(
        deepcopy(entry),
        target,
        attempts=2,
        attack_module=RecordingAttack(),
        attack_name="mock",
        attack_iterations=5,
        attempts_bar=bar,
        global_lock=threading.Lock(),
    )
    assert len(rows) == 1 and rows[0]["id"] == 42
    assert bar.n == bar.total == 1
    assert "attempt_history" not in rows[0]


def test_resume_legacy_attack_only(entry, tmp_path):
    path = tmp_path / "results.jsonl"
    write_jsonl_file(path, [dict(entry, id="42-attack", attack_name="old", attempts=3)])
    ids, _, calls, entries = tester._load_results_file(path, LegacyAttack(), 20)
    assert {str(i) for i in ids} == {"42"}
    assert calls == 3 and entries == 1


@pytest.mark.parametrize("trace", [False, True])
def test_backtrack_preserves_abandoned_attempt(monkeypatch, entry, trace, tmp_path):
    from flask import Flask

    from spikee.attacks import crescendo

    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(trace))
    from spikee.results import extract_results
    from spikee.viewer.blueprints.results import _process_standardised_conversation

    monkeypatch.setattr(crescendo, "get_llm", lambda *a, **kw: object())
    attack = crescendo.Crescendo()
    prompts = iter(["abandoned", "replacement"])
    monkeypatch.setattr(attack, "_generate_question", lambda *a: next(prompts))
    monkeypatch.setattr(attack, "_is_refusal", lambda *a: True)
    target = Target(success_at=2)
    returned = invoke_attack(
        attack.attack,
        entry,
        target,
        lambda e, r: r == "success",
        2,
        None,
        None,
        "model=mock",
    )
    count, success, payload, response = returned
    assert (count, success, response) == (2, True, "success")
    assert [c[1]["backtrack"] for c in target.calls] == [False, True]
    graph = json.loads(payload["conversation"])
    assert len(graph) == 5
    assert len(graph["0"]["children"]) == 2
    assert [graph[str(i)]["data"]["content"] for i in (1, 2, 3, 4)] == [
        "abandoned",
        "refused",
        "replacement",
        "success",
    ]
    # The existing row format keeps the entire graph through serialization,
    # result analysis, extraction, and rendering of abandoned branches.
    row = tester._attack_result(
        entry,
        count,
        success,
        payload,
        response,
        "crescendo",
        "model=mock",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / "conversation.jsonl"
    write_jsonl_file(path, [row])
    saved = read_jsonl_file(path)
    processor = ResultProcessor(saved, str(path))
    assert processor.total_entries == processor.successful_groups == 1
    assert processor.total_attempts == 2
    extract_results(
        SimpleNamespace(
            result_file=[str(path)],
            result_folder=None,
            query="success = true",
            tag="conversation",
        )
    )
    extracted = read_jsonl_file(next((tmp_path / "results").glob("extract*.jsonl")))
    assert len(extracted) == 1
    assert json.loads(extracted[0]["conversation"]) == graph
    app = Flask(__name__)
    app.jinja_env.globals["truncate_length"] = 400
    with app.app_context():
        rendered = _process_standardised_conversation(extracted[0]["conversation"])
    assert all(
        text in rendered for text in ("abandoned", "refused", "replacement", "success")
    )


@pytest.mark.parametrize("trace", [False, True])
def test_generator_error_preserves_conversation(monkeypatch, entry, trace):
    from spikee.attacks import crescendo

    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(trace))

    monkeypatch.setattr(crescendo, "get_llm", lambda *a, **kw: object())
    attack = crescendo.Crescendo()
    prompts = iter(["first"])
    monkeypatch.setattr(attack, "_generate_question", lambda *a: next(prompts))
    monkeypatch.setattr(attack, "_is_refusal", lambda *a: False)
    target = Target()
    count, success, payload, _response = invoke_attack(
        attack.attack,
        entry,
        target,
        lambda e, r: False,
        3,
        None,
        None,
        "model=mock",
    )
    assert count == 1 and success is False
    graph = json.loads(payload["conversation"])
    assert len(graph) == 3
    assert graph["1"]["data"]["content"] == "first"
    assert graph["2"]["data"]["content"] == "refused"
    assert count == len(target.calls) == 1


def test_repeated_guardrail_categories_count_dataset_groups(entry):
    rows = [
        dict(
            entry,
            id=f"42-attack-{i}",
            attack_name="mock",
            attempts=1,
            success=False,
            guardrail=True,
            guardrail_categories={"policy": True},
            error="blocked",
        )
        for i in range(1, 4)
    ]
    p = ResultProcessor(rows, "history")
    assert p.total_entries == 1 and p.total_attempts == 3
    assert p.guardrail_groups == 1
    assert p.guardrail_categories == {"policy": 1}
    assert p.attack_types["mock"]["guardrail"] == 1


def test_cli_rejudge_keeps_history_ids_and_parentage(run_spikee, workspace_dir, entry):
    rows = [
        dict(
            entry,
            id=f"42-attack-{i}",
            input=f"candidate-{i}",
            response="reply",
            attack_name="mock",
            attack_parent_id=42,
            attack_parent_long_id=entry["long_id"],
            attack_attempt=i,
            attempts=1,
            success=None,
            judge_name="test_judge",
        )
        for i in range(1, 4)
    ]
    path = workspace_dir / "results" / "results_history.jsonl"
    path.parent.mkdir(exist_ok=True)
    write_jsonl_file(path, rows)
    run_spikee(
        [
            "results",
            "rejudge",
            "--result-file",
            str(path),
            "--judge-options",
            "test_judge:mode=success",
        ],
        cwd=workspace_dir,
    )
    output = next((workspace_dir / "results").glob("rejudge*.jsonl"))
    rejudged = read_jsonl_file(output)
    assert [r["id"] for r in rejudged] == [r["id"] for r in rows]
    assert all(r["attack_parent_id"] == 42 and r["attempts"] == 1 for r in rejudged)
    p = ResultProcessor(rejudged, str(output))
    assert p.successful_groups == p.total_entries == 1
    assert p.total_attempts == 3


@pytest.mark.parametrize(
    "module_name,class_name",
    [("crescendo", "Crescendo"), ("echo_chamber", "EchoChamber"), ("goat", "GOAT")],
)
@pytest.mark.parametrize(
    "scenario", ["failure", "success", "refusal", "target_error", "seed_error"]
)
@pytest.mark.parametrize("invocations", [1, 3])
def test_conversation_attacks_ignore_candidate_history_environment(
    monkeypatch, entry, module_name, class_name, scenario, invocations
):
    import importlib
    import random

    prefix = (
        "spikee.data.workspace.attacks" if module_name == "goat" else "spikee.attacks"
    )
    module = importlib.import_module(f"{prefix}.{module_name}")
    monkeypatch.setattr(module, "get_llm", lambda *a, **kw: object())
    monkeypatch.setattr(module.uuid, "uuid4", lambda: "test-session")
    results = []
    for trace in (False, True):
        random.seed(12)
        monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(trace))
        attack = getattr(module, class_name)()
        assert "return_all_attempts" not in inspect.signature(attack.attack).parameters
        target = Target(
            success_at=3 if scenario == "success" else None,
            error_at=2
            if scenario == "target_error"
            else 1
            if scenario == "seed_error"
            else None,
        )
        target.get_target = lambda target=target: target
        if module_name == "crescendo":
            monkeypatch.setattr(
                attack,
                "_generate_question",
                lambda *a, target=target: f"turn-{len(target.calls) + 1}",
            )
            monkeypatch.setattr(attack, "_is_refusal", lambda *a: scenario == "refusal")
        elif module_name == "goat":
            monkeypatch.setattr(
                attack,
                "_generate_question",
                lambda *a, target=target: {
                    "observation": "",
                    "thought": "",
                    "strategy": "",
                    "next_question": f"turn-{len(target.calls) + 1}",
                },
            )
            monkeypatch.setattr(
                attack,
                "evaluate_refusal",
                lambda *a: (scenario == "refusal", "refusal reason"),
            )
        else:
            original_call = target.process_input

            def echo_response(*args, original_call=original_call, **kwargs):
                response = original_call(*args, **kwargs)
                return (
                    ('{"sentences": ["one", "two", "three"]}', {})
                    if module.TARGET_SEED_SENTENCES_PROMPT in args[0]
                    else response
                )

            target.process_input = echo_response
            monkeypatch.setattr(
                attack, "get_targeted_objective", lambda *a: "objective"
            )
            monkeypatch.setattr(
                attack, "get_keywords", lambda *a, **kw: ["one", "two", "three"]
            )
            monkeypatch.setattr(
                attack, "evaluate_seed_alignment", lambda *a: (False, "")
            )
            monkeypatch.setattr(attack, "choose_seed_sentence", lambda *a: (1, ""))
            monkeypatch.setattr(
                attack,
                "get_next_question",
                lambda *a, target=target: f"turn-{len(target.calls) + 1}",
            )
            monkeypatch.setattr(
                attack,
                "evaluate_success",
                lambda llm, obj, response: (response == "success", "High", ""),
            )
            monkeypatch.setattr(
                attack,
                "evaluate_refusal",
                lambda *a: (scenario == "refusal", "refusal reason"),
            )
        judge_calls = []

        def judge(e, response, judge_calls=judge_calls):
            judge_calls.append(response)
            return response == "success"

        monkeypatch.setattr(tester, "call_judge", judge)
        bar = Bar(invocations * 5)
        rows = tester.process_entry(
            dict(entry, text=entry["content"]),
            target,
            attempts=invocations,
            attack_name=module_name,
            attack_module=attack,
            attack_iterations=5,
            attack_options="model=mock",
            attack_only=True,
            attempts_bar=bar,
            global_lock=threading.Lock(),
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == "42-attack"
        assert "attack_attempt" not in row
        assert "attempt_history" not in row
        if "conversation" in row:
            graph = json.loads(row["conversation"])
            assert len(graph) > 1
            if scenario == "refusal":
                assert any(len(node["children"]) > 1 for node in graph.values())
                assert any(
                    node["data"].get("content") == "refused" for node in graph.values()
                )
        elif not (module_name == "goat" and scenario in ("target_error", "seed_error")):
            pytest.fail("Attack discarded its conversation graph")
        if module_name == "goat" and scenario == "target_error":
            assert len(target.calls) == 2  # An error must still stop outer invocations.
        calls = [
            (prompt, kwargs.get("backtrack", False)) for prompt, kwargs in target.calls
        ]
        results.append(
            (
                {
                    key: row.get(key)
                    for key in (
                        "input",
                        "response",
                        "conversation",
                        "objective",
                        "success",
                        "error",
                        "attempts",
                    )
                },
                calls,
                judge_calls,
                bar.n,
                bar.total,
            )
        )
    assert results[0] == results[1]


SINGLE_ATTACKS = [
    ("anti_spotlighting", "AntiSpotlightingAttack"),
    ("best_of_n", "BestOfNAttack"),
    ("random_suffix_search", "RandomSuffixSearch"),
    ("prompt_decomposition", "PromptDecompositionAttack"),
    ("llm_jailbreaker", "LLMJailbreaker"),
    ("llm_multi_language_jailbreaker", "LLMMultiLanguageJailbreaker"),
    ("llm_poetry_jailbreaker", "LLMPoetryJailbreaker"),
    ("rag_poisoner", "RAGPoisoner"),
    ("sample_attack", "SampleAttack"),
]


@pytest.mark.parametrize("module_name,class_name", SINGLE_ATTACKS)
@pytest.mark.parametrize("success_at,error_at", [(None, None), (3, None), (None, 2)])
def test_single_turn_history_preserves_execution(
    monkeypatch, entry, module_name, class_name, success_at, error_at
):
    import importlib
    import random

    import numpy as np

    prefix = (
        "spikee.data.workspace.attacks"
        if module_name == "sample_attack"
        else "spikee.attacks"
    )
    module = importlib.import_module(f"{prefix}.{module_name}")
    monkeypatch.setattr(module, "get_llm", lambda *a, **kw: object(), raising=False)
    if module_name == "random_suffix_search":
        monkeypatch.setattr(
            module.tiktoken,
            "get_encoding",
            lambda name: SimpleNamespace(
                n_vocab=100, decode=lambda tokens: str(tokens)
            ),
        )
    results = []
    for trace in (False, True):
        random.seed(12)
        monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(trace))
        np.random.seed(12)
        attack = getattr(module, class_name)()
        target = Target(success_at=success_at, error_at=error_at)
        for method in (
            "_generate_jailbreak_attack",
            "_generate_multilingual_jailbreak_attack",
            "_generate_rag_attack",
        ):
            if hasattr(attack, method):
                monkeypatch.setattr(
                    attack,
                    method,
                    lambda *a, target=target: f"candidate-{len(target.calls) + 1}",
                )
        judge_calls = []

        def judge(e, response, judge_calls=judge_calls):
            judge_calls.append(response)
            return response == "success"

        data = dict(entry, text=entry["content"])
        returned = attack.attack(
            data,
            target,
            judge,
            5,
            attack_option="model=mock",
        )
        results.append((returned, target.calls, judge_calls))
    summary, detailed = results[0][0], results[1][0]
    assert isinstance(summary, tuple) and isinstance(detailed, tuple)
    assert len(summary) == len(detailed) == 4
    assert "attempt_history" not in summary[2]
    history = detailed[2]["attempt_history"]
    assert summary[:2] == detailed[:2]
    assert summary[2]["input"] == detailed[2]["input"]
    assert summary[3] == detailed[3]
    assert results[0][1:] == results[1][1:]
    assert len(history) == summary[0]
    assert history[-1]["input"] == detailed[2]["input"]
    assert history[-1]["response"] == detailed[3]
    if error_at:
        assert history[error_at - 1]["error"] == "target failed"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "attack_name", ["best_of_n", "mock_attack", "mock_attack_legacy"]
)
def test_cli_nested_history_and_legacy_modules(
    run_spikee, workspace_dir, monkeypatch, enabled, attack_name
):
    from .utils import spikee_generate_cli, spikee_test_cli

    monkeypatch.setenv("SPIKEE_ATTACK_HISTORY", str(enabled))
    dataset = spikee_generate_cli(run_spikee, workspace_dir)
    entries = read_jsonl_file(dataset)[:2]
    write_jsonl_file(dataset, entries)
    paths, _ = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="always_refuse",
        datasets=[dataset],
        additional_args=[
            "--attack",
            attack_name,
            "--attack-only",
            "--attack-iterations",
            "3",
            "--attempts",
            "2",
            "--threads",
            "2",
            "--no-auto-resume",
        ],
    )
    rows = read_jsonl_file(paths[0])
    assert len(rows) == len(entries) == 2
    assert all(r["attempts"] == 6 for r in rows)
    if enabled and attack_name == "best_of_n":
        assert all(len(r["attempt_history"]) == 6 for r in rows)
        assert [h["invocation"] for h in rows[0]["attempt_history"]] == [
            1,
            1,
            1,
            2,
            2,
            2,
        ]
    else:
        assert all("attempt_history" not in r for r in rows)
    resumed = run_spikee(
        [
            "test",
            "--target",
            "always_refuse",
            "--dataset",
            str(dataset),
            "--attack",
            attack_name,
            "--attack-only",
            "--resume-file",
            str(paths[0]),
            "--no-auto-resume",
        ],
        cwd=workspace_dir,
    )
    assert "All entries have already been processed" in resumed.stdout


def test_nested_history_extract_and_rejudge(run_spikee, workspace_dir, entry):
    history = [
        {"input": "first", "response": "refused", "success": False},
        {"input": "last", "response": "reply", "success": True},
    ]
    row = dict(
        entry,
        id="42-attack",
        input="last",
        response="reply",
        attack_name="mock",
        attempts=2,
        success=True,
        judge_name="test_judge",
        attempt_history=history,
    )
    path = workspace_dir / "results" / "results_nested.jsonl"
    path.parent.mkdir(exist_ok=True)
    write_jsonl_file(path, [row])
    run_spikee(
        ["results", "extract", "--result-file", str(path), "--query", "success = true"],
        cwd=workspace_dir,
    )
    extracted = read_jsonl_file(
        next((workspace_dir / "results").glob("extract*.jsonl"))
    )
    assert len(extracted) == 1 and extracted[0]["attempt_history"] == history
    assert extracted[0]["attempts"] == 2
    run_spikee(
        [
            "results",
            "rejudge",
            "--result-file",
            str(path),
            "--judge-options",
            "test_judge:mode=failure",
        ],
        cwd=workspace_dir,
    )
    rejudged = read_jsonl_file(next((workspace_dir / "results").glob("rejudge*.jsonl")))
    assert rejudged[0]["attempt_history"] == history and rejudged[0]["id"] == row["id"]
