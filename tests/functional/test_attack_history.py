"""History is an alternate representation of the same attack execution."""

import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest
from werkzeug.datastructures import MultiDict

from spikee import tester
from spikee.attacks.multi_turn import MultiTurnAttack
from spikee.utilities.attack import invoke_attack
from spikee.utilities.files import read_jsonl_file, write_jsonl_file
from spikee.utilities.hinting import AttackAttempt, Image
from spikee.utilities.results import ResultProcessor, extract_entries
from spikee.viewer.blueprints._forms import TestForm as RunForm


class Bar:
    def __init__(self, total):
        self.total = total
        self.n = 0

    def update(self, n=1):
        self.n += n

    def refresh(self):
        pass


class Target:
    config = {"single-turn": True, "multi-turn": True, "backtrack": True}

    def __init__(self, success_at=None, error_at=None):
        self.calls = []
        self.success_at = success_at
        self.error_at = error_at

    def process_input(self, input_text, system_message=None, **kwargs):
        self.calls.append((deepcopy(input_text), kwargs))
        if len(self.calls) == self.error_at:
            raise RuntimeError("target failed")
        return ("success" if len(self.calls) == self.success_at else "refused"), {}


class RecordingAttack:
    def attack(
        self,
        entry,
        target_module,
        call_judge,
        max_iterations,
        attempts_bar=None,
        bar_lock=None,
        attack_options=None,
        return_all_attempts=False,
    ):
        history = []
        for i in range(1, max_iterations + 1):
            payload = f"candidate-{i}"
            try:
                response, _ = target_module.process_input(payload)
                success = call_judge(entry, response)
            except Exception as exc:
                if return_all_attempts:
                    history.append(AttackAttempt(payload, "", False, error=str(exc)))
                    return history
                raise
            if return_all_attempts:
                history.append(AttackAttempt(payload, response, success))
            if attempts_bar:
                with bar_lock:
                    attempts_bar.update(1)
            if success:
                if attempts_bar:
                    attempts_bar.total -= max_iterations - i
                if return_all_attempts:
                    return history
                return i, True, payload, response
        if return_all_attempts:
            return history
        return max_iterations, False, payload, response


class LegacyAttack:
    def attack(
        self,
        entry,
        target_module,
        call_judge,
        max_iterations,
        attempts_bar=None,
        bar_lock=None,
    ):
        return RecordingAttack().attack(
            entry, target_module, call_judge, max_iterations, attempts_bar, bar_lock
        )


@pytest.fixture
def entry():
    return dict(
        id=42,
        long_id="dataset-entry",
        content="objective",
        content_type="text",
        judge_name="test",
        judge_args={},
        judge_options="",
    )


def run_attack(
    monkeypatch,
    entry,
    attack=None,
    *,
    trace=False,
    attempts=1,
    iterations=5,
    success_at=None,
    error_at=None,
    attack_only=True,
):
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
        attack_return_all_attempts=trace,
    )
    return rows, target, bar


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("success_at,expected", [(None, 15), (1, 1), (7, 7)])
def test_attempt_counts_and_early_stop(monkeypatch, entry, trace, success_at, expected):
    rows, target, bar = run_attack(
        monkeypatch, entry, trace=trace, attempts=3, success_at=success_at
    )
    assert len(target.calls) == expected
    assert sum(r["attempts"] for r in rows) == expected
    assert bar.n == bar.total == expected
    assert len(rows) == (expected if trace else 1)
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["long_id"] for r in rows}) == len(rows)
    assert rows[-1]["success"] == (success_at is not None)
    if trace:
        assert [r["attack_attempt"] for r in rows] == list(range(1, expected + 1))
        assert all(r["attempts"] == 1 for r in rows)
        assert rows[-1]["entry_complete"]
        assert not any(r["entry_complete"] for r in rows[:-1])
    else:
        assert rows[0]["id"] == "42-attack"
        assert "entry_complete" not in rows[0]


@pytest.mark.parametrize("trace", [False, True])
def test_legacy_without_parameter_still_runs(monkeypatch, entry, trace):
    rows, target, bar = run_attack(
        monkeypatch, entry, LegacyAttack(), trace=trace, attempts=2
    )
    assert len(rows) == 1
    assert rows[0]["id"] == "42-attack"
    assert rows[0]["attempts"] == len(target.calls) == 10
    assert bar.n == bar.total == 10


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
        attack_return_all_attempts=True,
    )
    assert len(rows) == 1 and rows[0]["id"] == 42
    assert bar.n == bar.total == 1
    assert rows[0]["entry_complete"]


def test_intermediate_results_survive_error(monkeypatch, entry):
    rows, target, bar = run_attack(monkeypatch, entry, trace=True, error_at=3)
    assert [r["input"] for r in rows] == ["candidate-1", "candidate-2", "candidate-3"]
    assert rows[0]["response"] == "refused"
    assert rows[-1]["error"] == "target failed"
    assert sum(r["attempts"] for r in rows) == len(target.calls) == 3
    assert bar.total == bar.n == 3


def test_summary_and_history_statistics_match(monkeypatch, entry):
    summary, _, _ = run_attack(monkeypatch, entry, success_at=4)
    history, _, _ = run_attack(monkeypatch, entry, trace=True, success_at=4)
    base = dict(entry, attack_name="None", attempts=1, success=False)
    processors = [
        ResultProcessor([base, *rows], "example") for rows in (summary, history)
    ]
    for p in processors:
        p.generate_output(combined=True)
        assert p.total_entries == p.successful_groups == 1
        assert p.total_attempts == 5
        assert p.attack_types["mock"] == dict(
            total=1, successes=1, attempts=4, guardrail=0
        )
    assert processors[0]._breakdowns == processors[1]._breakdowns
    assert processors[0].generate_overview() == processors[1].generate_overview()


@pytest.mark.parametrize("complete", [False, True])
def test_resume_complete_and_partial_attack_only(
    monkeypatch, entry, tmp_path, complete
):
    rows, _, _ = run_attack(monkeypatch, entry, trace=True)
    path = tmp_path / "results.jsonl"
    write_jsonl_file(path, rows if complete else rows[:-1])
    ids, retained, calls, entries = tester._load_results_file(
        path, RecordingAttack(), 99
    )
    assert calls == (5 if complete else 0)
    assert entries == (1 if complete else 0)
    assert retained == (rows if complete else [])
    assert {str(i) for i in ids} == ({"42"} if complete else set())


def test_resume_legacy_attack_only(entry, tmp_path):
    path = tmp_path / "results.jsonl"
    write_jsonl_file(path, [dict(entry, id="42-attack", attack_name="old", attempts=3)])
    ids, _, calls, entries = tester._load_results_file(path, LegacyAttack(), 20)
    assert {str(i) for i in ids} == {"42"}
    assert calls == 3 and entries == 1


def test_multiturn_snapshots_and_unjudged_turns(entry):
    entry["content"] = ["one", "two", "three"]
    attack = MultiTurnAttack()
    target = Target(success_at=3)
    history = attack.attack(
        entry, target, lambda e, r: r == "success", 3, return_all_attempts=True
    )
    assert len(history) == 3
    assert [r.success for r in history] == [None, None, True]
    assert [len(r.input["conversation"]) for r in history] == [2, 4, 6]
    assert [r.input["input"] for r in history] == entry["content"]
    assert not extract_entries({"success": None}, "failure")
    legacy = attack.attack(entry, Target(success_at=3), lambda e, r: r == "success", 3)
    assert legacy[:2] == (3, True)
    limited = attack.attack(entry, Target(), lambda e, r: False, 2)
    assert limited[0] == 2


def test_backtrack_preserves_abandoned_attempt(monkeypatch, entry):
    from spikee.attacks import crescendo

    monkeypatch.setattr(crescendo, "get_llm", lambda *a, **kw: object())
    attack = crescendo.Crescendo()
    prompts = iter(["abandoned", "replacement"])
    monkeypatch.setattr(attack, "_generate_question", lambda *a: next(prompts))
    monkeypatch.setattr(attack, "_is_refusal", lambda *a: True)
    target = Target(success_at=2)
    records = attack.attack(
        entry,
        target,
        lambda e, r: r == "success",
        2,
        attack_option="model=mock",
        return_all_attempts=True,
    )
    assert [r.input["input"] for r in records] == ["abandoned", "replacement"]
    assert [c[1]["backtrack"] for c in target.calls] == [False, True]
    first = json.loads(records[0].input["conversation"])
    last = json.loads(records[1].input["conversation"])
    assert len(first) == 3 and len(last) == 5
    assert sum(r.attempts for r in records) == 2


def test_options_signatures_and_explicit_history(entry):
    def plural(
        entry,
        target,
        judge,
        iters,
        bar,
        lock,
        attack_options=None,
        *,
        return_all_attempts=False,
    ):
        assert attack_options == "chosen"
        return (
            [AttackAttempt("a", "b", False)]
            if return_all_attempts
            else (1, False, "a", "b")
        )

    assert isinstance(
        invoke_attack(plural, entry, None, None, 1, None, None, "chosen", True), list
    )

    def singular(entry, target, judge, iters, bar, lock, attack_option=None):
        return attack_option

    assert (
        invoke_attack(singular, entry, None, None, 1, None, None, "chosen", True)
        == "chosen"
    )
    assert (
        "return_all_attempts" in inspect.signature(RecordingAttack().attack).parameters
    )


def test_trace_state_is_local_to_invocation(entry):
    attack = RecordingAttack()

    def run(i):
        return attack.attack(
            dict(entry, id=i),
            Target(success_at=i),
            lambda e, r: r == "success",
            5,
            return_all_attempts=True,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, [1, 2, 3]))
    assert [len(r) for r in results] == [1, 2, 3]
    assert len({id(r) for history in results for r in history}) == 6


def test_viewer_form_flag():
    form = MultiDict(dict(target="mock", datasets="example.jsonl", attack="best_of_n"))
    assert "--attack-return-all-attempts" not in RunForm.from_form(form).to_cli_args()
    form["attack_return_all_attempts"] = "on"
    assert "--attack-return-all-attempts" in RunForm.from_form(form).to_cli_args()


def test_mixed_results_extract_compare_and_viewer(monkeypatch, entry, tmp_path):
    from spikee.results import dataset_comparison, extract_results
    from spikee.viewer.app import create_app
    from spikee.viewer.blueprints import results as viewer_results
    from spikee.viewer.blueprints import _cache

    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    history, _, _ = run_attack(monkeypatch, entry, trace=True, success_at=3)
    legacy = dict(
        history[-1],
        id="99-attack",
        long_id="legacy-mock",
        attempts=20,
        success=False,
        attack_name="mock",
    )
    for key in [
        k
        for k in legacy
        if k.startswith("attack_") and k not in ("attack_name", "attack_options")
    ]:
        del legacy[key]
    rows = [*history, legacy]
    path = tmp_path / "results" / "results_history.jsonl"
    write_jsonl_file(path, rows)
    dataset = tmp_path / "dataset.jsonl"
    write_jsonl_file(dataset, [entry])
    # Comparison sees dynamic success even when no standard row exists.
    dataset_comparison(
        SimpleNamespace(
            dataset=str(dataset),
            result_file=[str(path)],
            result_folder=None,
            skip_validation=False,
            success_definition="gt",
            success_threshold=0.5,
            number=0,
            tag="history",
        )
    )
    # Extraction renumbers rows but retains their common dataset identity.
    extract_results(
        SimpleNamespace(
            result_file=[str(path)],
            result_folder=None,
            category="custom",
            custom_search="attack_name:mock",
            tag="history",
        )
    )
    extracted_path = next((tmp_path / "results").glob("extract*.jsonl"))
    extracted = read_jsonl_file(extracted_path)
    processor = ResultProcessor(extracted, str(extracted_path))
    assert processor.total_entries == 2
    assert processor.total_attempts == 23
    assert processor.successful_groups == 1
    assert len({r["id"] for r in extracted}) == 4
    # Full application route coverage, with no background module or network work.
    monkeypatch.setattr(_cache, "warm_cache", lambda: None)
    app = create_app(db_path=str(tmp_path / "jobs.sqlite"))
    app.config["TESTING"] = True
    monkeypatch.setattr(viewer_results, "loaded_files", {"history": path})
    client = app.test_client()
    response = client.get("/results/entries?result_file=history&per_page=2")
    assert response.status_code == 200
    assert b"42-attack-1" in response.data and b"42-attack-2" in response.data
    assert b"42-attack-3" not in response.data
    response = client.get("/results/entry/history-42-attack-2?result_file=history")
    assert response.status_code == 200 and b"Attack Attempt" in response.data
    response = client.post(
        "/results/entry/history-42-attack-2/toggle", data={"result_file": "history"}
    )
    assert response.status_code in (302, 303)
    updated = read_jsonl_file(path)
    assert updated[1]["success"]
    assert updated[0]["success"] == rows[0]["success"]
    assert updated[2]["success"] == rows[2]["success"]
    monkeypatch.setattr(viewer_results, "call_judge", lambda e, r: False)
    response = client.post(
        "/results/entry/history-42-attack-3/rejudge", data={"result_file": "history"}
    )
    assert response.status_code in (302, 303)
    assert read_jsonl_file(path)[2]["success"] is False


def test_explicit_multimodal_records_and_malformed_return(monkeypatch, entry):
    class Explicit:
        def attack(self, *args, return_all_attempts=False):
            return [AttackAttempt(Image("aGVsbG8="), "response", False)]

    rows, _, _ = run_attack(monkeypatch, entry, Explicit(), trace=True)
    assert rows[0]["input_type"] == "image"
    assert rows[0]["input"] == "aGVsbG8="

    class Malformed:
        def attack(self, *args, return_all_attempts=False):
            return [AttackAttempt("input", "output", False, attempts=-1)]

    rows, _, _ = run_attack(monkeypatch, entry, Malformed(), trace=True)
    assert "non-negative integer" in rows[0]["error"]
    assert rows[0]["attempts"] == 0


def test_generator_error_preserves_prior_attempts(monkeypatch, entry):
    from spikee.attacks import crescendo

    monkeypatch.setattr(crescendo, "get_llm", lambda *a, **kw: object())
    attack = crescendo.Crescendo()
    prompts = iter(["first"])
    monkeypatch.setattr(attack, "_generate_question", lambda *a: next(prompts))
    monkeypatch.setattr(attack, "_is_refusal", lambda *a: False)
    target = Target()
    records = attack.attack(
        entry,
        target,
        lambda e, r: False,
        3,
        attack_option="model=mock",
        return_all_attempts=True,
    )
    assert len(records) == 2
    assert records[0].input["input"] == "first"
    assert records[-1].attempts == 0
    assert records[-1].error is not None
    assert sum(r.attempts for r in records) == len(target.calls) == 1


def test_combined_files_do_not_merge_same_parent(monkeypatch, entry):
    history, _, _ = run_attack(monkeypatch, entry, trace=True, success_at=2)
    first = [dict(r, source_file="one") for r in history]
    second = [dict(r, source_file="two", success=False) for r in history]
    p = ResultProcessor(first + second, "combined")
    assert p.total_entries == 2 and p.total_attempts == 4
    assert p.successful_groups == 1
    assert p.attack_types["mock"]["total"] == 2


@pytest.mark.parametrize(
    "attack_name,expanded",
    [("best_of_n", True), ("mock_attack", False), ("mock_attack_legacy", False)],
)
def test_cli_trace_and_legacy_fallback(
    run_spikee, workspace_dir, attack_name, expanded
):
    from .utils import spikee_generate_cli, spikee_test_cli

    dataset = spikee_generate_cli(run_spikee, workspace_dir)
    # Keep the CLI regression small, while exercising concurrent entry writes.
    entries = read_jsonl_file(dataset)[:2]
    write_jsonl_file(dataset, entries)
    paths, result = spikee_test_cli(
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
            "--attack-return-all-attempts",
            "--threads",
            "2",
            "--no-auto-resume",
        ],
    )
    rows = read_jsonl_file(paths[0])
    assert len(rows) == len(entries) * (6 if expanded else 1)
    assert sum(r["attempts"] for r in rows) == len(entries) * 6
    assert len({r["id"] for r in rows}) == len(rows)
    p = ResultProcessor(rows, str(paths[0]))
    assert p.total_entries == len(entries)
    assert next(iter(p.attack_types.values()))["total"] == len(entries)
    if not expanded:
        assert "retaining its representative result" in result.stdout
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
            "--attack-return-all-attempts",
            "--resume-file",
            str(paths[0]),
            "--no-auto-resume",
        ],
        cwd=workspace_dir,
    )
    assert "All entries have already been processed" in resumed.stdout


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
def test_bundled_single_turn_formats_preserve_execution(
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
                    attack, method, lambda *a: f"candidate-{len(target.calls) + 1}"
                )
        judge_calls = []

        def judge(e, response):
            judge_calls.append(response)
            return response == "success"

        data = dict(entry, text=entry["content"])
        returned = attack.attack(
            data,
            target,
            judge,
            5,
            attack_option="model=mock",
            return_all_attempts=trace,
        )
        results.append((returned, target.calls, judge_calls))
    summary, history = results[0][0], results[1][0]
    assert isinstance(summary, tuple) and len(summary) == 4
    assert isinstance(history, list) and all(
        isinstance(r, AttackAttempt) for r in history
    )
    assert summary[0] == sum(r.attempts for r in history)
    assert summary[1] == any(r.success for r in history)
    assert results[0][1:] == results[1][1:]  # exact target calls and judge calls
    assert history[-1].input == summary[2]
    assert history[-1].response == summary[3]
    if error_at:
        assert history[error_at - 1].error == "target failed"


@pytest.mark.parametrize(
    "module_name,class_name",
    [("crescendo", "Crescendo"), ("echo_chamber", "EchoChamber"), ("goat", "GOAT")],
)
@pytest.mark.parametrize("success_at", [None, 3])
def test_bundled_adaptive_multiturn_formats_preserve_execution(
    monkeypatch, entry, module_name, class_name, success_at
):
    import importlib
    import random

    prefix = (
        "spikee.data.workspace.attacks" if module_name == "goat" else "spikee.attacks"
    )
    module = importlib.import_module(f"{prefix}.{module_name}")
    monkeypatch.setattr(module, "get_llm", lambda *a, **kw: object())
    results = []
    for trace in (False, True):
        random.seed(12)
        attack = getattr(module, class_name)()
        target = Target(success_at=success_at)
        target.get_target = lambda: target
        if module_name == "crescendo":
            monkeypatch.setattr(
                attack, "_generate_question", lambda *a: f"turn-{len(target.calls) + 1}"
            )
            monkeypatch.setattr(attack, "_is_refusal", lambda *a: False)
        elif module_name == "goat":
            monkeypatch.setattr(
                attack,
                "_generate_question",
                lambda *a: dict(
                    observation="",
                    thought="",
                    strategy="",
                    next_question=f"turn-{len(target.calls) + 1}",
                ),
            )
            monkeypatch.setattr(attack, "evaluate_refusal", lambda *a: (False, ""))
        else:
            original_call = target.process_input

            def echo_response(*args, **kwargs):
                response = original_call(*args, **kwargs)
                return (
                    ('{"sentences": ["one", "two", "three"]}', {})
                    if len(target.calls) == 1
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
                attack, "get_next_question", lambda *a: f"turn-{len(target.calls) + 1}"
            )
            monkeypatch.setattr(
                attack,
                "evaluate_success",
                lambda llm, obj, response: (response == "success", "High", ""),
            )
            monkeypatch.setattr(attack, "evaluate_refusal", lambda *a: (False, ""))
        judge_calls = []

        def judge(e, response):
            judge_calls.append(response)
            return response == "success"

        returned = attack.attack(
            dict(entry, text=entry["content"]),
            target,
            judge,
            5,
            attack_option="model=mock",
            return_all_attempts=trace,
        )
        # Session IDs differ between runs; compare actual prompts and backtracking.
        calls = [
            (prompt, kwargs.get("backtrack", False)) for prompt, kwargs in target.calls
        ]
        results.append((returned, calls, judge_calls))
    summary, history = results[0][0], results[1][0]
    assert isinstance(summary, tuple)
    assert isinstance(history, list) and all(
        isinstance(r, AttackAttempt) for r in history
    )
    assert results[0][1:] == results[1][1:]
    assert summary[0] == sum(r.attempts for r in history) == len(results[0][1])
    assert summary[1] == any(r.success for r in history)
    assert [
        r.input["input"] if isinstance(r.input, dict) else r.input for r in history
    ] == [c[0] for c in results[1][1]]
    if module_name != "echo_chamber":
        assert len(json.loads(history[0].input["conversation"])) < len(
            json.loads(history[-1].input["conversation"])
        )


def test_handled_errors_do_not_skip_repeated_invocations(monkeypatch, entry):
    from spikee.attacks.best_of_n import BestOfNAttack

    results = []
    for trace in (False, True):
        rows, target, bar = run_attack(
            monkeypatch,
            entry,
            BestOfNAttack(),
            trace=trace,
            iterations=3,
            attempts=2,
            error_at=3,
        )
        assert len(target.calls) == bar.n == bar.total == 6
        results.append(sum(r["attempts"] for r in rows))
    assert results == [6, 6]


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


def test_mixed_returns_across_invocations_preserve_all_counts(monkeypatch, entry):
    class MixedAttack:
        def __init__(self):
            self.invocation = 0

        def attack(self, *args, return_all_attempts=False):
            self.invocation += 1
            if self.invocation == 2:
                return [AttackAttempt("middle", "reply", False)]
            return 3, False, f"representative-{self.invocation}", "reply"

    rows, _, bar = run_attack(monkeypatch, entry, MixedAttack(), trace=True, attempts=3)
    assert [r["attempts"] for r in rows] == [3, 1, 3]
    assert [r["attack_result_format"] for r in rows] == [
        "representative",
        "attempt",
        "representative",
    ]
    assert bar.n == bar.total == 7
    assert ResultProcessor(rows, "mixed").total_attempts == 7
