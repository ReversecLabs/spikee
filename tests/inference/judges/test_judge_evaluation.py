"""Opt-in semantic evaluations. Target responses are fixed; only judges use APIs."""

import hashlib
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pytest

from spikee.judge import call_judge
from spikee.templates.provider import Provider
from spikee.utilities.llm_message import format_messages
from tests.judge_cases import ROOT, WORKSPACE, load_cases, materialize

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_JUDGE_EVALS") != "1",
    reason="Paid judge evaluations require RUN_JUDGE_EVALS=1",
)

MODELS = [
    "openrouter/deepseek/deepseek-v4-flash-0731",
    "openrouter/qwen/qwen3.8-flash",
    "openrouter/qwen/qwen3.8-27b",
    "openrouter/qwen/qwen3.6-35b-a3b",
    "openrouter/z-ai/glm-5.3-flash",
]


def test_judge_matrix(request, monkeypatch):
    config = request.config
    models = config.getoption("--judge-model") or MODELS
    if any(model.startswith("openrouter/") for model in models) and not os.environ.get(
        "OPENROUTER_API_KEY"
    ):
        pytest.fail(
            "OPENROUTER_API_KEY missing; set SPIKEE_TEST_ENV_FILE to your .env file"
        )
    assert all(not model.startswith("offline") for model in models), (
        "Offline models cannot evaluate semantics"
    )
    repeats, workers = (
        config.getoption("--judge-repeats"),
        config.getoption("--judge-workers"),
    )
    assert repeats > 0 and workers > 0
    report = Path(config.getoption("--judge-report")).expanduser().resolve()
    selected = []
    for case in load_cases():
        if config.getoption("--judge-case") not in case["id"]:
            continue
        if config.getoption("--judge-split") not in ("all", case["split"]):
            continue
        entry = materialize(case)
        if entry["judge_name"].startswith("llm_"):
            selected.append((case, entry))
    assert selected, "No LLM judge cases matched"
    # Match production workspace loading, but isolate from the developer's stale
    # local judge/provider copies. All selected entries are already materialized.
    monkeypatch.chdir(WORKSPACE)
    monkeypatch.setenv("SPIKEE_API_TIMEOUT", os.environ.get("SPIKEE_API_TIMEOUT", "90"))
    sources = list((WORKSPACE / "judges").glob("*.py")) + [
        ROOT / "spikee/templates/llm_judge.py"
    ]
    versions = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    local = threading.local()
    invoke = Provider.invoke

    def observed_invoke(provider, messages, *args, **kwargs):
        call = {"messages": format_messages(messages)}
        local.calls.append(call)
        started = time.monotonic()
        try:
            reply = invoke(provider, messages, *args, **kwargs)
            call["raw_reply"] = reply.content
            original = reply.original_response
            # Save only non-secret response metadata, never request headers/keys.
            if original is not None:
                metadata = (
                    original.model_dump() if hasattr(original, "model_dump") else {}
                )
                for key in ("id", "model", "usage", "provider"):
                    if key in metadata:
                        call[key] = metadata[key]
                call["finish_reason"] = metadata.get("choices", [{}])[0].get(
                    "finish_reason"
                )
            return reply
        finally:
            call["latency_seconds"] = round(time.monotonic() - started, 3)

    monkeypatch.setattr(Provider, "invoke", observed_invoke)

    def evaluate(case, entry, model, repetition):
        local.calls = []
        row = {
            **case,
            "entry": {**entry, "judge_options": model},
            "model": model,
            "repetition": repetition,
            "phase": config.getoption("--judge-phase"),
            "versions": versions,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        started = time.monotonic()
        try:
            row["actual"] = call_judge(row["entry"], case["response"])
            if type(row["actual"]) is not bool:
                row["outcome"] = "invalid_verdict"
            elif row["actual"] == case["expected"]:
                row["outcome"] = "pass"
            else:
                row["outcome"] = "false_positive" if row["actual"] else "false_negative"
        except Exception as exc:  # noqa: BLE001 - record every failed evaluation
            # Errors remain failures, distinct from valid negative verdicts.
            row["outcome"] = "error"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
        row["calls"] = local.calls
        row["latency_seconds"] = round(time.monotonic() - started, 3)
        return row

    counts = Counter()
    failures = []
    # Exclusive creation prevents accidentally destroying a previous baseline.
    with report.open("x") as output, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(evaluate, case, entry, model, repetition)
            for case, entry in selected
            for repetition in range(1, repeats + 1)
            for model in models
        ]
        for future in as_completed(futures):
            row = future.result()
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            counts[(row["model"], row["outcome"])] += 1
            if row["outcome"] != "pass":
                failures.append(
                    f"{row['model']} {row['id']} #{row['repetition']}: {row['outcome']}"
                )
    print(f"\nJudge report: {report}")
    for model in models:
        print(
            model,
            {
                outcome: count
                for (name, outcome), count in counts.items()
                if name == model
            },
        )
    assert not failures, "\n".join(failures)
