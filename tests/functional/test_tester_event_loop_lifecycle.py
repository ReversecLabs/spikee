"""Worker loop ownership and shutdown, without external model calls."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from spikee import tester

from .utils import spikee_generate_cli, spikee_test_cli


@pytest.fixture
def lifecycle(monkeypatch):
    state = SimpleNamespace(loops=[], closed_by={}, pools=[], release=threading.Event())
    new_event_loop = asyncio.new_event_loop

    def tracked_loop():
        loop = new_event_loop()
        state.loops.append((loop, threading.get_ident()))
        close = loop.close

        def tracked_close():
            state.closed_by[loop] = threading.get_ident()
            close()

        loop.close = tracked_close
        return loop

    class TrackedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state.pools.append(self)

    monkeypatch.setattr(asyncio, "new_event_loop", tracked_loop)
    monkeypatch.setattr(asyncio.events, "new_event_loop", tracked_loop)
    monkeypatch.setattr(tester, "ThreadPoolExecutor", TrackedExecutor)
    monkeypatch.setattr(tester, "tqdm", lambda **kwargs: Mock())
    yield state
    state.release.set()
    for pool in state.pools:
        pool.shutdown(wait=True, cancel_futures=True)
    # Keep a failing regression from leaking its own resources.
    for loop, _ in state.loops:
        if not loop.is_closed():
            loop.close()


def run_entries(tmp_path, workers=4, count=26):
    output = tmp_path / "results.jsonl"
    tester._run_threaded(
        entries=[{"id": i} for i in range(count)],
        target_module=None,
        attempts=1,
        attack_name="",
        attack_module=None,
        attack_iters=0,
        attack_options=None,
        attack_only=False,
        num_threads=workers,
        total_attempts=count,
        initial_attempts=0,
        output_file=output,
        total_dataset_size=count,
        initial_processed=0,
        initial_success=0,
        initial_guardrail=0,
    )
    return output


def assert_closed_by_owner(state):
    assert state.loops
    assert all(loop.is_closed() for loop, _ in state.loops)
    assert all(state.closed_by[loop] == owner for loop, owner in state.loops)


def test_cli_exits_without_event_loop_shutdown_errors(
    run_spikee, workspace_dir, monkeypatch
):
    monkeypatch.setenv("PYTHONWARNINGS", "default::ResourceWarning")
    dataset = spikee_generate_cli(run_spikee, workspace_dir)
    _, result = spikee_test_cli(
        run_spikee,
        workspace_dir,
        target="always_success",
        datasets=[dataset],
        additional_args=["--threads", "4", "--no-auto-resume"],
    )
    assert "unclosed event loop" not in result.stderr
    assert "Exception ignored" not in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("error", [None, ValueError, ImportError])
def test_entry_loops_close_on_success_and_errors(
    monkeypatch, tmp_path, lifecycle, workers, error
):
    barrier = threading.Barrier(workers)

    def process(entry, *args):
        loop = asyncio.get_event_loop()
        assert not loop.is_closed()
        if entry["id"] < workers:
            barrier.wait(timeout=30)
        if error:
            raise error("fixture failure")
        return {"id": entry["id"], "success": True}

    monkeypatch.setattr(tester, "process_entry", process)
    if error is ImportError:
        with pytest.raises(SystemExit, match="1"):
            run_entries(tmp_path, workers)
    else:
        output = run_entries(tmp_path, workers)
        if error is None:
            assert len(output.read_text().splitlines()) == 26
    for pool in lifecycle.pools:
        pool.shutdown(wait=True)
    assert len({owner for _, owner in lifecycle.loops}) == workers
    assert_closed_by_owner(lifecycle)


def test_interrupt_leaves_running_entry_open_until_it_finishes(
    monkeypatch, tmp_path, lifecycle
):
    started = threading.Event()
    processed = []

    def process(entry, *args):
        loop = asyncio.get_event_loop()
        started.set()
        assert lifecycle.release.wait(timeout=30)
        assert not loop.is_closed()
        processed.append(entry["id"])
        return {"success": True}

    def interrupt(futures):
        assert started.wait(timeout=30)
        raise KeyboardInterrupt

    monkeypatch.setattr(tester, "process_entry", process)
    monkeypatch.setattr(tester, "as_completed", interrupt)
    run_entries(tmp_path, workers=1)
    assert len(lifecycle.loops) == 1
    assert not lifecycle.loops[0][0].is_closed()
    lifecycle.release.set()
    lifecycle.pools[0].shutdown(wait=True)
    assert processed == [0]  # Queued entries were cancelled.
    assert_closed_by_owner(lifecycle)


def test_loop_shutdown_finishes_async_resources(monkeypatch, tmp_path, lifecycle):
    finalized = []
    resources = []
    executor_threads = []

    async def pending():
        try:
            await asyncio.Future()
        finally:
            finalized.append("task")

    async def generator():
        try:
            yield 1
        finally:
            finalized.append("generator")

    async def start():
        resources.append(asyncio.create_task(pending()))
        stream = generator()
        resources.append(stream)
        await anext(stream)
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: executor_threads.append(threading.current_thread())
        )

    def process(entry, *args):
        asyncio.get_event_loop().run_until_complete(start())
        return {"success": True}

    monkeypatch.setattr(tester, "process_entry", process)
    try:
        run_entries(tmp_path, workers=1, count=1)
        assert_closed_by_owner(lifecycle)
        assert sorted(finalized) == ["generator", "task"]
        assert all(not thread.is_alive() for thread in executor_threads)
    finally:
        # Also clean up if run against the leaking implementation.
        for loop, _ in lifecycle.loops:
            if not loop.is_closed():
                tasks = asyncio.all_tasks(loop)
                for task in tasks:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
