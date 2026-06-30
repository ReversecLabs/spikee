# spikee/viewer/job_queue.py
"""
Centralised in-memory job queue for the Spikee viewer.

All blueprints that create jobs (Generate, Test) import the module-level
`job_queue` singleton. The Jobs blueprint reads from it to list and stream jobs.
"""

from __future__ import annotations

import os
import sys
import html
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Job:
    id: str
    type: str                   # "generate" | "test"
    name: str                   # human-readable label
    status: str                 # "running" | "success" | "failed"
    created_at: datetime
    args: list                  # CLI args list (after "spikee --quiet")
    log: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    returncode: Optional[int] = None
    process: Optional[object] = None  # subprocess.Popen


class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, type: str, name: str, args: list) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            type=type,
            name=name,
            status="running",
            created_at=datetime.now(),
            args=args,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        """Return all jobs, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def has_running(self) -> bool:
        with self._lock:
            return any(j.status == "running" for j in self._jobs.values())


def spawn_job(job: Job) -> None:
    """
    Start the spikee subprocess for the given job.
    Streams stdout/stderr into job.log in a background daemon thread.
    Sets job.status to "success" or "failed" when the process exits.
    """
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "spikee", "--quiet"] + job.args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.getcwd(),
        )
        job.process = proc
    except Exception as e:
        with job.lock:
            job.log.append(f"[Error] Failed to start subprocess: {e}")
        job.status = "failed"
        job.returncode = -1
        return

    def _reader():
        try:
            for line in proc.stdout:
                with job.lock:
                    job.log.append(line.rstrip())
        except Exception as e:
            with job.lock:
                job.log.append(f"[Error] Log reader error: {e}")
        finally:
            proc.wait()
            job.status = "success" if proc.returncode == 0 else "failed"
            job.returncode = proc.returncode

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def sse_stream(job: Job):
    """
    Generator that yields SSE-formatted log lines for the given job.
    Sends `event: done` when the job exits.
    Used by the Jobs blueprint's /stream endpoint.
    """
    last = 0
    while True:
        with job.lock:
            lines = list(job.log[last:])

        for line in lines:
            yield f"data: {html.escape(line)}\n\n"

        last += len(lines)

        if job.status != "running" and last >= len(job.log):
            yield "event: done\ndata:\n\n"
            return

        time.sleep(0.2)


# Module-level singleton — imported by all blueprints
job_queue = JobQueue()
