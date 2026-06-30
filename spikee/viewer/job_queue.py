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
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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


def _find_spikee_exe() -> str:
    """
    Locate the spikee console-script executable installed in the same environment
    as the running Python interpreter. Falls back to 'spikee' on PATH.
    """
    # Prefer the sibling Scripts/bin directory of the current interpreter
    scripts_dir = Path(sys.executable).parent
    for candidate in ("spikee", "spikee.exe"):
        full = scripts_dir / candidate
        if full.is_file():
            return str(full)
    # Fall back to PATH lookup
    found = shutil.which("spikee")
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate the 'spikee' executable. "
        "Ensure spikee is installed in the active environment."
    )


def spawn_job(job: Job) -> None:
    """
    Start the spikee subprocess for the given job.
    Streams stdout/stderr into job.log in a background daemon thread.
    Sets job.status to "success" or "failed" when the process exits.
    """
    try:
        spikee_exe = _find_spikee_exe()
        proc = subprocess.Popen(
            [spikee_exe, "--quiet"] + job.args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,          # binary — we decode manually to handle \r correctly
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
        buf = ""
        try:
            while True:
                raw = proc.stdout.read(256)
                if not raw:
                    break
                chunk = raw.decode("utf-8", errors="replace")
                buf += chunk
                # Process all complete "lines" handling \r\n, \n, and bare \r.
                # Binary mode preserves \r so progress-bar overwrite works correctly.
                while True:
                    rn = buf.find("\r\n")
                    r  = buf.find("\r")
                    n  = buf.find("\n")

                    candidates = [
                        (pos, kind)
                        for pos, kind in [(rn, "rn"), (r, "r"), (n, "n")]
                        if pos != -1
                    ]
                    if not candidates:
                        break  # wait for more data

                    pos, kind = min(candidates, key=lambda x: x[0])
                    line = buf[:pos]

                    with job.lock:
                        if kind == "r" and job.log:
                            # bare \r → overwrite last entry (tqdm progress bar)
                            job.log[-1] = line
                        else:
                            job.log.append(line)

                    buf = buf[pos + (2 if kind == "rn" else 1):]

            # flush any remaining buffered text
            if buf:
                with job.lock:
                    job.log.append(buf)
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
