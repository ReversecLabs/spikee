# spikee/viewer/job_queue.py
"""
Centralised in-memory job queue for the Spikee viewer.

All blueprints that create jobs (Generate, Test) import the module-level
`job_queue` singleton. The Jobs blueprint reads from it to list and stream jobs.

When a database path is provided via init_job_queue(db_path=...), jobs are
persisted to a SQLite database (stdlib sqlite3, no extra dependencies).
"""

from __future__ import annotations

import html
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    args        TEXT NOT NULL,
    log         TEXT NOT NULL,
    returncode  INTEGER
);
"""


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
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._db_path: Optional[str] = db_path

        if db_path is not None:
            # Ensure parent directories exist
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_from_db()

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection (safe to call from any thread)."""
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        """Create the jobs table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def _load_from_db(self) -> None:
        """Restore all jobs from the database into memory on startup."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at ASC"
            ).fetchall()

        for row in rows:
            log_lines = row["log"].split("\n") if row["log"] else []
            job = Job(
                id=row["id"],
                type=row["type"],
                name=row["name"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                args=json.loads(row["args"]),
                log=log_lines,
                returncode=row["returncode"],
            )

            # Any job that was still "running" when the server last shut down
            # can never be re-attached — mark it failed.
            if job.status == "running":
                job.status = "failed"
                job.log.append("[Server restarted — job interrupted]")
                self._db_update(job)

            self._jobs[job.id] = job

    def _db_insert(self, job: Job) -> None:
        """Insert a new job row (called when a job is created)."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (id, type, name, status, created_at, args, log, returncode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id,
                    job.type,
                    job.name,
                    job.status,
                    job.created_at.isoformat(),
                    json.dumps(job.args),
                    "",
                    None,
                ),
            )
            conn.commit()

    def _db_update(self, job: Job) -> None:
        """Update status, log and returncode for an existing job row."""
        with job.lock:
            status = job.status
            log_text = "\n".join(job.log)
            returncode = job.returncode
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, log=?, returncode=? WHERE id=?",
                (status, log_text, returncode, job.id),
            )
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def create(self, type: str, name: str, args: list) -> Job:
        """Create a new job, register it in memory (and DB if enabled), and return it."""
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
        if self._db_path is not None:
            self._db_insert(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Return the job with the given ID, or None if not found."""
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        """Return all jobs, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def has_running(self) -> bool:
        """Return True if any job is currently in the 'running' state."""
        with self._lock:
            return any(j.status == "running" for j in self._jobs.values())


def init_job_queue(db_path: Optional[str] = None) -> None:
    """
    Initialise the module-level job_queue singleton with the given db_path.
    Mutates the existing instance in-place so that blueprints that have already
    imported `job_queue` by name continue to reference the correct object.
    Called from create_app() before blueprints are registered.
    """
    job_queue._db_path = db_path
    job_queue._jobs = {}
    job_queue._lock = threading.Lock()

    if db_path is not None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        job_queue._init_db()
        job_queue._load_from_db()


def _find_spikee_exe() -> str:
    """
    Locate the spikee console-script executable installed in the same environment
    as the running Python interpreter. Falls back to 'spikee' on PATH.
    """
    scripts_dir = Path(sys.executable).parent
    for candidate in ("spikee", "spikee.exe"):
        full = scripts_dir / candidate
        if full.is_file():
            return str(full)
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
        if job_queue._db_path is not None:
            job_queue._db_update(job)
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
            with job.lock:
                job.status = "success" if proc.returncode == 0 else "failed"
                job.returncode = proc.returncode
            if job_queue._db_path is not None:
                job_queue._db_update(job)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def sse_stream(job: Job) -> Generator[str, None, None]:
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


# Module-level singleton — blueprints import this name directly.
# init_job_queue() may replace it before blueprints are registered.
job_queue = JobQueue()
