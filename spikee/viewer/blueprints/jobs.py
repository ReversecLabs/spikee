# spikee/viewer/blueprints/jobs.py
"""Jobs Blueprint — job list, detail, log polling, and cancel."""

from __future__ import annotations

from flask import Blueprint, Response, abort, jsonify, redirect, render_template, url_for

from spikee.utilities.files import extract_resource_name
from spikee.viewer.job_queue import job_queue

jobs_bp = Blueprint("jobs", __name__)


# ── Routes ────────────────────────────────────────────────────────────────────

@jobs_bp.route("/")
@jobs_bp.route("")
def index() -> str:
    """Render the jobs list page."""
    return render_template("jobs/list.html", jobs=job_queue.all())


@jobs_bp.route("/list-partial")
def list_partial() -> str:
    """HTMX polling target — returns tbody rows only."""
    return render_template("jobs/list_partial.html", jobs=job_queue.all())


@jobs_bp.route("/<job_id>")
def detail(job_id: str) -> str:
    """Render the detail page for a single job, including log output and result links."""
    job = job_queue.get(job_id)
    if job is None:
        abort(404, description=f"Job '{job_id}' not found.")

    # For generate jobs, scan the log for the output dataset filename
    dataset_filename = None
    if job.type == "generate":
        prefix = "Dataset generated and saved to "
        with job.lock:
            log_lines = list(job.log)
        for line in log_lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                # Normalise Windows backslashes; path is like datasets/foo.jsonl
                raw_path = stripped[len(prefix):].strip().replace("\\", "/")
                # Strip the leading "datasets/" directory component
                if raw_path.startswith("datasets/"):
                    dataset_filename = raw_path[len("datasets/"):]
                else:
                    dataset_filename = raw_path
                break

    # For test jobs, scan the log for all output results filenames (one per dataset)
    result_keys = []
    if job.type == "test":
        prefix = "[Done] Testing finished. Results saved to "
        with job.lock:
            log_lines = list(job.log)
        for line in log_lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                raw_path = stripped[len(prefix):].strip().replace("\\", "/")
                # raw_path is like "results/foo.jsonl" or "results/sub/foo.jsonl"
                parts = raw_path.split("/")
                if len(parts) >= 3:  # results/subfolder/file.jsonl
                    result_keys.append(f"{parts[-2]}/{extract_resource_name(parts[-1])}")
                elif len(parts) == 2:  # results/file.jsonl
                    result_keys.append(extract_resource_name(parts[-1]))

    return render_template("jobs/detail.html", job=job, dataset_filename=dataset_filename, result_keys=result_keys)


@jobs_bp.route("/<job_id>/cancel", methods=["POST"])
def cancel(job_id: str) -> Response:
    """Terminate a running job subprocess."""
    job = job_queue.get(job_id)
    if job is None:
        abort(404, description=f"Job '{job_id}' not found.")
    with job.lock:
        status = job.status
        process = job.process
    if status == "running" and process:
        process.terminate()
    return redirect(url_for("jobs.detail", job_id=job_id))


@jobs_bp.route("/<job_id>/rerun", methods=["POST"])
def rerun(job_id: str) -> Response:
    """Create and immediately start a new job with the same args."""
    from spikee.viewer.job_queue import spawn_job
    original = job_queue.get(job_id)
    if original is None:
        abort(404, description=f"Job '{job_id}' not found.")
    with original.lock:
        job_type = original.type
        job_name = original.name
        job_args = list(original.args)
    new_job = job_queue.create(type=job_type, name=job_name, args=job_args)
    spawn_job(new_job)
    return redirect(url_for("jobs.detail", job_id=new_job.id))


@jobs_bp.route("/<job_id>/log")
def log(job_id: str) -> Response:
    """Polling endpoint — returns current log lines and status as JSON."""
    job = job_queue.get(job_id)
    if job is None:
        abort(404)
    with job.lock:
        log_snapshot = list(job.log)
        status = job.status
    return jsonify({"status": status, "log": log_snapshot})


@jobs_bp.route("/<job_id>/stream")
def stream(job_id: str) -> Response:
    """SSE log stream — not yet implemented."""
    abort(501, description="SSE streaming is not yet implemented.")

