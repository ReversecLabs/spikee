# spikee/viewer/blueprints/jobs.py
"""Jobs Blueprint — Phase 2 complete. SSE stream + cancel backend wired in Phase 4b."""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, redirect, render_template, url_for

from spikee.viewer.job_queue import job_queue

jobs_bp = Blueprint("jobs", __name__)


# ── Routes ────────────────────────────────────────────────────────────────────

@jobs_bp.route("/")
@jobs_bp.route("")
def index():
    return render_template("jobs/list.html", jobs=job_queue.all())


@jobs_bp.route("/list-partial")
def list_partial():
    """HTMX polling target — returns tbody rows only."""
    return render_template("jobs/list_partial.html", jobs=job_queue.all())


@jobs_bp.route("/<job_id>")
def detail(job_id):
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

    return render_template("jobs/detail.html", job=job, dataset_filename=dataset_filename)


@jobs_bp.route("/<job_id>/cancel", methods=["POST"])
def cancel(job_id):
    """Phase 4b: terminate a running job subprocess."""
    job = job_queue.get(job_id)
    if job is None:
        abort(404, description=f"Job '{job_id}' not found.")
    if job.status == "running" and job.process:
        job.process.terminate()
    return redirect(url_for("jobs.detail", job_id=job_id))


@jobs_bp.route("/<job_id>/log")
def log(job_id):
    """Polling endpoint — returns current log lines and status as JSON."""
    job = job_queue.get(job_id)
    if job is None:
        abort(404)
    with job.lock:
        log_snapshot = list(job.log)
        status = job.status
    return jsonify({"status": status, "log": log_snapshot})


@jobs_bp.route("/<job_id>/stream")
def stream(job_id):
    """Phase 4b: SSE log stream. Not yet implemented."""
    abort(501, description="SSE streaming will be activated in Phase 4b.")

